# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Rafael Omodei and EasySlice Print contributors
"""Plan (draft) records, preview objects and the glue between records and the
core cut pipeline."""

import math

import bmesh
import bpy
from bpy.app.handlers import persistent
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from . import overlay
from .core import connectors, cutting, mesh_utils, section, surfaces
from .core.units import mm_to_bu
from .props import connector_props

PLAN_COLLECTION = "_ESP_Plan"
BACKUP_COLLECTION = "ESP_Backup"
BUILT_PREFIX = "ESP_Built_"
GREEN = (0.30, 0.90, 0.40, 0.40)
ORANGE = (1.00, 0.55, 0.15, 0.85)
PREFIX = {'STRAIGHT': "Plane Cut", 'CURVED': "Curve Cut", 'FREEHAND': "Freehand Cut"}

_suspend = 0


class ContactData:
    """What a cut tool produces for one contact (everything in world space)."""

    def __init__(self, kind):
        self.kind = kind
        self.verts = []
        self.faces = []
        self.points = []  # editable control points (curve / freehand)
        self.view_dir = Vector((0.0, 0.0, -1.0))
        self.depth_range = (0.0, 0.0)
        self.extend = 0.0
        self.margin = 0.0
        self.detail = surfaces.SURFACE_DETAIL  # spline samples per control point segment
        self.hit = None  # where the stroke touched the model
        self.through = None  # direction through the model at `hit`
        self.center_hint = None  # alternative: a point inside the part
        self.anchor = Vector((0.0, 0.0, 0.0))
        self.plane_co = None  # plane cuts: a point on the cut plane ...
        self.plane_no = None  # ... and its normal, so the section can be rebuilt
        self.span = None  # (axis, lo, hi): the stroke, measured from plane_co
        self.cutter = None  # what the boolean subtracts, when it is not the patch itself
        self.is_cut_face = False  # the patch is the real printed cut face, edge to edge
        self.grown = 0.0  # a plane section's outline sits this far past the model
        self.regions_skipped = 0  # regions the plane crosses that the stroke missed


# ----------------------------------------------------------------------------
# preferences / units
# ----------------------------------------------------------------------------
def prefs(context):
    try:
        return context.preferences.addons[__package__].preferences
    except (KeyError, AttributeError):
        return None


def pref(context, name, default):
    p = prefs(context)
    return getattr(p, name, default) if p is not None else default


def mm(context):
    return mm_to_bu(context.scene, prefs(context))


# ----------------------------------------------------------------------------
# collections
# ----------------------------------------------------------------------------
def get_or_create_collection(scene, name, hide_render=True):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
    if col.name not in scene.collection.children:
        scene.collection.children.link(col)
    col.hide_render = hide_render
    return col


def plan_collection(scene):
    return get_or_create_collection(scene, PLAN_COLLECTION)


def backup_collection(scene):
    return get_or_create_collection(scene, BACKUP_COLLECTION)


def built_collection(scene, base_name):
    return get_or_create_collection(scene, BUILT_PREFIX + base_name, hide_render=False)


def move_to_backup(scene, obj):
    homes = [c.name for c in obj.users_collection]
    for c in obj.users_collection:
        c.objects.unlink(obj)
    backup_collection(scene).objects.link(obj)
    obj["esp_backup_home"] = homes
    obj.hide_set(True)
    obj.hide_render = True


def restore_from_backup(scene, obj):
    homes = list(obj.get("esp_backup_home", []))
    bc = bpy.data.collections.get(BACKUP_COLLECTION)
    if not homes and (bc is None or obj.name not in bc.objects):
        # never stashed (an approved part, say): leave its collections alone
        obj.hide_set(False)
        obj.hide_viewport = False
        obj.hide_render = False
        return
    if bc is not None and obj.name in bc.objects:
        bc.objects.unlink(obj)
    linked = False
    for name in homes:
        col = bpy.data.collections.get(name)
        if col is None and name == scene.collection.name:
            col = scene.collection
        if col is not None and obj.name not in col.objects:
            col.objects.link(obj)
            linked = True
    if not linked and obj.name not in scene.collection.objects:
        scene.collection.objects.link(obj)
    if "esp_backup_home" in obj:
        del obj["esp_backup_home"]
    obj.hide_set(False)
    obj.hide_viewport = False
    obj.hide_render = False


def ensure_evaluable(obj):
    """Make sure a (possibly hidden) object is evaluated by the depsgraph."""
    state = (obj.hide_viewport, obj.hide_get())
    obj.hide_viewport = False
    obj.hide_set(False)
    return state


def restore_visibility(obj, state):
    obj.hide_viewport = state[0]
    obj.hide_set(state[1])


def resolve_target(context):
    """The mesh the cut tools act on: active mesh object (never a preview), else the plan's base."""
    obj = context.active_object
    if obj is not None and obj.type == 'MESH' and "esp_preview" not in obj and obj.visible_get():
        return obj
    base = base_of(context.scene.esp)
    if base is not None and base.type == 'MESH':
        return base
    return None


BUILD_STASH = "esp_stashed_by_build"


def stash_built_source(scene, obj):
    """A source a build has cut goes to the backup, marked as the build's to give back."""
    move_to_backup(scene, obj)
    obj[BUILD_STASH] = True


def build_stashed(context):
    """Every source the current build put in the backup - also those of cuts since unticked."""
    col = bpy.data.collections.get(BACKUP_COLLECTION)
    if col is None:
        return []
    return [o for o in col.objects if o.get(BUILD_STASH)]


def give_back_sources(context):
    """Back to how the plan was before the build: every source in view again. -> the sources."""
    sources = plan_sources(context)
    for o in build_stashed(context):
        if o not in sources:
            sources.append(o)
    for o in sources:
        restore_from_backup(context.scene, o)
        if BUILD_STASH in o:
            del o[BUILD_STASH]
    return sources


# ----------------------------------------------------------------------------
# object references
# ----------------------------------------------------------------------------
# The plan holds the objects it works with by pointer (see `props.ESP_CutRecord`), so
# a rename in the outliner, of the model or of a preview, changes nothing. A pointer
# also keeps its object in the file: deleting one in the viewport only unlinks it from
# its collections. So a referenced object counts only while it is in a collection, and
# `release_deleted` lets go of those deleted for Blender to drop them.
def alive(obj):
    """`obj` if it is still in the scene's collections, else None."""
    if obj is None:
        return None
    try:
        return obj if obj.users_collection else None
    except ReferenceError:
        return None


def collection_alive(col):
    """`col` if a scene still holds it, else None (the built parts' collection)."""
    if col is None:
        return None
    try:
        for sc in bpy.data.scenes:
            if col in sc.collection.children_recursive:
                return col
    except ReferenceError:
        pass
    return None


def base_of(settings):
    return alive(settings.base_obj)


def built_col_of(settings):
    return collection_alive(settings.built_col)


def target_of(rec):
    return alive(rec.target_obj)


def record_target(context, rec):
    """The object `rec` was drawn on, else the plan's base."""
    return target_of(rec) or base_of(context.scene.esp)


def surface_of(rec, i):
    return alive(getattr(rec, f"surface_obj_{'ab'[i]}"))


def pin_of(rec, i):
    return alive(getattr(rec, f"pin_obj_{'ab'[i]}"))


def extra_pin_of(e):
    return alive(e.pin_obj)


REF_FIELDS = ("target_obj", "surface_obj_a", "surface_obj_b", "pin_obj_a", "pin_obj_b")


def _ref_fields(settings):
    """(owner, pointer attribute) of every object reference in the plan."""
    yield settings, "base_obj"
    for rec in settings.cuts:
        for f in REF_FIELDS:
            yield rec, f
        for e in rec.extras:
            yield e, "pin_obj"


def release_deleted(settings):
    """Let go of every object deleted from the scene, so the file does not keep it."""
    for owner, attr in _ref_fields(settings):
        obj = getattr(owner, attr)
        if obj is not None and alive(obj) is None:
            setattr(owner, attr, None)
    col = settings.built_col
    if col is not None and collection_alive(col) is None:
        settings.built_col = None


# Files saved before the pointers kept a name in a string property; the name of each
# one, and the pointer it becomes. The strings stay in the file as plain ID properties
# (nothing declares them any more), so they are read, and dropped, from there.
LEGACY_RECORD = {
    "target": "target_obj",
    "surface_a": "surface_obj_a",
    "surface_b": "surface_obj_b",
    "pin_a": "pin_obj_a",
    "pin_b": "pin_obj_b",
}
# the custom properties the name-based plan tagged objects with to follow a rename
LEGACY_OBJECT_TAGS = ("esp_ref_name", "esp_ref_uid")


def _take_legacy(owner, key):
    """Pop the old string property `key` off `owner`. -> the name, or None"""
    if key not in owner:
        return None
    value = owner.get(key)
    del owner[key]
    return value if isinstance(value, str) and value else None


def _adopt(owner, attr, data, name):
    if name and getattr(owner, attr) is None:
        found = data.get(name)
        if found is not None:
            setattr(owner, attr, found)


def migrate_scene(scene):
    """Turn the object names a file saved before the pointers into pointers. -> changed"""
    settings = getattr(scene, "esp", None)
    if settings is None:
        return False
    objects = bpy.data.objects
    changed = False
    for key, attr, data in (
        ("base_object", "base_obj", objects),
        ("built_collection", "built_col", bpy.data.collections),
    ):
        if key in settings:
            _adopt(settings, attr, data, _take_legacy(settings, key))
            changed = True
    for rec in settings.cuts:
        for key, attr in LEGACY_RECORD.items():
            if key in rec:
                _adopt(rec, attr, objects, _take_legacy(rec, key))
                changed = True
        for e in rec.extras:
            if "pin" in e:
                _adopt(e, "pin_obj", objects, _take_legacy(e, "pin"))
                changed = True
    return changed


def migrate_file():
    """Bring every scene of the open file to object pointers. -> changed"""
    changed = False
    for scene in bpy.data.scenes:
        changed = migrate_scene(scene) or changed
    for obj in bpy.data.objects:
        for key in LEGACY_OBJECT_TAGS:
            if key in obj:
                del obj[key]
                changed = True
    return changed


@persistent
def load_post_handler(_filepath):
    migrate_file()
    schedule_walls()


@persistent
def undo_redo_handler(scene, *_args):
    """An undo brings back the walls stored before the step, measured on older geometry."""
    schedule_walls()


def _migrate_after_register():
    """The add-on enabled with a file already open: no load handler runs for it."""
    try:
        migrate_file()
    except Exception:
        pass
    return None


def plan_sources(context):
    """Every live model the plan cuts into, the base first.

    Usually that is just the base. Once a build is approved the plan is empty and
    new cuts land on the parts themselves, so each part a record was drawn on is a
    source of its own and the next build must cut *it*, not the untouched original.
    """
    settings = context.scene.esp
    out = []
    base = base_of(settings)
    if base is not None and base.type == 'MESH':
        out.append(base)
    for rec in settings.cuts:
        if not rec.enabled:
            continue
        obj = target_of(rec)
        if obj is not None and obj.type == 'MESH' and obj not in out:
            out.append(obj)
    return out


# ----------------------------------------------------------------------------
# cross sections (plane cuts)
# ----------------------------------------------------------------------------
SECTION_MARGIN = 0.01  # of the section's own bounding box


def straight_section(context, target, plane_co, plane_no, span=None, reference=None):
    """The model's own cross section on a plane, grown by a hair. None if it fails.

    This is what a plane cut uses as its surface: the area the print will actually be
    cut through, taken from the model's geometry, instead of a rectangle drawn around
    the mouse stroke. `span` is the stroke - every region of the model the line ran
    across is cut, and only those, so one leg goes and the other stays.

    An open or non manifold mesh may not section at all; the caller falls back to the
    old rectangle in that case rather than leaving the user without a cut.
    """
    if target is None or target.type != 'MESH':
        return None
    state = ensure_evaluable(target)
    try:
        depsgraph = context.evaluated_depsgraph_get()
        res = section.plane_section(target, plane_co, plane_no, span=span, reference=reference, depsgraph=depsgraph)
    except Exception:
        return None
    finally:
        restore_visibility(target, state)
    if res is None:
        return None
    verts, faces = res
    margin = max(section.section_extent(verts) * SECTION_MARGIN, mesh_utils.object_world_diagonal(target) * 0.002)
    grown_v, grown_f = section.grow_section(verts, faces, margin, plane_no)
    return section.SectionResult(grown_v, grown_f, islands=res.islands, kept=res.kept, cutter=res.cutter, grown=margin)


def fill_straight_contact(data, patch, span):
    """Put a section into a contact: the face the user sees *and* the quad the boolean gets.

    Both travel on the same `SectionResult`, and they are not interchangeable. Copying
    only the first leaves the boolean subtracting the section itself, whose rim lies on
    the model tangentially - the exact solver returns the mesh nearly untouched, so the
    cut reports success, names two parts and separates nothing.
    """
    data.verts, data.faces = patch
    data.cutter = patch.cutter
    data.span = span
    data.is_cut_face = True
    data.grown = getattr(patch, "grown", 0.0)
    data.regions_skipped = patch.islands - patch.kept


# ----------------------------------------------------------------------------
# cut surfaces that follow the model (curve cuts)
# ----------------------------------------------------------------------------
def column_runs(context, target, columns, direction, ref):
    """Per column, the runs of material along `direction`, measured from `ref`.

    Every crossing along the ray is collected, so a closed mesh gives entry/exit pairs:
    one run for a solid limb, two for a hollow one. -> list of lists of (lo, hi).
    """
    diag = mesh_utils.object_world_diagonal(target)
    d = Vector(direction).normalized()
    eps = diag * 1e-4
    state = ensure_evaluable(target)
    out = []
    try:
        depsgraph = context.evaluated_depsgraph_get()
        for p in columns:
            start = Vector(p) - d * diag * 2.0
            base = (start - Vector(ref)).dot(d)
            hits = mesh_utils.object_ray_hits(target, start, d, eps, depsgraph, max_dist=diag * 4.0)
            out.append([(base + hits[i], base + hits[i + 1]) for i in range(0, len(hits) - 1, 2)])
    finally:
        restore_visibility(target, state)
    return out


def ribbon_surfaces(context, target, points, view, extend, detail, pad_factor=0.02):
    """The pair of surfaces a curve cut needs. -> (preview, cutter, band) or None.

    `preview` follows the model: every column of the ribbon runs exactly as deep as the
    material under it, and a column with nothing under it is dropped, so a stroke drawn
    past the end of a limb stops at the limb. `cutter` is what the boolean subtracts:
    the same ribbon at one depth for its whole length, reaching past the model on both
    sides - a rim in free space, for the same reason `section.clip_rect` exists.

    `band` is the run of material the stroke is standing on, widened half way into the
    empty gaps beside it. That is what keeps a curve drawn across the near leg from
    reaching through the far one, which one depth range taken from the furthest hit
    anywhere under the stroke could not do.
    """
    if target is None or target.type != 'MESH':
        return None
    d = Vector(view).normalized()
    try:
        columns = surfaces.ribbon_samples(points, extend, detail)
    except ValueError:
        return None
    ref = sum((Vector(p) for p in points), Vector((0.0, 0.0, 0.0))) / len(points)
    try:
        runs = column_runs(context, target, columns, d, ref)
    except Exception:
        return None
    flat = [r for col in runs for r in col]
    if not flat:
        return None
    diag = mesh_utils.object_world_diagonal(target)
    ts = sorted((Vector(p) - ref).dot(d) for p in points)
    anchor = ts[len(ts) // 2]
    band = section.band_around(flat, anchor, diag * pad_factor)
    if band is None:
        return None
    # `column_runs` measures along the ray from one shared reference; a ribbon column
    # is extruded from its own point, so every range has to be rebased onto it
    trimmed, flat_band = [], []
    for col, p in zip(runs, columns):
        offset = (Vector(p) - ref).dot(d)
        flat_band.append((band[0] - offset, band[1] - offset))
        within = [(max(lo, band[0]), min(hi, band[1])) for lo, hi in col if hi > band[0] and lo < band[1]]
        # one span from the first entry to the last exit: a column crossing a hollow
        # limb still has to hold the two halves of the surface together
        if within:
            trimmed.append((min(a for a, _b in within) - offset, max(b for _a, b in within) - offset))
        else:
            trimmed.append(None)
    try:
        preview = surfaces.ribbon_patch(points, d, 0.0, extend, detail=detail, depth_ranges=trimmed)
    except ValueError:
        return None
    cutter = surfaces.ribbon_patch(points, d, 0.0, extend, detail=detail, depth_ranges=flat_band)
    return preview, cutter, band


# ----------------------------------------------------------------------------
# preview objects
# ----------------------------------------------------------------------------
def _material(name, rgba):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = False
    mat.diffuse_color = rgba
    try:
        mat.surface_render_method = 'BLENDED'
    except Exception:
        pass
    return mat


def _preview_object(scene, name, mesh, kind, color, mat_name):
    obj = bpy.data.objects.new(name, mesh)
    plan_collection(scene).objects.link(obj)
    obj.display_type = 'SOLID'
    obj.show_in_front = True
    obj.hide_render = True
    obj.color = color
    obj["esp_preview"] = kind
    mesh.materials.append(_material(mat_name, color))
    return obj


def surface_origin_point(context, data, target):
    """Where the origin of a generated cut surface goes.

    By default the surface's own median point, so R / S on it pivot on the surface
    instead of on some point out in the scene. 'OBJECT' puts it on the origin of the
    model being cut, which is what you want when several cut surfaces should share
    one pivot.
    """
    if context.scene.esp.surface_origin == 'OBJECT' and target is not None:
        return target.matrix_world.translation.copy()
    if data.verts:
        return surfaces.patch_center(data.verts)
    return Vector((0.0, 0.0, 0.0))


def create_surface_object(context, name, data, target=None):
    """The patch arrives in world space; it is stored local to `origin` so the preview
    object carries a real origin of its own (mesh AND the editable control points)."""
    origin = surface_origin_point(context, data, target)
    me = mesh_utils.mesh_from_pydata(name, [Vector(v) - origin for v in data.verts], data.faces)
    obj = _preview_object(context.scene, name, me, 'surface', GREEN, "ESP_Preview_Surface")
    obj.matrix_world = Matrix.Translation(origin)
    obj["esp_kind"] = data.kind
    flat_pts = [c for p in data.points for c in (Vector(p) - origin)]
    obj["esp_points"] = flat_pts
    obj["esp_points_orig"] = list(flat_pts)
    obj["esp_view"] = list(data.view_dir)
    obj["esp_depth"] = list(data.depth_range)
    obj["esp_extend"] = data.extend
    obj["esp_margin"] = data.margin
    obj["esp_detail"] = data.detail
    if data.plane_no is not None and data.plane_co is not None:
        # stored local, so G / R / S on the preview move the plane with the object and
        # the section can be taken again from wherever the user parked it
        obj["esp_plane_co"] = list(Vector(data.plane_co) - origin)
        obj["esp_plane_no"] = list(data.plane_no)
        obj["esp_ref"] = list(Vector(data.anchor) - origin)
        obj["esp_cut_face"] = data.is_cut_face
        obj["esp_skipped"] = data.regions_skipped
        obj["esp_grown"] = data.grown
        if data.span is not None:
            obj["esp_span_u"] = list(data.span[0])
            obj["esp_span"] = [data.span[1], data.span[2]]
    if data.hit is not None:
        obj["esp_hit"] = list(data.hit)
        obj["esp_through"] = list(data.through)
    if data.center_hint is not None:
        obj["esp_center_hint"] = list(data.center_hint)
    obj["esp_cut_face"] = data.is_cut_face
    set_cutter(obj, [Vector(p) - origin for p in data.cutter[0]] if data.cutter else None, data.cutter)
    obj["esp_mw"] = [v for row in obj.matrix_world for v in row]
    return obj


def surface_points(obj):
    flat = list(obj.get("esp_points", []))
    return [Vector(flat[i : i + 3]) for i in range(0, len(flat) - 2, 3)]


def set_surface_points(obj, points):
    obj["esp_points"] = [c for p in points for c in p]


def _clear_point(target, p, margin, depsgraph):
    """`p` moved out of the model until it clears the surface by `margin`."""
    p = Vector(p)
    # lifting a point clear of the nearest surface can drop it into a second one (a
    # loop round a thigh has the other leg right there), so look again
    for _ in range(3):
        found, loc, nor, depth = mesh_utils.object_surface_depth(target, p, depsgraph)
        if not found or depth >= margin:
            break
        p = loc + nor * margin
    return p


def clear_of_model(context, target, points, margin):
    """Move each point out of the model until it clears the surface by `margin`.

    The tool no longer does this to a rim: a point the user drew is where the cut goes,
    and lifting it off the surface - along one normal at a time, which in a crease
    bounces the point between the two faces that meet there and carries it millimetres
    out of the crease - is what put the cut face off the traced line. The clearance is
    a skirt on the cutter now (`skirt_ring`, which lifts skirt vertices with the same
    `_clear_point`). This is kept as the pipeline it replaced, so a test can measure
    the two against each other.
    """
    if target is None or context is None or margin <= 0.0:
        return [Vector(p) for p in points]
    state = ensure_evaluable(target)
    try:
        depsgraph = context.evaluated_depsgraph_get()
        return [_clear_point(target, p, margin, depsgraph) for p in points]
    finally:
        restore_visibility(target, state)


def settle_on_model(context, target, points, detail):
    """The rim: the drawn points, subdivided, and put onto the surface between them.

    `surfaces.loop_boundary` with the model to hand. The rim is straight between two
    drawn points, and where a stroke jumps across a junction - off a leg onto the body
    in one mouse move - that straight segment runs through the material. The drawn
    points are on the surface and never move; a point between them that is inside the
    model is cast out along the two drawn normals' average, the way the mouse ray that
    would have sampled it went, so the whole segment lands on one side of the limb
    instead of the closest wall, which flips from side to side across it. A segment in
    the air is left alone.
    """
    if target is None or context is None:
        return surfaces.loop_boundary(points, detail)
    pts = surfaces.decimate_polyline([Vector(p) for p in points], surfaces.MAX_LOOP_SAMPLES)
    detail = max(1, min(detail, surfaces.MAX_LOOP_SAMPLES // len(pts)))
    state = ensure_evaluable(target)
    try:
        depsgraph = context.evaluated_depsgraph_get()
        diag = mesh_utils.object_world_diagonal(target)
        hair = diag * 1e-5  # float noise on a point that is on the surface
        normals = []
        for p in pts:
            found, _loc, nor, _depth = mesh_utils.object_surface_depth(target, p, depsgraph)
            normals.append(nor if found else Vector((0.0, 0.0, 1.0)))
        m = len(pts)
        out = []
        for i in range(m):
            a, b = pts[i], pts[(i + 1) % m]
            axis = normals[i] + normals[(i + 1) % m]
            if axis.length < 1e-6:
                axis = normals[i]
            axis.normalize()
            for k in range(detail):
                q = a.lerp(b, k / detail)
                if k > 0:
                    found, loc, _nor, depth = mesh_utils.object_surface_depth(target, q, depsgraph)
                    if found and depth < -hair:
                        hits = []
                        for d in (axis, -axis):
                            hit, hloc, _n, dist = mesh_utils.object_ray_cast(target, q, d, depsgraph, max_dist=diag)
                            if hit:
                                hits.append((dist, hloc))
                        q = min(hits)[1] if hits else loc
                out.append(q)
        out = surfaces.dedupe_polyline(out, 1e-9)
        if len(out) < 3:
            raise ValueError("loop is degenerate")
        return out
    finally:
        restore_visibility(target, state)


def _clearance(target, q, depsgraph):
    """Signed distance from `q` to the model: positive outside, negative inside."""
    found, loc, _nor, depth = mesh_utils.object_surface_depth(target, q, depsgraph)
    if not found:
        return 1e30
    dist = (Vector(q) - loc).length
    return dist if depth >= 0.0 else -dist


def _march_clear(target, p, d, margin, depsgraph, want):
    """`p` moved along `d`, at least by |d| and further until it clears the model by `want`.

    Growing the offset along its own direction is what keeps the skirt from stepping
    sideways: a sideways lift is what `_clear_point` does, and two neighbours lifted
    sideways along different surfaces are how a skirt folds.
    """
    length = d.length
    if length < 1e-9:
        return Vector(p)
    u = d / length
    best, best_clear = Vector(p) + d, -1e30
    t = length
    step = margin * 0.125
    while t <= margin * 3.0 + 1e-9:
        q = Vector(p) + u * t
        clear = _clearance(target, q, depsgraph)
        if clear >= want:
            return q
        if clear > best_clear:
            best, best_clear = q, clear
        t += step
    return best


def skirt_ring(context, target, verts, faces, rim, margin):
    """Where the cutter continues past each rim vertex: `margin` out, clear of the model.

    The membrane's rim is on the surface, and the cutter has to leave it. Each rim
    vertex gets one vertex beyond it, and the way out is tried in order: straight on in
    the membrane's own surface (the cut continued past the model - right on a limb,
    right along a crease, where it leaves the wedge of air between the two faces);
    then that direction leaned towards the surface normal; then the normal itself,
    which is what the rim used to be pushed along. A candidate is lifted clear of the
    model and then checked from outside, with a ray cast back at the rim vertex: if the
    ray meets the model before it gets there, the skirt face would run through material
    - through the hem a crease is traced under - and the next way out is tried.

    Chosen one vertex at a time, the offsets jump between neighbours: where the rim
    passes from one surface onto another, and from one facet of the model to the next,
    since the lift follows the facet's normal. A jump larger than the rim spacing folds
    the skirt, so the offsets are smoothed along the rim (`surfaces.unfold_skirt`),
    grown along their own direction until they clear the model again, and the few that
    still break the rule are smoothed locally. The rim is not touched by any of it.
    """
    out = surfaces.rim_outward(verts, faces, rim)
    pts = [Vector(verts[i]) for i in range(rim)]
    if target is None or context is None or margin <= 0.0:
        return [pts[i] + out[i] * margin for i in range(rim)]
    state = ensure_evaluable(target)
    try:
        depsgraph = context.evaluated_depsgraph_get()
        tol = margin * 0.3  # a rim vertex is on the surface, and the model is faceted
        ring = []
        for i in range(rim):
            p = pts[i]
            found, _loc, nor, _depth = mesh_utils.object_surface_depth(target, p, depsgraph)
            ways = [out[i]]
            if found:
                lean = out[i] + nor
                if lean.length > 1e-6:
                    ways.append(lean.normalized())
                ways.append(nor)
            chosen = None
            for d in ways:
                if d.length < 1e-9:
                    continue
                q = _clear_point(target, p + d * margin, margin, depsgraph)
                back = p - q
                dist = back.length
                if dist <= 1e-9:
                    continue
                chosen = q
                hit, _l, _n, _d = mesh_utils.object_ray_cast(
                    target, q, back / dist, depsgraph, max_dist=max(dist - tol, 0.0)
                )
                if not hit:
                    break
            ring.append(chosen if chosen is not None else p + out[i] * margin)
        ring = surfaces.unfold_skirt(pts, ring, passes=200)
        want = margin * 0.5
        ring = [_march_clear(target, p, q - p, margin, depsgraph, want) for p, q in zip(pts, ring)]
        return surfaces.unfold_skirt(pts, ring, local=True)
    finally:
        restore_visibility(target, state)


def loop_surface(context, target, points, detail, margin, passes=None):
    """The freehand membrane and its cutter. -> (verts, faces, cutter)

    The membrane ends exactly on the drawn loop: that is the printed cut face, and the
    outline the connector is measured against. Between two drawn points the rim is
    put onto the surface where it would otherwise run through the model
    (`settle_on_model`). `cutter` is the same surface continued `margin` past the rim
    (`skirt_ring` + `surfaces.with_skirt`), which is what the boolean subtracts; it is
    None when there is no clearance to add.
    """
    boundary = settle_on_model(context, target, points, detail)
    verts, faces = surfaces.loop_patch(points, detail=detail, passes=passes, boundary=boundary)
    rim = len(boundary)
    if margin <= 0.0:
        return verts, faces, None
    ring = skirt_ring(context, target, verts, faces, rim, margin)
    return verts, faces, surfaces.with_skirt(verts, faces, rim, ring)


# A freehand cutter has to stand clear of the model by a real distance, not a hair: where
# the slab wall leaves the surface almost tangentially the boolean welds the two halves
# back together, and the part comes out in one piece with the cut face already carved
# into it.
#
# The clearance used to be paid for in traced detail: the rim itself was pushed out by
# it, so every millimetre of it was a millimetre the cut face had moved off the line.
# The rim stays on the line now and the clearance is a skirt past it, so the multiple
# is set only by what fails first - and that is not the boolean: a loop round the top
# of a thigh separates the model cleanly at 0.8 mm, but every loose piece then votes to
# the same side of a membrane that local, and the cut is reported as one that never
# split the part. Measured over loops round a waist, a thigh, a shin, a hip junction and
# a neck, classification holds from about 1.2 mm on that worst case; this multiple
# leaves a factor of two on it. Raise Surface Margin if a loop still fails to split.
LOOP_MARGIN_SCALE = 1.5


def loop_margin(context, src, points, diag):
    """How far a freehand cutter has to reach past its rim, clear of the model.

    Scaled to the loop, so the same slider means the same thing on a 20 mm figure and
    on a 200 mm one, and floored at the kerf: a skirt inside the kerf is no skirt at all.
    """
    u = mm(context)
    radius = surfaces.polyline_length(points, closed=True) / (2.0 * math.pi)
    return max(
        radius * context.scene.esp.surface_margin * LOOP_MARGIN_SCALE,
        diag * 0.002,
        src.cut_gap_mm * u * 3.0,
        0.6 * u,
    )


def rebuild_surface(obj, draft=False, context=None, target=None):
    """Regenerate the patch from its control points.

    `draft` halves the spline resolution and cuts the membrane relaxation short; it is
    what runs while a control point is being dragged, so the preview keeps up with the
    mouse. The full surface is rebuilt as soon as the drag ends.
    """
    kind = obj.get("esp_kind", 'STRAIGHT')
    pts = surface_points(obj)
    detail = int(obj.get("esp_detail", surfaces.SURFACE_DETAIL))
    if draft:
        detail = max(1, detail // 2)
    if kind == 'CURVED' and len(pts) >= 2:
        # the control points are local to the object; the stored view direction is world
        mw = obj.matrix_basis
        world_view = Vector(obj["esp_view"]).normalized()
        built = None
        if context is not None and target is not None:
            built = ribbon_surfaces(context, target, [mw @ p for p in pts], world_view, obj["esp_extend"], detail)
        if built is not None:
            preview, cutter, band = built
            inv = mw.inverted_safe()
            verts = [inv @ v for v in preview[0]]
            faces = preview[1]
            obj["esp_depth"] = list(band)
            set_cutter(obj, [inv @ Vector(p) for p in cutter[0]], cutter)
        else:
            # no model to measure against: one depth for the whole ribbon, as before
            view = (mw.to_3x3().inverted_safe() @ world_view).normalized()
            verts, faces = surfaces.ribbon_patch(
                pts, view, 0.0, obj["esp_extend"], depth_range=tuple(obj["esp_depth"]), detail=detail
            )
            set_cutter(obj, None, None)
    elif kind == 'FREEHAND' and len(pts) >= 3:
        # the skirt is measured against the model on the full rebuild; a drag keeps up
        # with the mouse instead, and the drag ends in a full rebuild anyway
        margin = 0.0 if draft else float(obj.get("esp_margin", 0.0))
        world = [obj.matrix_basis @ p for p in pts]
        verts, faces, cutter = loop_surface(context, target, world, detail, margin, passes=6 if draft else None)
        inv = obj.matrix_basis.inverted_safe()
        verts = [inv @ Vector(v) for v in verts]
        if not draft:
            set_cutter(obj, [inv @ Vector(p) for p in cutter[0]] if cutter else None, cutter)
    elif kind == 'STRAIGHT':
        patch = rebuilt_section(obj, context, target)
        if patch is None:
            return
        verts, faces = patch
    else:
        return
    me = obj.data
    me.clear_geometry()
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.update()


def rebuilt_section(obj, context, target):
    """Section the model again on the preview's current plane. Local space, or None.

    A plane cut carries no control points; what it carries is the plane itself, so a
    moved or rotated preview re-slices the model rather than dragging a stale outline
    around. Scaling a section has no effect - the section is the model's own area, and
    there is nothing to scale it to.
    """
    if context is None or target is None or "esp_plane_no" not in obj:
        return None
    mw = obj.matrix_basis
    rot = mw.to_3x3()
    no = (rot.inverted_safe().transposed() @ Vector(obj["esp_plane_no"])).normalized()
    co = mw @ Vector(obj["esp_plane_co"])
    ref = mw @ Vector(obj["esp_ref"]) if "esp_ref" in obj else None
    span = None
    if "esp_span_u" in obj:
        lo, hi = obj["esp_span"]
        span = ((rot @ Vector(obj["esp_span_u"])).normalized(), lo, hi)
    patch = straight_section(context, target, co, no, span, ref)
    if patch is None:
        return None
    obj["esp_skipped"] = patch.islands - patch.kept
    obj["esp_grown"] = getattr(patch, "grown", 0.0)
    inv = mw.inverted_safe()
    cutter = patch.cutter
    set_cutter(obj, [inv @ Vector(p) for p in cutter[0]] if cutter else None, cutter)
    verts, faces = patch
    return [inv @ v for v in verts], faces


def set_cutter(obj, local_verts, cutter):
    """Store what the boolean should subtract, in the preview's own space.

    Only a surface whose own rim would sit on the model needs one: a plane cut's
    section, a curve cut's silhouette hugging ribbon, and a freehand loop, whose
    membrane ends on the line drawn on the surface and whose cutter is that membrane
    with a skirt past the rim. Faces are stored triangulated, whatever they were.
    """
    if not local_verts or cutter is None:
        for key in ("esp_cutter", "esp_cutter_f", "esp_cutter_n"):
            if key in obj:
                del obj[key]
        return
    tris = []
    for f in cutter[1]:
        for k in range(1, len(f) - 1):
            tris.extend((f[0], f[k], f[k + 1]))
    obj["esp_cutter"] = [c for p in local_verts for c in p]
    obj["esp_cutter_f"] = tris
    obj["esp_cutter_n"] = 3


def cutter_world_patch(obj):
    """(verts, faces) the boolean should subtract, or the preview's own mesh.

    A plane cut hands the boolean a quad and a curve cut a ribbon at one depth, both
    reaching past the model instead of stopping on it; see `section.clip_rect` for why
    a rim on the surface is the thing to avoid. A freehand loop hands it the membrane
    plus its skirt. A surface built by the fallbacks subtracts exactly what it shows.
    """
    flat_co = list(obj.get("esp_cutter", []))
    flat_f = list(obj.get("esp_cutter_f", []))
    n = int(obj.get("esp_cutter_n", 4))  # plans saved before the skirt stored quads
    if len(flat_co) < 9 or len(flat_f) < n:
        return surface_world_patch(obj)
    mw = obj.matrix_basis
    verts = [mw @ Vector(flat_co[i : i + 3]) for i in range(0, len(flat_co) - 2, 3)]
    faces = [tuple(flat_f[i : i + n]) for i in range(0, len(flat_f) - n + 1, n)]
    return verts, faces


def surface_world_patch(obj):
    mw = obj.matrix_basis  # previews are never parented; basis is always current
    verts = [mw @ v.co for v in obj.data.vertices]
    faces = [tuple(p.vertices) for p in obj.data.polygons]
    return verts, faces


def layout_key(offsets):
    """Rounded, flat form of a layout - what a pin stores and is compared by."""
    return [round(c, 5) for o in (offsets or [(0.0, 0.0)]) for c in o]


def _store_layout(obj, layout):
    obj["esp_layout"] = layout_key(layout.offsets)
    obj["esp_layout_count"] = len(layout.offsets)
    obj["esp_layout_scale"] = layout.scale


def create_pin_object(context, name, shape_id, matrix, layout=None):
    layout = layout or PinLayout()
    shape, custom = connectors.resolve_shape(shape_id)
    bm = connectors.unit_layout_bmesh(shape, custom, layout.offsets)
    me = mesh_utils.bmesh_to_mesh(bm, name)
    bm.free()
    obj = _preview_object(context.scene, name, me, 'pin', ORANGE, "ESP_Preview_Pin")
    obj.matrix_world = matrix
    obj["esp_shape"] = shape_id
    _store_layout(obj, layout)
    return obj


def update_pin_mesh(obj, shape_id, layout=None):
    layout = layout or PinLayout()
    shape, custom = connectors.resolve_shape(shape_id)
    bm = connectors.unit_layout_bmesh(shape, custom, layout.offsets)
    me = mesh_utils.bmesh_to_mesh(bm, obj.name)
    bm.free()
    me.materials.append(_material("ESP_Preview_Pin", ORANGE))
    old = obj.data
    obj.data = me
    mesh_utils.remove_mesh(old)
    obj["esp_shape"] = shape_id
    _store_layout(obj, layout)


def pin_needs_rebuild(obj, shape_id, layout):
    """A new shape or a new layout replaces the preview mesh (and any edit-mode reshape)."""
    if obj.get("esp_shape") != shape_id:
        return True
    stored = obj.get("esp_layout")
    old = list(stored) if stored is not None else [0.0, 0.0]
    new = layout_key(layout.offsets)
    return len(old) != len(new) or any(abs(a - b) > 1e-4 for a, b in zip(old, new))


def pin_unit_mesh(pin):
    """The connector geometry to build, taken from the preview pin itself.

    Whatever the user did to the pin in edit mode - reshaped it, scaled it, added a
    second lobe - is in this mesh and nowhere else, so the build has to read it here
    instead of regenerating the shape from its name. Only a pin left with nothing the
    boolean could use falls back to the stored shape.
    """
    if pin is None or pin.type != 'MESH' or pin.data is None:
        return None
    if not pin.data.polygons:
        return None
    return pin.data


def flat(m):
    return [m[i][j] for i in range(4) for j in range(4)]


def unflat(vals):
    return Matrix([[vals[i * 4 + j] for j in range(4)] for i in range(4)])


# ----------------------------------------------------------------------------
# frames & sizes
# ----------------------------------------------------------------------------
_bvh_cache = {}


def _model_bvh(target, depsgraph):
    """BVH of the evaluated model, kept while the model is unchanged.

    Every change to a connector setting lays the pins out again, and on a dense model
    building the tree each time was most of the wait - a width slider lagged by
    seconds. The key is the object, its mesh size and a few vertices: an edit to the
    model makes a new one.
    """
    ev = target.evaluated_get(depsgraph)
    me = ev.data
    n = len(me.vertices)
    if n == 0:
        return None
    probe = tuple(round(c, 6) for i in (0, n // 2, n - 1) for c in me.vertices[i].co)
    key = (getattr(target, "session_uid", target.name), n, len(me.polygons), probe)
    bvh = _bvh_cache.get(key)
    if bvh is None:
        try:
            bvh = BVHTree.FromObject(ev, depsgraph)
        except Exception:
            return None
        if len(_bvh_cache) >= 4:
            _bvh_cache.clear()
        _bvh_cache[key] = bvh
    return bvh


def material_clearance(context, target):
    """A `clearance(p)` for the section helpers: distance from `p` to the model's surface.

    Negative when `p` is not inside the material. The target has to be evaluable while
    it is used (see `ensure_evaluable`). None when there is no mesh to measure against.
    One BVH of the evaluated model is built up front: the section helpers ask for
    thousands of points, and a dense model would otherwise rebuild it for every one.
    """
    if target is None or target.type != 'MESH':
        return None
    depsgraph = context.evaluated_depsgraph_get()
    bvh = _model_bvh(target, depsgraph)
    if bvh is None:
        return None
    mw = target.matrix_world.copy()
    inv = mw.inverted_safe()
    nmat = inv.to_3x3().transposed()

    def clearance(p):
        p = Vector(p)
        loc, nor, _i, _d = bvh.find_nearest(inv @ p)
        if loc is None:
            return 1e30
        loc_w = mw @ loc
        if (p - loc_w).dot(nmat @ nor) > 0.0:
            return -1.0
        return (p - loc_w).length

    return clearance


def contact_frame(
    context, target, verts, faces, hit=None, through=None, center_hint=None, is_cut_face=False, grown=0.0
):
    """-> (centre, normal, inscribed diameter) for the connector on this cut surface.

    When the surface is the real printed cut face - a plane section, a ribbon trimmed
    to the model, a freehand loop drawn on it - the answer is exact and cheap: the
    largest circle that fits inside it. A surface that covers more than the printed
    face (a fallback rectangle, an untrimmed ribbon) says nothing about where the
    material is, and the ray based estimate is still the best available.

    A plane section's outline is pushed `grown` past the model so the boolean clears
    it; the circle is measured against the section less that much, or a pin sized from
    it breaks out of a thin wall. A ribbon or a freehand membrane can span a hollow - a
    loop round a tube closes over its bore - so those are measured against the model
    itself as well (`material_clearance`), which keeps the pin in the wall.
    """
    if is_cut_face:
        normal = surfaces.patch_normal(verts, faces)
        clearance = None
        state = None
        if grown <= 0.0 and target is not None:
            state = ensure_evaluable(target)
            clearance = material_clearance(context, target)
        try:
            found = section.inscribed_circle(verts, faces, normal, clearance=clearance)
        finally:
            if state is not None:
                restore_visibility(target, state)
        if found is not None:
            return found[0], normal, max(found[1] - 2.0 * grown, found[1] * 0.2)
    bvh = mesh_utils.bvh_from_pydata(verts, faces)
    diag = mesh_utils.object_world_diagonal(target)
    state = ensure_evaluable(target)
    try:
        depsgraph = context.evaluated_depsgraph_get()
        if center_hint is not None:
            return cutting.estimate_pin_frame(target, depsgraph, bvh, diag, center_hint=center_hint)
        return cutting.estimate_pin_frame(target, depsgraph, bvh, diag, surface_point=hit, through_dir=through)
    finally:
        restore_visibility(target, state)


def preset_size_mm(context, preset, inscribed_bu):
    factor = connectors.preset_factor(preset)
    width_mm = max(1.0, inscribed_bu * factor / mm(context))
    return width_mm, width_mm * connectors.HEIGHT_RATIO


def contact_size_bu(context, src, inscribed_bu, depth_bu=0.0):
    """(width, height) in BU for a settings source (record or scene defaults).

    A preset pin is also kept shorter than the material is deep (`depth_limit_bu`): it
    reaches `height` into both parts, and on a cut close to an end of the model a pin
    sized from the face alone came out through the far side. A Custom height is left as
    typed; the panel warns when it is too long.
    """
    if src.size_preset == 'CUSTOM':
        return src.pin_width_mm * mm(context), src.pin_height_mm * mm(context)
    w_mm, h_mm = preset_size_mm(context, src.size_preset, inscribed_bu)
    w, h = w_mm * mm(context), h_mm * mm(context)
    limit = depth_limit_bu(context, src, depth_bu)
    if limit is not None and h > limit:
        h = max(limit, DEPTH_MIN_PIN_MM * mm(context))
    return w, h


# Material left beyond the tip of a pin or the bottom of its socket (mm), and the shortest
# pin a preset makes on a part too shallow for anything longer.
DEPTH_WALL_MM = 0.8
DEPTH_MIN_PIN_MM = 0.5


def depth_limit_bu(context, src, depth_bu):
    """Longest pin the material allows: the depth less the socket's tip extra and a wall."""
    if depth_bu <= 0.0:
        return None
    u = mm(context)
    tip = src.tip_extra_mm * u if src.asymmetric else 0.0
    return depth_bu - tip - DEPTH_WALL_MM * u


def contact_depth(context, target, center, normal):
    """How deep the material runs from `center` on the cut, along +/-`normal`; the smaller.

    0.0 when it cannot be measured (no target, or a ray that never leaves the model).
    """
    if target is None or target.type != 'MESH':
        return 0.0
    diag = mesh_utils.object_world_diagonal(target)
    state = ensure_evaluable(target)
    try:
        depsgraph = context.evaluated_depsgraph_get()
        depths = []
        for d in (Vector(normal), -Vector(normal)):
            hits = mesh_utils.object_ray_hits(
                target, Vector(center), d, diag * 1e-5, depsgraph, max_dist=diag * 2.0, limit=1
            )
            if not hits:
                return 0.0
            depths.append(hits[0])
        return min(depths)
    finally:
        restore_visibility(target, state)


def _contact_attr(rec, base, i):
    return getattr(rec, f"{base}_{'ab'[i]}")


def _set_contact_attr(rec, base, i, value):
    setattr(rec, f"{base}_{'ab'[i]}", value)


def auto_pin_matrix(context, rec, i, layout=None):
    center = Vector(_contact_attr(rec, "center", i))
    normal = Vector(_contact_attr(rec, "normal", i))
    w, h = contact_size_bu(context, rec, _contact_attr(rec, "inscribed", i), _contact_attr(rec, "depth", i))
    k = layout.scale if layout is not None else 1.0
    matrix = connectors.connector_matrix(center, cutting.protrude_direction(normal, rec.pin_side), w * k, h * k)
    region = _main_region(context, rec, i)
    return fit_preview(context, rec, matrix, layout, region[0] if region is not None else None)


# ----------------------------------------------------------------------------
# several connectors on one cut face
# ----------------------------------------------------------------------------
class PinLayout:
    """Where the connectors of one contact go.

    `offsets` are (x, y) positions in the connector's unit space (1.0 = one connector
    width, measured in the pin's own frame), `scale` is how much the preset size was
    reduced to make them fit, `asked` is how many the user wanted.
    """

    def __init__(self, offsets=None, scale=1.0, asked=1):
        self.offsets = offsets or [(0.0, 0.0)]
        self.scale = scale
        self.asked = asked

    @property
    def placed(self):
        return len(self.offsets)


# A preset sized pin may shrink this far to fit the count asked for; a Custom one never does.
LAYOUT_SHRINK = (1.0, 0.85, 0.72, 0.6, 0.5)
# Material kept round every socket: a share of the pin radius, but never under this (mm).
LAYOUT_RIM_RATIO = 0.5
LAYOUT_RIM_MIN_MM = 0.8

_layout_cache = {}


def _line_layout(center, x_axis, inscribed, count, r, rim):
    """Fallback for a surface that is not the real cut face: a row across the inscribed circle."""
    half = inscribed * 0.5 - r - rim
    if half <= 0.0:
        return []
    step = 2.0 * half / (count - 1)
    if step < 2.0 * r + rim:
        return []
    return [center + x_axis * (-half + step * j) for j in range(count)]


def compute_layout(
    context, src, verts, faces, is_cut_face, center, normal, inscribed, target=None, grown=0.0, region=None
):
    """PinLayout for `src.pin_count` connectors on one cut surface (world space input).

    The pins are spread over the face as far apart as the material allows (see
    `section.spread_points`), each keeping a rim of material round its socket. When
    they do not fit, preset sized pins get smaller first, then fewer are placed.

    `region` - (FaceRegions, k) - keeps them on one stretch of a face an earlier cut
    of the plan runs across (see `record_regions`).
    """
    asked = max(1, int(getattr(src, "pin_count", 1)))
    if asked <= 1 or not src.add_pin:
        return PinLayout(asked=asked)
    u = mm(context)
    w, _h = contact_size_bu(context, src, inscribed)
    center = Vector(center)
    normal = Vector(normal)
    frame = connectors.connector_frame(center, cutting.protrude_direction(normal, src.pin_side))
    x_axis = frame.col[0].xyz.normalized()
    y_axis = frame.col[1].xyz.normalized()
    scales = (1.0,) if src.size_preset == 'CUSTOM' else LAYOUT_SHRINK
    key = (
        tuple(round(c, 6) for c in center),
        tuple(round(c, 6) for c in normal),
        round(inscribed, 6),
        round(w, 6),
        asked,
        src.pin_side,
        scales,
        bool(is_cut_face),
        round(grown, 6),
        target.name if target is not None else "",
        len(verts),
        len(faces),
        tuple(round(c, 6) for c in (verts[0] if verts else (0.0, 0.0, 0.0))),
        tuple(round(c, 6) for c in (verts[-1] if verts else (0.0, 0.0, 0.0))),
        (region[0].key, region[1]) if region is not None else None,
    )
    hit = _layout_cache.get(key)
    if hit is not None:
        return PinLayout(list(hit[0]), hit[1], asked)
    found = None
    # a plane section is measured less its grown outline, anything else against the model
    state = None
    if region is not None:
        clearance = region[0].clearance(region[1])
    else:
        state = ensure_evaluable(target) if target is not None and is_cut_face and grown <= 0.0 else None
        clearance = material_clearance(context, target) if state is not None else None
    try:
        found = _search_layout(
            asked,
            scales,
            w,
            u,
            is_cut_face,
            (verts, faces, normal, grown),
            (center, x_axis, y_axis),
            inscribed,
            clearance,
        )
    finally:
        if state is not None:
            restore_visibility(target, state)
    if found is None:
        found = ([(0.0, 0.0)], 1.0)
    if len(_layout_cache) > 64:
        _layout_cache.clear()
    _layout_cache[key] = found
    return PinLayout(list(found[0]), found[1], asked)


def _search_layout(asked, scales, w, u, is_cut_face, patch, frame, inscribed, clearance):
    """Most pins first, biggest first: the first layout that fits. -> (offsets, scale) or None."""
    verts, faces, normal, grown = patch
    center, x_axis, y_axis = frame
    for count in range(asked, 1, -1):
        for k in scales:
            r = w * k * 0.5
            rim = max(r * LAYOUT_RIM_RATIO, LAYOUT_RIM_MIN_MM * u)
            pts = []
            if is_cut_face:
                pts = section.spread_points(
                    verts, faces, normal, count, r + rim + grown, 2.0 * r + rim, clearance=clearance
                )
            if not pts:
                pts = _line_layout(center, x_axis, inscribed, count, r, rim)
            if pts:
                size = w * k
                return [((p - center).dot(x_axis) / size, (p - center).dot(y_axis) / size) for p in pts], k
    return None


def record_layout(context, rec, i):
    """The layout for contact `i` of a cut record, measured on its preview surface."""
    if max(1, int(rec.pin_count)) <= 1 or not rec.add_pin:
        return PinLayout(asked=max(1, int(rec.pin_count)))
    sobj = surface_of(rec, i)
    if sobj is None:
        return PinLayout(asked=rec.pin_count)
    verts, faces = surface_world_patch(sobj)
    return compute_layout(
        context,
        rec,
        verts,
        faces,
        bool(sobj.get("esp_cut_face", False)),
        _contact_attr(rec, "center", i),
        _contact_attr(rec, "normal", i),
        _contact_attr(rec, "inscribed", i),
        target=record_target(context, rec),
        grown=float(sobj.get("esp_grown", 0.0)),
        region=_main_region(context, rec, i),
    )


def _main_region(context, rec, i):
    if i != 0 or rec.region < 0:
        return None
    regions = record_regions(context, rec)
    return (regions, rec.region) if regions is not None else None


def apply_preset(context, rec):
    """Write the preset size of contact A into the width/height fields."""
    if rec.size_preset == 'CUSTOM':
        return
    w, h = contact_size_bu(context, rec, rec.inscribed_a, rec.depth_a)
    rec.pin_width_mm = w / mm(context)
    rec.pin_height_mm = h / mm(context)


def printer_clearance_mm(context):
    """The clearance this printer needs on each side of a pin, from the preferences."""
    return pref(context, "printer_clearance_mm", 0.10)


def preset_clearance_mm(context, fit_preset):
    return printer_clearance_mm(context) * connectors.fit_factor(fit_preset)


_applying_fit = False


def apply_fit(context, src):
    """Write the Fit preset into the gap field, so the number in the panel is the real one.

    `src` is a cut record or the scene defaults - both carry the connector settings. The
    write lands on a property that has an update callback of its own, which calls back in
    here, so the guard and the tolerance check keep it from looping.
    """
    global _applying_fit
    if src.fit_preset == 'CUSTOM' or _applying_fit:
        return
    target = preset_clearance_mm(context, src.fit_preset)
    if abs(src.clearance_mm - target) <= 1e-6:
        return
    _applying_fit = True
    try:
        src.clearance_mm = target
    finally:
        _applying_fit = False


def copy_connector_props(src, dst):
    for key in connector_props():
        try:
            setattr(dst, key, getattr(src, key))
        except TypeError:
            pass


def contact_count(rec):
    return 2 if rec.two_contact else 1


def record_pins(rec):
    """Every connector preview of `rec` still in the scene: its own, then the extras."""
    pins = (rec.pin_obj_a, rec.pin_obj_b, *(e.pin_obj for e in rec.extras))
    return [p for p in map(alive, pins) if p is not None]


def record_objects(rec):
    """Every preview object `rec` owns - also one deleted from the scene but still held."""
    out = []
    for obj in (rec.surface_obj_a, rec.surface_obj_b, rec.pin_obj_a, rec.pin_obj_b, *(e.pin_obj for e in rec.extras)):
        if obj is not None and obj not in out:
            out.append(obj)
    return out


def unique_record_name(settings, prefix):
    names = {r.name for r in settings.cuts}
    n = 1
    while f"{prefix} {n:03d}" in names:
        n += 1
    return f"{prefix} {n:03d}"


# ----------------------------------------------------------------------------
# records
# ----------------------------------------------------------------------------
def add_record(context, target, cut_type, contacts):
    global _suspend
    settings = context.scene.esp
    _suspend += 1
    try:
        if base_of(settings) is None:
            # the object under the cut is the plan's source, even when it is a part of
            # an earlier build: its `esp_base` points at the model before those cuts,
            # and rooting the plan there would undo every cut already approved
            settings.base_obj = target
        rec = settings.cuts.add()
        rec.name = unique_record_name(settings, "Base Split" if len(contacts) == 2 else PREFIX[cut_type])
        rec.cut_type = cut_type
        rec.target_obj = target
        rec.two_contact = len(contacts) == 2
        active = settings.cuts[settings.active_cut] if 0 <= settings.active_cut < len(settings.cuts) - 1 else None
        copy_connector_props(active if active is not None else settings, rec)
        for i, data in enumerate(contacts):
            sobj = create_surface_object(context, f"ESP_Surface_{rec.name}_{'AB'[i]}", data, target)
            _set_contact_attr(rec, "surface_obj", i, sobj)
            center, normal, inscribed = contact_frame(
                context,
                target,
                data.verts,
                data.faces,
                data.hit,
                data.through,
                data.center_hint,
                data.is_cut_face,
                data.grown,
            )
            _set_contact_attr(rec, "center", i, center)
            _set_contact_attr(rec, "normal", i, normal)
            _set_contact_attr(rec, "inscribed", i, inscribed)
            _set_contact_attr(rec, "depth", i, contact_depth(context, target, center, normal))
        orient_second_contact(context, target, rec)
        rec.anchor = contacts[0].anchor
        extras = place_by_region(context, rec, target)
        apply_preset(context, rec)
        apply_fit(context, rec)
        for i in range(len(contacts)):
            layout = record_layout(context, rec, i)
            matrix = auto_pin_matrix(context, rec, i, layout)
            pin = create_pin_object(context, f"ESP_Pin_{rec.name}_{'AB'[i]}", rec.shape, matrix, layout)
            _set_contact_attr(rec, "pin_obj", i, pin)
            _set_contact_attr(rec, "pin_auto", i, flat(matrix))
            pin.hide_viewport = not rec.add_pin
        sync_extra_pins(context, rec, extras, target)
        settings.active_cut = len(settings.cuts) - 1
    finally:
        _suspend -= 1
    schedule_walls()
    return rec


def on_record_settings_changed(context, rec):
    global _suspend
    if _suspend:
        return
    schedule_walls()
    _suspend += 1
    try:
        apply_preset(context, rec)
        apply_fit(context, rec)
        for i in range(contact_count(rec)):
            pin = pin_of(rec, i)
            if pin is None:
                continue
            layout = record_layout(context, rec, i)
            if pin_needs_rebuild(pin, rec.shape, layout):
                update_pin_mesh(pin, rec.shape, layout)
            else:
                _store_layout(pin, layout)
            old_auto = unflat(_contact_attr(rec, "pin_auto", i))
            new_auto = auto_pin_matrix(context, rec, i, layout)
            delta = user_delta(old_auto, pin.matrix_basis)
            pin.matrix_world = _frame(new_auto) @ delta @ _scale(new_auto)
            _set_contact_attr(rec, "pin_auto", i, flat(new_auto))
            pin.hide_viewport = not (rec.add_pin and rec.show)
        for j in range(len(rec.extras)):
            _update_extra_pin(context, rec, j)
    finally:
        _suspend -= 1
    schedule_cuts_after(rec)


def _is_identity(m, tol=1e-5):
    ident = Matrix.Identity(4)
    return all(abs(m[i][j] - ident[i][j]) < tol for i in range(4) for j in range(4))


def _frame(m):
    """Rotation + translation part of a matrix (scale removed)."""
    return m.normalized()


def _scale(m):
    s = m.to_scale()
    return Matrix.Diagonal((s.x, s.y, s.z, 1.0))


def user_delta(auto, current):
    """What the user did to the pin, expressed in the pin's unscaled local frame.

    auto = F @ S  (frame @ scale); current = F @ D @ S  ->  D = F^-1 @ current @ S^-1
    Identity when the pin was never touched. Survives size/side/shape changes.
    """
    d = _frame(auto).inverted_safe() @ current @ _scale(auto).inverted_safe()
    return Matrix.Identity(4) if _is_identity(d) else d


def user_moved_pin(rec, i):
    pin = pin_of(rec, i)
    if pin is None:
        return False
    return not _is_identity(user_delta(unflat(_contact_attr(rec, "pin_auto", i)), pin.matrix_basis))


def reset_pins(context, rec):
    """Back to the automatic placement AND the stock shape.

    The build follows the preview pin's own geometry, so a reshape in edit mode is a
    change like a move or a scale is - and this is the way back from all three.
    """
    global _suspend
    _suspend += 1
    schedule_walls()
    try:
        for i in range(contact_count(rec)):
            pin = pin_of(rec, i)
            if pin is None:
                continue
            layout = record_layout(context, rec, i)
            update_pin_mesh(pin, rec.shape, layout)
            auto = auto_pin_matrix(context, rec, i, layout)
            pin.matrix_world = auto
            _set_contact_attr(rec, "pin_auto", i, flat(auto))
        for j in range(len(rec.extras)):
            _update_extra_pin(context, rec, j, reset=True)
    finally:
        _suspend -= 1


def faces_away(context, target, patches, c0, n0, c1, n1, cells=40, budget=20000):
    """True when the second contact's + side is the part without the pins.

    The pins go into side A: the + side of the FIRST contact's surface. The second
    contact's normal comes from its own surface, which faces whichever way that loop or
    curve was drawn. Which of its sides is A is a question of what is connected to what:
    the material is flooded on a grid from just past the second surface, never crossing
    either surface, until it runs into the first one - and the side of the first it runs
    into answers it. When the flood from the + side never meets the first surface (a leg
    that only touches the second), the - side is flooded instead. -> True / False, or
    None when neither side reaches the first surface.
    """
    h = mesh_utils.object_world_diagonal(target) / cells
    if h <= 0.0:
        return None
    clearance = material_clearance(context, target)
    if clearance is None:
        return None
    n0, n1 = Vector(n0).normalized(), Vector(n1).normalized()
    steps = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

    def first_contact_side(start):
        """+1 / -1: the side of the first surface the material from `start` runs into."""
        if clearance(start) <= 0.0:
            return None
        seen = {(0, 0, 0)}
        queue = [(0, 0, 0)]
        head = 0
        while head < len(queue) and len(seen) < budget:
            key = queue[head]
            head += 1
            here = start + Vector(key) * h
            for dx, dy, dz in steps:
                nxt = (key[0] + dx, key[1] + dy, key[2] + dz)
                if nxt in seen:
                    continue
                there = start + Vector(nxt) * h
                d = there - here
                hit0 = patches[0].ray_cast(here, d.normalized(), d.length)
                if hit0[0] is not None:
                    return 1 if (here - hit0[0]).dot(n0) >= 0.0 else -1
                if patches[1].ray_cast(here, d.normalized(), d.length)[0] is not None:
                    continue
                if clearance(there) <= 0.0:
                    continue
                seen.add(nxt)
                queue.append(nxt)
        return None

    plus = first_contact_side(Vector(c1) + n1 * h * 0.5)
    if plus is not None:
        return plus < 0
    minus = first_contact_side(Vector(c1) - n1 * h * 0.5)
    if minus is not None:
        return minus > 0
    return None


def second_contact_faces_away(context, target, rec):
    """`faces_away` for a two-contact record, on its preview surfaces."""
    if contact_count(rec) < 2 or target is None or target.type != 'MESH':
        return None
    patches = []
    for i in range(2):
        sobj = surface_of(rec, i)
        if sobj is None:
            return None
        patches.append(mesh_utils.bvh_from_pydata(*surface_world_patch(sobj)))
    state = ensure_evaluable(target)
    try:
        return faces_away(context, target, patches, rec.center_a, rec.normal_a, rec.center_b, rec.normal_b)
    finally:
        restore_visibility(target, state)


def orient_second_contact(context, target, rec):
    """Turn the second contact's normal so its + side is the pin part, like the first's."""
    if second_contact_faces_away(context, target, rec):
        rec.normal_b = tuple(-Vector(rec.normal_b))


def contact_side(verts, faces, normal):
    """+1 when the cutter's own faces point to `normal`'s side, -1 when they point away."""
    if not faces:
        return 1
    return 1 if surfaces.patch_normal(verts, faces).dot(Vector(normal)) >= 0.0 else -1


# ----------------------------------------------------------------------------
# a cut across earlier cuts: one connector for every part it is going to cut
# ----------------------------------------------------------------------------
# A build applies the cuts in order. When a later cut runs across an earlier one, the
# build cuts every part it crosses (`parts_crossed`), and each needs a connector of its
# own. Those are worked out here, while planning, from the same geometry the build
# will meet - so the preview shows every connector the build makes, and the build makes
# the ones the preview shows (`spec_for_part` takes them).
MIN_REGION_SAMPLES = 4


class FaceRegions:
    """A cut face divided by earlier cuts: which stretch a point is on, and room on it."""

    def __init__(self, key, samples, labels, radius, material, blockers, obstacles=()):
        from mathutils.kdtree import KDTree

        self.key = key
        self.obstacles = obstacles  # (bvh of an earlier cut's connector, room its socket takes)
        self.labels = labels
        self.radius = radius
        self.material = material
        self.blockers = blockers  # (bvh of the earlier surface, half its kerf)
        self.tree = KDTree(len(samples))
        for i, q in enumerate(samples):
            self.tree.insert(q, i)
        self.tree.balance()
        sizes = {}
        for k in labels:
            sizes[k] = sizes.get(k, 0) + 1
        self.big = sorted(k for k, n in sizes.items() if n >= MIN_REGION_SAMPLES)

    def label_at(self, p):
        _co, i, dist = self.tree.find(p)
        if i is None or dist > self.radius:
            return None
        return self.labels[i]

    def nearest_label(self, p):
        _co, i, _dist = self.tree.find(p)
        return None if i is None else self.labels[i]

    def depth(self, context, target, centre, normal):
        """`contact_depth`, also stopping at the earlier cuts and their connectors on the way."""
        depth = contact_depth(context, target, centre, normal)
        for d in (Vector(normal).normalized(), -Vector(normal).normalized()):
            for bvh, room in (*self.blockers, *self.obstacles):
                loc, _nor, _i, dist = bvh.ray_cast(Vector(centre), d)
                if loc is not None:
                    reach = max(dist - room, 0.0)
                    depth = reach if depth <= 0.0 else min(depth, reach)
        return depth

    def clearance(self, k):
        """`clearance(p)` for the section helpers, confined to stretch `k` of the face."""

        def room(p):
            p = Vector(p)
            if self.label_at(p) != k:
                return -1.0
            r = self.material(p)
            if r <= 0.0:
                return -1.0
            for bvh, half in self.blockers:
                loc, _nor, _i, dist = bvh.find_nearest(p)
                if loc is not None:
                    r = min(r, dist - half)
            for bvh, room in self.obstacles:
                loc, nor, _i, dist = bvh.find_nearest(p)
                if loc is None:
                    continue
                if (p - loc).dot(nor) < 0.0:
                    return -1.0  # inside an earlier connector's socket
                r = min(r, dist - room)
            return r

        return room


_regions_cache = {}


def _surface_key(obj):
    return (obj.name, tuple(round(v, 6) for v in flat(obj.matrix_world)), len(obj.data.vertices))


def _socket_half(pin):
    """The half of a connector that sits in a socket - local z >= 0 - as a closed BVH.

    The other half is fused into the material of its own part: a later cut's pin may run
    through it as through any material, but not into the socket round the first half.
    """
    bm = bmesh.new()
    bm.from_mesh(pin.data)
    bmesh.ops.bisect_plane(
        bm,
        geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
        plane_co=(0.0, 0.0, 0.0),
        plane_no=(0.0, 0.0, 1.0),
        clear_inner=True,
    )
    bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary])
    bmesh.ops.transform(bm, matrix=pin.matrix_world, verts=bm.verts)
    bm.normal_update()
    if not bm.faces:
        bm.free()
        return None
    bvh = BVHTree.FromBMesh(bm)
    bm.free()
    return bvh


def record_regions(context, rec):
    """FaceRegions of the record's cut face, or None when no earlier cut divides it."""
    if contact_count(rec) != 1:
        return None
    settings = context.scene.esp
    sobj = surface_of(rec, 0)
    target = record_target(context, rec)
    if sobj is None or target is None or target.type != 'MESH' or not len(sobj.data.polygons):
        return None
    u = mm(context)
    # what an answer depends on, cheap to gather: the cache is asked on every settings
    # change, and building the trees for a miss is what takes the time
    surfaces_before, pins_before = [], []
    for other in settings.cuts:
        if other.name == rec.name:
            break
        if not other.enabled:
            continue
        for i in range(contact_count(other)):
            o = surface_of(other, i)
            if o is not None and len(o.data.polygons):
                surfaces_before.append((o, other.cut_gap_mm * u * 0.5))
        if other.add_pin:
            socket = other.clearance_mm * u + (other.tip_extra_mm * u if other.asymmetric else 0.0)
            for pin in record_pins(other):
                if len(pin.data.polygons):
                    pins_before.append((pin, socket))
    if not surfaces_before:
        return None
    key = (
        _surface_key(sobj),
        tuple((*_surface_key(o), round(half, 7)) for o, half in surfaces_before),
        tuple((*_surface_key(o), round(room, 7)) for o, room in pins_before),
        target.name,
        len(target.data.vertices),
    )
    if key in _regions_cache:
        return _regions_cache[key]
    verts, faces = surface_world_patch(sobj)
    own = mesh_utils.bvh_from_pydata(verts, faces)
    blockers = []
    for o, half in surfaces_before:
        bvh = mesh_utils.bvh_from_pydata(*surface_world_patch(o))
        if own.overlap(bvh):
            blockers.append((bvh, half))
    obstacles = []
    for pin, room in pins_before:
        bvh = _socket_half(pin)
        if bvh is not None:
            obstacles.append((bvh, room))
    result = None
    if blockers:
        state = ensure_evaluable(target)
        try:
            material = material_clearance(context, target)
            if material is not None:
                samples, labels, radius = section.face_regions(
                    verts,
                    faces,
                    surfaces.patch_normal(verts, faces),
                    [b for b, _h in blockers],
                    lambda q: material(q) > 0.0,
                )
                if labels and max(labels) > 0:
                    result = FaceRegions(key, samples, labels, radius, material, blockers, obstacles)
                    if len(result.big) < 2:
                        result = None
        finally:
            restore_visibility(target, state)
    if len(_regions_cache) > 16:
        _regions_cache.clear()
    _regions_cache[key] = result
    return result


def pin_fits(context, rec, inscribed):
    """Whether a connector of the record's size fits in a circle of this diameter at all.

    A preset never goes below 1 mm, so a sliver narrower than that has no room for one -
    the plan shows none there, and the build makes none.
    """
    w, _h = contact_size_bu(context, rec, inscribed)
    return w <= inscribed


def place_by_region(context, rec, target):
    """Put the main connector on the stretch of the face the cut's anchor is on.

    -> [(centre, inscribed, label)] for every other stretch with room for a connector,
    for `sync_extra_pins`. With no earlier cut dividing the face, nothing changes.
    """
    regions = record_regions(context, rec)
    if regions is None:
        rec.region = -1
        return []
    sobj = surface_of(rec, 0)
    verts, faces = surface_world_patch(sobj)
    face_normal = surfaces.patch_normal(verts, faces)
    normal = Vector(rec.normal_a)
    anchor = regions.nearest_label(Vector(rec.anchor))
    if anchor not in regions.big:
        anchor = regions.big[0]
    found = {}
    for k in regions.big:
        hit = section.inscribed_circle(verts, faces, face_normal, clearance=regions.clearance(k))
        if hit is not None and pin_fits(context, rec, hit[1]):
            found[k] = hit
    if anchor not in found and found:
        # no room on the stretch the cut was drawn on: the build cuts that part without
        # a connector, and the cut's own connector goes on the roomiest other stretch
        anchor = max(found, key=lambda k: found[k][1])
    rec.region = anchor
    if anchor in found:
        centre, inscribed = found[anchor]
        rec.center_a = centre
        rec.inscribed_a = inscribed
        rec.depth_a = regions.depth(context, target, centre, normal)
    return [(found[k][0], found[k][1], k) for k in sorted(found) if k != anchor]


def _extra_layout(context, rec, e):
    regions = record_regions(context, rec)
    sobj = surface_of(rec, 0)
    if regions is None or sobj is None or max(1, int(rec.pin_count)) <= 1 or not rec.add_pin:
        return PinLayout(asked=max(1, int(rec.pin_count)))
    verts, faces = surface_world_patch(sobj)
    return compute_layout(context, rec, verts, faces, True, e.center, e.normal, e.inscribed, region=(regions, e.region))


def _extra_matrix(context, rec, e, layout):
    w, h = contact_size_bu(context, rec, e.inscribed, e.depth)
    k = layout.scale
    matrix = connectors.connector_matrix(
        Vector(e.center), cutting.protrude_direction(Vector(e.normal), rec.pin_side), w * k, h * k
    )
    return fit_preview(context, rec, matrix, layout, record_regions(context, rec))


def _update_extra_pin(context, rec, j, reset=False):
    """Bring extra connector `j`'s preview in line with the record, keeping a user's move."""
    e = rec.extras[j]
    layout = _extra_layout(context, rec, e)
    auto = _extra_matrix(context, rec, e, layout)
    pin = extra_pin_of(e)
    if pin is None:
        pin = create_pin_object(context, f"ESP_Pin_{rec.name}_P{j + 2}", rec.shape, auto, layout)
        e.pin_obj = pin
    else:
        if reset or pin_needs_rebuild(pin, rec.shape, layout):
            update_pin_mesh(pin, rec.shape, layout)
        else:
            _store_layout(pin, layout)
        if reset:
            pin.matrix_world = auto
        else:
            delta = user_delta(unflat(e.pin_auto), pin.matrix_basis)
            pin.matrix_world = _frame(auto) @ delta @ _scale(auto)
    e.pin_auto = flat(auto)
    pin.hide_viewport = not (rec.add_pin and rec.show)


def sync_extra_pins(context, rec, items, target):
    """One extra connector per item of `place_by_region`, previews made or dropped to match."""
    while len(rec.extras) > len(items):
        pin = rec.extras[-1].pin_obj
        if pin is not None:  # one deleted by hand but still held goes too
            mesh_utils.remove_object(pin)
        rec.extras.remove(len(rec.extras) - 1)
    while len(rec.extras) < len(items):
        rec.extras.add()
    for j, (centre, inscribed, k) in enumerate(items):
        e = rec.extras[j]
        e.center = centre
        e.normal = rec.normal_a
        e.inscribed = inscribed
        regions = record_regions(context, rec)
        e.depth = (
            regions.depth(context, target, centre, Vector(rec.normal_a))
            if regions is not None
            else contact_depth(context, target, centre, Vector(rec.normal_a))
        )
        e.region = k
        _update_extra_pin(context, rec, j)


def refresh_cuts_after(context, rec):
    """Measure again every cut after `rec`: how earlier cuts divide their faces changed."""
    settings = context.scene.esp
    schedule_walls()  # a cut in or out of the build is a wall more or less for every socket
    after = False
    for other in settings.cuts:
        if after:
            try:
                refresh_record_frames(context, other, rebuild=False)
            except Exception:
                pass
        elif other.name == rec.name:
            after = True


def refresh_record_frames(context, rec, rebuild=True):
    """Re-estimate centre/normal/size from the (possibly edited) cut surfaces."""
    schedule_walls()
    target = record_target(context, rec)
    if target is None:
        return
    for i in range(contact_count(rec)):
        sobj = surface_of(rec, i)
        if sobj is None:
            continue
        if rebuild and sobj.get("esp_kind") == 'STRAIGHT':
            rebuild_surface(sobj, context=context, target=target)
        verts, faces = surface_world_patch(sobj)
        hit = Vector(sobj["esp_hit"]) if "esp_hit" in sobj else None
        through = Vector(sobj["esp_through"]) if "esp_through" in sobj else None
        hint = Vector(sobj["esp_center_hint"]) if "esp_center_hint" in sobj else None
        if hit is not None and sobj.get("esp_kind") == 'STRAIGHT':
            # a moved plane: re-anchor the hit onto the new plane along the view direction
            hint, hit = _reanchor_plane(verts, faces, hit, through, hint)
        try:
            center, normal, inscribed = contact_frame(
                context,
                target,
                verts,
                faces,
                hit,
                through,
                hint,
                bool(sobj.get("esp_cut_face", False)),
                float(sobj.get("esp_grown", 0.0)),
            )
        except Exception:
            continue
        _set_contact_attr(rec, "center", i, center)
        _set_contact_attr(rec, "normal", i, normal)
        _set_contact_attr(rec, "inscribed", i, inscribed)
        _set_contact_attr(rec, "depth", i, contact_depth(context, target, center, normal))
        sobj["esp_mw"] = flat(sobj.matrix_world)
    orient_second_contact(context, target, rec)
    sync_extra_pins(context, rec, place_by_region(context, rec, target), target)
    on_record_settings_changed(context, rec)


def _reanchor_plane(verts, faces, hit, through, hint):
    n = surfaces.patch_normal(verts, faces)
    c = surfaces.patch_center(verts)
    d = Vector(through) if through is not None else n
    denom = d.dot(n)
    if abs(denom) < 1e-6:
        return c, None
    t = (c - hit).dot(n) / denom
    return None, hit + d * t


def set_record_visibility(rec, show):
    for i in range(contact_count(rec)):
        s = surface_of(rec, i)
        if s is not None:
            s.hide_viewport = not show
        p = pin_of(rec, i)
        if p is not None:
            p.hide_viewport = not (show and rec.add_pin)
    for e in rec.extras:
        p = extra_pin_of(e)
        if p is not None:
            p.hide_viewport = not (show and rec.add_pin)


def on_active_changed(context):
    settings = context.scene.esp
    if not (0 <= settings.active_cut < len(settings.cuts)):
        return
    rec = settings.cuts[settings.active_cut]
    col = bpy.data.collections.get(PLAN_COLLECTION)
    if col is None:
        return
    for obj in col.objects:
        if obj is None:
            continue
        kind = obj.get("esp_preview")
        if kind in ('pin', 'surface'):
            obj.color = ORANGE if kind == 'pin' else GREEN
    for obj in record_objects(rec):
        obj.color = (1.0, 0.85, 0.2, 0.9) if obj.get("esp_preview") == 'pin' else (0.5, 1.0, 0.55, 0.5)


def remove_record(context, index):
    settings = context.scene.esp
    if not (0 <= index < len(settings.cuts)):
        return
    rec = settings.cuts[index]
    schedule_walls()
    for obj in record_objects(rec):
        mesh_utils.remove_object(obj)
    settings.cuts.remove(index)
    settings.active_cut = min(index, len(settings.cuts) - 1)
    # the cuts after it no longer have it dividing their faces
    for later in list(settings.cuts)[index:]:
        try:
            refresh_record_frames(context, later, rebuild=False)
        except Exception:
            pass


def remove_all_records(context):
    settings = context.scene.esp
    while len(settings.cuts):
        remove_record(context, len(settings.cuts) - 1)
    col = bpy.data.collections.get(PLAN_COLLECTION)
    if col is not None:
        for obj in list(col.objects):
            mesh_utils.remove_object(obj)
        bpy.data.collections.remove(col)
    settings.active_cut = -1


def set_plan_hidden(context, hidden):
    col = bpy.data.collections.get(PLAN_COLLECTION)
    if col is not None:
        col.hide_viewport = hidden


def record_spec(context, rec, settings, remesh=True):
    apply_fit(context, rec)
    contacts = []
    shape, custom = connectors.resolve_shape(rec.shape)
    for i in range(contact_count(rec)):
        sobj = surface_of(rec, i)
        if sobj is None:
            raise cutting.CutError(f"Preview surface of '{rec.name}' is missing")
        verts, faces = cutter_world_patch(sobj)
        pin = pin_of(rec, i)
        pm = pin.matrix_basis.copy() if (rec.add_pin and pin is not None) else None
        contacts.append(
            cutting.ContactSpec(
                verts,
                faces,
                rec.add_pin,
                pm,
                shape,
                custom,
                pin_mesh=pin_unit_mesh(pin) if pm is not None else None,
                regions_skipped=int(sobj.get("esp_skipped", 0)),
                preview=surface_world_patch(sobj),
                side=contact_side(verts, faces, _contact_attr(rec, "normal", i)),
            )
        )
    u = mm(context)
    return cutting.CutSpec(
        contacts=contacts,
        oriented=len(contacts) > 1,
        gap=rec.cut_gap_mm * u,
        clearance=rec.clearance_mm * u,
        tip_extra=(rec.tip_extra_mm * u) if rec.asymmetric else 0.0,
        pin_side=rec.pin_side,
        solver=pref(context, "solver", 'AUTO'),
        remesh=remesh and settings.remesh_enable,
        remesh_voxel=settings.remesh_voxel_mm * u,
        remesh_adaptivity=settings.remesh_adaptivity,
        remesh_smooth=settings.remesh_smooth,
        diagnose=settings.diagnose_failed,
    )


def parts_crossed(context, parts, spec, threshold):
    """The parts the cut runs through, in the order given.

    An earlier cut of the same build leaves the model in several parts, and a later cut
    can run across more than one of them - a full width plane under a cut that split the
    model in two. Each of those has to be cut, not only the one nearest the anchor.

    A part counts when some of the cut face lies in its material deeper than
    `threshold`. The cutter itself is no test: it reaches past the model on purpose, and
    across the kerf of an earlier cut it touches the neighbouring part without cutting
    anything there - every such part would cost a boolean that can only fail.
    """
    samples = []
    for c in spec.contacts:
        verts, faces = c.shown_patch()
        # the same points the plan divides the face with (`section.face_regions`): a part
        # the preview gives a connector is a part the build cuts
        samples += section.face_points(verts, faces, surfaces.patch_normal(verts, faces))[0]
    out = []
    for part in parts:
        if part.type != 'MESH':
            continue
        state = ensure_evaluable(part)
        try:
            clearance = material_clearance(context, part)
            if clearance is not None and any(clearance(p) > threshold for p in samples):
                out.append(part)
        finally:
            restore_visibility(part, state)
    return out


def _footprint(c, matrix):
    """Where the connector meets the cut face: its outline at local z = 0, in world space."""
    if c.pin_mesh is not None:
        pts = {(round(v.co.x, 5), round(v.co.y, 5)) for v in c.pin_mesh.vertices}
    else:
        bm = connectors.unit_layout_bmesh(c.shape, c.custom_obj, c.offsets)
        pts = {(round(v.co.x, 5), round(v.co.y, 5)) for v in bm.verts}
        bm.free()
    return [matrix @ Vector((x, y, 0.0)) for x, y in pts]


def spec_for_part(context, rec, spec, part):
    """`spec` with a connector that sits in `part`, for a cut that runs through several.

    The record's connector - where the user left it, as they shaped it - is kept when
    the whole of it is inside this part: every pin, edge to edge, not just its middle.
    The connectors the plan shows for the other parts the cut crosses (`rec.extras`)
    are taken the same way. A part none of them is in gets one made here: the record's
    shape and size settings, placed and sized on the stretch of the cut face that is
    this part's material. When there is no room for one, the part is cut without.
    """
    import dataclasses

    c = spec.contacts[0]
    if not c.add_pin or c.pin_matrix is None:
        return spec
    # the record's connector first, then those it shows on the other parts it crosses
    candidates = [c]
    for e in rec.extras:
        pin = extra_pin_of(e)
        mesh = pin_unit_mesh(pin)
        if mesh is not None:
            candidates.append(dataclasses.replace(c, pin_matrix=pin.matrix_basis.copy(), pin_mesh=mesh, offsets=None))
    state = ensure_evaluable(part)
    try:
        clearance = material_clearance(context, part)
        if clearance is None:
            return spec
        for cand in candidates:
            if all(clearance(p) > 0.0 for p in _footprint(cand, cand.pin_matrix)):
                return spec if cand is c else dataclasses.replace(spec, contacts=[cand])
        verts, faces = c.shown_patch()
        normal = Vector(rec.normal_a)
        found = section.inscribed_circle(verts, faces, surfaces.patch_normal(verts, faces), clearance=clearance)
    finally:
        restore_visibility(part, state)
    if found is None or not pin_fits(context, rec, found[1]):
        own = dataclasses.replace(c, add_pin=False, pin_matrix=None, pin_mesh=None, offsets=None)
        return dataclasses.replace(spec, contacts=[own])
    centre, inscribed = found
    layout = compute_layout(context, rec, verts, faces, True, centre, normal, inscribed, target=part)
    depth = contact_depth(context, part, centre, normal)
    w, h = contact_size_bu(context, rec, inscribed, depth)
    matrix = connectors.connector_matrix(
        centre, cutting.protrude_direction(normal, rec.pin_side), w * layout.scale, h * layout.scale
    )
    own = dataclasses.replace(
        c, pin_matrix=matrix, pin_mesh=None, offsets=layout.offsets if layout.placed > 1 else None
    )
    return dataclasses.replace(spec, contacts=[own])


def _depth_scale(context, rec, matrix, footprint, unit_reach, part, depsgraph, extra=()):
    """How much of a preset pin's height the material leaves: 1.0, or less to shorten it.

    Rays go both ways along the pin's axis from every point of its footprint, to the
    part's own surface - and to every `extra` (bvh, room): surfaces the part will have
    by then that `part` does not show yet (the plan's earlier cuts, their sockets).
    """
    z = matrix.col[2].xyz
    h = z.length
    if h < 1e-12:
        return 1.0
    reach = unit_reach * h
    diag = mesh_utils.object_world_diagonal(part)
    depth = None
    for p in footprint:
        for d in (z.normalized(), -z.normalized()):
            hits = mesh_utils.object_ray_hits(part, p, d, diag * 1e-5, depsgraph, max_dist=reach * 4.0, limit=1)
            if hits:
                depth = hits[0] if depth is None else min(depth, hits[0])
            for bvh, room in extra:
                loc, _nor, _i, dist = bvh.ray_cast(p, d, reach * 4.0)
                if loc is not None:
                    gone = max(dist - room, 0.0)
                    depth = gone if depth is None else min(depth, gone)
    limit = depth_limit_bu(context, rec, depth or 0.0)
    if limit is None or reach <= limit:
        return 1.0
    return max(limit, DEPTH_MIN_PIN_MM * mm(context)) / reach


def fit_depth(context, rec, spec, part):
    """`spec` with every preset pin kept shorter than `part` is deep where the pin goes.

    The height a preset pin gets when it is planned is measured on the model, at the
    middle of the cut face. The part a build actually cuts can be shallower: an earlier
    cut of the same build may have sliced it thin, or carved a socket right behind this
    cut face - and a pin as long as the model allowed then runs into that socket, and
    the pin sitting in it. Measured again on the part, across the whole footprint of
    the connector, the pin is shortened to leave a wall before anything it would reach.
    The preview makes the same measurement in advance (`fit_preview`).
    """
    import dataclasses

    if rec.size_preset == 'CUSTOM':
        return spec
    out = []
    changed = False
    state = ensure_evaluable(part)
    try:
        depsgraph = context.evaluated_depsgraph_get()
        for c in spec.contacts:
            if not c.add_pin or c.pin_matrix is None:
                out.append(c)
                continue
            if c.pin_mesh is not None:
                unit_reach = max((abs(v.co.z) for v in c.pin_mesh.vertices), default=1.0)
            else:
                unit_reach = 1.0
            k = _depth_scale(context, rec, c.pin_matrix, _footprint(c, c.pin_matrix), unit_reach, part, depsgraph)
            if k >= 1.0:
                out.append(c)
                continue
            out.append(dataclasses.replace(c, pin_matrix=c.pin_matrix @ Matrix.Diagonal((1.0, 1.0, k, 1.0))))
            changed = True
    finally:
        restore_visibility(part, state)
    return dataclasses.replace(spec, contacts=out) if changed else spec


def fit_preview(context, rec, matrix, layout, regions=None):
    """A preview pin's matrix, as short as the build will make it (`fit_depth`).

    On a cut the plan's earlier cuts divide (`regions`), their surfaces and sockets are
    in the way as well - the part the build cuts will have them.
    """
    if rec.size_preset == 'CUSTOM' or not rec.add_pin:
        return matrix
    target = record_target(bpy.context, rec)
    if target is None or target.type != 'MESH':
        return matrix
    shape, custom = connectors.resolve_shape(rec.shape)
    bm = connectors.unit_layout_bmesh(shape, custom, layout.offsets if layout is not None else None)
    footprint = [matrix @ Vector(xy) for xy in {(round(v.co.x, 5), round(v.co.y, 5), 0.0) for v in bm.verts}]
    unit_reach = max((abs(v.co.z) for v in bm.verts), default=1.0)
    bm.free()
    extra = (*regions.blockers, *regions.obstacles) if regions is not None else ()
    state = ensure_evaluable(target)
    try:
        k = _depth_scale(context, rec, matrix, footprint, unit_reach, target, context.evaluated_depsgraph_get(), extra)
    finally:
        restore_visibility(target, state)
    return matrix if k >= 1.0 else matrix @ Matrix.Diagonal((1.0, 1.0, k, 1.0))


# ----------------------------------------------------------------------------
# the wall left round a socket
# ----------------------------------------------------------------------------
# A socket is a hole: where it runs close to the outside of the part, or to another
# cut or socket, the wall between them prints as a skin that splits or never prints.
# Each connector preview carries the thinnest wall measured round its socket, and
# what that wall is against; the panel warns below the Minimum Wall preference.
WALL_KEY = "esp_wall_mm"  # thinnest wall round the socket (mm); 0 = it breaks out
WALL_NEAR_KEY = "esp_wall_near"  # against what: "" the outside of the part, else a cut or connector
WALL_CAP_MM = 5.0  # walls thicker than this are not measured further
WALL_TOLERANCE_MM = 0.05  # a wall the add-on itself leaves at the minimum is not flagged


def min_wall_mm(context):
    """The thinnest wall this printer prints reliably, from the preferences."""
    return pref(context, "min_wall_mm", 0.8)


def thin_wall(context, pin):
    """(wall in mm, what it is against) when `pin`'s socket leaves too thin a wall, else None."""
    if pin is None:
        return None
    wall = pin.get(WALL_KEY)
    if wall is None or wall >= min_wall_mm(context) - WALL_TOLERANCE_MM:
        return None
    return float(wall), str(pin.get(WALL_NEAR_KEY, ""))


def record_thin_walls(context, rec):
    """[(label, wall mm, against)] for every connector of `rec` with too thin a wall."""
    if not rec.add_pin:
        return []
    out = []
    for i in range(contact_count(rec)):
        found = thin_wall(context, pin_of(rec, i))
        if found is not None:
            out.append((f"Contact {i + 1}" if rec.two_contact else "", *found))
    for j, e in enumerate(rec.extras):
        found = thin_wall(context, extra_pin_of(e))
        if found is not None:
            out.append((f"Connector on part {j + 2}", *found))
    return out


def socket_room(context, rec):
    """How much bigger than its pin a record's socket is: (per side, at the tip), BU."""
    u = mm(context)
    return rec.clearance_mm * u, (rec.tip_extra_mm * u if rec.asymmetric else 0.0)


def socket_bmesh(context, rec, pin):
    """The hole preview `pin` leaves (world), as a bmesh the caller frees; None without a pin.

    The socket is the shape the build carves - the pin's own mesh, grown by the
    clearance and the tip extra - and only its half in the socket part is a hole: the
    other half is the pin's root, inside the material of the pin part, and a socket of
    another cut running into a root runs into solid material.
    """
    me = pin_unit_mesh(pin)
    if me is None:
        return None
    side, tip = socket_room(context, rec)
    matrix = pin.matrix_basis.copy()
    bm = connectors.connector_bmesh(None, None, matrix, side, tip, unit_mesh=me)
    bmesh.ops.bisect_plane(
        bm,
        geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
        dist=1e-7,
        plane_co=matrix.translation,
        plane_no=matrix.col[2].xyz.normalized(),
        clear_inner=True,
    )
    return bm


def _segment_points(p, q, step, ends=True):
    n = max(1, math.ceil((q - p).length / step))
    first, last = (0, n + 1) if ends else (1, n)
    return [p.lerp(q, k / n) for k in range(first, last)]


def _triangle_points(a, b, c, step):
    """Points over triangle abc, about `step` apart: its edges, then rows across it.

    The rows run along the longest edge, so a long thin face - the side of a round pin -
    gets its edges and little else, instead of a grid of the long edge squared.
    """
    edges = sorted(((a, b, c), (b, c, a), (c, a, b)), key=lambda e: -(e[1] - e[0]).length)
    p, q, apex = edges[0]
    pts = _segment_points(p, q, step) + _segment_points(q, apex, step, False) + _segment_points(apex, p, step, False)
    pts.append(apex)
    base = (q - p).length
    if base < 1e-12:
        return pts
    height = (q - p).cross(apex - p).length / base
    rows = int(height / step)
    for k in range(1, rows + 1):
        t = k / (rows + 1)
        pts += _segment_points(p.lerp(apex, t), q.lerp(apex, t), step, False)
    return pts


def socket_samples(context, rec, pin, step):
    """Points on the wall of the hole preview `pin` leaves (world), about `step` apart.

    -> (points, (lo, hi) bounds), or ([], None).
    """
    bm = socket_bmesh(context, rec, pin)
    if bm is None:
        return [], None
    try:
        tris = bmesh.ops.triangulate(bm, faces=bm.faces[:])["faces"]
        corners = [[v.co.copy() for v in f.verts] for f in tris]
        bounds = _bounds([v.co for v in bm.verts]) if bm.verts else None
    finally:
        bm.free()
    points = []
    for a, b, c in corners:
        points += _triangle_points(a, b, c, step)
    return points, bounds


def _bounds(points):
    xs, ys, zs = zip(*points)
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def _wall_obstacles(context, settings):
    """What else a socket's wall can run against: every other cut's surface and socket.

    -> [(object, bvh, room, (lo, hi), record name)]. A cut surface takes half its kerf
    with it; a socket is measured as the hole it is (`socket_bmesh`). Cuts left out of the build do not
    count; the order of the cuts does not matter - a later cut passing a socket thins
    its wall just the same.
    """
    out = []
    u = mm(context)
    for rec in settings.cuts:
        if not rec.enabled:
            continue
        for i in range(contact_count(rec)):
            sobj = surface_of(rec, i)
            if sobj is not None and len(sobj.data.polygons):
                verts, faces = surface_world_patch(sobj)
                bvh = mesh_utils.bvh_from_pydata(verts, faces)
                out.append((sobj, bvh, rec.cut_gap_mm * u * 0.5, _bounds(verts), rec.name))
        if rec.add_pin:
            for pin in record_pins(rec):
                bm = socket_bmesh(context, rec, pin)
                if bm is None:
                    continue
                try:
                    if not bm.faces:
                        continue
                    bounds = _bounds([v.co for v in bm.verts])
                    bvh = BVHTree.FromBMesh(bm)
                finally:
                    bm.free()
                out.append((pin, bvh, 0.0, bounds, rec.name))
    return out


def socket_wall(context, rec, pin, obstacles=()):
    """(thinnest wall in BU, what it is against) round the socket of preview `pin`.

    The wall is measured from points all over the socket to the outside of the part
    (the model the cut was drawn on) and to the other cuts' surfaces and sockets
    (`obstacles`, see `_wall_obstacles`), capped at WALL_CAP_MM. A point of the socket
    outside the material is a socket that breaks out: a wall of 0. None when there is
    nothing to measure.
    """
    target = record_target(context, rec)
    if pin is None or target is None or target.type != 'MESH':
        return None
    u = mm(context)
    cap = WALL_CAP_MM * u
    step = max(min_wall_mm(context) * u * 0.5, cap * 0.02)
    points, bounds = socket_samples(context, rec, pin, step)
    if not points:
        return None
    best, near = cap, ""
    own = set(record_objects(rec))
    lo, hi = bounds
    lo, hi = lo - Vector((cap,) * 3), hi + Vector((cap,) * 3)
    for obj, bvh, room, (olo, ohi), name in obstacles:
        if obj in own and obj.get("esp_preview") == 'surface':
            continue  # the cut's own face is where the socket opens
        if obj == pin or any(olo[k] > hi[k] or ohi[k] < lo[k] for k in range(3)):
            continue
        label = "another connector of this cut" if obj in own else name
        for p in points:
            loc, _nor, _i, dist = bvh.find_nearest(p, best + room)
            if loc is not None and dist - room < best:
                best, near = max(dist - room, 0.0), label
    state = ensure_evaluable(target)
    try:
        bvh = _model_bvh(target, context.evaluated_depsgraph_get())
        if bvh is not None:
            mw = target.matrix_world.copy()
            inv = mw.inverted_safe()
            nmat = inv.to_3x3().transposed()
            reach = best / max(min(abs(c) for c in mw.to_scale()), 1e-9)
            for p in points:
                loc, nor, _i, _d = bvh.find_nearest(inv @ p, reach)
                if loc is None:
                    continue
                loc_w = mw @ loc
                gap = p - loc_w
                wall = 0.0 if gap.dot(nmat @ nor) > 0.0 else gap.length
                if wall < best:
                    best, near = wall, ""
                    if wall <= 0.0:
                        break
    finally:
        restore_visibility(target, state)
    return best, near


def measure_walls(context, rec, obstacles=None):
    """Store on every connector preview of `rec` the wall left round its socket."""
    if obstacles is None:
        obstacles = _wall_obstacles(context, context.scene.esp)
    u = mm(context)
    for pin in record_pins(rec):
        found = socket_wall(context, rec, pin, obstacles) if rec.add_pin and rec.enabled else None
        if found is None:
            for key in (WALL_KEY, WALL_NEAR_KEY):
                if key in pin:
                    del pin[key]
            continue
        wall, near = found
        pin[WALL_KEY] = round(wall / u, 3)
        pin[WALL_NEAR_KEY] = near


def measure_all_walls(context):
    settings = context.scene.esp
    obstacles = _wall_obstacles(context, settings)
    for rec in settings.cuts:
        measure_walls(context, rec, obstacles)


def schedule_walls():
    """Measure every socket's wall again once the plan stops changing.

    Every record shares the walls: a cut moved, or a connector made wider, can thin
    the wall of a socket of any other cut. Debounced like `schedule_cuts_after`.
    """
    if not bpy.app.timers.is_registered(_refresh_walls):
        bpy.app.timers.register(_refresh_walls, first_interval=0.3)


def _refresh_walls():
    scene = bpy.context.scene
    if scene is None or not len(scene.esp.cuts) or scene.esp.built:
        return None
    try:
        measure_all_walls(bpy.context)
    except Exception:
        pass
    overlay.redraw()
    return None


def crossing_threshold(context, rec, spec):
    """How deep in a part the cut face must run for the part to count as crossed."""
    grown = max(
        (float(o.get("esp_grown", 0.0)) for o in record_objects(rec) if o.get("esp_preview") == 'surface'),
        default=0.0,
    )
    return max(2.0 * grown, 2.0 * spec.gap, 0.5 * mm(context))


def quick_spec(context, target, contacts):
    """CutSpec for Quick mode straight from the tool output and the scene defaults."""
    settings = context.scene.esp
    apply_fit(context, settings)
    shape, custom = connectors.resolve_shape(settings.shape)
    frames = [
        contact_frame(
            context,
            target,
            data.verts,
            data.faces,
            data.hit,
            data.through,
            data.center_hint,
            data.is_cut_face,
            data.grown,
        )
        for data in contacts
    ]
    if len(contacts) == 2:
        # the second surface faces the way it was drawn: turned so its + side is the pin
        # part, as a plan does it (`orient_second_contact`)
        patches = [mesh_utils.bvh_from_pydata(d.verts, d.faces) for d in contacts]
        state = ensure_evaluable(target)
        try:
            away = faces_away(context, target, patches, frames[0][0], frames[0][1], frames[1][0], frames[1][1])
        finally:
            restore_visibility(target, state)
        if away:
            centre, normal, inscribed = frames[1]
            frames[1] = (centre, -Vector(normal), inscribed)
    specs = []
    for data, (center, normal, inscribed) in zip(contacts, frames):
        pm = None
        offsets = None
        if settings.add_pin:
            layout = compute_layout(
                context,
                settings,
                data.verts,
                data.faces,
                data.is_cut_face,
                center,
                normal,
                inscribed,
                target=target,
                grown=data.grown,
            )
            depth = contact_depth(context, target, center, normal)
            w, h = contact_size_bu(context, settings, inscribed, depth)
            w, h = w * layout.scale, h * layout.scale
            pm = connectors.connector_matrix(center, cutting.protrude_direction(normal, settings.pin_side), w, h)
            offsets = layout.offsets if layout.placed > 1 else None
        verts, faces = data.cutter if data.cutter is not None else (data.verts, data.faces)
        specs.append(
            cutting.ContactSpec(
                verts,
                faces,
                settings.add_pin,
                pm,
                shape,
                custom,
                offsets=offsets,
                regions_skipped=data.regions_skipped,
                preview=(data.verts, data.faces),
                side=contact_side(verts, faces, normal),
            )
        )
    u = mm(context)
    return cutting.CutSpec(
        contacts=specs,
        oriented=len(specs) > 1,
        gap=settings.cut_gap_mm * u,
        clearance=settings.clearance_mm * u,
        tip_extra=(settings.tip_extra_mm * u) if settings.asymmetric else 0.0,
        pin_side=settings.pin_side,
        solver=pref(context, "solver", 'AUTO'),
        diagnose=settings.diagnose_failed,
    )


# ----------------------------------------------------------------------------
# follow moved cut planes (G/R/S on the preview) -> re-place the pin
# ----------------------------------------------------------------------------
_dirty = {}  # surface object name -> the transform it carried when it was marked


def _refresh_dirty():
    """Re-place the pin (and, for a plane cut, re-take the section) once the drag stops.

    Re-sectioning the model is far too heavy to run on every mouse move of a G, so the
    timer keeps re-arming itself while the preview is still moving and only does the
    work once its transform has held still for a tick.
    """
    global _dirty
    scene = bpy.context.scene
    if scene is None:
        _dirty = {}
        return None
    moving = False
    for name, seen in list(_dirty.items()):
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        cur = tuple(flat(obj.matrix_basis))
        if cur != seen:
            _dirty[name] = cur
            moving = True
    if moving:
        return 0.1
    names = list(_dirty)
    _dirty = {}
    moved = False
    for rec in scene.esp.cuts:
        if any(o is not None and o.name in names for o in (surface_of(rec, 0), surface_of(rec, 1))):
            moved = True
            try:
                refresh_record_frames(bpy.context, rec)
            except Exception:
                pass
        elif moved:
            # a cut after a moved one: the moved surface may divide its face differently
            try:
                refresh_record_frames(bpy.context, rec, rebuild=False)
            except Exception:
                pass
    return None


@persistent
def depsgraph_handler(scene, depsgraph):
    settings = getattr(scene, "esp", None)
    if settings is None:
        return
    migrate_scene(scene)  # a scene appended from an older file, say
    release_deleted(settings)  # a preview, or the model, deleted in the viewport
    if not len(settings.cuts):
        return
    changed = False
    for rec in settings.cuts:
        for i in range(2):
            obj = surface_of(rec, i)
            if obj is None:
                continue
            cur = flat(obj.matrix_basis)
            old = obj.get("esp_mw")
            if old is None or any(abs(a - b) > 1e-6 for a, b in zip(cur, old)):
                obj["esp_mw"] = cur
                if old is not None:
                    _dirty[obj.name] = tuple(cur)
                    changed = True
                    schedule_walls()
    if changed and not bpy.app.timers.is_registered(_refresh_dirty):
        bpy.app.timers.register(_refresh_dirty, first_interval=0.05)
    # a connector moved by hand: the cuts after it keep their own clear of its socket
    for rec in settings.cuts:
        for obj in record_pins(rec):
            cur = flat(obj.matrix_basis)
            old = obj.get("esp_mw")
            if old is None or any(abs(a - b) > 1e-6 for a, b in zip(cur, old)):
                obj["esp_mw"] = cur
                if old is not None:
                    schedule_cuts_after(rec)
                    schedule_walls()


# cuts whose connectors changed: the divided cuts after them are measured again shortly
_pending_after = set()


def schedule_cuts_after(rec):
    """Measure the divided cuts after `rec` again once its settings stop changing.

    Debounced: a width slider being dragged changes the record dozens of times a second,
    and every cut after it with an extra connector would be measured again each time.
    """
    settings = bpy.context.scene.esp if bpy.context.scene is not None else None
    if settings is None:
        return
    names = [r.name for r in settings.cuts]
    if rec.name not in names:
        return
    later = settings.cuts[names.index(rec.name) + 1 :]
    if not any(r.region >= 0 or len(r.extras) for r in later):
        return
    _pending_after.add(rec.name)
    if not bpy.app.timers.is_registered(_refresh_pending_after):
        bpy.app.timers.register(_refresh_pending_after, first_interval=0.3)


def _refresh_pending_after():
    scene = bpy.context.scene
    names = set(_pending_after)
    _pending_after.clear()
    if scene is None:
        return None
    settings = scene.esp
    order = [r.name for r in settings.cuts]
    first = min((order.index(n) for n in names if n in order), default=None)
    if first is None:
        return None
    for rec in list(settings.cuts)[first + 1 :]:
        if rec.region >= 0 or len(rec.extras):
            try:
                refresh_record_frames(bpy.context, rec, rebuild=False)
            except Exception:
                pass
    return None


def register():
    if depsgraph_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(depsgraph_handler)
    if load_post_handler not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(load_post_handler)
    for handlers in (bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        if undo_redo_handler not in handlers:
            handlers.append(undo_redo_handler)
    # bpy.data is out of reach while add-ons register: the file open now is brought
    # over as soon as Blender is running
    if not bpy.app.timers.is_registered(_migrate_after_register):
        bpy.app.timers.register(_migrate_after_register, first_interval=0.0)


def unregister():
    if depsgraph_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(depsgraph_handler)
    if load_post_handler in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(load_post_handler)
    for handlers in (bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        if undo_redo_handler in handlers:
            handlers.remove(undo_redo_handler)
    for timer in (_refresh_dirty, _refresh_pending_after, _migrate_after_register, _refresh_walls):
        if bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)
    section.free_cache()
