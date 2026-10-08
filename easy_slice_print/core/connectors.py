# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Rafael Omodei and EasySlice Print contributors
"""Connector (pin/socket) shapes.

Every connector is a *unit* mesh: it fits in x,y in [-0.5, 0.5] and spans
z in [-1, 1]. z == 0 is the cut surface, +z is the half that protrudes into
the socket part, -z is the half embedded in the pin part.
The final connector is unit_mesh @ Matrix(location, rotation, scale(w, w, h)).
"""

import bmesh
import bpy
from mathutils import Matrix, Vector

from . import mesh_utils

LIB_COLLECTION = "ESP_Connectors"
CUSTOM_PREFIX = "OBJ:"
TEMPLATE_PROP = "esp_connector_template"

BUILTIN_SHAPES = [
    ('CYLINDER', "Cylinder", "Round pin", 'MESH_CYLINDER'),
    ('TAPERED', "Tapered", "Cone-like pin, easier to insert", 'MESH_CONE'),
    ('HEX', "Hexagon", "Six sided pin, locks rotation", 'MESH_CIRCLE'),
    ('BOX', "Box", "Square pin, locks rotation", 'MESH_CUBE'),
]
BUILTIN_IDS = {s[0] for s in BUILTIN_SHAPES}

SIZE_PRESETS = [
    ('SMALL', "Small", "~30% of the cut width", 0.30),
    ('MEDIUM', "Medium", "~45% of the cut width", 0.45),
    ('LARGE', "Large", "~60% of the cut width", 0.60),
    ('CUSTOM', "Custom", "Type width and height", 0.0),
]
HEIGHT_RATIO = 1.2  # protrusion height = diameter * ratio

# How tight the printed joint should be, as a multiple of the printer's own clearance.
# The value is the gap on EACH side, so the socket ends up twice that wider than the pin.
FIT_PRESETS = [
    ('PRESS', "Press", "Tight: tap it home, holds on its own", 0.5),
    ('SNUG', "Snug", "Firm by hand - the usual choice", 1.0),
    ('EASY', "Easy", "Slides together without force", 1.5),
    ('LOOSE', "Loose", "Free fit, leaves room for glue", 2.5),
    ('CUSTOM', "Custom", "Type the gap yourself", 0.0),
]


def fit_factor(preset):
    for ident, _n, _d, f in FIT_PRESETS:
        if ident == preset:
            return f
    return 1.0


def preset_factor(preset):
    for ident, _n, _d, f in SIZE_PRESETS:
        if ident == preset:
            return f
    return 0.45


# ----------------------------------------------------------------------------
# unit meshes
# ----------------------------------------------------------------------------
def unit_connector_bmesh(shape, custom_obj=None):
    bm = bmesh.new()
    if shape == 'CYLINDER':
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=32, radius1=0.5, radius2=0.5, depth=2.0)
    elif shape == 'TAPERED':
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=32, radius1=0.62, radius2=0.38, depth=2.0)
    elif shape == 'HEX':
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=6, radius1=0.5, radius2=0.5, depth=2.0)
    elif shape == 'BOX':
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(1.0, 1.0, 2.0), verts=bm.verts)
    elif shape == 'CUSTOM' and custom_obj is not None and custom_obj.type == 'MESH':
        bm.from_mesh(custom_obj.data)
        _normalise_unit(bm)
    else:
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=32, radius1=0.5, radius2=0.5, depth=2.0)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def unit_bmesh_from_mesh(me):
    """Bmesh from an existing connector mesh, taken verbatim.

    The preview pin carries the shape the user actually sees - every edit-mode change
    to it included - and it already lives in unit space, so re-fitting it the way an
    imported custom object is fitted would undo exactly those edits.
    """
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def _normalise_unit(bm):
    """Fit arbitrary geometry into the unit connector box (xy centred, z in [-1,1])."""
    if not bm.verts:
        return
    xs = [v.co.x for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    zs = [v.co.z for v in bm.verts]
    cx = (min(xs) + max(xs)) * 0.5
    cy = (min(ys) + max(ys)) * 0.5
    cz = (min(zs) + max(zs)) * 0.5
    ex = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    ez = max(max(zs) - min(zs), 1e-9)
    for v in bm.verts:
        v.co.x = (v.co.x - cx) / ex
        v.co.y = (v.co.y - cy) / ex
        v.co.z = (v.co.z - cz) / ez * 2.0


def _vert_islands(bm):
    """Connected groups of vertices - one per pin when a connector has several."""
    seen = set()
    out = []
    for v in bm.verts:
        if v.index in seen:
            continue
        stack = [v]
        seen.add(v.index)
        group = []
        while stack:
            cur = stack.pop()
            group.append(cur)
            for e in cur.link_edges:
                o = e.other_vert(cur)
                if o.index not in seen:
                    seen.add(o.index)
                    stack.append(o)
        out.append(group)
    return out


def _side_offset(v, radial_extra):
    """How far, in x/y, to move vertex `v` so every side face it touches moves out by `radial_extra`.

    The side faces are the ones that are not caps (their normals are not near +/-Z);
    each one's outward direction in the x/y plane is `m`, and the move `d` is the one
    with `d . m == radial_extra` for all of them - two faces meeting at a corner make
    the corner move along its bisector, by more than `radial_extra`, which is exactly
    what keeps both faces `radial_extra` away from the pin.
    """
    ms = []
    for f in v.link_faces:
        n = f.normal
        m = Vector((n.x, n.y))
        if abs(n.z) > 0.95 or m.length < 1e-6:
            continue
        ms.append(m.normalized())
    if not ms:
        return None
    a11 = sum(m.x * m.x for m in ms)
    a12 = sum(m.x * m.y for m in ms)
    a22 = sum(m.y * m.y for m in ms)
    bx = radial_extra * sum(m.x for m in ms)
    by = radial_extra * sum(m.y for m in ms)
    det = a11 * a22 - a12 * a12
    if det < 1e-6 * len(ms) ** 2:
        # every face it touches faces the same way (a vertex in the middle of a flat side)
        avg = sum(ms, Vector((0.0, 0.0)))
        return avg.normalized() * radial_extra if avg.length > 1e-9 else None
    d = Vector(((a22 * bx - a12 * by) / det, (a11 * by - a12 * bx) / det))
    if d.length > radial_extra * 4.0:  # a needle-sharp corner: do not shoot it out
        d = d.normalized() * radial_extra * 4.0
    return d


def _grow_for_socket(bm, scale, radial_extra, tip_extra):
    """Widen the unit shape so the socket clears the pin by `radial_extra` per side.

    Every side face of the shape is moved out by `radial_extra`, measured at the pin's
    real size (`scale`), whatever the shape: a cylinder, the flats of a hexagon, the
    slope of a tapered pin, or a connector edited in edit mode - a lobe stretched, or
    duplicated off to one side. Scaling the shape about its centre did this only for
    a shape as wide everywhere as at its widest: a hexagon's flats got 87% of the gap,
    and a tapered pin, widest at its buried end, got 61% at its tip.

    The tip of every separate piece (each pin of a multi-pin layout) moves `tip_extra`
    further out, so the socket is deeper than the pin.
    """
    if not bm.verts:
        return
    sx, sy, sz = scale
    to_real = Matrix.Diagonal((sx, sy, sz, 1.0))
    bmesh.ops.transform(bm, matrix=to_real, verts=bm.verts)
    bm.normal_update()
    if radial_extra > 0.0:
        moves = [(v, _side_offset(v, radial_extra)) for v in bm.verts]
        for v, d in moves:
            if d is not None:
                v.co.x += d.x
                v.co.y += d.y
    if tip_extra > 0.0:
        bm.verts.index_update()
        for group in _vert_islands(bm):
            ztip = max(v.co.z for v in group)
            if ztip <= 1e-9:
                continue
            f = (ztip + tip_extra) / ztip
            for v in group:
                if v.co.z > 0.0:
                    v.co.z *= f
    bmesh.ops.transform(bm, matrix=to_real.inverted(), verts=bm.verts)


def unit_layout_bmesh(shape, custom_obj=None, offsets=None):
    """The unit shape repeated at every `offsets` (x, y) position, in unit space.

    One entry, or none, is the plain single connector at the origin. Offsets are in
    the connector's own unit space: 1.0 is one connector width.
    """
    bm = unit_connector_bmesh(shape, custom_obj)
    offsets = [tuple(o) for o in (offsets or [])]
    if len(offsets) <= 1:
        if offsets and (abs(offsets[0][0]) > 1e-9 or abs(offsets[0][1]) > 1e-9):
            bmesh.ops.translate(bm, vec=(offsets[0][0], offsets[0][1], 0.0), verts=bm.verts)
        return bm
    original = list(bm.verts) + list(bm.edges) + list(bm.faces)
    base_verts = list(bm.verts)
    for ox, oy in offsets[1:]:
        dup = bmesh.ops.duplicate(bm, geom=original)
        verts = [g for g in dup["geom"] if isinstance(g, bmesh.types.BMVert)]
        bmesh.ops.translate(bm, vec=(ox, oy, 0.0), verts=verts)
    ox, oy = offsets[0]
    bmesh.ops.translate(bm, vec=(ox, oy, 0.0), verts=base_verts)
    return bm


def connector_mesh(shape, custom_obj, matrix, name, radial_extra=0.0, tip_extra=0.0, unit_mesh=None, offsets=None):
    """World space connector mesh. Extras (in BU) grow the shape for the socket.

    `unit_mesh` is the already built unit shape to use - the preview pin's own mesh -
    and it wins over `shape`/`custom_obj`, which only describe how to build one.
    `offsets` repeats a built shape at several unit-space (x, y) positions (Quick mode
    with more than one connector; a preview pin already carries its copies).
    """
    if unit_mesh is not None:
        bm = unit_bmesh_from_mesh(unit_mesh)
    else:
        bm = unit_layout_bmesh(shape, custom_obj, offsets)
    sx, sy, sz = matrix.to_scale()
    sx = abs(sx) or 1e-9
    sy = abs(sy) or 1e-9
    sz = abs(sz) or 1e-9
    if radial_extra > 0.0 or tip_extra > 0.0:
        _grow_for_socket(bm, (sx, sy, sz), radial_extra, tip_extra)
    bmesh.ops.transform(bm, matrix=matrix, verts=bm.verts)
    if matrix.is_negative:  # a mirrored matrix (`flip_matrix`) turns the faces inside out
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    me = mesh_utils.bmesh_to_mesh(bm, name)
    bm.free()
    return me


# ----------------------------------------------------------------------------
# frames
# ----------------------------------------------------------------------------
def connector_frame(center, protrude_dir, up_hint=None):
    """Rotation+translation matrix whose +Z is `protrude_dir`."""
    z = Vector(protrude_dir).normalized()
    if up_hint is None:
        up_hint = Vector((0.0, 0.0, 1.0)) if abs(z.z) < 0.9 else Vector((0.0, 1.0, 0.0))
    x = Vector(up_hint).cross(z)
    if x.length < 1e-6:
        x = Vector((1.0, 0.0, 0.0)).cross(z)
    x.normalize()
    y = z.cross(x).normalized()
    m = Matrix(
        (
            (x.x, y.x, z.x, center[0]),
            (x.y, y.y, z.y, center[1]),
            (x.z, y.z, z.z, center[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    return m


def connector_matrix(center, protrude_dir, width_bu, height_bu, up_hint=None):
    return connector_frame(center, protrude_dir, up_hint) @ Matrix.Diagonal((width_bu, width_bu, height_bu, 1.0))


def flip_matrix(matrix):
    """Same location, scale and layout, protrusion direction reversed: local Z mirrored.

    A half turn about local X would reverse Z too, but it also mirrors Y - every pin of a
    multi-pin layout, and any shape not symmetric in Y, would land somewhere else than
    the preview shows, possibly outside the part. The mirrored matrix is negative, which
    `connector_mesh` turns back right side out.
    """
    return matrix @ Matrix.Diagonal((1.0, 1.0, -1.0, 1.0))


# ----------------------------------------------------------------------------
# library of custom shapes
# ----------------------------------------------------------------------------
def library_collection(create=False):
    col = bpy.data.collections.get(LIB_COLLECTION)
    if col is None and create:
        col = bpy.data.collections.new(LIB_COLLECTION)
    return col


def ensure_library(context):
    scene = context.scene
    col = library_collection(create=True)
    if col.name not in scene.collection.children:
        scene.collection.children.link(col)
    created = []
    for ident, label, _d, _i in BUILTIN_SHAPES:
        name = f"ESP_Connector_{label}"
        if name in col.objects:
            continue
        bm = unit_connector_bmesh(ident)
        me = mesh_utils.bmesh_to_mesh(bm, name)
        bm.free()
        obj = bpy.data.objects.new(name, me)
        obj[TEMPLATE_PROP] = ident
        obj.display_type = 'WIRE'
        col.objects.link(obj)
        created.append(obj)
    col.hide_render = True
    return col, created


def library_objects():
    col = library_collection()
    if col is None:
        return []
    return [o for o in col.objects if o.type == 'MESH']


_enum_cache = []


def shape_enum_items(self, context):
    global _enum_cache
    items = [(i, n, d, ic, k) for k, (i, n, d, ic) in enumerate(BUILTIN_SHAPES)]
    for k, obj in enumerate(library_objects()):
        items.append(
            (CUSTOM_PREFIX + obj.name, obj.name, "Custom connector from the library", 'OUTLINER_OB_MESH', 100 + k)
        )
    _enum_cache = items
    return _enum_cache


def resolve_shape(identifier):
    """-> (shape_key, custom_object_or_None)"""
    if identifier in BUILTIN_IDS:
        return identifier, None
    if identifier and identifier.startswith(CUSTOM_PREFIX):
        obj = bpy.data.objects.get(identifier[len(CUSTOM_PREFIX) :])
        if obj is not None and obj.type == 'MESH':
            return 'CUSTOM', obj
    return 'CYLINDER', None
