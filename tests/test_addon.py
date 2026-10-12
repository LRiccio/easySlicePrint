# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Rafael Omodei and EasySlice Print contributors
"""Headless add-on tests: registration, plan workflow, build/return/approve, explode, export, panel drawing.
Run:  blender -b --python tests/test_addon.py
"""

import math
import os
import sys
import tempfile

import bmesh
import bpy
from mathutils import Matrix, Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import easy_slice_print  # noqa: E402
from easy_slice_print import overlay, plan, ui  # noqa: E402
from easy_slice_print.core import connectors, cutting, mesh_utils, surfaces  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


class DummyLayout:
    """Records UI calls; raises on unknown properties like Blender would."""

    def __init__(self):
        self.active = True
        self.enabled = True
        self.scale_y = 1.0

    def __getattr__(self, name):
        def call(*args, **kwargs):
            if name in ("prop",):
                data, prop = args[0], args[1]
                if not hasattr(data, prop) and prop not in data.bl_rna.properties:
                    raise AttributeError(f"layout.prop: {data} has no property '{prop}'")
            if name == "operator":
                idname = args[0]
                mod, op = idname.split('.')
                if not hasattr(getattr(bpy.ops, mod), op):
                    raise AttributeError(f"layout.operator: unknown operator '{idname}'")
                return DummyOp()
            return DummyLayout()

        return call


class DummyOp:
    def __setattr__(self, k, v):
        pass


def draw_all_panels(context):
    for cls in ui.CLASSES:
        if not hasattr(cls, "draw"):
            continue
        if hasattr(cls, "poll") and not cls.poll(context):
            continue
        fake = type("P", (), {})()
        fake.layout = DummyLayout()
        cls.draw(fake, context)
        if hasattr(cls, "draw_header"):
            cls.draw_header(fake, context)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = 'METRIC'
    sc.unit_settings.scale_length = 0.001
    sc.unit_settings.length_unit = 'MILLIMETERS'
    return sc


def make_cylinder(name="Figure", radius=10.0, height=60.0):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=True, segments=48, radius1=radius, radius2=radius, depth=height)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=3, use_grid_fill=True)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def plane_contact(z, diag, hit_x=10.0):
    d = plan.ContactData('STRAIGHT')
    d.verts, d.faces = surfaces.plane_patch(Vector((0, 0, z)), Vector((0, 0, 1)), Vector((1, 0, 0)), diag * 1.3)
    d.view_dir = Vector((-1, 0, 0))
    d.hit = Vector((hit_x, 0, z))
    d.through = Vector((-1, 0, 0))
    d.anchor = d.hit.copy()
    return d


def curve_contact(obj, diag, y_end=14.0):
    pts = [Vector((10.0, -14 + (14 + y_end) * i / 19, 15.0 + 3.0 * math.sin(i / 19 * math.pi * 2))) for i in range(20)]
    d = plan.ContactData('CURVED')
    d.points = pts
    d.view_dir = Vector((-1, 0, 0))
    d.extend = diag * 0.1
    built = plan.ribbon_surfaces(bpy.context, obj, pts, d.view_dir, d.extend, d.detail)
    if built is not None:
        preview, cutter, band = built
        d.verts, d.faces = preview
        d.cutter = cutter
        d.depth_range = band
        d.is_cut_face = True
    else:
        d.depth_range = (-2.0, 22.0)
        d.verts, d.faces = surfaces.ribbon_patch(pts, d.view_dir, 0.0, d.extend, depth_range=d.depth_range)
    d.hit = pts[10].copy()
    d.through = d.view_dir
    d.anchor = d.hit.copy()
    return d


def make_two_legs(name="Legs", at=(-30.0, 30.0), radius=10.0, height=60.0):
    bm = bmesh.new()
    for x in at:
        leg = bmesh.new()
        bmesh.ops.create_cone(
            leg, cap_ends=True, cap_tris=True, segments=48, radius1=radius, radius2=radius, depth=height
        )
        bmesh.ops.translate(leg, verts=list(leg.verts), vec=Vector((x, 0, 0)))
        bm.from_mesh(mesh_utils.bmesh_to_mesh(leg, f"_leg{x}"))
        leg.free()
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.context.view_layer.update()
    return obj


def world_bounds(obj):
    pts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return mn, mx


def failing_build():
    """`esp.build` when it is expected to fail: in background mode the error report raises."""
    try:
        return bpy.ops.esp.build()
    except RuntimeError:
        return {'CANCELLED'}


def test_build_shows_why_a_cut_failed():
    """A build that fails puts the diagnosis up; the next build, or the X, takes it down."""
    print("== a failed build paints where the cut stays joined")
    sc = reset_scene()
    obj = make_cylinder()
    s = sc.esp
    s.mode = 'PLAN'
    diag = mesh_utils.object_world_diagonal(obj)
    ctx = bpy.context
    # a plane that reaches only half way across the cylinder
    d = plan.ContactData('STRAIGHT')
    d.verts, d.faces = surfaces.rect_patch(
        Vector((-10, 0, 5)), Vector((0, 0, 1)), Vector((1, 0, 0)), 10.0, (-30.0, 30.0)
    )
    d.view_dir = Vector((-1, 0, 0))
    d.hit = Vector((-10, 0, 5))
    d.through = Vector((1, 0, 0))
    d.anchor = d.hit.copy()
    rec = plan.add_record(ctx, obj, 'STRAIGHT', [d])
    rec.add_pin = False
    res = failing_build()
    check(res == {'CANCELLED'} and not s.built, "the half cut does not build")
    shown = overlay.shown()
    check(shown is not None and shown.cut and shown.label == rec.name, "the diagnosis is up, named after the cut")
    check(shown.surface is not None, "the cut surface is painted")
    check(not overlay.hidden(), "the map shows while the cut does")
    rec.show = False
    check(overlay.hidden() and overlay.shown() is not None, "the eye hides the map with the surface, keeping it")
    rec.show = True
    check(not overlay.hidden(), "and brings it back")
    plan.set_plan_hidden(ctx, True)
    check(overlay.hidden(), "a hidden plan hides the map")
    plan.set_plan_hidden(ctx, False)
    check(not overlay.hidden(), "a shown plan shows it")
    pos, colors, _idx = shown.surface
    reds = [p for p, c in zip(pos, colors) if c[0] > 0.9 and c[1] < 0.4]
    greens = [p for p, c in zip(pos, colors) if c[1] > 0.8 and c[0] < 0.5]
    check(reds and max(p[0] for p in reds) > -1.0, "the surface is red where it stops inside the cylinder")
    check(greens and min(p[0] for p in greens) < -15.0, "and green where it cut through")
    draw_all_panels(ctx)
    check(True, "panels draw with the diagnosis box")
    res = bpy.ops.esp.hide_diagnosis()
    check(res == {'FINISHED'} and overlay.shown() is None, "the X takes it down")
    # Ctrl+Z on the cut: the map is not in the undo history, so it has to follow the cut by hand
    failing_build()
    check(overlay.shown() is not None, "failed again, map up again")
    overlay._undo_post(sc, None)
    check(overlay.shown() is not None, "an undo that keeps the cut keeps the map")
    plan.remove_record(ctx, 0)
    overlay._undo_post(sc, None)
    check(overlay.shown() is None, "an undo that takes the cut away takes the map with it")
    rec = plan.add_record(ctx, obj, 'STRAIGHT', [d])
    rec.add_pin = False
    # a plane across the whole cylinder builds, and puts no map up
    plan.remove_record(ctx, 0)
    rec = plan.add_record(ctx, obj, 'STRAIGHT', [plane_contact(5.0, diag)])
    rec.add_pin = False
    bpy.ops.esp.build()
    check(overlay.shown() is None, "a build that failed earlier leaves nothing behind once it works")
    bpy.ops.esp.return_to_plan()
    # a curve that stops half way across: what is painted is the surface on screen - the
    # ribbon trimmed to the model - not the flat ribbon the boolean is handed
    plan.remove_record(ctx, 0)
    half = curve_contact(obj, diag, y_end=0.0)
    check(half.cutter is not None, "the half curve has a separate cutter")
    rec = plan.add_record(ctx, obj, 'CURVED', [half])
    rec.add_pin = False
    res = failing_build()
    shown = overlay.shown()
    check(res == {'CANCELLED'} and shown is not None, "the half curve fails and is painted")
    pos, _colors, _idx = shown.surface
    reach = max(abs(p[0]) for p in pos)
    past = max(abs(v.x) for v in half.cutter[0])
    check(reach < 11.0 < past, f"painted on the trimmed preview (x to {reach:.1f}), not the cutter (x to {past:.1f})")
    check(len(pos) >= len(half.verts), "every preview vertex is in the painted surface")
    bpy.ops.esp.hide_diagnosis()
    # with the diagnosis off the build just fails
    s.diagnose_failed = False
    plan.remove_record(ctx, 0)
    rec = plan.add_record(ctx, obj, 'STRAIGHT', [d])
    rec.add_pin = False
    res = failing_build()
    check(res == {'CANCELLED'} and overlay.shown() is None, "with Show Why a Cut Failed off there is no map")
    s.diagnose_failed = True


def test_plane_section_preview():
    """A plane cut's preview is the model's cross section, and it follows the plane."""
    print("== plane cut surface tracks the model's cross section")
    reset_scene()
    legs = make_two_legs()

    ref = Vector((-30.0, 0.0, 5.0))
    span = (Vector((1, 0, 0)), -45.0, -15.0)
    patch = plan.straight_section(bpy.context, legs, Vector((0, 0, 5)), Vector((0, 0, 1)), span, ref)
    check(patch is not None, "the plane sections the model")
    d = plan.ContactData('STRAIGHT')
    plan.fill_straight_contact(d, patch, span)
    d.view_dir = Vector((-1, 0, 0))
    d.plane_co = Vector((0, 0, 5))
    d.plane_no = Vector((0, 0, 1))
    d.hit = Vector((-20.0, 0.0, 5.0))
    d.through = Vector((1, 0, 0))
    d.anchor = ref.copy()
    rec = plan.add_record(bpy.context, legs, 'STRAIGHT', [d])

    sobj = rec.surface_obj_a
    check(sobj is not None, "the preview surface was created")
    mn, mx = world_bounds(sobj)
    check(abs((mx.x - mn.x) - 20.2) < 0.6, f"preview is one leg wide, not the whole model ({mx.x - mn.x:.1f} ~ 20)")
    check(mx.x < -10.0, "preview sits on the leg the cut was drawn on")
    check(abs(mn.z - 5.0) < 0.1 and abs(mx.z - 5.0) < 0.1, "preview sits on the cut plane")

    cverts, _cfaces = plan.cutter_world_patch(sobj)
    check(len(cverts) == 4, f"the boolean is handed a quad, not the section ({len(cverts)} verts)")
    check(max(v.x for v in cverts) < 5.0, "and the quad stops in the gap before the other leg")

    # G on the preview: the plane moves, so the section is taken again where it landed
    sobj.matrix_world = Matrix.Translation(Vector((0.0, 0.0, 12.0))) @ sobj.matrix_world
    bpy.context.view_layer.update()
    plan.refresh_record_frames(bpy.context, rec)
    mn, mx = world_bounds(sobj)
    check(abs(mn.z - 17.0) < 0.1 and abs(mx.z - 17.0) < 0.1, f"moved preview re-sections at z=17 (got {mn.z:.1f})")
    check(abs((mx.x - mn.x) - 20.2) < 0.6, f"moved preview is still one leg wide ({mx.x - mn.x:.1f} ~ 20)")
    check(mx.x < -10.0, "moved preview stayed on the same leg")
    cverts, _cfaces = plan.cutter_world_patch(sobj)
    check(all(abs(v.z - 17.0) < 0.1 for v in cverts), "the quad followed the plane to z=17")
    check(max(v.x for v in cverts) < 5.0, "and still spares the other leg")

    centre = Vector(rec.center_a)
    check(abs(centre.z - 17.0) < 0.5 and centre.x < -10.0, f"the pin followed the section (centre {centre})")


def make_two_rods(name="Rods", at=(-40.0, 40.0), radius=10.0, height=80.0):
    """Two cylinders one behind the other along +Y - the far one must survive a cut."""
    bm = bmesh.new()
    for y in at:
        rod = bmesh.new()
        bmesh.ops.create_cone(
            rod, cap_ends=True, cap_tris=True, segments=48, radius1=radius, radius2=radius, depth=height
        )
        bmesh.ops.translate(rod, verts=list(rod.verts), vec=Vector((0.0, y, 0.0)))
        bm.from_mesh(mesh_utils.bmesh_to_mesh(rod, f"_rod{y}"))
        rod.free()
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.context.view_layer.update()
    return obj


def test_curve_cut_stops_at_the_model():
    """A curve drawn over the near rod must not reach through to the one behind it."""
    print("== curve cut follows the model instead of one flat depth")
    reset_scene()
    rods = make_two_rods()
    view = Vector((0, 1, 0))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    pts = []
    for i in range(20):
        probe = Vector((-14.0 + 28.0 * i / 19, -80.0, 4.0 * math.sin(i / 19 * math.pi)))
        hit, loc, _n, _d = mesh_utils.object_ray_cast(rods, probe, view, depsgraph, max_dist=400.0)
        pts.append(loc if hit else probe)

    built = plan.ribbon_surfaces(bpy.context, rods, pts, view, 6.0, 3)
    check(built is not None, "the stroke finds material to follow")
    preview, cutter, band = built
    check(max(v.y for v in cutter[0]) < 25.0, "the surface the boolean gets stops in the gap, not on the far rod")
    check(min(v.y for v in cutter[0]) < -50.0, "which still reaches out past the near rod")
    py = [v.y for v in preview[0]]
    check(max(py) < -25.0, f"the preview stops on the near rod's own back ({max(py):.1f} < -25)")

    vol0 = mesh_utils.mesh_volume(rods.data)
    spec = cutting.CutSpec(contacts=[cutting.ContactSpec(cutter[0], cutter[1], add_pin=False)], gap=0.17)
    a, b, _secs = cutting.perform_cut(bpy.context, rods, spec, ("A", "B"), bpy.context.scene.collection)
    check(
        mesh_utils.manifold_report(a.data)[:2] == (0, 0) and mesh_utils.manifold_report(b.data)[:2] == (0, 0),
        "both halves are closed manifold",
    )
    kept = mesh_utils.mesh_volume(a.data) + mesh_utils.mesh_volume(b.data)
    check(abs(kept - vol0) / vol0 < 0.02, f"volume preserved ({kept:.0f} of {vol0:.0f})")
    spans = []
    for part in (a, b):
        far = [part.matrix_world @ v.co for v in part.data.vertices if (part.matrix_world @ v.co).y > 20.0]
        spans.append((max(p.z for p in far) - min(p.z for p in far)) if far else 0.0)
    check(abs(max(spans) - 80.0) < 0.5, f"the far rod came through whole ({max(spans):.1f} ~ 80)")


def test_freehand_connector_fits_the_loop():
    """The pin on a freehand cut is measured from the loop, not from a bundle of rays."""
    print("== freehand connector is measured from the loop it was drawn on")
    reset_scene()
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=True, segments=64, radius1=12.0, radius2=12.0, depth=100.0)
    me = bpy.data.meshes.new("Rod")
    bm.to_mesh(me)
    bm.free()
    rod = bpy.data.objects.new("Rod", me)
    bpy.context.scene.collection.objects.link(rod)
    bpy.context.view_layer.update()

    # the loop as the tool keeps it: on the surface, where it was drawn
    pts = []
    for i in range(40):
        ang = 2.0 * math.pi * i / 40
        nrm = Vector((math.cos(ang), math.sin(ang), 0.0))
        pts.append(nrm * 12.0 + Vector((0.0, 0.0, 6.0 * math.cos(ang))))
    verts, faces = surfaces.loop_patch(pts, detail=3)

    centre, _normal, inscribed = plan.contact_frame(
        bpy.context, rod, verts, faces, center_hint=Vector((0, 0, 0)), is_cut_face=True
    )
    check(abs(inscribed - 24.0) < 1.0, f"pin sized to the rod ({inscribed:.2f} ~ 24)")
    check(centre.xy.length < 1.0, f"and placed in the middle of the cut face ({centre})")


def freehand_contact(obj, height=5.0, tilt=3.0, samples=96):
    """The contact `ESP_OT_cut_freehand.close_loop` produces for a slanted loop round `obj`."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    diag = mesh_utils.object_world_diagonal(obj)
    locs = []
    for k in range(samples):
        a = 2.0 * math.pi * k / samples
        d = Vector((math.cos(a), math.sin(a), 0.0))
        aim = Vector((0.0, 0.0, height + tilt * math.cos(a)))
        hit, loc, _n, _d = mesh_utils.object_ray_cast(obj, aim + d * diag, -d, depsgraph, max_dist=diag * 4.0)
        if hit:
            locs.append(loc)
    settings = bpy.context.scene.esp
    margin = plan.loop_margin(bpy.context, settings, locs, diag)
    verts, faces, cutter = plan.loop_surface(bpy.context, obj, locs, settings.surface_detail, margin)
    d = plan.ContactData('FREEHAND')
    d.points = locs
    d.margin = margin
    d.detail = settings.surface_detail
    d.verts, d.faces = verts, faces
    d.cutter = cutter
    d.is_cut_face = True
    d.center_hint = Vector((0.0, 0.0, height))
    d.anchor = locs[0]
    return d


def wall_z(obj, x_sign):
    """The z range of the part's wall at x = +-10 (the loop is at z = 8 there, and 2 on the far side)."""
    zs = [v.co.z for v in obj.data.vertices if v.co.x * x_sign > 9.5 and abs(v.co.y) < 1.0]
    return min(zs), max(zs)


def test_freehand_cut_lands_on_the_loop():
    """Quick and Plan mode both cut a freehand loop where it was drawn, through the cutter."""
    print("== a freehand loop cuts on its line in quick mode and in the plan")
    sc = reset_scene()
    obj = make_cylinder("Loop")
    s = sc.esp
    s.mode = 'QUICK'
    s.add_pin = False
    from easy_slice_print.ops_tools import quick_cut

    d = freehand_contact(obj)
    check(d.cutter is not None and len(d.cutter[0]) > len(d.verts), "the contact carries a cutter with a skirt")
    a, b, _secs = quick_cut(bpy.context, obj, [d])
    check({a.name, b.name} == {"Loop_UPPER", "Loop_LOWER"}, f"quick freehand names {a.name}/{b.name}")
    lower = a if a.name.endswith("LOWER") else b
    upper = b if lower is a else a
    top_near, top_far = wall_z(lower, 1)[1], wall_z(lower, -1)[1]
    bot_near, bot_far = wall_z(upper, 1)[0], wall_z(upper, -1)[0]
    print(f"  lower ends at z={top_near:.2f}/{top_far:.2f}, upper starts at z={bot_near:.2f}/{bot_far:.2f}")
    check(abs(top_near - 7.9) < 0.2 and abs(top_far - 1.9) < 0.2, "the lower part ends on the slanted line")
    check(abs(bot_near - 8.1) < 0.2 and abs(bot_far - 2.1) < 0.2, "and the upper part starts on it")

    sc = reset_scene()
    obj = make_cylinder("Planned")
    s = sc.esp
    s.mode = 'PLAN'
    s.keep_original = True
    ctx = bpy.context
    plan.add_record(ctx, obj, 'FREEHAND', [freehand_contact(obj)])
    rec = s.cuts[-1]
    sobj = rec.surface_obj_a
    check(sobj is not None and int(sobj.get("esp_cutter_n", 0)) == 3, "the plan stores the cutter triangulated")
    shown = plan.surface_world_patch(sobj)
    cut_v, cut_f = plan.cutter_world_patch(sobj)
    check(len(cut_v) > len(shown[0]) and all(len(f) == 3 for f in cut_f), "and reads it back as membrane plus skirt")
    rim = plan.surface_points(sobj)
    depsgraph = ctx.evaluated_depsgraph_get()
    drawn = max(abs(mesh_utils.object_surface_depth(obj, sobj.matrix_world @ p, depsgraph)[3]) for p in rim)
    check(drawn < 1e-3, f"the stored control points sit on the surface ({drawn:.4f} mm)")
    plan.rebuild_surface(sobj, context=ctx, target=obj)
    again = plan.cutter_world_patch(sobj)
    check(len(again[0]) == len(cut_v), "a rebuild keeps the skirt on the cutter")
    res = bpy.ops.esp.build()
    col = s.built_col
    check(res == {'FINISHED'} and col is not None and len(col.objects) == 2, "the plan builds two parts from it")
    if col is not None and len(col.objects) == 2:
        lower = min(col.objects, key=lambda o: world_bounds(o)[0].z)
        top_near, top_far = wall_z(lower, 1)[1], wall_z(lower, -1)[1]
        check(abs(top_near - 7.9) < 0.2 and abs(top_far - 1.9) < 0.2, "and the built part ends on the drawn line")


def test_printer_fit():
    """The Fit preset must decide how much wider the socket comes out than the pin."""
    print("== printer fit / clearance")
    sc = reset_scene()
    base = plan.printer_clearance_mm(bpy.context)
    check(abs(base - 0.10) < 1e-9, f"printer clearance defaults to 0.10 mm (got {base})")
    check(sc.esp.fit_preset == 'SNUG', f"new cuts start on the Snug fit (got {sc.esp.fit_preset})")

    expected = {'PRESS': 0.05, 'SNUG': 0.10, 'EASY': 0.15, 'LOOSE': 0.25}
    for preset, want in expected.items():
        got = plan.preset_clearance_mm(bpy.context, preset)
        check(abs(got - want) < 1e-9, f"{preset} fit -> {want:.2f} mm per side (got {got:.3f})")

    # the preset has to land in the field the panel shows and the build reads
    for preset, want in expected.items():
        sc.esp.fit_preset = preset
        check(abs(sc.esp.clearance_mm - want) < 1e-6, f"{preset} written into the gap field")
    sc.esp.fit_preset = 'CUSTOM'
    sc.esp.clearance_mm = 0.42
    plan.apply_fit(bpy.context, sc.esp)
    check(abs(sc.esp.clearance_mm - 0.42) < 1e-6, "Custom fit keeps the typed gap")

    # what actually gets printed: the socket cavity against the pin, same call the build makes
    matrix = connectors.connector_matrix(Vector((0, 0, 0)), Vector((0, 0, 1)), 6.0, 7.0)
    for gap in (0.05, 0.10, 0.25):
        pin = connectors.connector_mesh('CYLINDER', None, matrix, "pin")
        socket = connectors.connector_mesh('CYLINDER', None, matrix, "socket", radial_extra=gap)
        pmn, pmx = mesh_utils.mesh_bounds(pin)

        # measured wall to wall: the side faces' distance from the axis, not the corners
        def wall(me):
            return min(p.normal.dot(p.center) for p in me.polygons if abs(p.normal.z) < 0.5)

        widened = 2.0 * (wall(socket) - wall(pin))
        check(abs(widened - 2.0 * gap) < 1e-4, f"gap {gap:.2f} -> socket {widened:.3f} mm wider (want {2 * gap:.2f})")
        # the pin itself must never change: only the socket opens up
        check(abs((pmx.x - pmn.x) - 6.0) < 1e-4, f"pin stays {pmx.x - pmn.x:.3f} mm across at gap {gap:.2f}")
        for m in (pin, socket):
            mesh_utils.remove_mesh(m)


def test_version():
    """The sidebar must never show a version other than the one Blender installed."""
    print("== version")
    import tomllib

    from easy_slice_print import version as ver

    with open(os.path.join(ROOT, "easy_slice_print", "blender_manifest.toml"), "rb") as fh:
        manifest = tomllib.load(fh)
    check(manifest["version"] == ver.VERSION, f"version {ver.VERSION} == manifest {manifest['version']}")
    check(manifest["version"] == ui.VERSION, "the panel reads the manifest version")
    check(ver.STAGE in ("", "alpha", "beta"), f"stage {ver.STAGE!r}")


def test_register():
    print("== register")
    easy_slice_print.register()
    check(hasattr(bpy.types.Scene, "esp"), "scene.esp registered")
    check(hasattr(bpy.ops.esp, "cut_straight") and hasattr(bpy.ops.esp, "build"), "operators registered")


def test_plan_workflow():
    print("== plan workflow")
    sc = reset_scene()
    obj = make_cylinder()
    s = sc.esp
    s.mode = 'PLAN'
    diag = mesh_utils.object_world_diagonal(obj)
    ctx = bpy.context
    rec1 = plan.add_record(ctx, obj, 'STRAIGHT', [plane_contact(-10.0, diag)])
    rec2 = plan.add_record(ctx, obj, 'CURVED', [curve_contact(obj, diag)])
    check(len(s.cuts) == 2 and s.active_cut == 1, "two records added")
    check(
        rec1.surface_obj_a is not None and rec1.pin_obj_a is not None,
        "preview objects exist",
    )
    check(abs(rec1.inscribed_a - 20.0) < 1.0, f"record inscribed diameter {rec1.inscribed_a:.2f}")
    check(abs(rec1.pin_width_mm - 9.0) < 0.5, f"preset width from inscribed: {rec1.pin_width_mm:.2f} mm")
    # change settings on the record -> preview follows
    pin = rec1.pin_obj_a
    old_scale = pin.scale.copy()
    rec1.size_preset = 'LARGE'
    check(pin.scale.x > old_scale.x, "preset change rescaled the pin preview")
    rec1.pin_side = 'B'
    check(pin.matrix_world.col[2].xyz.z > 0.5, "swap side flipped the pin direction (+Z now up)")
    rec1.shape = 'HEX'
    check(pin.get("esp_shape") == 'HEX' and len(pin.data.polygons) == 8, "shape change rebuilt the pin mesh")
    # user moves the pin -> delta preserved on later changes
    pin.location.x += 2.0
    rec1.size_preset = 'SMALL'
    check(abs(pin.location.x - 2.0) < 1e-3, "user offset preserved through size change")
    # reshape the pin the way edit mode does -> the build spec has to carry that mesh
    for v in pin.data.vertices:
        v.co.x *= 3.0
        v.co.y *= 3.0
    spec = plan.record_spec(ctx, rec1, s, remesh=False)
    check(spec.contacts[0].pin_mesh is pin.data, "the spec carries the pin's own edited mesh")
    built = connectors.connector_mesh(
        spec.contacts[0].shape, None, spec.contacts[0].pin_matrix, "chk", unit_mesh=spec.contacts[0].pin_mesh
    )
    mn, mx = mesh_utils.mesh_bounds(built)
    check(mx.x - mn.x > pin.scale.x * 2.0, f"and builds the widened connector ({mx.x - mn.x:.2f} mm across)")
    plan.reset_pins(ctx, rec1)
    check(abs(pin.location.x) < 1e-3, "reset pin transform")
    stock = len(pin.data.polygons) == 8 and max(v.co.x for v in pin.data.vertices) < 0.6
    check(stock, "reset restores the stock shape")
    draw_all_panels(ctx)
    check(True, "panels draw (plan mode, with record)")
    # disable the curve cut, build
    rec2.enabled = False
    res = bpy.ops.esp.build()
    check(res == {'FINISHED'} and s.built, "build finished")
    col = s.built_col
    check(col is not None and len(col.objects) == 2, f"built collection has 2 parts ({len(col.objects) if col else 0})")
    check(
        obj.hide_get() and obj.name in bpy.data.collections[plan.BACKUP_COLLECTION].objects, "original hidden in backup"
    )
    for o in col.objects:
        non, boundary, _t = mesh_utils.manifold_report(o.data)
        check(non == 0 and boundary == 0, f"{o.name} closed manifold")
    # add the curve cut back and rebuild -> 3 parts
    rec2.enabled = True
    res = bpy.ops.esp.build()
    col = s.built_col
    check(res == {'FINISHED'} and len(col.objects) == 3, f"rebuild with 2 cuts -> 3 parts ({len(col.objects)})")
    names = sorted(o.name for o in col.objects)
    check(names == ["Figure_PART_001", "Figure_PART_002", "Figure_PART_003"], f"part names {names}")
    draw_all_panels(ctx)
    # explode / collapse
    s.explode_distance_mm = 5.0
    res = bpy.ops.esp.explode()
    check(res == {'FINISHED'} and s.exploded, "exploded")
    moved = [o for o in col.objects if o.location.length > 1e-6]
    check(len(moved) >= 2, "parts moved apart")
    res = bpy.ops.esp.collapse()
    check(all(o.location.length < 1e-6 for o in col.objects), "collapsed back")
    # export: the default folder is relative to the .blend, and this one was never saved
    cwd = os.getcwd()
    os.chdir(tempfile.mkdtemp(prefix="esp_cwd_"))
    try:
        s.export_folder = "//ESP_Export/"
        try:
            res = bpy.ops.esp.export_parts()
        except RuntimeError as e:  # an operator that reports an error raises from a script
            res = {'CANCELLED'} if "Save the .blend" in str(e) else {str(e)}
        check(
            res == {'CANCELLED'} and not os.path.exists("ESP_Export"), "an unsaved file does not export next to Blender"
        )
    finally:
        os.chdir(cwd)
    from easy_slice_print.ops_misc import fbx_global_scale

    us = sc.unit_settings
    us.scale_length = 1.0
    check(abs(fbx_global_scale(sc, 1000.0) - 1.0) < 1e-9, "FBX: a metre scene is written as it is")
    check(abs(fbx_global_scale(sc, 1.0) - 0.001) < 1e-9, "FBX: with 1 BU = 1 mm a unit comes out as a millimetre")
    us.scale_length = 0.001
    check(abs(fbx_global_scale(sc, 1.0) - 1.0) < 1e-6, "FBX: a millimetre scene is written as it is")
    tmp = tempfile.mkdtemp(prefix="esp_export_")
    s.export_folder = tmp
    for fmt in ('STL', 'OBJ', 'FBX'):
        s.export_format = fmt
        res = bpy.ops.esp.export_parts()
        files = [f for f in os.listdir(tmp) if f.lower().endswith(fmt.lower())]
        check(res == {'FINISHED'} and len(files) == 3, f"exported 3 {fmt} files")
    # return to plan
    res = bpy.ops.esp.return_to_plan()
    check(
        res == {'FINISHED'} and not s.built and bpy.data.collections.get("ESP_Built_Figure") is None, "returned to plan"
    )
    check(not obj.hide_get(), "original visible again")
    # build & approve without keeping the original
    s.keep_original = False
    bpy.ops.esp.build()
    res = bpy.ops.esp.approve()
    check(
        res == {'FINISHED'} and len(s.cuts) == 0 and bpy.data.objects.get("Figure") is None,
        "approve cleared plan and removed original",
    )
    check(len(bpy.data.collections["ESP_Built_Figure"].objects) == 3, "parts survived approve")
    draw_all_panels(ctx)


def section_contact(obj, z):
    """A plane cut at height z taken from the model's own cross section, as the tool makes it."""
    d = plan.ContactData('STRAIGHT')
    co, no = Vector((0, 0, z)), Vector((0, 0, 1))
    d.plane_co, d.plane_no = co, no
    d.center_hint, d.anchor = co, co
    patch = plan.straight_section(bpy.context, obj, co, no, None, co)
    plan.fill_straight_contact(d, patch, None)
    return d


def make_slab(name="Slab", size=(60.0, 10.0, 40.0)):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=size, verts=bm.verts)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def loose_pieces(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.index_update()
    n = len(connectors._vert_islands(bm))
    bm.free()
    return n


def test_several_connectors():
    print("== several connectors on one cut")
    sc = reset_scene()
    ctx = bpy.context
    s = sc.esp
    s.mode = 'PLAN'
    s.pin_count = 3
    obj = make_slab()
    rec = plan.add_record(ctx, obj, 'STRAIGHT', [section_contact(obj, 0.0)])
    pin = rec.pin_obj_a
    check(rec.pin_count == 3, "a new cut takes the Count of the settings")
    check(loose_pieces(pin.data) == 3, f"the preview shows three pins ({loose_pieces(pin.data)})")
    check(pin.get("esp_layout_count") == 3, "and records that all three fit")
    xs = sorted(
        (pin.matrix_world @ Vector((ox, oy, 0))).x
        for ox, oy in zip(list(pin["esp_layout"])[::2], list(pin["esp_layout"])[1::2])
    )
    check(
        xs[0] < -15.0 and xs[2] > 15.0, f"spread along the long side of the 60 x 10 face ({[round(x, 1) for x in xs]})"
    )

    moved = pin.location.x + 1.0
    pin.location.x = moved
    rec.fit_preset = 'EASY'
    check(abs(pin.location.x - moved) < 1e-3, "moving the group by hand survives a settings change")
    rec.pin_count = 2
    check(loose_pieces(pin.data) == 2, "a new Count rebuilds the preview")
    rec.pin_count = 3

    res = bpy.ops.esp.build()
    col = s.built_col
    check(res == {'FINISHED'} and col is not None and len(col.objects) == 2, "the build makes two parts")
    for o in col.objects:
        check(loose_pieces(o.data) == 1, f"{o.name} is one piece, every pin fused to it")
        non, boundary, _t = mesh_utils.manifold_report(o.data)
        check(non == 0 and boundary == 0, f"{o.name} closed manifold")
    a, b = sorted(col.objects, key=lambda o: mesh_utils.mesh_volume(o.data))
    check(
        mesh_utils.mesh_volume(b.data) - mesh_utils.mesh_volume(a.data) > 3 * 50.0, "three pins moved material across"
    )
    bpy.ops.esp.return_to_plan()

    # more than the face can take: preset pins shrink, then fewer are placed, and the panel says so
    rec.size_preset = 'CUSTOM'
    rec.pin_width_mm = 8.0
    rec.pin_count = 8
    check(pin.get("esp_layout_count") < 8, f"only {pin.get('esp_layout_count')} of 8 custom 8 mm pins fit")
    check(loose_pieces(pin.data) == pin.get("esp_layout_count"), "and the preview shows the ones that do")
    draw_all_panels(ctx)

    # one connector is still the connector it always was
    rec.pin_count = 1
    check(loose_pieces(pin.data) == 1 and list(pin["esp_layout"]) == [0.0, 0.0], "Count 1 is one centred pin")

    # Quick mode takes the same setting
    s.mode = 'QUICK'
    s.pin_count = 2
    from easy_slice_print.ops_tools import quick_cut

    slab = make_slab("QuickSlab")
    a, b, _secs = quick_cut(ctx, slab, [section_contact(slab, 0.0)])
    pinned = a if mesh_utils.mesh_volume(a.data) > mesh_utils.mesh_volume(b.data) else b
    lo, _hi = mesh_utils.mesh_bounds(pinned.data)
    stubs = [v.co for v in pinned.data.vertices if v.co.z < lo.z + 0.01] if lo.z < -0.5 else []
    check(lo.z < -0.5 and max(v.x for v in stubs) - min(v.x for v in stubs) > 20.0, "quick cut builds two pins apart")
    check(loose_pieces(a.data) == 1 and loose_pieces(b.data) == 1, "both quick parts are one piece")


def make_tube(name="Tube", outer=15.0, inner=11.0, height=60.0, segments=64):
    def cylinder(radius, depth, mesh_name):
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=radius, radius2=radius, depth=depth)
        me = bpy.data.meshes.new(mesh_name)
        bm.to_mesh(me)
        bm.free()
        return me

    solid = cylinder(outer, height, name + "_solid")
    bore = cylinder(inner, height * 1.2, name + "_bore")
    me = mesh_utils.boolean_mesh(bpy.context, solid, bore, 'DIFFERENCE', 'EXACT')
    mesh_utils.remove_mesh(solid)
    mesh_utils.remove_mesh(bore)
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    return obj


def test_connector_stays_in_the_material():
    """The pin is measured against the material, not against a surface that overstates it.

    A plane section is grown past the model so the boolean clears it: measured on that,
    a 4 mm wall looked 8 mm thick and a Large pin broke out of it. A freehand loop round
    a tube closes over the bore, so the pin went into the hollow.
    """
    print("== the connector stays in the material")
    reset_scene()
    tube = make_tube("ThinTube", outer=100.0, inner=96.0, height=80.0, segments=256)
    d = section_contact(tube, 0.0)
    _centre, _normal, inscribed = plan.contact_frame(
        bpy.context, tube, d.verts, d.faces, None, None, d.center_hint, True, d.grown
    )
    check(abs(inscribed - 4.0) < 0.3, f"a 4 mm wall measures 4 mm ({inscribed:.2f})")

    reset_scene()
    tube = make_tube("Bore", outer=15.0, inner=11.0, height=60.0)
    d = freehand_contact(tube, height=0.0, tilt=0.0)
    centre, _normal, inscribed = plan.contact_frame(
        bpy.context, tube, d.verts, d.faces, None, None, d.center_hint, True, d.grown
    )
    r = centre.xy.length
    check(11.0 < r < 15.0, f"a loop round a tube puts the pin in the wall (r {r:.2f})")
    check(inscribed < 4.5, f"sized to the wall, not to the bore ({inscribed:.2f})")


def test_pin_fits_the_depth_of_the_part():
    """A cut close to the end of the model gets a pin that stays inside it.

    The pin reaches its height into both parts, and a height taken from the width of the
    face alone came out through the far side of a thin slice - or, with the pin on the
    other side, its socket punched a hole through it.
    """
    print("== the pin fits the depth of the part")
    for side in ('A', 'B'):
        sc = reset_scene()
        s = sc.esp
        s.mode = 'QUICK'
        s.pin_side = side
        cube = make_slab(f"Block{side}", size=(40.0, 40.0, 40.0))
        a, b, _secs = __import__("easy_slice_print.ops_tools", fromlist=["quick_cut"]).quick_cut(
            bpy.context, cube, [section_contact(cube, 16.0)]
        )
        top = a if mesh_utils.mesh_bounds(a.data)[1].z > mesh_utils.mesh_bounds(b.data)[1].z else b
        bottom = b if top is a else a
        mn, mx = mesh_utils.mesh_bounds(top.data)
        check(mx.z < 20.01, f"side {side}: nothing sticks out of the top ({mx.z:.2f})")
        roof = [v for v in top.data.vertices if abs(v.co.z - 20.0) < 1e-3]
        check(len(roof) == 4, f"side {side}: the 4 mm slice keeps its top face whole ({len(roof)} vertices)")
        check(loose_pieces(top.data) == 1 and loose_pieces(bottom.data) == 1, f"side {side}: both parts whole")

    # a Custom height that is too long is left alone, and the panel says so
    sc = reset_scene()
    sc.esp.mode = 'PLAN'
    cube = make_slab("Custom", size=(40.0, 40.0, 40.0))
    rec = plan.add_record(bpy.context, cube, 'STRAIGHT', [section_contact(cube, 16.0)])
    check(rec.depth_a > 3.9 and rec.depth_a < 4.1, f"the record knows the slice is 4 mm deep ({rec.depth_a:.2f})")
    check(rec.pin_height_mm < 3.2, f"the preset height was cut to fit ({rec.pin_height_mm:.2f} mm)")
    rec.size_preset = 'CUSTOM'
    rec.pin_height_mm = 10.0
    labels = []

    class Spy(DummyLayout):
        def label(self, text="", **_kw):
            labels.append(text)
            return DummyLayout()

    from easy_slice_print import ui

    ui._depth_note(Spy(), bpy.context, rec)
    check(any("breaks through" in t for t in labels), f"the panel warns ({labels})")


def test_clear_plan_after_a_build_keeps_the_parts():
    """Clear Plan on a built plan keeps its parts, and the next plan cuts those parts."""
    print("== clear plan after a build")
    sc = reset_scene()
    sc.esp.mode = 'PLAN'
    cube = make_slab("Cube", size=(20.0, 20.0, 60.0))
    plan.add_record(bpy.context, cube, 'STRAIGHT', [section_contact(cube, 0.0)])
    check(bpy.ops.esp.build() == {'FINISHED'}, "built")
    bpy.ops.esp.clear_plan('EXEC_DEFAULT')
    kept = sorted(o.name for o in bpy.data.objects if o.name.startswith("Cube_PART"))
    check(kept == ["Cube_PART_001", "Cube_PART_002"], f"the parts are kept ({kept})")
    check(sc.esp.base_obj is None, "the plan no longer points at the original")
    upper = max((bpy.data.objects[n] for n in kept), key=lambda o: mesh_utils.mesh_bounds(o.data)[1].z)
    plan.add_record(bpy.context, upper, 'STRAIGHT', [section_contact(upper, 15.0)])
    check(sc.esp.base_obj == upper, f"a new cut on a part makes that part the source ({sc.esp.base_obj})")
    check(bpy.ops.esp.build() == {'FINISHED'}, "the new plan builds")
    check(
        cube.hide_get() and cube.name in bpy.data.collections[plan.BACKUP_COLLECTION].objects, "the original stays put"
    )
    col = sc.esp.built_col
    tops = sorted(round(mesh_utils.mesh_bounds(o.data)[1].z, 1) for o in col.objects)
    check(len(col.objects) == 2 and tops[-1] == 30.0 and tops[0] < 16.0, f"the part was cut, not the model ({tops})")
    bpy.ops.esp.return_to_plan()
    lower = [n for n in kept if bpy.data.objects.get(n) is not None]
    check(len(lower) == 2, f"Back to Plan leaves the earlier parts alone ({lower})")


def test_a_cancelled_build_goes_back_to_the_plan():
    """Esc on a build leaves no parts behind for the next build to pick up."""
    print("== a cancelled build goes back to the plan")
    sc = reset_scene()
    s = sc.esp
    s.mode = 'PLAN'
    bar = make_slab("Bar", size=(20.0, 20.0, 80.0))
    plan.add_record(bpy.context, bar, 'STRAIGHT', [section_contact(bar, -15.0)])
    plan.add_record(bpy.context, bar, 'STRAIGHT', [section_contact(bar, 15.0)])
    from easy_slice_print.ops_plan import ESP_OT_build

    class Build:
        build_steps = ESP_OT_build.build_steps
        _build_cuts = ESP_OT_build._build_cuts

        def report(self, *_args):
            pass

    steps = Build().build_steps(bpy.context)
    for label in steps:
        if label.startswith("cut 2/2"):
            break
    steps.close()  # what Esc does to the job
    parts = [o for o in bpy.data.objects if "_PART_" in o.name]
    check(not parts, f"no parts are left behind ({[o.name for o in parts]})")
    check(not any(r.built for r in s.cuts) and not s.built, "nothing is marked built")
    check(not bar.hide_get(), "the model is back in view")
    check(bpy.ops.esp.build() == {'FINISHED'}, "the next build runs")
    col = s.built_col
    names = sorted(o.name for o in col.objects)
    check(names == ["Bar_PART_001", "Bar_PART_002", "Bar_PART_003"], f"and makes exactly its own parts ({names})")


def test_back_to_plan_gives_every_source_back():
    """Back to Plan restores every part the build cut, also the source of a cut unticked since."""
    print("== back to plan gives every source back")
    sc = reset_scene()
    s = sc.esp
    s.mode = 'PLAN'
    cube = make_slab("Cube", size=(20.0, 20.0, 60.0))
    plan.add_record(bpy.context, cube, 'STRAIGHT', [section_contact(cube, 0.0)])
    r2 = plan.add_record(bpy.context, cube, 'STRAIGHT', [section_contact(cube, 15.0)])
    r3 = plan.add_record(bpy.context, cube, 'STRAIGHT', [section_contact(cube, -15.0)])
    r2.enabled = r3.enabled = False
    bpy.ops.esp.build()
    bpy.ops.esp.approve()  # the two unbuilt cuts move onto the two halves
    sources = sorted({r.target_obj.name for r in s.cuts})
    check(len(sources) == 2, f"the kept cuts sit on two parts ({sources})")
    for r in s.cuts:
        r.enabled = True
    check(bpy.ops.esp.build() == {'FINISHED'}, "built again")
    s.cuts[1].enabled = False  # unticked after the build (cuts[0] holds the base)
    bpy.ops.esp.return_to_plan()
    backup = bpy.data.collections[plan.BACKUP_COLLECTION].objects
    for name in sources:
        obj = bpy.data.objects.get(name)
        check(obj is not None and name not in backup and not obj.hide_get(), f"{name} is back in view")


def test_a_built_plan_is_not_edited_blind():
    """While built the plan's previews are hidden, so the plan's editing buttons wait."""
    print("== a built plan is not edited blind")
    sc = reset_scene()
    sc.esp.mode = 'PLAN'
    cube = make_slab("Cube", size=(20.0, 20.0, 60.0))
    plan.add_record(bpy.context, cube, 'STRAIGHT', [section_contact(cube, 0.0)])
    ops = ("new_cut", "select_pin", "reset_pin", "swap_pin_side", "edit_surface", "refresh_pins", "cut_straight")
    check(all(getattr(bpy.ops.esp, op).poll() for op in ops), "every editing button works on a plan")
    bpy.ops.esp.build()
    bpy.context.view_layer.objects.active = bpy.data.objects["Cube_PART_001"]
    blocked = [op for op in ops if not getattr(bpy.ops.esp, op).poll()]
    check(blocked == list(ops), f"none of them while it is built ({blocked})")
    bpy.ops.esp.return_to_plan()
    check(all(getattr(bpy.ops.esp, op).poll() for op in ops), "and all of them again after Back to Plan")


def test_the_plan_holds_its_objects_by_pointer():
    """Renaming the model, or a preview, after planning does not break the plan."""
    print("== the plan holds its objects by pointer")
    sc = reset_scene()
    s = sc.esp
    s.mode = 'PLAN'
    cube = make_slab("Statue", size=(20.0, 20.0, 60.0))
    rec = plan.add_record(bpy.context, cube, 'STRAIGHT', [section_contact(cube, 0.0)])
    pin, surface = rec.pin_obj_a, rec.surface_obj_a
    cube.name = "Statue_final"
    pin.name = "my pin"
    surface.name = "my plane"
    # a copy carries everything the original does; the plan must still mean the original
    copy = cube.copy()
    sc.collection.objects.link(copy)
    cube.name = "Statue_v2"
    check(bpy.ops.esp.build() == {'FINISHED'}, "the plan still builds")
    check(s.base_obj == cube and rec.target_obj == cube, f"on the renamed model, not its copy ({s.base_obj})")
    check(rec.pin_obj_a == pin and rec.surface_obj_a == surface, "renamed previews are the plan's still")
    col = s.built_col
    check(col is not None and len(col.objects) == 2, f"two parts ({len(col.objects) if col else 0})")
    col.name = "my parts"
    check(bpy.ops.esp.return_to_plan() == {'FINISHED'}, "back to the plan")
    check(bpy.data.collections.get("my parts") is None, "the renamed parts' collection went with its parts")

    # deleted in the viewport: the pointer kept the object in the file, but not in the scene
    for o in bpy.context.view_layer.objects:
        o.select_set(o == pin)
    bpy.context.view_layer.objects.active = pin
    bpy.ops.object.delete()
    check(plan.pin_of(rec, 0) is None, "a deleted connector preview is gone for the plan")
    plan.depsgraph_handler(sc, None)  # what Blender runs after the delete
    check(rec.pin_obj_a is None and pin.users == 0, f"and let go of, for Blender to drop ({pin.users} users)")
    for o in bpy.context.view_layer.objects:
        o.select_set(o == cube)
    bpy.context.view_layer.objects.active = cube
    bpy.ops.object.delete()
    check(plan.base_of(s) is None and plan.resolve_target(bpy.context) is not cube, "a deleted model is gone too")
    try:
        res = bpy.ops.esp.build()
    except RuntimeError:
        res = {'CANCELLED'}
    check(res == {'CANCELLED'}, "and the plan does not build on it")


def test_a_file_saved_with_names_opens_with_its_plan():
    """A .blend saved when the plan named its objects comes back with every reference."""
    print("== a file saved with names opens with its plan")
    sc = reset_scene()
    s = sc.esp
    s.mode = 'PLAN'
    cube = make_slab("Statue", size=(40.0, 20.0, 60.0))
    d1 = plan.ContactData('STRAIGHT')
    co, no = Vector((0, 0, 0)), Vector((1, 0, 0))
    d1.plane_co, d1.plane_no, d1.center_hint, d1.anchor = co, no, co, co
    plan.fill_straight_contact(d1, plan.straight_section(bpy.context, cube, co, no, None, co), None)
    plan.add_record(bpy.context, cube, 'STRAIGHT', [d1])
    rec = plan.add_record(bpy.context, cube, 'STRAIGHT', [section_contact(cube, 10.0)])
    check(bpy.ops.esp.build() == {'FINISHED'}, "built")
    # what a file of the name-based plan holds: names in strings, a tag on each object
    names = {"base_object": s.base_obj.name, "built_collection": s.built_col.name}
    s.base_obj = s.built_col = None
    for k, v in names.items():
        s[k] = v
    old = []
    for r in s.cuts:
        refs = {k: getattr(r, a).name if getattr(r, a) else "" for k, a in plan.LEGACY_RECORD.items()}
        refs["extras"] = [e.pin_obj.name for e in r.extras]
        old.append(refs)
        for k, a in plan.LEGACY_RECORD.items():
            setattr(r, a, None)
            r[k] = refs[k]
        for e in r.extras:
            e["pin"] = e.pin_obj.name
            e.pin_obj = None
    cube["esp_ref_name"] = cube.name
    extras = len(rec.extras)
    check(extras > 0, f"the second cut shows connectors on both halves ({extras})")
    path = os.path.join(tempfile.mkdtemp(), "named_plan.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)

    s = bpy.context.scene.esp
    objects = bpy.data.objects
    check(s.base_obj == objects["Statue"], f"the base is pointed at ({s.base_obj})")
    check(s.built_col == bpy.data.collections[names["built_collection"]], "and the built parts")
    for r, refs in zip(s.cuts, old):
        for k, a in plan.LEGACY_RECORD.items():
            want = objects.get(refs[k]) if refs[k] else None
            check(getattr(r, a) == want, f"{r.name}: {k} -> {a} ({getattr(r, a)})")
            check(k not in r, f"{r.name}: the old '{k}' is dropped")
        check([e.pin_obj.name for e in r.extras] == refs["extras"], f"{r.name}: its extra connectors")
        check(all("pin" not in e for e in r.extras), f"{r.name}: their old names are dropped")
    check("base_object" not in s and "esp_ref_name" not in objects["Statue"], "nothing of the names is left")
    check(bpy.ops.esp.return_to_plan() == {'FINISHED'}, "back to the plan")
    check(bpy.data.collections.get(names["built_collection"]) is None, "the parts it found went")
    check(bpy.ops.esp.build() == {'FINISHED'}, "and the plan builds again")
    check(len(s.built_col.objects) == 4, f"four parts ({len(s.built_col.objects)})")
    check(plan.migrate_file() is False, "a second pass finds nothing to bring over")


def test_a_thin_socket_wall_is_flagged():
    """A socket that leaves too little material round it says so, and how much is left."""
    print("== a thin socket wall is flagged")
    sc = reset_scene()
    s = sc.esp
    s.mode = 'PLAN'
    slab = make_slab("Plank", size=(60.0, 10.0, 40.0))  # 10 mm thick
    rec = plan.add_record(bpy.context, slab, 'STRAIGHT', [section_contact(slab, 0.0)])
    plan.measure_all_walls(bpy.context)
    check(not plan.record_thin_walls(bpy.context, rec), "a preset pin keeps a wall the printer can print")
    wall = rec.pin_obj_a.get(plan.WALL_KEY)
    check(wall is not None and wall >= 0.8 - plan.WALL_TOLERANCE_MM, f"and its wall is measured ({wall})")

    rec.size_preset = 'CUSTOM'
    rec.pin_width_mm = 9.0  # 0.5 mm each side, less the clearance
    rec.pin_height_mm = 6.0
    plan.measure_all_walls(bpy.context)
    expect = 0.5 - rec.clearance_mm
    found = plan.record_thin_walls(bpy.context, rec)
    check(len(found) == 1, f"a 9 mm pin in a 10 mm plank is flagged ({found})")
    if found:
        _where, wall, near = found[0]
        check(abs(wall - expect) < 0.06 and near == "", f"with the wall left: {wall:.2f} mm, not {expect:.2f}")
    draw_all_panels(bpy.context)  # the note and the list icon draw

    rec.pin_width_mm = 6.0
    rec.pin_obj_a.location.y += 3.0  # moved by hand, half out of the plank
    plan.measure_all_walls(bpy.context)
    found = plan.record_thin_walls(bpy.context, rec)
    check(found and found[0][1] == 0.0, f"a socket off the edge breaks out ({found})")
    plan.reset_pins(bpy.context, rec)

    # another cut planned close to the socket: the connector moves clear of it; moved
    # back by hand, the wall against the cut is flagged
    pin_x = rec.pin_obj_a.matrix_world.translation.x
    x = pin_x + 3.6  # 0.6 mm past the 6 mm pin
    co, no = Vector((x, 0.0, 10.0)), Vector((1.0, 0.0, 0.0))
    d = plan.ContactData('STRAIGHT')
    d.plane_co, d.plane_no, d.center_hint, d.anchor = co, no, co, co
    plan.fill_straight_contact(d, plan.straight_section(bpy.context, slab, co, no, None, co), None)
    s.add_pin = False  # only its surface near the socket
    other = plan.add_record(bpy.context, slab, 'STRAIGHT', [d])
    plan.measure_all_walls(bpy.context)
    wall = rec.pin_obj_a.get(plan.WALL_KEY)
    check(
        not plan.record_thin_walls(bpy.context, rec) and wall >= 0.8 - plan.WALL_TOLERANCE_MM,
        f"the connector moved clear of the later cut ({wall} mm)",
    )
    rec.pin_obj_a.location.x += pin_x - rec.pin_obj_a.matrix_world.translation.x
    bpy.context.view_layer.update()
    plan.measure_all_walls(bpy.context)
    found = plan.record_thin_walls(bpy.context, rec)
    expect = 0.6 - rec.clearance_mm - other.cut_gap_mm * 0.5
    check(len(found) == 1 and found[0][2] == other.name, f"the wall is against the other cut ({found})")
    if found:
        check(abs(found[0][1] - expect) < 0.06, f"{found[0][1]:.2f} mm left, not {expect:.2f}")
    other.enabled = False
    plan.measure_all_walls(bpy.context)
    check(not plan.record_thin_walls(bpy.context, rec), "a cut left out of the build is not in the way")

    # a later cut across an earlier one: its socket running through the earlier pin's root
    # runs through solid material, and is fine; and the earlier connector is placed clear
    # of the later cut, which used to slice right past its socket
    for z in (10.0, 0.0):
        sc = reset_scene()
        sc.esp.mode = 'PLAN'
        box = make_slab("Box", size=(40.0, 20.0, 60.0))
        d = plan.ContactData('STRAIGHT')
        co, no = Vector((0, 0, 0)), Vector((1, 0, 0))
        d.plane_co, d.plane_no, d.center_hint, d.anchor = co, no, co, co
        plan.fill_straight_contact(d, plan.straight_section(bpy.context, box, co, no, None, co), None)
        first = plan.add_record(bpy.context, box, 'STRAIGHT', [d])
        later = plan.add_record(bpy.context, box, 'STRAIGHT', [section_contact(box, z)])
        plan.measure_all_walls(bpy.context)
        check(not plan.record_thin_walls(bpy.context, later), f"z={z:.0f}: the later cut's sockets are fine")
        found = plan.record_thin_walls(bpy.context, first)
        wall = first.pin_obj_a.get(plan.WALL_KEY)
        check(
            not found and wall >= 0.8 - plan.WALL_TOLERANCE_MM, f"z={z:.0f}: the earlier socket keeps its wall ({wall})"
        )
    pin = first.pin_obj_a
    pin.location.z -= pin.matrix_world.translation.z  # moved by hand onto the later cut
    bpy.context.view_layer.update()
    plan.measure_all_walls(bpy.context)
    found = plan.record_thin_walls(bpy.context, first)
    check(found and found[0][1] == 0.0 and found[0][2] == later.name, f"moved onto it, it runs into the cut ({found})")
    draw_all_panels(bpy.context)

    # the measurement follows the plan by itself, a moment after a change
    if bpy.app.timers.is_registered(plan._refresh_walls):
        bpy.app.timers.unregister(plan._refresh_walls)
    later.enabled = False
    check(bpy.app.timers.is_registered(plan._refresh_walls), "a change schedules the walls")
    plan._refresh_walls()
    check(not plan.record_thin_walls(bpy.context, first), "and they are measured again")


def test_connectors_leave_the_minimum_wall():
    """A preset connector is sized and spread to leave Minimum Wall round every socket."""
    print("== connectors leave the minimum wall")
    real = plan.min_wall_mm
    try:
        for wall in (0.8, 1.2):
            plan.min_wall_mm = lambda _context, wall=wall: wall
            sc = reset_scene()
            s = sc.esp
            s.mode = 'PLAN'
            s.size_preset = 'LARGE'  # 60% of a 4 mm face would leave 0.7 mm a side, less the clearance
            plank = make_slab("Plank", size=(60.0, 4.0, 40.0))
            rec = plan.add_record(bpy.context, plank, 'STRAIGHT', [section_contact(plank, 0.0)])
            plan.measure_all_walls(bpy.context)
            got = rec.pin_obj_a.get(plan.WALL_KEY)
            what = f"Minimum Wall {wall}: a Large pin on a 4 mm plank"
            check(rec.pin_width_mm < 0.6 * 4.0, f"{what} is made thinner ({rec.pin_width_mm:.2f} mm)")
            check(got is not None and got >= wall - plan.WALL_TOLERANCE_MM, f"{what} leaves the wall ({got} mm)")
            check(not plan.record_thin_walls(bpy.context, rec), f"{what} is not flagged")
            u = plan.mm(bpy.context)
            tip = rec.tip_extra_mm * u if rec.asymmetric else 0.0
            check(
                abs(plan.depth_limit_bu(bpy.context, rec, 10.0 * u) - (10.0 * u - tip - wall * u)) < 1e-6,
                f"Minimum Wall {wall}: and past the tip of a pin on a shallow part",
            )

            # several on one face: the rim between two sockets is the wall too
            s = reset_scene().esp
            s.mode = 'PLAN'
            s.pin_count = 3
            slab = make_slab("Slab", size=(22.0, 10.0, 40.0))  # tight: the rims decide
            rec = plan.add_record(bpy.context, slab, 'STRAIGHT', [section_contact(slab, 0.0)])
            pin = rec.pin_obj_a
            flat = list(pin["esp_layout"])
            centres = [pin.matrix_world @ Vector((x, y, 0.0)) for x, y in zip(flat[::2], flat[1::2])]
            socket = rec.pin_width_mm * u * 0.5 + rec.clearance_mm * u
            gaps = [(a - b).length - 2.0 * socket for i, a in enumerate(centres) for b in centres[i + 1 :]]
            check(
                len(centres) > 1 and min(gaps) >= wall * u - 1e-4,
                f"Minimum Wall {wall}: {len(centres)} sockets, {min(gaps, default=0):.2f} mm apart",
            )
            plan.measure_all_walls(bpy.context)
            got = pin.get(plan.WALL_KEY)
            check(got is not None and got >= wall - plan.WALL_TOLERANCE_MM, f"and to the outside ({got} mm)")
    finally:
        plan.min_wall_mm = real


def test_a_connector_stops_short_of_a_later_cut():
    """A pin reaching towards a later cut of the plan stops short of it, its walls kept."""
    print("== a connector stops short of a later cut")
    sc = reset_scene()
    s = sc.esp
    s.mode = 'PLAN'
    block = make_slab("Block", size=(40.0, 40.0, 60.0))
    rec = plan.add_record(bpy.context, block, 'STRAIGHT', [section_contact(block, 0.0)])
    pin = rec.pin_obj_a
    reach = abs(pin.matrix_world.col[2].xyz.length)
    check(reach > 8.0, f"alone, the pin reaches {reach:.1f} mm")
    later = plan.add_record(bpy.context, block, 'STRAIGHT', [section_contact(block, 6.0)])
    plan.add_record(bpy.context, block, 'STRAIGHT', [section_contact(block, -6.0)])
    u = plan.mm(bpy.context)
    reach = pin.matrix_world.col[2].xyz.length / u
    room = 6.0 - later.cut_gap_mm * 0.5 - (rec.tip_extra_mm if rec.asymmetric else 0.0) - 0.8
    check(reach <= room + 0.05, f"with cuts 6 mm either side, it stops at {reach:.2f} mm (room {room:.2f})")
    plan.measure_all_walls(bpy.context)
    check(not plan.record_thin_walls(bpy.context, rec), f"and keeps its walls ({pin.get(plan.WALL_KEY)})")
    plan.remove_record(bpy.context, 2)
    plan.remove_record(bpy.context, 1)
    check(pin.matrix_world.col[2].xyz.length / u > 8.0, "the later cuts gone, it is long again")


def test_a_cut_across_earlier_cuts_cuts_every_part():
    """A later cut runs across every part an earlier cut of the same plan made.

    A full width plane under a cut that had split the model in two used to cut only
    the half nearest its anchor - "3 parts from 2 cuts" - and its connector, placed on
    the whole section, could land in the other half as a floating pin with no socket.
    """
    print("== a cut across earlier cuts cuts every part")
    from mathutils.bvhtree import BVHTree

    for z, count in ((10.0, 1), (10.0, 3), (0.0, 1)):
        sc = reset_scene()
        s = sc.esp
        s.mode = 'PLAN'
        s.pin_count = count
        box = make_slab("Box", size=(40.0, 20.0, 60.0))
        d1 = plan.ContactData('STRAIGHT')
        co, no = Vector((0, 0, 0)), Vector((1, 0, 0))
        d1.plane_co, d1.plane_no, d1.center_hint, d1.anchor = co, no, co, co
        plan.fill_straight_contact(d1, plan.straight_section(bpy.context, box, co, no, None, co), None)
        plan.add_record(bpy.context, box, 'STRAIGHT', [d1])
        plan.add_record(bpy.context, box, 'STRAIGHT', [section_contact(box, z)])
        what = f"plane at z={z:.0f}, {count} pin(s)"
        check(bpy.ops.esp.build() == {'FINISHED'}, f"{what}: built")
        col = s.built_col
        parts = list(col.objects)
        check(len(parts) == 4, f"{what}: four parts from two cuts ({len(parts)})")
        for o in parts:
            check(loose_pieces(o.data) == 1, f"{what}: {o.name} is one piece - no floating pin")
        # no pin reaches into another part - a socket an earlier cut carved, and its pin
        trees = {
            o.name: BVHTree.FromPolygons([v.co for v in o.data.vertices], [p.vertices for p in o.data.polygons])
            for o in parts
        }
        clashes = [(a, b) for a in trees for b in trees if a < b and trees[a].overlap(trees[b])]
        check(not clashes, f"{what}: no part runs into another ({clashes})")
        uppers = [o for o in parts if mesh_utils.mesh_bounds(o.data)[1].z > 29.0]
        with_pins = [o for o in uppers if mesh_utils.mesh_bounds(o.data)[0].z < z - 1.0]
        check(len(with_pins) == 2, f"{what}: both upper parts carry a pin ({[o.name for o in with_pins]})")


def make_figure(name="Figure2"):
    """Two legs standing on a base, one closed mesh: what a Base Split is for."""

    def solid(build, mesh_name):
        bm = bmesh.new()
        build(bm)
        me = bpy.data.meshes.new(mesh_name)
        bm.to_mesh(me)
        bm.free()
        return me

    base = solid(
        lambda bm: (
            bmesh.ops.create_cube(bm, size=1.0),
            bmesh.ops.scale(bm, vec=(60.0, 30.0, 8.0), verts=bm.verts),
            bmesh.ops.translate(bm, vec=(0.0, 0.0, -3.0), verts=bm.verts),
        ),
        name + "_base",
    )
    legs = solid(
        lambda bm: [
            bmesh.ops.create_cone(
                bm,
                cap_ends=True,
                segments=32,
                radius1=6.0,
                radius2=6.0,
                depth=40.0,
                matrix=Matrix.Translation((x, 0.0, 20.0)),
            )
            for x in (-15.0, 15.0)
        ],
        name + "_legs",
    )
    me = mesh_utils.boolean_mesh(bpy.context, base, legs, 'UNION', 'EXACT')
    mesh_utils.remove_mesh(base)
    mesh_utils.remove_mesh(legs)
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    return obj


def plane_section_contact(obj, co, no, span=None):
    """A plane cut through `co`; `span` limits it to what a stroke ran across, as the tool does."""
    d = plan.ContactData('STRAIGHT')
    co, no = Vector(co), Vector(no)
    d.plane_co, d.plane_no, d.center_hint, d.anchor = co, no, co, co
    plan.fill_straight_contact(d, plan.straight_section(bpy.context, obj, co, no, span, co), span)
    return d


def test_base_split_preview_points_like_the_build():
    """Both pins of a Base Split are previewed pointing from the legs into the base.

    The second contact drawn the other way round faces down instead of up; its preview
    pin used to point into the legs, and only the build turned it the right way.
    """
    print("== base split preview points like the build")
    sc = reset_scene()
    sc.esp.mode = 'PLAN'
    fig = make_figure()
    stroke = (Vector((1.0, 0.0, 0.0)), -8.0, 8.0)  # a stroke across one leg
    d0 = plane_section_contact(fig, (-15.0, 0.0, 3.0), (0.0, 0.0, 1.0), stroke)
    d1 = plane_section_contact(fig, (15.0, 0.0, 3.0), (0.0, 0.0, -1.0), stroke)  # drawn the other way
    rec = plan.add_record(bpy.context, fig, 'STRAIGHT', [d0, d1])
    dirs = [p.matrix_world.col[2].xyz.z for p in (rec.pin_obj_a, rec.pin_obj_b)]
    check(dirs[0] < 0.0 and dirs[1] < 0.0, f"both preview pins point down into the base ({dirs[0]:.1f}, {dirs[1]:.1f})")
    check(rec.normal_b[2] > 0.0, "the second contact's + side is the legs, like the first's")
    plan.refresh_record_frames(bpy.context, rec)
    pin_b = rec.pin_obj_b
    check(pin_b.matrix_world.col[2].xyz.z < 0.0, "and still after the frames are measured again")
    check(bpy.ops.esp.build() == {'FINISHED'}, "the split builds")
    parts = list(sc.esp.built_col.objects)
    legs = [o for o in parts if mesh_utils.mesh_bounds(o.data)[1].z > 30.0]
    base = [o for o in parts if o not in legs]
    lo, hi = mesh_utils.mesh_bounds(legs[0].data) if len(legs) == 1 else (Vector(), Vector())
    check(len(legs) == 1 and hi.x - lo.x > 35.0, "both legs come off as the pin part")
    check(len(base) == 1 and loose_pieces(base[0].data) == 1, "the base is the other part, whole")
    stubs = [v.co for v in legs[0].data.vertices if v.co.z < 2.5] if legs else []
    check(any(v.x < 0 for v in stubs) and any(v.x > 0 for v in stubs), "each leg carries its pin down into the base")
    bpy.ops.esp.return_to_plan()

    # drawn the same way round, nothing is turned
    sc = reset_scene()
    sc.esp.mode = 'PLAN'
    fig = make_figure("Figure3")
    d0 = plane_section_contact(fig, (-15.0, 0.0, 3.0), (0.0, 0.0, 1.0), stroke)
    d1 = plane_section_contact(fig, (15.0, 0.0, 3.0), (0.0, 0.0, 1.0), stroke)
    rec = plan.add_record(bpy.context, fig, 'STRAIGHT', [d0, d1])
    check(rec.normal_b[2] > 0.0, "a second contact facing the right way is left as it is")
    check(rec.pin_obj_b.matrix_world.col[2].xyz.z < 0.0, "and its pin points down")


def test_the_preview_shows_every_connector_a_build_makes():
    """A cut across an earlier one shows a connector on each part it will cut, and Build makes those."""
    print("== the preview shows every connector a build makes")
    sc = reset_scene()
    s = sc.esp
    s.mode = 'PLAN'
    box = make_slab("Box", size=(40.0, 20.0, 60.0))
    first = plan.add_record(bpy.context, box, 'STRAIGHT', [plane_section_contact(box, (0, 0, 0), (1, 0, 0))])
    rec = plan.add_record(bpy.context, box, 'STRAIGHT', [section_contact(box, 10.0)])
    check(len(rec.extras) == 1, f"one more connector, for the other half ({len(rec.extras)})")
    s.active_cut = 1
    draw_all_panels(bpy.context)  # the plan panel lists the extra connector
    pins = [rec.pin_obj_a] + [e.pin_obj for e in rec.extras]
    xs = sorted(p.matrix_world.translation.x for p in pins)
    check(xs[0] < -2.0 and xs[1] > 2.0, f"one connector in each half ({[round(x, 1) for x in xs]})")

    first.enabled = False
    check(len(rec.extras) == 0, "with the first cut left out, the face is whole again: one connector")
    first.enabled = True
    check(len(rec.extras) == 1, "and back when it is in")

    # moved by hand, the extra connector is built where it was left
    pins = [rec.pin_obj_a] + [e.pin_obj for e in rec.extras]
    extra = rec.extras[0].pin_obj
    extra.location.y += 2.0
    bpy.context.view_layer.update()
    planned = sorted((p.matrix_world.translation.x, p.matrix_world.translation.y) for p in pins)
    tips = sorted(round(p.matrix_world.translation.z - p.matrix_world.col[2].xyz.length, 1) for p in pins)
    check(bpy.ops.esp.build() == {'FINISHED'}, "built")
    col = s.built_col
    built = []
    for o in col.objects:
        if mesh_utils.mesh_bounds(o.data)[1].z < 29.0:
            continue  # a lower part: the pins hang from the upper ones
        stub = [v.co for v in o.data.vertices if v.co.z < 9.5]
        if stub:  # the middle of the pin below the cut face
            built.append(
                (
                    (min(v.x for v in stub) + max(v.x for v in stub)) / 2,
                    (min(v.y for v in stub) + max(v.y for v in stub)) / 2,
                )
            )
    built.sort()
    close = len(built) == len(planned) and all(
        abs(a[0] - b[0]) < 0.6 and abs(a[1] - b[1]) < 0.6 for a, b in zip(built, planned)
    )
    check(close, f"the pins are built where the preview shows them ({built} vs {planned})")
    lows = sorted(
        round(mesh_utils.mesh_bounds(o.data)[0].z, 1) for o in col.objects if mesh_utils.mesh_bounds(o.data)[1].z > 29.0
    )
    check(
        all(abs(a - b) < 0.2 for a, b in zip(lows, tips)), f"and as long as the preview shows them ({lows} vs {tips})"
    )

    bpy.ops.esp.return_to_plan()

    # the earlier cut's connector moved and made bigger: the later cut's preview follows it
    pin1 = first.pin_obj_a
    pin1.location.z += 14.0
    bpy.context.view_layer.update()
    plan.depsgraph_handler(sc, None)  # what Blender runs after the move
    first.size_preset = 'LARGE'
    plan._refresh_pending_after()  # what the timer runs a moment later
    planned = sorted(tuple(round(c, 1) for c in p.matrix_world.translation) for p in plan.record_pins(rec))
    used = []
    real_fit = plan.fit_depth

    def spy(context, record, spec, part):
        out = real_fit(context, record, spec, part)
        if record.name == rec.name:
            used.extend(tuple(round(c, 1) for c in c_.pin_matrix.translation) for c_ in out.contacts if c_.add_pin)
        return out

    plan.fit_depth = spy
    try:
        bpy.ops.esp.build()
    finally:
        plan.fit_depth = real_fit
    check(sorted(used) == planned, f"the build follows the refreshed preview ({sorted(used)} vs {planned})")
    bpy.ops.esp.return_to_plan()

    plan.remove_record(bpy.context, 0)
    check(len(s.cuts[0].extras) == 0, "deleting the first cut takes the extra connector away")

    # three cuts: x, y, then z - the last one shows a connector on each of its four parts,
    # none of them inside an earlier connector, and the build makes those four
    sc = reset_scene()
    s = sc.esp
    s.mode = 'PLAN'
    box = make_slab("Box3", size=(40.0, 20.0, 60.0))
    plan.add_record(bpy.context, box, 'STRAIGHT', [plane_section_contact(box, (0, 0, 0), (1, 0, 0))])
    plan.add_record(bpy.context, box, 'STRAIGHT', [plane_section_contact(box, (0, 0, 0), (0, 1, 0))])
    third = plan.add_record(bpy.context, box, 'STRAIGHT', [section_contact(box, 10.0)])
    check(len(third.extras) == 3, f"four connectors for four parts ({1 + len(third.extras)})")
    check(bpy.ops.esp.build() == {'FINISHED'}, "built")
    col = s.built_col
    check(len(col.objects) == 8, f"eight parts ({len(col.objects)})")
    uppers = [o for o in col.objects if mesh_utils.mesh_bounds(o.data)[1].z > 29.0]
    with_pins = [o for o in uppers if mesh_utils.mesh_bounds(o.data)[0].z < 9.0]
    check(len(with_pins) == 4, f"each upper part carries its pin ({len(with_pins)})")

    # a sliver too narrow for a pin: the plan shows none on it, and the build makes none
    sc = reset_scene()
    s = sc.esp
    s.mode = 'PLAN'
    box = make_slab("Sliver", size=(40.0, 20.0, 60.0))
    plan.add_record(bpy.context, box, 'STRAIGHT', [plane_section_contact(box, (19.1, 0, 0), (1, 0, 0))])
    d = section_contact(box, 10.0)
    d.anchor = Vector((19.8, 0.0, 10.0))  # drawn over the sliver
    rec = plan.add_record(bpy.context, box, 'STRAIGHT', [d])
    shown = 1 + len(rec.extras)
    check(shown == 1 and rec.pin_obj_a.matrix_world.translation.x < 18.0, "one connector, off the sliver")
    check(bpy.ops.esp.build() == {'FINISHED'}, "the sliver builds")


def test_cut_after_approve():
    """A cut drawn on an approved part cuts that part, not the model it came from."""
    print("== cutting an approved part")
    sc = reset_scene()
    obj = make_cylinder("Chain")  # z from -30 to 30
    s = sc.esp
    s.mode = 'PLAN'
    s.keep_original = True
    diag = mesh_utils.object_world_diagonal(obj)
    ctx = bpy.context
    plan.add_record(ctx, obj, 'STRAIGHT', [plane_contact(-10.0, diag)])
    bpy.ops.esp.build()
    bpy.ops.esp.approve()
    parts = sorted(bpy.data.collections["ESP_Built_Chain"].objects, key=lambda o: world_bounds(o)[0].z)
    check(len(parts) == 2 and obj.hide_get(), "approved 2 parts, original stashed")
    upper = parts[1]  # -10 .. 30
    ctx.view_layer.objects.active = upper
    plan.add_record(ctx, upper, 'STRAIGHT', [plane_contact(10.0, diag)])
    check(s.base_obj == upper, f"plan rooted at the approved part ({s.base_obj})")
    res = bpy.ops.esp.build()
    col = s.built_col
    check(res == {'FINISHED'} and col is not None and len(col.objects) == 2, "second build made 2 parts")
    # the part starts at z=-10 and its connector hangs a little below that; the whole
    # model would reach z=-30, so a bottom near -30 means the original got cut instead
    low = min(world_bounds(o)[0].z for o in col.objects)
    check(low > -25.0, f"cut the approved part, not the whole model (lowest z {low:.1f})")
    check(upper.hide_get(), "the cut part went to the backup")
    check(bpy.data.objects.get("Chain") is not None, "the original is untouched")
    check(bpy.data.objects.get(parts[0].name) is not None, "the other approved part survived")
    draw_all_panels(ctx)
    res = bpy.ops.esp.return_to_plan()
    check(res == {'FINISHED'} and not upper.hide_get(), "back to plan restored the approved part")


def test_approve_keeps_unbuilt_cuts():
    """Approve drops the cuts it built; a cut that was not built stays, on the part it sits on."""
    print("== approve keeps the cuts that were not built")
    sc = reset_scene()
    obj = make_cylinder("Keep")  # z from -30 to 30
    s = sc.esp
    s.mode = 'PLAN'
    s.keep_original = False
    diag = mesh_utils.object_world_diagonal(obj)
    ctx = bpy.context
    low = plan.add_record(ctx, obj, 'STRAIGHT', [plane_contact(-10.0, diag)])
    high = plan.add_record(ctx, obj, 'STRAIGHT', [plane_contact(10.0, diag)])
    high.enabled = False
    high_name = high.name  # the record slot moves once the built cut is removed
    bpy.ops.esp.build()
    check(low.built and not high.built, "only the ready cut was built")
    res = bpy.ops.esp.approve()
    check(res == {'FINISHED'} and len(s.cuts) == 1 and s.cuts[0].name == high_name, "the unbuilt cut survived approve")
    check(bpy.data.objects.get("Keep") is None, "the original went away (Keep Original off)")
    rec = s.cuts[0]
    parts = sorted(bpy.data.collections["ESP_Built_Keep"].objects, key=lambda o: world_bounds(o)[0].z)
    upper = parts[1]  # -10 .. 30, where the z=10 cut sits
    check(rec.target_obj == upper, f"the kept cut now targets the part under it ({rec.target_obj})")
    check(s.base_obj == upper, f"the plan is rooted at that part ({s.base_obj})")
    check(rec.surface_obj_a is not None, "its preview surface is still there")
    col = bpy.data.collections.get(plan.PLAN_COLLECTION)
    check(col is not None and not col.hide_viewport, "the plan is visible again")
    check(s.last_message.startswith("Parts approved, 1 unbuilt cut(s) kept"), f"message says so ({s.last_message})")
    draw_all_panels(ctx)
    rec.enabled = True
    res = bpy.ops.esp.build()
    col = s.built_col
    check(res == {'FINISHED'} and col is not None and len(col.objects) == 2, "the kept cut builds on the part")
    lowest = min(world_bounds(o)[0].z for o in col.objects)
    check(lowest > -25.0, f"it cut the part, not the whole model (lowest z {lowest:.1f})")
    check(bpy.data.objects.get(parts[0].name) is not None, "the other approved part survived")
    res = bpy.ops.esp.approve()
    check(res == {'FINISHED'} and len(s.cuts) == 0, "approving the last cut empties the plan")
    check(bpy.data.collections.get(plan.PLAN_COLLECTION) is None, "and drops the plan collection")
    check(s.last_message == "Parts approved", f"plain message when nothing is kept ({s.last_message})")


def test_two_contact_surfaces_edit_by_points():
    """Each contact of a two-contact curve or freehand cut is edited by its own points; planes by G/R/S."""
    print("== two-contact cuts edit each surface")
    from easy_slice_print.ops_plan import edit_surface_of

    sc = reset_scene()
    obj = make_cylinder("Twin")
    s = sc.esp
    s.mode = 'PLAN'
    diag = mesh_utils.object_world_diagonal(obj)
    ctx = bpy.context
    rec = plan.add_record(ctx, obj, 'CURVED', [curve_contact(obj, diag), curve_contact(obj, diag, y_end=8.0)])
    check(rec.two_contact and rec.surface_obj_a and rec.surface_obj_b, "a two-contact curve cut has two surfaces")
    a, by_pts_a = edit_surface_of(rec, 0)
    b, by_pts_b = edit_surface_of(rec, 1)
    check(a is not None and a == rec.surface_obj_a and by_pts_a, "contact 1 edits by points")
    check(b is not None and b == rec.surface_obj_b and by_pts_b, "contact 2 edits by points, on its own surface")
    check(a != b and len(plan.surface_points(b)) == 20, "the second surface carries its own control points")
    draw_all_panels(ctx)
    loop = plan.add_record(ctx, obj, 'FREEHAND', [freehand_contact(obj), freehand_contact(obj, height=-5.0)])
    fa, by_a = edit_surface_of(loop, 0)
    fb, by_b = edit_surface_of(loop, 1)
    check(by_a and by_b and fa != fb, "a two-contact freehand cut edits both loops by points")
    flat = plan.add_record(ctx, obj, 'STRAIGHT', [plane_contact(-10.0, diag), plane_contact(10.0, diag)])
    pa, by_pa = edit_surface_of(flat, 0)
    pb, by_pb = edit_surface_of(flat, 1)
    check(pa is not None and pb is not None and not by_pa and not by_pb and pa != pb, "two planes are moved as objects")
    single = plan.add_record(ctx, obj, 'CURVED', [curve_contact(obj, diag)])
    sa, _ = edit_surface_of(single, 0)
    sb, _ = edit_surface_of(single, 1)
    check(sa is not None and sb == sa, "a one-contact cut ignores the 2nd index")
    draw_all_panels(ctx)


def test_quick_mode():
    print("== quick mode")
    sc = reset_scene()
    obj = make_cylinder("Quick")
    s = sc.esp
    s.mode = 'QUICK'
    s.size_preset = 'MEDIUM'
    diag = mesh_utils.object_world_diagonal(obj)
    from easy_slice_print.ops_tools import quick_cut, quick_cut_steps

    # Esc on the job before the parts exist leaves nothing behind, not even their collection
    steps = quick_cut_steps(bpy.context, obj, [plane_contact(0.0, diag)])
    next(steps)
    steps.close()
    check(bpy.data.collections.get("ESP_Built_Quick") is None, "a cancelled quick cut leaves no empty collection")
    a, b, secs = quick_cut(bpy.context, obj, [plane_contact(0.0, diag)])
    check({a.name, b.name} == {"Quick_UPPER", "Quick_LOWER"}, f"quick names {a.name}/{b.name}")
    check(obj.hide_get(), "original kept hidden")
    upper = a if a.name.endswith("UPPER") else b
    mn, _mx = mesh_utils.mesh_bounds(upper.data)
    check(mn.z < -3.0, f"upper part carries the pin (min z {mn.z:.2f})")
    draw_all_panels(bpy.context)
    # two contacts in quick mode with skip of original
    s.keep_original = False
    s.two_contact = True
    obj2 = make_cylinder("Quick2")
    a, b, secs = quick_cut(bpy.context, obj2, [plane_contact(-20.0, diag), plane_contact(-20.0, diag)])
    check(bpy.data.objects.get("Quick2") is None, "original deleted when Keep Original is off")


class FakeStroke:
    """Enough of the Plane tool for `make_contact` to run without a 3D viewport.

    Screen x maps straight to world x on the cut plane and the view looks down -Y, so a
    drag from x0 to x1 is a horizontal plane cut across that stretch of the model.
    """

    def __init__(self, target, plane_z, hit):
        self.target = target
        self.plane_z = plane_z
        self.bcenter = Vector((0.0, 0.0, plane_z))
        self.contact_hit = hit
        self.data = None

    def at_depth(self, p, _center):
        return Vector((p[0], 0.0, self.plane_z))

    def view_dir(self, _p):
        return Vector((0.0, -1.0, 0.0))

    def cast(self, _context, _p):
        return True, self.contact_hit, Vector((0.0, 0.0, 1.0)), 0.0

    def view_moved(self):
        return False

    def reset_stroke(self):
        self.data = None

    def contact_done(self, _context, data):
        self.data = data  # keep it instead of cutting, so the contact can be inspected
        return {'FINISHED'}


def test_quick_plane_section():
    """The Plane tool puts the section's quad on the contact, so Quick mode cuts.

    The tool used to copy the section into the contact and drop the quad that came with
    it. Quick mode reads the quad straight off the contact, so it booleaned the cross
    section itself: its rim lies on the model tangentially, the exact solver hands back
    a mesh barely touched, and the cut reports success, names two parts and separates
    nothing. Plan mode was spared because it re-sections on the preview and picks the
    quad up there.
    """
    print("== the plane tool hands the boolean a quad, not the raw section")
    sc = reset_scene()
    legs = make_two_legs()
    sc.esp.mode = 'QUICK'
    sc.esp.size_preset = 'MEDIUM'

    from easy_slice_print.ops_tools import ESP_OT_cut_straight, quick_cut

    # drag across the leg at x=-30 only, at z=5
    tool = FakeStroke(legs, 5.0, Vector((-20.0, 0.0, 5.0)))
    ESP_OT_cut_straight.make_contact(tool, bpy.context, (-45.0, 0.0), (-15.0, 0.0))
    d = tool.data
    check(d is not None, "the stroke produced a contact")
    check(d.is_cut_face, "the surface is the model's own cut face")
    check(len(d.verts) > 8, f"the section is an outline, not a quad ({len(d.verts)} verts)")
    check(d.cutter is not None, "the contact carries the section's cutter")

    spec = plan.quick_spec(bpy.context, legs, [d])
    cverts = spec.contacts[0].verts
    check(len(cverts) == 4, f"and the boolean is handed that quad ({len(cverts)} verts)")
    check(max(v.x for v in cverts) < 5.0, "which stops in the gap before the other leg")

    a, b, _secs = quick_cut(bpy.context, legs, [d])
    lower = a if a.name.endswith("LOWER") else b
    # the leg the stroke missed goes into one of the parts whole, so only the cut leg
    # says anything about the cut: its half has to stop at the plane, pin aside
    mw = lower.matrix_world
    cut_leg = [(mw @ v.co).z for v in lower.data.vertices if (mw @ v.co).x < -10.0]
    check(bool(cut_leg), "the lower part kept the cut leg")
    check(max(cut_leg) < 8.0, f"and it came apart at the plane (its leg tops out at {max(cut_leg):.1f})")


class FakeArea:
    def __init__(self, regions):
        self.regions = regions
        self.header = "set"

    def header_text_set(self, text):
        self.header = text

    def tag_redraw(self):
        pass


class FakeRegion:
    def __init__(self, kind, x, y, w, h):
        self.type, self.x, self.y, self.width, self.height = kind, x, y, w, h
        self.data = None

    def as_pointer(self):
        return id(self)


class FakeEvent:
    def __init__(self, kind='MOUSEMOVE', value='NOTHING', x=500, y=400):
        self.type, self.value, self.mouse_x, self.mouse_y = kind, value, x, y
        self.alt = self.ctrl = self.shift = False


def armed_tool(target):
    """A cut tool as `invoke` leaves it, without a 3D viewport to run in."""
    from easy_slice_print import ops_tools

    class Tool(ops_tools.CutToolBase):
        bl_idname = "esp.fake_tool"

        def __init__(self):
            self.reports = []

        def report(self, kind, msg):
            self.reports.append(msg)

    tool = Tool()
    view = FakeRegion('WINDOW', 0, 0, 1000, 800)
    tool.area = FakeArea([view])
    tool.window = None
    tool.region = view
    tool.rv3d = None
    tool.target = target
    tool.contacts = []
    tool.contacts_needed = 1
    tool.reset_stroke()
    tool._cursor_on = False
    tool.superseded = False
    tool._handle = bpy.types.SpaceView3D.draw_handler_add(lambda: None, (), 'WINDOW', 'POST_PIXEL')
    ops_tools._armed = tool
    return tool


def test_an_armed_tool_cleans_up_after_itself():
    """Clicks off the view reach the interface, so the model can vanish under an armed tool.

    Undo or Delete from the outliner, or a second tool started from the sidebar: the
    tool has to notice and take its drawing, cursor and status text down with it,
    instead of failing on the next click and leaving them on screen for good.
    """
    print("== an armed tool cleans up after itself")
    reset_scene()
    from easy_slice_print import ops_tools

    obj = make_cylinder("Armed")
    tool = armed_tool(obj)
    check(tool._modal(bpy.context, FakeEvent()) == {'RUNNING_MODAL'}, "a live model keeps the tool armed")
    bpy.data.objects.remove(obj)
    res = tool._modal(bpy.context, FakeEvent())
    check(res == {'CANCELLED'}, "a deleted model ends the tool")
    check(tool._handle is None and tool.area.header is None, "and takes its drawing and header down")
    check(ops_tools._armed is None, "nothing is left armed")

    # a second tool takes over; the first goes quietly and leaves the new one's header alone
    obj = make_cylinder("Armed2")
    first = armed_tool(obj)
    second = armed_tool(obj)
    first.superseded = True  # what the second one's invoke does
    second.area = first.area
    res = first._modal(bpy.context, FakeEvent())
    check(res == {'CANCELLED'} and first._handle is None, "the replaced tool bows out")
    check(first.area.header == "set", "without clearing the header the new tool put up")
    check(ops_tools._armed is second, "the new tool is the armed one")

    # an error inside the tool still ends it cleanly
    def boom(*_args):
        raise RuntimeError("boom")

    second.on_move = boom
    res = second.modal(bpy.context, FakeEvent())
    check(res == {'CANCELLED'} and second._handle is None, "an unexpected error ends the tool cleanly")
    check(any("boom" in r for r in second.reports), "and says why")

    # Quad View: four views in one area; the tool works in the one under the mouse
    obj = make_cylinder("Quad")
    tool = armed_tool(obj)
    left, right = FakeRegion('WINDOW', 0, 0, 500, 800), FakeRegion('WINDOW', 500, 0, 500, 800)
    tool.area = FakeArea([left, right])
    tool.region = left
    tool._modal(bpy.context, FakeEvent(x=700))
    check(tool.region is right, "moving into another quadrant moves the tool there")
    tool.drawing = True
    tool._modal(bpy.context, FakeEvent(x=200))
    check(tool.region is right, "but not in the middle of a stroke")
    tool.end(bpy.context)


def edit_session(rec):
    """Edit Cut Surface as its `invoke` leaves it, run without a 3D viewport.

    The operator cannot be instantiated outside Blender's operator machinery, so its
    methods are borrowed onto a plain object carrying the state `invoke` sets up.
    """
    from easy_slice_print.ops_plan import ESP_OT_edit_surface

    class Session:
        pass

    for name, fn in vars(ESP_OT_edit_surface).items():
        if callable(fn) and not name.startswith("__") and name not in ("bl_rna",):
            setattr(Session, name, fn)
    ses = Session()
    ses.reports = []
    ses.report = lambda kind, msg: ses.reports.append(msg)
    ses.rec = rec
    ses.sobj = rec.surface_obj_a
    view = FakeRegion('WINDOW', 0, 0, 1000, 800)
    ses.area = FakeArea([view, FakeRegion('UI', 800, 0, 200, 800)])
    ses.window = None
    ses.region = view
    ses.rv3d = None
    ses.kind = ses.sobj.get("esp_kind")
    ses.closed = ses.kind == 'FREEHAND'
    ses.target = rec.target_obj
    ses.mw = ses.sobj.matrix_world.copy()
    ses.inv = ses.mw.inverted_safe()
    ses.points = plan.surface_points(ses.sobj)
    ses.orig = [p.copy() for p in ses.points]
    ses.history = [[p.copy() for p in ses.points]]
    ses.redo = []
    ses.hover = ses.drag = ses.slide = None
    ses.mouse = (0, 0)
    ses.changed = False
    ses._cursor_on = False
    ses._handle = bpy.types.SpaceView3D.draw_handler_add(lambda: None, (), 'WINDOW', 'POST_PIXEL')
    return ses


def test_edit_cut_surface_modal():
    print("== edit cut surface")
    sc = reset_scene()
    sc.esp.mode = 'PLAN'
    obj = make_cylinder("Edited")
    rec = plan.add_record(bpy.context, obj, 'FREEHAND', [freehand_contact(obj)])

    ses = edit_session(rec)
    res = ses._modal(bpy.context, FakeEvent('LEFTMOUSE', 'PRESS', x=900))
    check(res == {'PASS_THROUGH'}, "a click on the sidebar reaches the sidebar")

    # X while a point is being dragged would delete the point under the mouse
    count = len(ses.points)
    ses.hover = ses.drag = 3
    ses._modal(bpy.context, FakeEvent('X', 'PRESS'))
    check(len(ses.points) == count and ses.drag == 3, "X does nothing in the middle of a drag")

    # Enter in the middle of a drag: the last rebuild was a draft, the cut must not keep it
    calls = []
    real_commit = ses.commit
    ses.commit = lambda context, draft=False: (calls.append(draft), real_commit(context, draft))
    res = ses._modal(bpy.context, FakeEvent('RET', 'PRESS'))
    check(res == {'FINISHED'} and calls and calls[-1] is False, "Enter ends a drag with a full rebuild")
    check(ses._handle is None, "and the editor is closed")

    ses = edit_session(rec)
    plan.remove_record(bpy.context, 0)
    res = ses._modal(bpy.context, FakeEvent())
    check(res == {'CANCELLED'} and ses._handle is None, "a cut deleted under the editor closes it cleanly")


def test_cut_tools_leave_the_interface_alone():
    """An armed cut tool only takes the clicks that land on the 3D view itself.

    The modal used to eat every click in the window and set its pencil cursor
    everywhere, so the sidebar could not be used until the cut was finished or
    cancelled. The sidebar, toolbar and headers sit inside the view's own rectangle
    when Region Overlap is on, so the test covers that layout.
    """
    print("== cut tools leave the interface alone")
    from easy_slice_print.ops_tools import over_view

    class R:
        def __init__(self, kind, x, y, w, h):
            self.type, self.x, self.y, self.width, self.height = kind, x, y, w, h

    class E:
        def __init__(self, x, y):
            self.mouse_x, self.mouse_y = x, y

    view = R('WINDOW', 100, 0, 1000, 800)

    class Area:
        regions = [
            R('HEADER', 100, 776, 1000, 24),
            R('TOOLS', 100, 0, 40, 776),
            R('UI', 850, 0, 250, 776),
            R('HUD', 150, 20, 200, 120),  # Adjust Last Operation, floating over the view
            R('ASSET_SHELF', 0, 0, 1, 1),  # collapsed regions are left out
            view,
        ]

    check(over_view(Area, view, E(500, 400)), "the middle of the view is the view")
    check(not over_view(Area, view, E(900, 400)), "the sidebar drawn over the view is interface")
    check(not over_view(Area, view, E(120, 400)), "so is the toolbar")
    check(not over_view(Area, view, E(500, 790)), "and the header")
    check(not over_view(Area, view, E(200, 60)), "and the Adjust Last Operation panel")
    check(not over_view(Area, view, E(50, 400)), "another editor is not the view")
    check(not over_view(Area, None, E(500, 400)), "no view region, nothing is the view")


def test_unregister():
    print("== unregister")
    easy_slice_print.unregister()
    check(not hasattr(bpy.types.Scene, "esp"), "scene.esp removed")


if __name__ == "__main__":
    test_version()
    test_register()
    test_plan_workflow()
    test_several_connectors()
    test_connector_stays_in_the_material()
    test_pin_fits_the_depth_of_the_part()
    test_clear_plan_after_a_build_keeps_the_parts()
    test_a_cancelled_build_goes_back_to_the_plan()
    test_back_to_plan_gives_every_source_back()
    test_a_built_plan_is_not_edited_blind()
    test_the_plan_holds_its_objects_by_pointer()
    test_a_file_saved_with_names_opens_with_its_plan()
    test_a_thin_socket_wall_is_flagged()
    test_connectors_leave_the_minimum_wall()
    test_a_connector_stops_short_of_a_later_cut()
    test_a_cut_across_earlier_cuts_cuts_every_part()
    test_base_split_preview_points_like_the_build()
    test_the_preview_shows_every_connector_a_build_makes()
    test_cut_after_approve()
    test_approve_keeps_unbuilt_cuts()
    test_two_contact_surfaces_edit_by_points()
    test_quick_mode()
    test_quick_plane_section()
    test_printer_fit()
    test_curve_cut_stops_at_the_model()
    test_freehand_connector_fits_the_loop()
    test_freehand_cut_lands_on_the_loop()
    test_plane_section_preview()
    test_build_shows_why_a_cut_failed()
    test_cut_tools_leave_the_interface_alone()
    test_an_armed_tool_cleans_up_after_itself()
    test_edit_cut_surface_modal()
    test_unregister()
    print(f"\n{len(FAILS)} failure(s)")
    for f in FAILS:
        print("  -", f)
    sys.exit(1 if FAILS else 0)
