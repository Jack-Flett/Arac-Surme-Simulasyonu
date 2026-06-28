"""
scene.py – Clean grid town, no trees.

3 NS roads at x = -22, 0, +22
3 EW roads at z = -22, 0, +22
All width-8, full map length.

Lamps are placed explicitly:
  • One pole at each of the 4 pavement corners of every intersection  (9×4 = 36)
  Those 36 lamps are passed to the shader as point lights.
  Additional decorative poles (mid-road outer stretches) are rendered
  but not counted as shader lights (shader cap = 36).

build_town(box, cyl, cone) → (draw_calls, lamp_head_models, bboxes, lamp_pos, lamp_col)
"""

import numpy as np
from camera import translation_matrix, scale_matrix, rotation_y

# Lazy import — only available after renderer is loaded
def _make_uvbox(w, h, d, density):
    from renderer import make_box_uvworld
    return make_box_uvworld(w, h, d, uv_density=density)

# ── Visual constants ─────────────────────────────────────────────────────
_ROAD_Y   = 0.02
_PAVE_Y   = 0.01
_PAVE_W   = 2.0

_ROAD_COLOR   = (0.22, 0.22, 0.22)
_PAVE_COLOR   = (0.65, 0.63, 0.60)
_POLE_COLOR   = (0.28, 0.28, 0.30)

_LAMP_HEIGHT      = 5.0
_LAMP_LIGHT_COLOR = np.array([1.0, 0.82, 0.35], dtype=np.float32)

# ── Road network — clean 3×3 orthogonal grid ────────────────────────────
_ROADS_NS = [
    (-22, 0, 55, 8),
    (  0, 0, 55, 8),
    ( 22, 0, 55, 8),
]
_ROADS_EW = [
    (0, -22, 55, 8),
    (0,   0, 55, 8),
    (0,  22, 55, 8),
]

# ── Buildings  (cx, cz, w, h, d, R, G, B, heading_deg) ──────────────────────
# heading_deg is optional — omit or use 0.0 for axis-aligned.
# Rotated buildings use an expanded AABB envelope for collision (conservative but correct).
_BUILDINGS = [
    # ── SW corner block ──
    ( -48, -48,  9,  9,  9,  0.65, 0.40, 0.35,  12.0),   # angled
    ( -35, -48,  9,  7,  9,  0.50, 0.52, 0.68,   0.0),
    ( -48, -35,  9, 11,  9,  0.60, 0.58, 0.48,  -8.0),   # angled
    ( -35, -35,  9,  8,  9,  0.72, 0.55, 0.38,   0.0),

    # ── NW corner block ──
    ( -48,  35,  9,  8,  9,  0.55, 0.62, 0.48,   0.0),
    ( -35,  35,  9, 10,  9,  0.70, 0.52, 0.40, -10.0),   # angled
    ( -48,  48,  9,  7,  9,  0.62, 0.58, 0.52,   0.0),
    ( -35,  48,  9,  9,  9,  0.50, 0.55, 0.60,   7.0),   # angled

    # ── SE corner block ──
    (  35, -48,  9, 10,  9,  0.70, 0.50, 0.35,   0.0),
    (  48, -48,  9,  8,  9,  0.48, 0.58, 0.52,  11.0),   # angled
    (  35, -35,  9,  9,  9,  0.65, 0.45, 0.42,   0.0),
    (  48, -35,  9,  7,  9,  0.55, 0.50, 0.65,  -9.0),   # angled

    # ── NE corner block ──
    (  35,  35,  9,  9,  9,  0.60, 0.60, 0.60,   8.0),   # angled
    (  48,  35,  9,  8,  9,  0.72, 0.60, 0.42,   0.0),
    (  35,  48,  9, 11,  9,  0.50, 0.58, 0.65,   0.0),
    (  48,  48,  9,  7,  9,  0.68, 0.44, 0.40, -13.0),   # angled

    # ── West edge blocks ──
    ( -47, -11,  8,  6,  5,  0.72, 0.72, 0.50,  17.0),   # angled
    ( -34, -11,  8,  7,  5,  0.62, 0.52, 0.40,   0.0),
    ( -47,  11,  8,  8,  5,  0.65, 0.60, 0.55,   0.0),
    ( -34,  11,  8,  6,  5,  0.50, 0.55, 0.60,   0.0),

    # ── East edge blocks ──
    (  34, -11,  8,  7,  5,  0.75, 0.50, 0.35,   0.0),
    (  47, -11,  8,  6,  5,  0.55, 0.55, 0.68,   0.0),
    (  34,  11,  8,  8,  5,  0.65, 0.45, 0.42, -21.0),   # angled
    (  47,  11,  8,  7,  5,  0.45, 0.60, 0.55,   0.0),

    # ── South edge blocks ──
    ( -11, -48,  5,  8,  9,  0.68, 0.62, 0.50,   0.0),
    ( -11, -35,  5,  7,  9,  0.50, 0.62, 0.55,   0.0),
    (  11, -48,  5,  7,  9,  0.70, 0.55, 0.40,   0.0),
    (  11, -35,  5,  9,  9,  0.62, 0.52, 0.40,   0.0),

    # ── North edge blocks ──
    ( -11,  35,  5,  9,  9,  0.58, 0.50, 0.62,   0.0),
    ( -11,  48,  5,  7,  9,  0.62, 0.55, 0.45,   0.0),
    (  11,  35,  5,  8,  9,  0.60, 0.45, 0.45,   0.0),
    (  11,  48,  5,  6,  9,  0.72, 0.50, 0.45,   0.0),

    # ── Four tiny centre-adjacent blocks ──
    ( -11, -11,  4,  7,  4,  0.70, 0.65, 0.50,   0.0),
    ( -11,  11,  4,  6,  4,  0.55, 0.48, 0.42,   0.0),
    (  11, -11,  4,  8,  4,  0.60, 0.45, 0.45,   0.0),
    (  11,  11,  4,  7,  4,  0.68, 0.60, 0.45,   0.0),
]

# ── Lamp positions ────────────────────────────────────────────────────────
# Pavement centre is 1 unit outside the road edge.
# Road half-width = 4, pavement centre = 4 + 1 = 5 from road centre.
# So for NS road at cx: lamp x = cx ± 5
#    for EW road at cz: lamp z = cz ± 5
#
# INTERSECTION LAMPS (shader lights) — one at each of the 4 pavement corners
# of every NS×EW road crossing. 3 NS × 3 EW × 4 corners = 36.
_NS_X = [-22, 0, 22]
_EW_Z = [-22, 0, 22]
_PAVE_OFFSET = 5  # distance from road centre to pavement lamp

_INTERSECTION_LAMPS = [
    (cx + dx, cz + dz)
    for cx in _NS_X
    for cz in _EW_Z
    for dx in (+_PAVE_OFFSET, -_PAVE_OFFSET)
    for dz in (+_PAVE_OFFSET, -_PAVE_OFFSET)
]  # 36 lamps — matches MAX_LAMPS 36 in fragment.glsl

# MID-ROAD DECORATIVE POLES — on the outer stretches between intersections and map edge.
# These are rendered (visual) but not added as shader lights (already at cap).
# Outer z-segments: z in [-55,-22] → mid=-38.5, z in [22,55] → mid=38.5
# Outer x-segments: x in [-55,-22] → mid=-38.5, x in [22,55] → mid=38.5
_OUTER_Z_MIDS = [-38.5, 38.5]
_OUTER_X_MIDS = [-38.5, 38.5]

_OUTER_NS_POLES = [
    (cx + dx, mz)
    for cx in _NS_X
    for dx in (+_PAVE_OFFSET, -_PAVE_OFFSET)
    for mz in _OUTER_Z_MIDS
]
_OUTER_EW_POLES = [
    (mx, cz + dz)
    for cz in _EW_Z
    for dz in (+_PAVE_OFFSET, -_PAVE_OFFSET)
    for mx in _OUTER_X_MIDS
]

_MAX_LAMPS = 36   # must match #define MAX_LAMPS in fragment.glsl


# ── Scene builder ────────────────────────────────────────────────────────

def build_town(box, cyl, cone):
    calls  = []   # each entry: [model_matrix, color, mesh, tex_tag]
    bboxes = []

    # UV density: 1 texture tile per N world units
    _UV_ROAD  = 1/4    # 1 tile per 4 units — asphalt
    _UV_PAVE  = 1/2    # 1 tile per 2 units — pavement tiles (smaller tiles)
    _UV_BLDG  = 1/3    # 1 tile per 3 units — brick/concrete

    def add_plain(mesh, color, cx, cy, cz, w=1, h=1, d=1, heading=0.0, tex=None):
        """For untextured or pre-built-mesh objects (poles, etc.)."""
        M = translation_matrix(cx, cy, cz) @ rotation_y(heading) @ scale_matrix(w, h, d)
        calls.append([M, color, mesh, tex])

    def add_road(color, cx, cy, cz, w, h, d, tex):
        """Road/pavement slab — bakes real dimensions into UVs."""
        density = _UV_ROAD if tex == 'road' else _UV_PAVE
        mesh = _make_uvbox(w, h, d, density)
        M = translation_matrix(cx, cy, cz)
        calls.append([M, color, mesh, tex])

    def add_building(color, cx, cy, cz, w, h, d, heading=0.0):
        """Building box — UVs proportional to real size, optional rotation."""
        mesh = _make_uvbox(w, h, d, _UV_BLDG)
        M = translation_matrix(cx, cy, cz) @ rotation_y(heading)
        calls.append([M, color, mesh, 'building'])

    def _building_bbox(cx, cz, w, d, heading_deg, h):
        bmin = np.array([cx - w/2, 0.0,      cz - d/2], dtype=np.float32)
        bmax = np.array([cx + w/2, float(h), cz + d/2], dtype=np.float32)
        if heading_deg == 0.0:
            return (bmin, bmax)
        else:
            return (bmin, bmax, heading_deg)

    def road_ns(cx, cz_c, half_len, rw):
        length = half_len * 2
        add_road(_ROAD_COLOR, cx,            _ROAD_Y,        cz_c, rw,      0.02, length, 'road')
        add_road(_PAVE_COLOR, cx + rw/2 + 1, _PAVE_Y,        cz_c, _PAVE_W, 0.01, length, 'pavement')
        add_road(_PAVE_COLOR, cx - rw/2 - 1, _PAVE_Y,        cz_c, _PAVE_W, 0.01, length, 'pavement')

    def road_ew(cx_c, cz, half_len, rw):
        length = half_len * 2
        add_road(_ROAD_COLOR, cx_c, _ROAD_Y + 0.01,  cz,             length,  0.02, rw,      'road')
        add_road(_PAVE_COLOR, cx_c, _PAVE_Y + 0.01,  cz + rw/2 + 1, length,  0.01, _PAVE_W, 'pavement')
        add_road(_PAVE_COLOR, cx_c, _PAVE_Y + 0.01,  cz - rw/2 - 1, length,  0.01, _PAVE_W, 'pavement')

    # Roads
    for (cx, cz, hl, rw) in _ROADS_NS:
        road_ns(cx, cz, hl, rw)
    for (cx, cz, hl, rw) in _ROADS_EW:
        road_ew(cx, cz, hl, rw)

    # Buildings + collision boxes
    for b in _BUILDINGS:
        cx, cz, w, h, d = b[0], b[1], b[2], b[3], b[4]
        r, g, bl = b[5], b[6], b[7]
        heading  = b[8] if len(b) > 8 else 0.0
        add_building((r, g, bl), cx, h / 2.0, cz, w, h, d, heading)
        bboxes.append(_building_bbox(cx, cz, w, d, heading, h))

    # Invisible boundary walls — 4 thin slabs around the map edge (±55 units).
    # No mesh is rendered; only collision boxes are added.
    _WALL_H    = 20.0   # tall enough that nothing can jump over
    _WALL_T    =  1.0   # thickness
    _WALL_HALF = 56.0   # just outside the road half-length (55)
    _SPAN      = 114.0  # full width of map (2 × 57 to cover corners)
    _boundary_walls = [
        # (bmin, bmax)
        # North wall  (z = +55)
        (np.array([-_SPAN/2,  0.0,  _WALL_HALF],         dtype=np.float32),
         np.array([ _SPAN/2,  _WALL_H, _WALL_HALF+_WALL_T], dtype=np.float32)),
        # South wall  (z = -55)
        (np.array([-_SPAN/2,  0.0, -_WALL_HALF-_WALL_T], dtype=np.float32),
         np.array([ _SPAN/2,  _WALL_H, -_WALL_HALF],      dtype=np.float32)),
        # East wall   (x = +55)
        (np.array([ _WALL_HALF,         0.0, -_SPAN/2], dtype=np.float32),
         np.array([ _WALL_HALF+_WALL_T, _WALL_H,  _SPAN/2], dtype=np.float32)),
        # West wall   (x = -55)
        (np.array([-_WALL_HALF-_WALL_T, 0.0, -_SPAN/2], dtype=np.float32),
         np.array([-_WALL_HALF,         _WALL_H,  _SPAN/2], dtype=np.float32)),
    ]
    for bmin, bmax in _boundary_walls:
        bboxes.append((bmin, bmax))

    # ── Lamp poles ───────────────────────────────────────────────────────
    lamp_head_models = []
    lamp_positions   = []
    lamp_colors      = []

    POLE_R = 0.25  # collision half-width

    def _place_pole(lx, lz, is_light):
        """Render a lamp pole. Always shows a glowing head; only lit poles count as shader lights."""
        add_plain(cyl, _POLE_COLOR, lx, _LAMP_HEIGHT / 2, lz, w=0.18, h=_LAMP_HEIGHT, d=0.18, tex=None)
        head_M = (translation_matrix(lx, _LAMP_HEIGHT + 0.15, lz)
                  @ scale_matrix(0.55, 0.30, 0.55))
        lamp_head_models.append(head_M)   # always rendered with emissive glow
        if is_light:
            lamp_positions.append(np.array([lx, _LAMP_HEIGHT - 0.1, lz], dtype=np.float32))
            lamp_colors.append(_LAMP_LIGHT_COLOR.copy())
        # Collision box for every pole (light or decorative)
        bboxes.append((
            np.array([lx - POLE_R,  0.0,          lz - POLE_R], dtype=np.float32),
            np.array([lx + POLE_R,  _LAMP_HEIGHT,  lz + POLE_R], dtype=np.float32),
        ))

    # Intersection lamps — shader lights (36 total = MAX_LAMPS)
    for (lx, lz) in _INTERSECTION_LAMPS:
        _place_pole(lx, lz, is_light=True)

    # Emissive glow for ALL intersection lamp heads (registered lights)
    # (handled in main.py loop over lamp_head_models)

    # Outer mid-road poles — decorative only (no shader light slot needed)
    for (lx, lz) in _OUTER_NS_POLES:
        _place_pole(lx, lz, is_light=False)
    for (lx, lz) in _OUTER_EW_POLES:
        _place_pole(lx, lz, is_light=False)

    return calls, lamp_head_models, bboxes, lamp_positions, lamp_colors