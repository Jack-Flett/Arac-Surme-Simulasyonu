"""
main.py – Week 5 entry point.
Evening scene with street lamps and building collision.

Controls:
  F              – toggle Free-Fly / Chase camera
  Chase mode:
    W / S        – accelerate / brake
    A / D        – steer  (only while rolling)
  Free-fly mode:
    W/S/A/D      – fly camera
    Mouse        – look around
    Space/LShift – up / down
  Escape         – quit
"""

import sys
import ctypes
import pygame
from pygame.locals import *
from OpenGL.GL import *
import numpy as np

from camera   import Camera, ChaseCamera, perspective, translation_matrix
from renderer import (load_shaders, make_box, make_ground_plane,
                      make_cylinder, make_cone, make_box_uvworld,
                      load_texture, load_obj, load_obj_materials,
                      set_mat4, set_vec3, set_int, set_vec3_array)
from car      import Car
from scene    import build_town

# ── Debug AABB wireframe shader ─────────────────────────────────────────
_DBG_VERT = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uVP;
void main() { gl_Position = uVP * vec4(aPos, 1.0); }
"""
_DBG_FRAG = """
#version 330 core
uniform vec3 uColor;
out vec4 FragColor;
void main() { FragColor = vec4(uColor, 1.0); }
"""

def _build_debug_shader():
    def _compile(src, kind):
        s = glCreateShader(kind)
        glShaderSource(s, src)
        glCompileShader(s)
        if not glGetShaderiv(s, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(s).decode())
        return s
    prog = glCreateProgram()
    vs = _compile(_DBG_VERT, GL_VERTEX_SHADER)
    fs = _compile(_DBG_FRAG, GL_FRAGMENT_SHADER)
    glAttachShader(prog, vs); glAttachShader(prog, fs)
    glLinkProgram(prog)
    glDeleteShader(vs); glDeleteShader(fs)
    return prog

def _aabb_lines_static(bmin, bmax):
    """Return 24 vertices (12 edges x 2 endpoints) for one AABB wireframe."""
    x0, y0, z0 = float(bmin[0]), float(bmin[1]), float(bmin[2])
    x1, y1, z1 = float(bmax[0]), float(bmax[1]), float(bmax[2])
    corners = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ], dtype=np.float32)
    edges = [0,1, 1,2, 2,3, 3,0,  4,5, 5,6, 6,7, 7,4,  0,4, 1,5, 2,6, 3,7]
    return corners[edges].flatten()


def _bbox_lines(bbox):
    """Draw either a plain AABB (2-tuple) or a rotated OBB (3-tuple)."""
    import math as _m
    if len(bbox) == 3:
        bmin, bmax, hdeg = bbox
        cx = (float(bmin[0])+float(bmax[0]))*0.5
        cz = (float(bmin[2])+float(bmax[2]))*0.5
        hw = (float(bmax[0])-float(bmin[0]))*0.5
        hd = (float(bmax[2])-float(bmin[2]))*0.5
        h  = _m.radians(hdeg)
        fx, fz =  _m.sin(h),  _m.cos(h)
        rx, rz =  _m.cos(h), -_m.sin(h)
        y0, y1 = float(bmin[1]), float(bmax[1])
        bc = [(cx+fx*hd+rx*hw, y0, cz+fz*hd+rz*hw),
              (cx+fx*hd-rx*hw, y0, cz+fz*hd-rz*hw),
              (cx-fx*hd-rx*hw, y0, cz-fz*hd-rz*hw),
              (cx-fx*hd+rx*hw, y0, cz-fz*hd+rz*hw)]
        tc = [(x, y1, z) for x,_,z in bc]
        pts = [np.array(p, dtype=np.float32) for p in bc+tc]
        v = []
        for i in range(4): v += list(pts[i])+list(pts[(i+1)%4])
        for i in range(4): v += list(pts[i+4])+list(pts[(i+1)%4+4])
        for i in range(4): v += list(pts[i])+list(pts[i+4])
        return np.array(v, dtype=np.float32)
    else:
        return _aabb_lines_static(bbox[0], bbox[1])

class DebugAABBDrawer:
    """Uploads all AABB wireframes to a single VBO and draws them with GL_LINES."""
    def __init__(self, shader):
        self.shader = shader
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.vertex_count = 0

    def upload(self, bboxes):
        verts = np.concatenate([_bbox_lines(bbox) for bbox in bboxes])
        self.vertex_count = len(verts) // 3
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def draw(self, view, proj, color=(1.0, 0.0, 1.0)):
        if self.vertex_count == 0:
            return
        vp = proj @ view
        glUseProgram(self.shader)
        loc_vp  = glGetUniformLocation(self.shader, "uVP")
        loc_col = glGetUniformLocation(self.shader, "uColor")
        glUniformMatrix4fv(loc_vp, 1, GL_TRUE, vp.astype(np.float32))
        glUniform3f(loc_col, *color)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_LINES, 0, self.vertex_count)
        glBindVertexArray(0)

# ── Config ──────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1280, 720
TITLE         = "Araç Sürme Simülasyonu"
FOV           = 75.0
NEAR, FAR     = 0.1, 600.0

CAR_COLOR  = (0.05, 0.35, 0.75)   # deep blue car
MAX_LAMPS  = 36                    # must match #define in fragment.glsl

# ── Evening atmosphere ───────────────────────────────────────────────────
SKY_COLOR    = (0.06, 0.05, 0.16)          # deep purple-blue dusk
AMBIENT      = (0.06, 0.05, 0.12)          # very dark blue-purple ambient
SUN_DIR      = (1.0, -0.15, 0.3)           # low on the western horizon
SUN_COLOR    = (0.28, 0.18, 0.08)          # dim warm orange (nearly set)

LAMP_EMISSIVE = (1.00, 0.85, 0.30)         # glow written onto the lamp head
LAMP_HEAD_COL = (1.00, 0.95, 0.65)         # base colour of the lamp head mesh


def main():
    pygame.init()
    pygame.display.set_caption(TITLE)

    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                    pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)

    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)

    glEnable(GL_DEPTH_TEST)
    glDisable(GL_CULL_FACE)
    print("OpenGL:", glGetString(GL_VERSION).decode())
    glClearColor(*SKY_COLOR, 1.0)

    shader = load_shaders("shaders/vertex.glsl", "shaders/fragment.glsl")
    ground = make_ground_plane(size=500.0, uv_scale=50.0)
    box    = make_box()
    cyl    = make_cylinder()
    cone   = make_cone()

    # Load textures  (files live in textures/ next to main.py)
    tex_grass    = load_texture("textures/grass.png")
    tex_road     = load_texture("textures/road.png")
    tex_pavement = load_texture("textures/pavement.png")
    tex_building = load_texture("textures/building.png")

    # Load car model  (models/car.obj)
    # Colours matched to the Blender material names:
    #   Carcamero_ma = green body, .027 = dark tyre, .031 = grey rim,
    #   .028 = black stripe/roof, .030 = red tail light, .029 = headlight,
    #   .026 = dark bumper trim, .032 = black interior, .025 = accent
    _CAR_COLORS = {
        'Carcamero_ma': (0.08, 0.55, 0.12),   # green body
        'Material.027':  (0.12, 0.12, 0.12),   # tyre rubber (very dark)
        'Material.031':  (0.55, 0.55, 0.60),   # wheel rim (silver-grey)
        'Material.028':  (0.10, 0.10, 0.10),   # black stripe / roof
        'Material.030':  (0.85, 0.08, 0.08),   # tail lights (red)
        'Material.029':  (0.90, 0.90, 0.85),   # headlights (near white)
        'Material.026':  (0.20, 0.20, 0.22),   # dark bumper trim
        'Material.032':  (0.08, 0.08, 0.08),   # black interior
        'Material.025':  (0.80, 0.65, 0.10),   # yellow accent
    }
    car_parts = load_obj_materials("models/car.obj", _CAR_COLORS)

    # Scale + offset so the model fits inside get_model_matrix()'s transform.
    # get_model_matrix() applies scale(WIDTH=2, HEIGHT=0.8, LENGTH=4) to whatever we draw,
    # so the local scale must divide out each game axis to reach the correct world size.
    # OBJ dims: W=3.592, H=2.424, L=8.449  → target world: W≈1.7, H≈1.15, L=4.0
    # 180° Y rotation to flip from tail-first to nose-first.
    _CAR_SX    =  0.2367   # 0.4734 / game_width  (2.0)
    _CAR_SY    =  0.5917   # 0.4734 / game_height (0.8)
    _CAR_SZ    =  0.1183   # 0.4734 / game_length (4.0)
    _CAR_Y_OFF = -0.4953   # shift model down so wheels sit at ground level
    _CAR_Z_OFF =  0.0354   # shift model to centre along length after 180° flip

    # Map scene tag → GL texture ID
    _TEX = {
        'road':      tex_road,
        'pavement':  tex_pavement,
        'building':  tex_building,
    }

    # Debug AABB wireframe
    dbg_shader = _build_debug_shader()
    dbg_drawer = DebugAABBDrawer(dbg_shader)
    debug_mode = False

    # Build town: normal geometry, glowing lamp heads, AABBs, lamp lights
    scene_calls, lamp_heads, building_boxes, lamp_pos, lamp_col = build_town(box, cyl, cone)

    # Cap lights at shader array size
    lamp_pos = lamp_pos[:MAX_LAMPS]
    lamp_col = lamp_col[:MAX_LAMPS]

    # ── Car + cameras ────────────────────────────────────────────────────
    car        = Car(position=(0.0, 0.0, 8.0), heading=0.0)
    chase_cam  = ChaseCamera(car)
    camera     = Camera(position=(0.0, 25.0, 35.0), yaw=-90.0, pitch=-35.0)
    chase_mode = False

    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)

    proj      = perspective(FOV, WIDTH / HEIGHT, NEAR, FAR)
    clock     = pygame.time.Clock()
    fps_timer = 0.0

    while True:
        dt = clock.tick(0) / 1000.0
        dt = min(dt, 0.05)

        # ── Events ──────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit(); sys.exit()
                if event.key == K_f:
                    chase_mode = not chase_mode
                if event.key == K_b:
                    debug_mode = not debug_mode
            if event.type == MOUSEMOTION:
                if chase_mode:
                    chase_cam.process_mouse(event.rel[0], event.rel[1])
                else:
                    camera.process_mouse(event.rel[0], event.rel[1])

        # ── Update ──────────────────────────────────────────────────────
        keys = pygame.key.get_pressed()
        if chase_mode:
            car.update(keys, dt, building_boxes)   # ← collision enabled
            chase_cam.update(car, dt)
            view     = chase_cam.get_view_matrix(car)
            view_pos = chase_cam.position
        else:
            camera.process_keyboard(keys, dt)
            view     = camera.get_view_matrix()
            view_pos = camera.position

        # ── Render ──────────────────────────────────────────────────────
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(shader)

        # Shared uniforms
        set_mat4(shader, "uView",       view)
        set_mat4(shader, "uProjection", proj)
        set_vec3(shader, "uLightDir",   SUN_DIR)
        set_vec3(shader, "uLightColor", SUN_COLOR)
        set_vec3(shader, "uAmbient",    AMBIENT)
        set_vec3(shader, "uViewPos",    view_pos)
        set_int (shader, "uUseTexture", 0)
        set_int (shader, "uNumLamps",   len(lamp_pos))
        set_vec3_array(shader, "uLampPos",   lamp_pos)
        set_vec3_array(shader, "uLampColor", lamp_col)
        set_vec3(shader, "uEmissive",   (0.0, 0.0, 0.0))

        # Ground — grass texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex_grass)
        set_int (shader, "uTexture",    0)
        set_int (shader, "uUseTexture", 1)
        set_mat4(shader, "uModel", translation_matrix(0, 0, 0))
        set_vec3(shader, "uColor", (1.0, 1.0, 1.0))   # white = texture shows as-is
        ground.draw()
        set_int(shader, "uUseTexture", 0)
        glBindTexture(GL_TEXTURE_2D, 0)

        # Town geometry — roads, pavements, buildings, poles
        for model, color, mesh, tex_tag in scene_calls:
            set_mat4(shader, "uModel", model)
            if tex_tag and tex_tag in _TEX:
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, _TEX[tex_tag])
                set_int (shader, "uTexture",    0)
                set_int (shader, "uUseTexture", 1)
                set_vec3(shader, "uColor", (1.0, 1.0, 1.0))
            else:
                set_int (shader, "uUseTexture", 0)
                set_vec3(shader, "uColor", color)
            mesh.draw()
        set_int(shader, "uUseTexture", 0)
        glBindTexture(GL_TEXTURE_2D, 0)

        # Car model — draw each material group with its own colour
        from camera import scale_matrix as _sm, translation_matrix as _tm, rotation_y as _ry
        game_M  = car.get_model_matrix()
        local_M = (_tm(0, _CAR_Y_OFF, _CAR_Z_OFF)
                   @ _ry(180.0)
                   @ _sm(_CAR_SX, _CAR_SY, _CAR_SZ))
        car_M   = game_M @ local_M
        set_int(shader, "uUseTexture", 0)
        for part_mesh, part_color in car_parts:
            set_mat4(shader, "uModel", car_M)
            set_vec3(shader, "uColor", part_color)
            part_mesh.draw()

        # Lamp heads — rendered with emissive so they appear to glow
        set_vec3(shader, "uEmissive", LAMP_EMISSIVE)
        set_vec3(shader, "uColor",    LAMP_HEAD_COL)
        for head_model in lamp_heads:
            set_mat4(shader, "uModel", head_model)
            box.draw()
        set_vec3(shader, "uEmissive", (0.0, 0.0, 0.0))

        # Debug: draw AABB wireframes on top (B to toggle)
        if debug_mode:
            # Rebuild every frame so the car OBB tracks its live position/heading
            bot, top = car.get_obb_corners_world()
            def _obb_lines(bc, tc):
                pts = [np.array(p, dtype=np.float32) for p in bc + tc]
                v = []
                for i in range(4): v += list(pts[i])       + list(pts[(i+1)%4])
                for i in range(4): v += list(pts[i+4])     + list(pts[(i+1)%4+4])
                for i in range(4): v += list(pts[i])       + list(pts[i+4])
                return np.array(v, dtype=np.float32)
            scene_verts = (np.concatenate([_bbox_lines(bbox)
                                           for bbox in building_boxes])
                           if building_boxes else np.array([], dtype=np.float32))
            all_verts = np.concatenate([scene_verts, _obb_lines(bot, top)])
            glBindVertexArray(dbg_drawer.vao)
            glBindBuffer(GL_ARRAY_BUFFER, dbg_drawer.vbo)
            glBufferData(GL_ARRAY_BUFFER, all_verts.nbytes, all_verts, GL_DYNAMIC_DRAW)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(0)
            glBindVertexArray(0)
            dbg_drawer.vertex_count = len(all_verts) // 3
            glDisable(GL_DEPTH_TEST)
            dbg_drawer.draw(view, proj, color=(1.0, 0.0, 1.0))
            glEnable(GL_DEPTH_TEST)

        pygame.display.flip()

        # Title bar
        fps_timer += dt
        if fps_timer >= 0.5:
            fps      = clock.get_fps()
            kph      = abs(car.speed) * 3.6
            mode_str = "Chase [F]" if chase_mode else "Free-Fly [F]"
            dbg_str  = "  |  BBOX [B]" if debug_mode else "  |  bbox [B]"
            pygame.display.set_caption(
                f"{TITLE}  |  {mode_str}  |  {kph:.0f} km/h  |  FPS: {fps:.0f}{dbg_str}"
            )
            fps_timer = 0.0


if __name__ == "__main__":
    main()