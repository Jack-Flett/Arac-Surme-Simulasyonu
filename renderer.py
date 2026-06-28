"""
renderer.py – low-level OpenGL helpers.

Provides:
  - compile_shader / link_program
  - Mesh class  (VAO/VBO upload + draw)
  - Geometry builders: make_box(), make_ground_plane()
  - Uniform setters
  - Texture loader (Pillow)
"""

import ctypes
import numpy as np
from OpenGL.GL import *
from PIL import Image


# -----------------------------------------------------------------------
# Shader helpers
# -----------------------------------------------------------------------

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compile error:\n{log}")
    return shader


def link_program(vert_src, frag_src):
    vert = compile_shader(vert_src, GL_VERTEX_SHADER)
    frag = compile_shader(frag_src, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vert)
    glAttachShader(prog, frag)
    glLinkProgram(prog)
    glDeleteShader(vert)
    glDeleteShader(frag)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        log = glGetProgramInfoLog(prog).decode()
        raise RuntimeError(f"Program link error:\n{log}")
    return prog


def load_shaders(vert_path, frag_path):
    with open(vert_path, 'r') as f:
        vert_src = f.read()
    with open(frag_path, 'r') as f:
        frag_src = f.read()
    return link_program(vert_src, frag_src)


# -----------------------------------------------------------------------
# Uniform setters
# -----------------------------------------------------------------------

def set_mat4(prog, name, matrix):
    loc = glGetUniformLocation(prog, name)
    # GL_FALSE + no .T: NumPy stores rows contiguously in memory.
    # OpenGL reads the flat bytes as columns — which is exactly correct
    # since our row-major NumPy layout matches OpenGL's expected byte order.
    glUniformMatrix4fv(loc, 1, GL_TRUE, matrix.astype(np.float32))


def set_vec3(prog, name, vec):
    loc = glGetUniformLocation(prog, name)
    glUniform3f(loc, float(vec[0]), float(vec[1]), float(vec[2]))


def set_int(prog, name, value):
    loc = glGetUniformLocation(prog, name)
    glUniform1i(loc, int(value))


def set_vec3_array(prog, name, vecs):
    """Upload a Python list of (x,y,z) sequences to a vec3 uniform array."""
    for i, v in enumerate(vecs):
        loc = glGetUniformLocation(prog, f"{name}[{i}]")
        if loc != -1:
            glUniform3f(loc, float(v[0]), float(v[1]), float(v[2]))


# -----------------------------------------------------------------------
# Texture loader
# -----------------------------------------------------------------------

def load_texture(path):
    """Load an image file and upload it to OpenGL. Returns texture ID."""
    img  = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA")
    data = np.array(img, dtype=np.uint8)

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 img.width, img.height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data)
    glGenerateMipmap(GL_TEXTURE_2D)

    # Repeating, linear filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glBindTexture(GL_TEXTURE_2D, 0)
    return tex


# -----------------------------------------------------------------------
# Mesh class
# -----------------------------------------------------------------------

class Mesh:
    """
    Holds a VAO for a piece of geometry.

    vertices: float32 array, interleaved [x,y,z, nx,ny,nz, u,v]  per vertex
    indices:  uint32 array of triangle indices (if None, draw as arrays)
    """

    def __init__(self, vertices, indices=None):
        self.vertex_count = len(vertices)         # 2-D array: len = number of rows = vertex count
        self.index_count  = len(indices) if indices is not None else 0
        self.has_indices  = indices is not None

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)

        # Upload vertex data
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        stride = 8 * 4   # 8 floats × 4 bytes

        # location 0: position (xyz)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # location 1: normal (xyz)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        # location 2: UV (uv)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)

        # Upload index data if provided
        if self.has_indices:
            self.ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        glBindVertexArray(0)

    def draw(self):
        glBindVertexArray(self.vao)
        if self.has_indices:
            glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        else:
            glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        glBindVertexArray(0)

    def delete(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        if self.has_indices:
            glDeleteBuffers(1, [self.ebo])


# -----------------------------------------------------------------------
# Geometry builders
# -----------------------------------------------------------------------

def _face(p0, p1, p2, p3, normal, uv_scale=1.0):
    """
    Build two CCW triangles for a quad face.
    p0=bottom-left, p1=bottom-right, p2=top-right, p3=top-left
    viewed from OUTSIDE (from the normal's side).
    CCW winding: tri1 = 0,2,1   tri2 = 0,3,2
    Returns 6 rows of [x,y,z, nx,ny,nz, u,v].
    """
    uvs = [(0,0),(uv_scale,0),(uv_scale,uv_scale),(0,uv_scale)]
    corners = [p0, p1, p2, p3]
    n = normal

    def v(idx):
        p = corners[idx]
        u, vv = uvs[idx]
        return [p[0],p[1],p[2], n[0],n[1],n[2], u, vv]

    return [v(0),v(2),v(1),  v(0),v(3),v(2)]


def make_box(width=1.0, height=1.0, depth=1.0, uv_scale=1.0):
    """
    Axis-aligned box centred at origin.
    Returns a Mesh.
    """
    w, h, d = width/2, height/2, depth/2
    rows = []

    # Front  (+Z)
    rows += _face([-w,-h, d],[ w,-h, d],[ w, h, d],[-w, h, d],[0,0,1], uv_scale)
    # Back   (-Z)
    rows += _face([ w,-h,-d],[-w,-h,-d],[-w, h,-d],[ w, h,-d],[0,0,-1], uv_scale)
    # Left   (-X)
    rows += _face([-w,-h,-d],[-w,-h, d],[-w, h, d],[-w, h,-d],[-1,0,0], uv_scale)
    # Right  (+X)
    rows += _face([ w,-h, d],[ w,-h,-d],[ w, h,-d],[ w, h, d],[1,0,0], uv_scale)
    # Top    (+Y)
    rows += _face([-w, h, d],[ w, h, d],[ w, h,-d],[-w, h,-d],[0,1,0], uv_scale)
    # Bottom (-Y)
    rows += _face([-w,-h,-d],[ w,-h,-d],[ w,-h, d],[-w,-h, d],[0,-1,0], uv_scale)

    verts = np.array(rows, dtype=np.float32)
    return Mesh(verts)


def make_box_uvworld(width=1.0, height=1.0, depth=1.0, uv_density=0.25):
    """
    Axis-aligned box where UVs tile at a fixed world-space density.
    uv_density = texture tiles per world unit (e.g. 0.25 → 1 tile per 4 units).
    Each face uses its own (u_scale, v_scale) so the texture never stretches.

    Face UV assignment (what makes sense visually):
      Front/Back (+Z/-Z): U = width,  V = height
      Left/Right (-X/+X): U = depth,  V = height
      Top/Bottom (+Y/-Y): U = width,  V = depth
    All multiplied by uv_density.
    """
    def _face_uv(p0, p1, p2, p3, normal, us, vs):
        uvs = [(0,0),(us,0),(us,vs),(0,vs)]
        corners = [p0,p1,p2,p3]
        n = normal
        def v(i):
            p = corners[i]; u,vv = uvs[i]
            return [p[0],p[1],p[2], n[0],n[1],n[2], u, vv]
        return [v(0),v(2),v(1), v(0),v(3),v(2)]

    W, H, D = width, height, depth
    w, h, d = W/2, H/2, D/2
    ud = uv_density
    rows = []
    # Front (+Z)
    rows += _face_uv([-w,-h,d],[w,-h,d],[w,h,d],[-w,h,d], [0,0,1],  W*ud, H*ud)
    # Back  (-Z)
    rows += _face_uv([w,-h,-d],[-w,-h,-d],[-w,h,-d],[w,h,-d], [0,0,-1], W*ud, H*ud)
    # Left  (-X)
    rows += _face_uv([-w,-h,-d],[-w,-h,d],[-w,h,d],[-w,h,-d], [-1,0,0], D*ud, H*ud)
    # Right (+X)
    rows += _face_uv([w,-h,d],[w,-h,-d],[w,h,-d],[w,h,d], [1,0,0],  D*ud, H*ud)
    # Top   (+Y)
    rows += _face_uv([-w,h,d],[w,h,d],[w,h,-d],[-w,h,-d], [0,1,0],  W*ud, D*ud)
    # Bottom(-Y)
    rows += _face_uv([-w,-h,-d],[w,-h,-d],[w,-h,d],[-w,-h,d], [0,-1,0], W*ud, D*ud)
    return Mesh(np.array(rows, dtype=np.float32))



def make_ground_plane(size=100.0, uv_scale=20.0):
    """
    Flat XZ plane centred at origin, normal pointing up (+Y).
    uv_scale tiles the texture across the plane.
    """
    h = size / 2
    # CCW when viewed from above (+Y): front-left -> front-right -> back-right -> back-left
    rows = _face(
        [-h, 0, h],[ h, 0, h],[ h, 0,-h],[-h, 0,-h],
        [0, 1, 0],
        uv_scale
    )
    verts = np.array(rows, dtype=np.float32)
    return Mesh(verts)


def make_cylinder(radius=0.5, height=1.0, segments=16, uv_scale=1.0):
    """
    Upright cylinder centred at origin. Used for tree trunks, wheels, etc.
    """
    rows = []
    h = height / 2
    for i in range(segments):
        a0 = 2 * np.pi * i / segments
        a1 = 2 * np.pi * (i+1) / segments
        x0, z0 = radius * np.cos(a0), radius * np.sin(a0)
        x1, z1 = radius * np.cos(a1), radius * np.sin(a1)

        # Side quad
        nx0, nz0 = np.cos(a0), np.sin(a0)
        nx1, nz1 = np.cos(a1), np.sin(a1)
        u0, u1 = i/segments, (i+1)/segments

        # two triangles for this segment
        rows += [
            [x0,-h,z0, nx0,0,nz0, u0*uv_scale, 0],
            [x1,-h,z1, nx1,0,nz1, u1*uv_scale, 0],
            [x1, h,z1, nx1,0,nz1, u1*uv_scale, uv_scale],
            [x0,-h,z0, nx0,0,nz0, u0*uv_scale, 0],
            [x1, h,z1, nx1,0,nz1, u1*uv_scale, uv_scale],
            [x0, h,z0, nx0,0,nz0, u0*uv_scale, uv_scale],
        ]

        # Top cap
        rows += [
            [0, h, 0,  0,1,0,  0.5, 0.5],
            [x0,h,z0,  0,1,0,  0.5+0.5*np.cos(a0), 0.5+0.5*np.sin(a0)],
            [x1,h,z1,  0,1,0,  0.5+0.5*np.cos(a1), 0.5+0.5*np.sin(a1)],
        ]
        # Bottom cap
        rows += [
            [0,-h, 0,  0,-1,0,  0.5, 0.5],
            [x1,-h,z1, 0,-1,0,  0.5+0.5*np.cos(a1), 0.5+0.5*np.sin(a1)],
            [x0,-h,z0, 0,-1,0,  0.5+0.5*np.cos(a0), 0.5+0.5*np.sin(a0)],
        ]

    verts = np.array(rows, dtype=np.float32)
    return Mesh(verts)


def make_cone(radius=0.5, height=1.0, segments=12, uv_scale=1.0):
    """Upright cone – used for simple trees."""
    rows = []
    h_base = 0.0
    h_tip  = height
    for i in range(segments):
        a0 = 2 * np.pi * i / segments
        a1 = 2 * np.pi * (i+1) / segments
        x0, z0 = radius * np.cos(a0), radius * np.sin(a0)
        x1, z1 = radius * np.cos(a1), radius * np.sin(a1)
        u0, u1 = i/segments, (i+1)/segments

        # Slope
        rows += [
            [0,  h_tip, 0,   0,1,0,  0.5, 1.0],
            [x0, h_base, z0, np.cos(a0),0,np.sin(a0), u0, 0],
            [x1, h_base, z1, np.cos(a1),0,np.sin(a1), u1, 0],
        ]
        # Base cap
        rows += [
            [0,  h_base, 0,  0,-1,0, 0.5, 0.5],
            [x1, h_base, z1, 0,-1,0, 0.5+0.5*np.cos(a1), 0.5+0.5*np.sin(a1)],
            [x0, h_base, z0, 0,-1,0, 0.5+0.5*np.cos(a0), 0.5+0.5*np.sin(a0)],
        ]

    verts = np.array(rows, dtype=np.float32)
    return Mesh(verts)

# -----------------------------------------------------------------------
# OBJ loader
# -----------------------------------------------------------------------

def load_obj(path):
    """
    Load a Wavefront OBJ file and return a Mesh.

    Handles:
      - v  (positions), vn (normals), vt (texture coords)
      - f  with triangles or quads, using v/vt/vn or v//vn or v indices
      - Multiple objects / groups in one file (merged into a single Mesh)
      - Missing normals or UVs (filled with zeros)

    Place car.obj in a models/ folder next to main.py.
    The returned mesh uses the same [x,y,z, nx,ny,nz, u,v] layout as
    every other mesh in this project, so it works with the same shader.
    """
    positions = []
    normals   = []
    uvs       = []
    vertices  = []

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            tok   = parts[0]

            if tok == 'v':
                positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tok == 'vn':
                normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tok == 'vt':
                uvs.append((float(parts[1]), float(parts[2])))
            elif tok == 'f':
                face_verts = parts[1:]
                parsed = []
                for fv in face_verts:
                    items = fv.split('/')
                    pi = int(items[0]) - 1
                    ti = int(items[1]) - 1 if len(items) > 1 and items[1] else -1
                    ni = int(items[2]) - 1 if len(items) > 2 and items[2] else -1
                    parsed.append((pi, ti, ni))
                # Fan-triangulate (handles tris and quads)
                for i in range(1, len(parsed) - 1):
                    for pi, ti, ni in [parsed[0], parsed[i], parsed[i+1]]:
                        x,  y,  z  = positions[pi]
                        nx, ny, nz = normals[ni] if ni >= 0 and ni < len(normals) else (0, 1, 0)
                        u,  v      = uvs[ti]     if ti >= 0 and ti < len(uvs)     else (0, 0)
                        vertices.append([x, y, z, nx, ny, nz, u, v])

    if not vertices:
        raise RuntimeError(f"OBJ file '{path}' contained no geometry.")

    return Mesh(np.array(vertices, dtype=np.float32))


# -----------------------------------------------------------------------
# Multi-material OBJ loader
# -----------------------------------------------------------------------

def load_obj_materials(path, material_colors=None):
    """
    Load a Wavefront OBJ file and return a list of (Mesh, color) tuples,
    one per material group.  material_colors is a dict mapping material
    name → (r, g, b).  Any material not in the dict gets a default grey.

    The returned list can be drawn with:
        for mesh, color in car_parts:
            set_vec3(shader, "uColor", color)
            set_mat4(shader, "uModel", model_matrix)
            mesh.draw()
    """
    if material_colors is None:
        material_colors = {}

    positions = []
    normals   = []
    uvs       = []

    # Collect faces per material: {mat_name: [vertex_rows]}
    groups   = {}   # mat_name → list of vertex rows
    cur_mat  = '__default__'

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            tok   = parts[0]

            if tok == 'v':
                positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tok == 'vn':
                normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tok == 'vt':
                uvs.append((float(parts[1]), float(parts[2])))
            elif tok == 'usemtl':
                cur_mat = parts[1]
                if cur_mat not in groups:
                    groups[cur_mat] = []
            elif tok == 'f':
                face_verts = parts[1:]
                parsed = []
                for fv in face_verts:
                    items = fv.split('/')
                    pi = int(items[0]) - 1
                    ti = int(items[1]) - 1 if len(items) > 1 and items[1] else -1
                    ni = int(items[2]) - 1 if len(items) > 2 and items[2] else -1
                    parsed.append((pi, ti, ni))
                if cur_mat not in groups:
                    groups[cur_mat] = []
                for i in range(1, len(parsed) - 1):
                    for pi, ti, ni in [parsed[0], parsed[i], parsed[i+1]]:
                        x,  y,  z  = positions[pi]
                        nx, ny, nz = normals[ni] if 0 <= ni < len(normals) else (0, 1, 0)
                        u,  v      = uvs[ti]     if 0 <= ti < len(uvs)     else (0, 0)
                        groups[cur_mat].append([x, y, z, nx, ny, nz, u, v])

    result = []
    default_color = (0.6, 0.6, 0.6)
    for mat_name, verts in groups.items():
        if not verts:
            continue
        mesh  = Mesh(np.array(verts, dtype=np.float32))
        color = material_colors.get(mat_name, default_color)
        result.append((mesh, color))
    return result