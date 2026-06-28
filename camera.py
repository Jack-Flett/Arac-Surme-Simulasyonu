import numpy as np
import math


class Camera:
    """
    Free-fly camera.
    WASD  – move forward/back/left/right
    Mouse – look around (yaw + pitch)
    Space / LShift – move up / down  (optional, handy for testing)

    Later (Week 2) we'll add a ChaseCamera subclass for the car.
    """

    def __init__(self, position=(0.0, 2.0, 5.0), yaw=-90.0, pitch=0.0):
        self.position = np.array(position, dtype=np.float32)
        self.yaw   = yaw    # degrees, -90 looks toward -Z
        self.pitch = pitch  # degrees, clamped to ±89

        self.move_speed  = 10.0   # units per second
        self.mouse_sens  = 0.1    # degrees per pixel

        self._update_vectors()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_keyboard(self, keys, dt):
        import pygame
        velocity = self.move_speed * dt

        if keys[pygame.K_w]:
            self.position += self.front * velocity
        if keys[pygame.K_s]:
            self.position -= self.front * velocity
        if keys[pygame.K_a]:
            self.position -= self.right * velocity
        if keys[pygame.K_d]:
            self.position += self.right * velocity
        if keys[pygame.K_SPACE]:
            self.position += self.world_up * velocity
        if keys[pygame.K_LSHIFT]:
            self.position -= self.world_up * velocity

    def process_mouse(self, dx, dy):
        self.yaw   += dx * self.mouse_sens
        self.pitch -= dy * self.mouse_sens          # minus: moving mouse up → pitch up
        self.pitch  = max(-89.0, min(89.0, self.pitch))
        self._update_vectors()

    def get_view_matrix(self):
        return look_at(self.position, self.position + self.front, self.up)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_vectors(self):
        yaw_r   = math.radians(self.yaw)
        pitch_r = math.radians(self.pitch)

        front = np.array([
            math.cos(yaw_r) * math.cos(pitch_r),
            math.sin(pitch_r),
            math.sin(yaw_r) * math.cos(pitch_r),
        ], dtype=np.float32)

        self.front     = front / np.linalg.norm(front)
        self.world_up  = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right          = np.cross(self.front, self.world_up)
        self.right     = right / np.linalg.norm(right)
        self.up        = np.cross(self.right, self.front)
        self.up       /= np.linalg.norm(self.up)


# ------------------------------------------------------------------
# Pure-NumPy matrix helpers  (column-major, matching OpenGL convention)
# ------------------------------------------------------------------

def look_at(eye, center, up):
    f = center - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = r
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3]  = -np.dot(r, eye)
    m[1, 3]  = -np.dot(u, eye)
    m[2, 3]  =  np.dot(f, eye)
    return m


def perspective(fov_deg, aspect, near, far):
    f   = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m   = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] =  f / aspect
    m[1, 1] =  f
    m[2, 2] =  (far + near) / (near - far)
    m[2, 3] =  (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def translation_matrix(x, y, z):
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def scale_matrix(sx, sy, sz):
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def rotation_y(angle_deg):
    a   = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] =  c;  m[0, 2] = s
    m[2, 0] = -s;  m[2, 2] = c
    return m


def rotation_x(angle_deg):
    a   = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[1, 1] =  c;  m[1, 2] = -s
    m[2, 1] =  s;  m[2, 2] =  c
    return m


def rotation_z(angle_deg):
    a   = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] =  c;  m[0, 1] = -s
    m[1, 0] =  s;  m[1, 1] =  c
    return m


# ------------------------------------------------------------------
# Chase camera  (follows a Car instance from behind)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Chase camera  (follows a Car instance from behind)
#
# Mouse drag in chase mode orbits the camera left/right around the car
# and tilts it up/down.  When the mouse is released the yaw offset
# gradually snaps back to zero (directly behind the car).
# ------------------------------------------------------------------

class ChaseCamera:
    DIST         = 10.0   # orbit radius (units from car centre)
    HEIGHT_BASE  =  3.0   # base height above car when pitch_offset = 0
    SNAP_SPEED   =  2.5   # how fast yaw snaps back (exp constant, s⁻¹)
    MOUSE_SENS   =  0.25  # degrees per pixel
    PITCH_MIN    = -20.0  # degrees — look down limit
    PITCH_MAX    =  40.0  # degrees — look up limit
    POS_LAG      =  8.0   # position smoothing (higher = tighter)

    def __init__(self, car):
        self.yaw_offset   = 0.0   # degrees offset from directly behind car
        self.pitch_offset = 15.0  # degrees above horizon (positive = looking down)
        self._mouse_active = False  # True while mouse is being dragged
        self._idle_timer   = 0.0   # seconds since last mouse movement

        # Initialise position directly behind car
        self.position = self._ideal_position(car)

    # ── Called from main event loop ─────────────────────────────────
    def process_mouse(self, dx, dy):
        """Call with pygame MOUSEMOTION rel values while in chase mode."""
        self.yaw_offset   -= dx * self.MOUSE_SENS   # negative: drag right → orbit right
        self.pitch_offset -= dy * self.MOUSE_SENS
        self.pitch_offset  = max(self.PITCH_MIN, min(self.PITCH_MAX, self.pitch_offset))
        self._idle_timer   = 0.0
        self._mouse_active = True

    # ── Called every frame ───────────────────────────────────────────
    PITCH_REST = 15.0   # degrees — resting pitch camera snaps back to

    def update(self, car, dt):
        # Count idle time; after 0.3 s of no mouse input start snapping back
        self._idle_timer += dt
        if self._idle_timer > 0.3:
            self._mouse_active = False

        if not self._mouse_active:
            # Wrap yaw to [-180, 180] so decay always takes the shortest path back
            self.yaw_offset = (self.yaw_offset + 180.0) % 360.0 - 180.0
            # Snap yaw back toward 0 (directly behind car)
            self.yaw_offset *= math.exp(-self.SNAP_SPEED * dt)
            if abs(self.yaw_offset) < 0.1:
                self.yaw_offset = 0.0
            # Snap pitch back toward resting angle
            diff = self.PITCH_REST - self.pitch_offset
            self.pitch_offset += diff * (1.0 - math.exp(-self.SNAP_SPEED * dt))
            if abs(diff) < 0.1:
                self.pitch_offset = self.PITCH_REST

        # Smoothly move camera position toward ideal orbit point
        ideal = self._ideal_position(car)
        t = 1.0 - math.exp(-self.POS_LAG * dt)
        self.position = self.position + (ideal - self.position) * t

    def get_view_matrix(self, car):
        target = car.position + np.array([0.0, 0.8, 0.0], dtype=np.float32)
        up     = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return look_at(self.position, target, up)

    # ── Internal ─────────────────────────────────────────────────────
    def _ideal_position(self, car):
        """Compute the target camera position for the current offsets."""
        # Total yaw in world space = car heading + our offset
        total_yaw = math.radians(car.heading + self.yaw_offset)
        pitch_r   = math.radians(self.pitch_offset)

        # Orbit vector: points FROM car TOWARD camera
        # In the car's local frame, "behind" is +heading direction reversed.
        # We rotate by total_yaw around Y, then lift by pitch.
        horiz_dist = self.DIST * math.cos(pitch_r)
        vert_dist  = self.DIST * math.sin(pitch_r)

        # "Behind" the car facing total_yaw
        ox = math.sin(total_yaw) * horiz_dist
        oz = math.cos(total_yaw) * horiz_dist

        return car.position + np.array([ox, vert_dist + self.HEIGHT_BASE * 0.3,
                                         oz], dtype=np.float32)