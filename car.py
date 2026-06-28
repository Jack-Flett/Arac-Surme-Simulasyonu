import math
import numpy as np
import pygame

from camera import translation_matrix, scale_matrix, rotation_y


class Car:
    WIDTH  = 2.0
    HEIGHT = 0.8
    LENGTH = 4.0

    MAX_SPEED   = 15.0
    MAX_REVERSE =  5.0
    ACCEL       = 10.0
    BRAKE       = 20.0
    FRICTION    =  5.0
    TURN_SPEED  = 75.0

    def __init__(self, position=(0.0, 0.0, 8.0), heading=0.0):
        self.position = np.array(
            [position[0], self.HEIGHT / 2, position[2]], dtype=np.float32
        )
        self.heading = float(heading)
        self.speed   = 0.0

    # ── Movement ────────────────────────────────────────────────────────

    def get_forward(self):
        h = math.radians(self.heading)
        return np.array([-math.sin(h), 0.0, -math.cos(h)], dtype=np.float32)

    def update(self, keys, dt, building_boxes=None):
        # Steering
        if abs(self.speed) > 0.3:
            steer = 0
            if keys[pygame.K_a]: steer =  1
            if keys[pygame.K_d]: steer = -1
            sign = 1.0 if self.speed > 0.0 else -1.0
            self.heading += steer * self.TURN_SPEED * sign * dt

        # Throttle / brake
        if keys[pygame.K_w]:
            self.speed = min(self.MAX_SPEED,    self.speed + self.ACCEL   * dt)
        elif keys[pygame.K_s]:
            self.speed = max(-self.MAX_REVERSE, self.speed - self.BRAKE   * dt)
        else:
            if self.speed > 0.0:
                self.speed = max(0.0, self.speed - self.FRICTION * dt)
            elif self.speed < 0.0:
                self.speed = min(0.0, self.speed + self.FRICTION * dt)

        old_pos = self.position.copy()
        self.position += self.get_forward() * self.speed * dt
        self.position[1] = self.HEIGHT / 2

        if building_boxes:
            self._resolve_collision(old_pos, building_boxes)

    # ── Collision ───────────────────────────────────────────────────────

    def _get_obb_axes_corners(self):
        """
        Return the car's OBB as:
          axes    – two XZ unit vectors (forward and right)
          corners – four XZ corner positions as (x, z) tuples
        """
        h   = math.radians(self.heading)
        fx, fz =  -math.sin(h), -math.cos(h)
        rx, rz =  -math.cos(h),  math.sin(h)
        hw = self.WIDTH  / 2
        hl = self.LENGTH / 2
        cx, cz = self.position[0], self.position[2]
        corners = [
            (cx + fx*hl + rx*hw,  cz + fz*hl + rz*hw),
            (cx + fx*hl - rx*hw,  cz + fz*hl - rz*hw),
            (cx - fx*hl - rx*hw,  cz - fz*hl - rz*hw),
            (cx - fx*hl + rx*hw,  cz - fz*hl + rz*hw),
        ]
        return (fx, fz), (rx, rz), corners

    def _get_mtv(self, bbox):
        """
        SAT test: car OBB vs obstacle (AABB or OBB).
        bbox is either (bmin, bmax) for an AABB, or (bmin, bmax, heading_deg) for an OBB.
        Returns (dx, dz) MTV to push the car out, or None if no overlap.
        """
        # --- Car axes and corners ---
        (fx, fz), (rx, rz), car_corners = self._get_obb_axes_corners()

        def project_pts(pts, ax, az):
            dots = [ax*x + az*z for x, z in pts]
            return min(dots), max(dots)

        # --- Obstacle corners ---
        if len(bbox) == 3:
            bmin, bmax, obs_heading = bbox
            oh = math.radians(obs_heading)
            ofx, ofz =  math.sin(oh), math.cos(oh)   # obstacle forward (note: building rotation_y)
            orx, orz =  math.cos(oh), -math.sin(oh)  # obstacle right
            ocx = (float(bmin[0]) + float(bmax[0])) * 0.5
            ocz = (float(bmin[2]) + float(bmax[2])) * 0.5
            ohw = (float(bmax[0]) - float(bmin[0])) * 0.5
            ohd = (float(bmax[2]) - float(bmin[2])) * 0.5
            obs_corners = [
                (ocx + ofx*ohd + orx*ohw,  ocz + ofz*ohd + orz*ohw),
                (ocx + ofx*ohd - orx*ohw,  ocz + ofz*ohd - orz*ohw),
                (ocx - ofx*ohd - orx*ohw,  ocz - ofz*ohd - orz*ohw),
                (ocx - ofx*ohd + orx*ohw,  ocz - ofz*ohd + orz*ohw),
            ]
            # SAT axes: car forward, car right, obstacle forward, obstacle right
            axes = [(fx, fz), (rx, rz), (ofx, ofz), (orx, orz)]
        else:
            bmin, bmax = bbox[0], bbox[1]
            ax0, az0 = float(bmin[0]), float(bmin[2])
            ax1, az1 = float(bmax[0]), float(bmax[2])
            obs_corners = [(ax0,az0),(ax1,az0),(ax1,az1),(ax0,az1)]
            axes = [(1.0, 0.0), (0.0, 1.0), (fx, fz), (rx, rz)]

        min_overlap = float('inf')
        mtv_ax, mtv_az = 0.0, 0.0

        for ax, az in axes:
            o_min, o_max = project_pts(car_corners,  ax, az)
            a_min, a_max = project_pts(obs_corners,  ax, az)
            if o_max <= a_min or a_max <= o_min:
                return None
            overlap = min(o_max - a_min, a_max - o_min)
            if overlap < min_overlap:
                min_overlap = overlap
                oc = (o_min + o_max) * 0.5
                ac = (a_min + a_max) * 0.5
                sign = 1.0 if oc > ac else -1.0
                mtv_ax = ax * overlap * sign
                mtv_az = az * overlap * sign

        return (mtv_ax, mtv_az)

    def _overlaps_any(self, bboxes):
        for bbox in bboxes:
            if self._get_mtv(bbox) is not None:
                return True
        return False

    def _resolve_collision(self, old_pos, bboxes):
        """
        MTV-based depenetration: push the car out of every overlapping box.
        Works for both AABB and OBB obstacles, forward and reverse.
        """
        mtvs = []
        for bbox in bboxes:
            mtv = self._get_mtv(bbox)
            if mtv is not None:
                mtvs.append(mtv)

        if not mtvs:
            return

        # Sum all push vectors (handles corner cases with multiple boxes)
        total_dx = sum(m[0] for m in mtvs)
        total_dz = sum(m[1] for m in mtvs)

        self.position[0] += total_dx
        self.position[2] += total_dz

        # Cancel speed if the car is moving toward the surface it just hit.
        # MTV pushes AWAY from wall, so if velocity opposes the push (dot < 0),
        # the car was moving INTO the wall → kill speed.
        push_len = math.sqrt(total_dx*total_dx + total_dz*total_dz)
        if push_len > 1e-6:
            nx, nz = total_dx / push_len, total_dz / push_len
            fwd = self.get_forward()
            vel_dot = fwd[0] * nx + fwd[2] * nz
            if (self.speed > 0 and vel_dot < 0) or (self.speed < 0 and vel_dot > 0):
                self.speed = 0.0

    # ── Rendering ───────────────────────────────────────────────────────

    def get_obb_corners_world(self):
        """Return the 4 XZ world corners of the car OBB (for debug drawing)."""
        _, _, corners = self._get_obb_axes_corners()
        y0, y1 = 0.0, self.HEIGHT
        return (
            [(x, y0, z) for x, z in corners],
            [(x, y1, z) for x, z in corners],
        )

    def get_model_matrix(self):
        T = translation_matrix(*self.position)
        R = rotation_y(self.heading)
        S = scale_matrix(self.WIDTH, self.HEIGHT, self.LENGTH)
        return T @ R @ S