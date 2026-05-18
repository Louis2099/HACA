"""Video overlay utilities for the Dodgeball-G1 task.

Public API
----------
configure_viewer_for_video(env_cfg, env_index)
    Call *before* gym.make().  Sets env_cfg.viewer to track the selected robot.

DodgeballVideoOverlay(env, env_index)
    A gymnasium.Wrapper placed *inside* RecordVideo:

        raw_env → DodgeballVideoOverlay → RecordVideo → RslRlVecEnvWrapper

    After every physics step it updates VisualizationMarkers in the USD stage
    so the markers appear in the rendered frame captured by RecordVideo.

    Markers:
      • Yellow small sphere   – CoM floor projection (on ground plane)
      • Yellow small sphere   – CoM at its actual 3-D world position
      • Yellow thin cylinder  – Vertical stick connecting floor CoM to 3-D CoM
      • Green spheres         – Grounded ankle centres (one per grounded foot)
      • Green thin cylinders  – Convex hull of all grounded foot-patch corners

    Each grounded foot is represented as a rectangular patch centred on the ankle
    link (foot_half_len × foot_half_width).  The support polygon drawn is the
    convex hull of all patch corners — identical to what the reward function uses.

    VisualizationMarkers create real USD geometry visible to all cameras,
    including the offscreen camera used for video recording.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import torch
import gymnasium as gym

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Camera helper — call before gym.make()
# ---------------------------------------------------------------------------

def configure_viewer_for_video(env_cfg, env_index: int = 0) -> None:
    """Configure env_cfg.viewer to track the selected robot for video recording.

    Uses Isaac Lab's built-in ``origin_type = "asset_root"`` mechanism so the
    camera follows the robot every frame without per-step manipulation.
    Works in headless + ``--enable_cameras`` mode.

    Eye and lookat offsets are chosen to show the full body, floor contact
    area, and the ground-projected CoM/support-polygon markers.
    """
    if not hasattr(env_cfg, "viewer") or env_cfg.viewer is None:
        return
    v = env_cfg.viewer
    v.origin_type = "asset_root"
    v.env_index = int(env_index)
    v.asset_name = "robot"
    # Slightly behind and above; lookat offset keeps torso and ground in frame.
    v.eye = (-3.0, -4.5, 3.2)
    v.lookat = (0.0, 0.3, 0.8)
    v.resolution = (1280, 720)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _convex_hull_2d(pts: np.ndarray) -> np.ndarray:
    """Return convex-hull vertices of a 2-D point set (CCW order).

    Falls back to unique input points if < 3 points or scipy is unavailable.
    """
    if len(pts) < 3:
        return pts
    try:
        from scipy.spatial import ConvexHull  # type: ignore
        return pts[ConvexHull(pts).vertices]
    except Exception:
        return pts


def _foot_patch_corners_np(
    ankle_xy: np.ndarray,   # [2]
    foot_yaw: float,        # radians
    half_len: float,
    half_width: float,
    toe_offset: float = 0.0,
) -> np.ndarray:            # [4, 2]
    """Numpy version of the foot-patch rectangle (matches the PyTorch reward helper).

    The patch is shifted ``toe_offset`` metres forward in foot-local +x so that
    the ankle-to-heel distance equals ``half_len − toe_offset`` and the
    ankle-to-toe distance equals ``half_len + toe_offset``.
    """
    cos_y, sin_y = float(np.cos(foot_yaw)), float(np.sin(foot_yaw))
    toe  = half_len + toe_offset
    heel = half_len - toe_offset
    local = np.array(
        [[ toe,   half_width],
         [ toe,  -half_width],
         [-heel, -half_width],
         [-heel,  half_width]],
        dtype=np.float32,
    )  # [4, 2]
    rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float32)
    return ankle_xy + local @ rot.T   # [4, 2]


def _quat_from_z_to_dir(d: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) that rotates the Z-axis to direction *d*.

    Uses the half-angle formula: q = normalize([1 + dz, -dy, dx, 0]).

    Special cases:
      • d ≈  Z → identity [1, 0, 0, 0]
      • d ≈ -Z → 180° around X [0, 1, 0, 0]
    """
    d = d / (np.linalg.norm(d) + 1e-12)
    dz = float(d[2])
    if dz > 1.0 - 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if dz < -1.0 + 1e-6:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    # Rodrigues half-angle via cross product with Z
    w = 1.0 + dz
    x = -float(d[1])
    y = float(d[0])
    z = 0.0
    norm = np.sqrt(w * w + x * x + y * y)
    return np.array([w / norm, x / norm, y / norm, z], dtype=np.float32)


def _cylinder_pose(
    a: np.ndarray,
    b: np.ndarray,
    floor_z: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (translation, orientation_wxyz, scale) for a unit cylinder (height=1,radius=1)
    connecting point *a* to point *b*, both snapped to floor_z."""
    a3 = np.array([a[0], a[1], floor_z], dtype=np.float32)
    b3 = np.array([b[0], b[1], floor_z], dtype=np.float32)
    diff = b3 - a3
    length = float(np.linalg.norm(diff))
    if length < 1e-4:
        length = 1e-4
    mid = (a3 + b3) * 0.5
    quat = _quat_from_z_to_dir(diff / length)
    # Scale: keep X/Y at 1 (prototype radius sets actual radius),
    # stretch Z to the segment length.
    scale = np.array([1.0, 1.0, length], dtype=np.float32)
    return mid, quat, scale


# ---------------------------------------------------------------------------
# Overlay wrapper
# ---------------------------------------------------------------------------

class DodgeballVideoOverlay(gym.Wrapper):
    """Updates USD VisualizationMarkers each step so CoM and support polygon
    appear in the recorded video.

    Parameters
    ----------
    env :
        Isaac Lab gymnasium environment to wrap.
    env_index :
        Which parallel environment to visualise (default: 0).
    force_threshold :
        Minimum contact-force magnitude (N) to count a body as grounded.
    diag_steps :
        Print per-step diagnostics for this many steps after each reset (0 = off).
    """

    # Prototype indices within the VisualizationMarkers
    _IDX_COM_FLOOR = 0   # large yellow sphere at CoM floor projection
    _IDX_COM_3D    = 1   # medium yellow sphere at actual 3-D CoM
    _IDX_CONTACT   = 2   # green sphere at each contact point (floor)
    _IDX_CYL_Y     = 3   # yellow thin cylinder (vertical CoM line)
    _IDX_CYL_G     = 4   # green thin cylinder (support polygon edges / spokes)

    def __init__(
        self,
        env: gym.Env,
        env_index: int = 0,
        force_threshold: float = 10.0,
        foot_half_len: float = 0.09,
        foot_half_width: float = 0.045,
        foot_toe_offset: float = 0.02,
        diag_steps: int = 50,
    ):
        super().__init__(env)
        self._env_index = int(env_index)
        self._force_threshold = force_threshold
        self._foot_half_len = foot_half_len
        self._foot_half_width = foot_half_width
        self._foot_toe_offset = foot_toe_offset
        self._diag_steps = diag_steps
        self._step_count = 0

        self._markers = None  # created lazily after USD stage is ready
        self._markers_init_failed = False
        # Mapping: sensor-local body index → robot body index (built on first use)
        self._sensor_to_robot_ids: list[int] | None = None

    # ------------------------------------------------------------------
    # Lazy marker initialisation
    # ------------------------------------------------------------------

    def _try_init_markers(self) -> bool:
        """Create VisualizationMarkers. Returns True on success."""
        if self._markers is not None:
            return True
        if self._markers_init_failed:
            return False
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/DodgeballOverlay",
                markers={
                    # 0: CoM floor projection — small yellow sphere, same size as 3-D marker
                    "com_floor": sim_utils.SphereCfg(
                        radius=0.04,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.9, 0.0), emissive_color=(0.4, 0.35, 0.0)
                        ),
                    ),
                    # 1: CoM actual 3-D position — yellow sphere (same radius as floor marker)
                    "com_3d": sim_utils.SphereCfg(
                        radius=0.04,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.9, 0.0), emissive_color=(0.3, 0.27, 0.0)
                        ),
                    ),
                    # 2: Contact point — green sphere on floor (small, to not obscure polygon)
                    "contact": sim_utils.SphereCfg(
                        radius=0.04,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 0.2), emissive_color=(0.0, 0.3, 0.06)
                        ),
                    ),
                    # 3: Yellow thin cylinder for vertical CoM stick
                    "cyl_yellow": sim_utils.CylinderCfg(
                        radius=0.025,
                        height=1.0,
                        axis="Z",
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.85, 0.0), emissive_color=(0.3, 0.25, 0.0)
                        ),
                    ),
                    # 4: Green thin cylinder for support polygon edges / spokes
                    "cyl_green": sim_utils.CylinderCfg(
                        radius=0.022,
                        height=1.0,
                        axis="Z",
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 0.2), emissive_color=(0.0, 0.3, 0.06)
                        ),
                    ),
                },
            )
            self._markers = VisualizationMarkers(cfg)
            log.info("[DodgeballVideoOverlay] VisualizationMarkers created at %s", self._markers.prim_path)
            return True
        except Exception as exc:
            warnings.warn(
                f"[DodgeballVideoOverlay] Could not create VisualizationMarkers: {exc}. "
                "CoM/support-polygon markers will not appear in the video.",
                stacklevel=2,
            )
            self._markers_init_failed = True
            return False

    # ------------------------------------------------------------------
    def step(self, action: Any):  # type: ignore[override]
        result = super().step(action)
        self._step_count += 1
        if self._try_init_markers():
            try:
                self._update_markers()
            except Exception as exc:
                log.debug("DodgeballVideoOverlay: marker update failed: %s", exc)
        return result

    # ------------------------------------------------------------------
    def _update_markers(self) -> None:
        base_env = self.env.unwrapped
        if not hasattr(base_env, "scene"):
            return

        try:
            robot = base_env.scene["robot"]
        except KeyError:
            return
        try:
            contact_sensor = base_env.scene["contact_forces"]
        except KeyError:
            contact_sensor = None

        idx = self._env_index

        # ── 1. Compute CoM ────────────────────────────────────────────────
        body_pos_w = robot.data.body_pos_w                              # [N, B, 3]
        body_quat_w = robot.data.body_quat_w                            # [N, B, 4]
        masses = robot.data.default_mass.to(body_pos_w.device)         # [N, B]
        total_mass = float(masses[idx].sum().clamp_min(1e-6).item())
        com_xyz_t = (masses[idx].unsqueeze(-1) * body_pos_w[idx]).sum(0) / total_mass
        com = np.array([
            float(com_xyz_t[0].item()),
            float(com_xyz_t[1].item()),
            float(com_xyz_t[2].item()),
        ], dtype=np.float32)

        FLOOR_Z = 0.025  # slightly above ground to avoid z-fighting

        # ── 2. Collect grounded foot patches ──────────────────────────────
        # Build sensor→robot body mapping once.
        # ContactSensor has no `body_ids` attribute; use body_names + find_bodies().
        foot_patches: list[np.ndarray] = []   # each entry: [4, 2] world-XY corners
        ankle_centers: list[np.ndarray] = []  # ankle XY for sphere markers

        if contact_sensor is not None:
            if self._sensor_to_robot_ids is None:
                try:
                    s_names = contact_sensor.body_names
                    robot_ids, _ = robot.find_bodies(s_names, preserve_order=True)
                    self._sensor_to_robot_ids = robot_ids
                except Exception as exc:
                    log.warning("DodgeballVideoOverlay: sensor→robot map failed: %s", exc)
                    self._sensor_to_robot_ids = list(range(contact_sensor.num_bodies))

            forces_w = contact_sensor.data.net_forces_w            # [N, C, 3]
            force_norms = torch.norm(forces_w[idx], dim=-1)         # [C]

            for local_i, robot_body_id in enumerate(self._sensor_to_robot_ids):
                if local_i >= force_norms.shape[0]:
                    break
                if float(force_norms[local_i].item()) > self._force_threshold:
                    bp = body_pos_w[idx, robot_body_id]
                    bq = body_quat_w[idx, robot_body_id]
                    ankle_xy = np.array(
                        [float(bp[0].item()), float(bp[1].item())],
                        dtype=np.float32,
                    )
                    q = np.array(
                        [float(bq[0].item()), float(bq[1].item()),
                         float(bq[2].item()), float(bq[3].item())],
                        dtype=np.float32,
                    )
                    # Yaw from quaternion (w,x,y,z)
                    yaw = float(np.arctan2(
                        2.0 * (q[0] * q[3] + q[1] * q[2]),
                        1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2),
                    ))
                    patch = _foot_patch_corners_np(
                        ankle_xy, yaw,
                        self._foot_half_len, self._foot_half_width,
                        self._foot_toe_offset,
                    )
                    foot_patches.append(patch)
                    ankle_centers.append(ankle_xy)

        # ── 3. Optional diagnostics ────────────────────────────────────────
        if self._diag_steps > 0 and self._step_count <= self._diag_steps:
            self._print_diagnostics(base_env, com, foot_patches, ankle_centers, idx)

        # ── 4. Build marker arrays ─────────────────────────────────────────
        translations: list[np.ndarray] = []
        orientations: list[np.ndarray] = []
        scales:       list[np.ndarray] = []
        proto_ids:    list[int]        = []

        identity_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        unit_scale  = np.ones(3, dtype=np.float32)

        # CoM floor projection (small yellow sphere)
        translations.append(np.array([com[0], com[1], FLOOR_Z], dtype=np.float32))
        orientations.append(identity_q)
        scales.append(unit_scale)
        proto_ids.append(self._IDX_COM_FLOOR)

        # CoM 3-D position (small yellow sphere at true height)
        com_z = max(float(com[2]), FLOOR_Z + 0.05)
        translations.append(np.array([com[0], com[1], com_z], dtype=np.float32))
        orientations.append(identity_q)
        scales.append(unit_scale)
        proto_ids.append(self._IDX_COM_3D)

        # Vertical yellow stick from floor CoM to 3-D CoM
        vert_len = max(com_z - FLOOR_Z, 0.05)
        translations.append(np.array([com[0], com[1], FLOOR_Z + vert_len * 0.5], dtype=np.float32))
        orientations.append(identity_q)
        scales.append(np.array([1.0, 1.0, vert_len], dtype=np.float32))
        proto_ids.append(self._IDX_CYL_Y)

        # Green spheres at each grounded ankle centre
        for ankle_xy in ankle_centers:
            translations.append(np.array([ankle_xy[0], ankle_xy[1], FLOOR_Z], dtype=np.float32))
            orientations.append(identity_q)
            scales.append(unit_scale)
            proto_ids.append(self._IDX_CONTACT)

        # Support polygon: convex hull of ALL foot-patch corners from grounded feet.
        # For 1 foot  → 4 corners → hull is the foot rectangle.
        # For 2 feet  → 8 corners → hull wraps both patches + area between them.
        if foot_patches:
            all_corners = np.vstack(foot_patches)             # [4n, 2]
            hull_verts  = _convex_hull_2d(all_corners)        # sorted CCW vertices
            m = len(hull_verts)
            for i in range(m):
                j = (i + 1) % m
                mid, q, sc = _cylinder_pose(hull_verts[i], hull_verts[j], FLOOR_Z)
                translations.append(mid)
                orientations.append(q)
                scales.append(sc)
                proto_ids.append(self._IDX_CYL_G)

        # ── 5. Call visualize ──────────────────────────────────────────────
        self._markers.visualize(
            translations=np.stack(translations, axis=0),
            orientations=np.stack(orientations, axis=0),
            scales=np.stack(scales, axis=0),
            marker_indices=proto_ids,
        )

    # ------------------------------------------------------------------
    def _print_diagnostics(
        self,
        base_env,
        com: np.ndarray,
        foot_patches: list[np.ndarray],
        ankle_centers: list[np.ndarray],
        idx: int,
    ) -> None:
        """Print a concise diagnostic row including foot patches and hull area."""
        try:
            robot = base_env.scene["robot"]
        except KeyError:
            return

        root_z = float(robot.data.root_pos_w[idx, 2].item())
        lowest_z = float(robot.data.body_pos_w[idx, :, 2].min().item())
        n_feet = len(foot_patches)

        # Build hull and compute distance + area
        if n_feet > 0:
            all_corners = np.vstack(foot_patches)    # [4n, 2]
            hull_verts  = _convex_hull_2d(all_corners)
            # Shoelace formula for polygon area
            hv = hull_verts
            n = len(hv)
            area = 0.5 * abs(
                sum(hv[i, 0] * hv[(i + 1) % n, 1] - hv[(i + 1) % n, 0] * hv[i, 1]
                    for i in range(n))
            )
            # Distance from CoM projection to hull (0 if inside)
            # Simple point-to-convex-polygon test using cross products
            inside = True
            for i in range(n):
                j = (i + 1) % n
                ab = hv[j] - hv[i]
                ap = com[:2] - hv[i]
                if ab[0] * ap[1] - ab[1] * ap[0] < -1e-6:
                    inside = False
                    break
            if inside:
                sp_dist = 0.0
            else:
                sp_dist = float("inf")
                for i in range(n):
                    j = (i + 1) % n
                    ab = hv[j] - hv[i]
                    t = float(np.clip(np.dot(com[:2] - hv[i], ab) / (np.dot(ab, ab) + 1e-8), 0, 1))
                    d = float(np.linalg.norm(com[:2] - (hv[i] + t * ab)))
                    sp_dist = min(sp_dist, d)
            sigma = 0.1
            rew = float(np.exp(-(sp_dist ** 2) / (sigma ** 2)))
            polygon_desc = (
                f"hull_pts={n}  area={area:.4f}m²  "
                f"sp_dist={sp_dist:.3f}m  com_rew~{rew:.3f}"
            )
        else:
            polygon_desc = "NO_CONTACTS"

        print(
            f"[DIAG step={self._step_count:3d}]"
            f"  root_z={root_z:.3f}  lowest_z={lowest_z:.3f}"
            f"  n_feet={n_feet}"
            f"  com=({com[0]:.3f},{com[1]:.3f},{com[2]:.3f})"
            f"  {polygon_desc}"
        )
