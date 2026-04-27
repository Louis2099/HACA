# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sim-to-sim cross-val watcher for velocity tracking policies.

Polls a training run's checkpoint directory and, for each new `model_*.pt`,
runs a short headless MuJoCo sweep (a few command schedules: forward,
strafe, yaw). Metrics (survival time, velocity tracking error, fall rate)
and a rendered mp4 per schedule are logged to the training's W&B run via a
custom step axis (`sim_to_sim/iter`), so the resulting curve sits alongside
the training reward curve on the same run page and is never dropped as
"out-of-order" against the training-side global step.

One-shot and continuous watch modes are both supported; watch mode anchors
at the latest existing checkpoint so it doesn't backfill historical ones.

Examples:
    # Continuous watcher, attached to a live training run:
    python scripts/sim2mujoco_watcher.py \\
        --log-dir logs/rsl_rl/<experiment>/<run> \\
        --config agile/data/policy/<policy>.yaml \\
        --mjcf agile/rl_env/assets/robot_menagerie/<robot>/mujoco/scene.xml \\
        --wandb-entity <entity> --wandb-project <project> \\
        --wandb-run-id <run_id> \\
        --every 1000 --poll-interval 30

    # One-shot eval of a single checkpoint:
    python scripts/sim2mujoco_watcher.py \\
        --once logs/rsl_rl/<experiment>/<run>/model_10000.pt \\
        --config agile/data/policy/<policy>.yaml \\
        --mjcf agile/rl_env/assets/robot_menagerie/<robot>/mujoco/scene.xml \\
        --track-body pelvis --no-wandb
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import imageio.v2 as iio
import mujoco
import numpy as np
import torch

from agile.sim2mujoco.actions import ActionProcessor
from agile.sim2mujoco.command_provider import (
    VelocityCommandProvider,
    create_command_provider,
)
from agile.sim2mujoco.observations import ObservationProcessor
from agile.sim2mujoco.policy import PolicyWrapper
from agile.sim2mujoco.simulation import MuJocoSimulation
from agile.sim2mujoco.utils import load_config

CKPT_RE = re.compile(r"^model_(\d+)\.pt$")

# Upper-bound per-rollout duration. `_run_single_rollout` keeps stepping the
# sim with zero-actions after a fall so the rendered video shows the fall +
# settle rather than freezing — but tracking-error stats only accumulate
# while the robot is upright.
SCHEDULE_DURATION_S = 12.0

# Default base height (m) used when a schedule entry omits it. Matches the
# CommandManager fallback used elsewhere in agile.sim2mujoco. Velocity-only
# policies don't read height from obs (PR #53 truncates it), so this is a
# no-op for them; velocity+height policies will receive it.
DEFAULT_HEIGHT_M = 0.72

# Schedule entries: (start_time_s, vx, vy, wz, height, label). Three schedules
# by default — one per locomotion axis — so each checkpoint gets three videos
# and per-axis metrics. Deliberately modest magnitudes: they're chosen so a
# reasonable policy survives end-to-end, while still separating good from
# bad checkpoints by fall time.
SCHEDULES: dict[str, list[tuple[float, float, float, float, float, str]]] = {
    "fwd": [
        (0.0, 0.0, 0.0, 0.0, DEFAULT_HEIGHT_M, "stand"),
        (2.0, 0.2, 0.0, 0.0, DEFAULT_HEIGHT_M, "fwd_slow"),
        (5.0, 0.3, 0.0, 0.0, DEFAULT_HEIGHT_M, "fwd_fast"),
        (9.0, 0.0, 0.0, 0.0, DEFAULT_HEIGHT_M, "stop"),
    ],
    "strafe": [
        (0.0, 0.0, 0.0, 0.0, DEFAULT_HEIGHT_M, "stand"),
        (2.0, 0.0, 0.15, 0.0, DEFAULT_HEIGHT_M, "strafe_py"),
        (5.0, 0.0, -0.15, 0.0, DEFAULT_HEIGHT_M, "strafe_my"),
        (9.0, 0.0, 0.0, 0.0, DEFAULT_HEIGHT_M, "stop"),
    ],
    "yaw": [
        (0.0, 0.0, 0.0, 0.0, DEFAULT_HEIGHT_M, "stand"),
        (2.0, 0.0, 0.0, 0.5, DEFAULT_HEIGHT_M, "yaw_p"),
        (5.0, 0.0, 0.0, -0.5, DEFAULT_HEIGHT_M, "yaw_m"),
        (9.0, 0.0, 0.0, 0.0, DEFAULT_HEIGHT_M, "stop"),
    ],
}


@dataclass
class RolloutResult:
    """Metrics + artifact for a single command-schedule rollout."""

    name: str
    survival_time_s: float
    fell: bool
    mean_base_height: float
    tracking_err_vx: float
    tracking_err_vy: float
    tracking_err_wz: float
    num_sim_steps: int
    video_path: str


@dataclass
class EvalResult:
    step: int
    checkpoint: str
    rollouts: list[RolloutResult]

    @property
    def fall_rate(self) -> float:
        return sum(r.fell for r in self.rollouts) / len(self.rollouts)

    @property
    def mean_survival_time_s(self) -> float:
        return float(np.mean([r.survival_time_s for r in self.rollouts]))

    @property
    def mean_tracking_err_vx(self) -> float:
        return float(np.mean([r.tracking_err_vx for r in self.rollouts]))

    @property
    def mean_tracking_err_vy(self) -> float:
        return float(np.mean([r.tracking_err_vy for r in self.rollouts]))

    @property
    def mean_tracking_err_wz(self) -> float:
        return float(np.mean([r.tracking_err_wz for r in self.rollouts]))


def _parse_step(p: Path) -> int | None:
    m = CKPT_RE.match(p.name)
    return int(m.group(1)) if m else None


def _scheduled_cmd(t: float, schedule):
    cur = schedule[0]
    for entry in schedule:
        if entry[0] <= t:
            cur = entry
        else:
            break
    return cur


def _init_base_pose(
    sim: MuJocoSimulation,
    init_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
):
    """Set the free-joint base orientation to `init_quat_wxyz` and snap the
    base so the lowest geom sits at z=0 — avoids the robot spawning through
    the floor for arbitrary MJCFs."""
    sim.mj_data.qpos[3:7] = init_quat_wxyz
    mujoco.mj_forward(sim.mj_model, sim.mj_data)

    min_z = float("inf")
    for gid in range(sim.mj_model.ngeom):
        if sim.mj_model.geom_group[gid] == 2 or sim.mj_model.geom_bodyid[gid] == 0:
            continue
        gpos = sim.mj_data.geom_xpos[gid]
        gmat = sim.mj_data.geom_xmat[gid].reshape(3, 3)
        if sim.mj_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = sim.mj_model.geom_dataid[gid]
            vadr, vnum = sim.mj_model.mesh_vertadr[mid], sim.mj_model.mesh_vertnum[mid]
            verts = (gmat @ sim.mj_model.mesh_vert[vadr : vadr + vnum].T).T + gpos
            z = verts[:, 2].min()
        else:
            z = gpos[2] - float(np.abs(gmat) @ sim.mj_model.geom_size[gid])[2]
        min_z = min(min_z, z)
    base_z = float(sim.mj_data.qpos[2] - min_z)
    sim.mj_data.qpos[:3] = (0.0, 0.0, base_z)
    sim.mj_data.qvel[:6] = 0.0
    mujoco.mj_forward(sim.mj_model, sim.mj_data)
    return base_z


def build_once(config_path: Path, mjcf_path: Path, device: torch.device):
    """Build config + a MuJoCo sim + obs/act processors. MjModel is expensive
    to reload, so the watcher reuses one sim across all checkpoints."""
    config = load_config(config_path)
    config["mjcf_path"] = str(mjcf_path)
    sim = MuJocoSimulation(config, device, enable_viewer=False, mjcf_path=mjcf_path)
    obs_processor = ObservationProcessor(config, sim.joint_names, device, command_manager=None)
    command_provider = create_command_provider(config, device, motion_tracker=obs_processor.motion_tracker)
    if not isinstance(command_provider, VelocityCommandProvider):
        raise NotImplementedError(
            f"sim2mujoco_watcher only supports VelocityCommandProvider, got {type(command_provider).__name__}"
        )
    sim.command_manager = command_provider.manager
    obs_processor.command_manager = command_provider.manager
    act_processor = ActionProcessor(config, sim.joint_names, device)
    return config, sim, obs_processor, act_processor, command_provider


def _make_camera(mj_model, track_body: str | None):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(mj_model, cam)
    cam.distance = 3.0
    cam.elevation = -15.0
    cam.azimuth = 135.0
    if track_body:
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, track_body)
        if bid == -1:
            print(f"[warn] --track-body '{track_body}' not found in MJCF; using free camera")
        else:
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = bid
    return cam


def _run_single_rollout(
    *,
    sim: MuJocoSimulation,
    obs_processor: ObservationProcessor,
    act_processor: ActionProcessor,
    command_provider,
    policy,
    schedule_name: str,
    schedule: list[tuple[float, float, float, float, str]],
    video_path: Path,
    fall_height_m: float,
    track_body: str | None,
    init_fn,
    video_width: int = 640,
    video_height: int = 360,
    fps: int = 25,
) -> RolloutResult:
    """Run one schedule. After a fall the policy is replaced with zero
    actions so the sim stays stable, but the sim + renderer keep running to
    end-of-schedule so all rollouts produce equal-length videos (easy to
    compare across checkpoints). Tracking-error stats stop accumulating
    once the robot is down — only live motion contributes."""
    sim.reset()
    init_fn(sim)
    obs_processor.reset()
    if hasattr(policy, "reset"):
        policy.reset()

    control_dt = sim.dt
    num_steps = int(SCHEDULE_DURATION_S / control_dt)

    cam = _make_camera(sim.mj_model, track_body)
    sim.mj_model.vis.global_.offwidth = video_width
    sim.mj_model.vis.global_.offheight = video_height
    renderer = mujoco.Renderer(sim.mj_model, height=video_height, width=video_width)

    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(str(video_path), fps=fps, codec="libx264", quality=7)
    frame_stride = max(1, int(round(1.0 / (fps * control_dt))))

    base_heights: list[float] = []
    sq_err_vx, sq_err_vy, sq_err_wz = 0.0, 0.0, 0.0
    n_err_vx, n_err_vy, n_err_wz = 0, 0, 0
    fell = False
    survival_time = SCHEDULE_DURATION_S
    policy_device = getattr(policy, "device", "cpu")
    zero_actions = torch.zeros(act_processor.total_action_dim, device=policy_device)

    try:
        for step in range(num_steps):
            t = step * control_dt
            _, cmd_vx, cmd_vy, cmd_wz, cmd_h, _ = _scheduled_cmd(t, schedule)
            command_provider.manager.set_command(linear_x=cmd_vx, linear_y=cmd_vy, angular_z=cmd_wz, height=cmd_h)

            sim_state = sim.get_state()
            if fell:
                actions = zero_actions
            else:
                obs = obs_processor.compute(sim_state, noise_scale=0.0)
                with torch.no_grad():
                    actions = policy(obs)
                obs_processor.set_last_action(actions)
            joint_cmd = act_processor.process(actions)
            for _ in range(sim.decimation):
                sim.step(joint_cmd)

            post = sim.get_state()
            bz = float(post.root_pos[2].item())
            base_heights.append(bz)

            if not fell:
                # Accumulate tracking error on every step, including commanded-zero
                # periods (stand / stop). Under cmd=0 the squared term reduces to
                # |measured|² so the "stillness" failure mode (small stepping in
                # place that the training-side `track_lin_vel_xy_exp` reward fails
                # to suppress) is captured as drift in tracking_err_*.
                meas_vx = float(post.root_lin_vel[0].item())
                meas_vy = float(post.root_lin_vel[1].item())
                meas_wz = float(post.root_ang_vel[2].item())
                sq_err_vx += (cmd_vx - meas_vx) ** 2
                sq_err_vy += (cmd_vy - meas_vy) ** 2
                sq_err_wz += (cmd_wz - meas_wz) ** 2
                n_err_vx += 1
                n_err_vy += 1
                n_err_wz += 1

            if step % frame_stride == 0:
                renderer.update_scene(sim.mj_data, camera=cam)
                writer.append_data(renderer.render())

            if not fell and bz < fall_height_m:
                fell = True
                survival_time = t
    finally:
        writer.close()
        renderer.close()

    return RolloutResult(
        name=schedule_name,
        survival_time_s=survival_time,
        fell=fell,
        mean_base_height=float(np.mean(base_heights)) if base_heights else 0.0,
        tracking_err_vx=float(np.sqrt(sq_err_vx / n_err_vx)) if n_err_vx else 0.0,
        tracking_err_vy=float(np.sqrt(sq_err_vy / n_err_vy)) if n_err_vy else 0.0,
        tracking_err_wz=float(np.sqrt(sq_err_wz / n_err_wz)) if n_err_wz else 0.0,
        num_sim_steps=num_steps,
        video_path=str(video_path),
    )


def run_eval(
    *,
    config: dict,
    sim: MuJocoSimulation,
    obs_processor: ObservationProcessor,
    act_processor: ActionProcessor,
    command_provider,
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    init_fn,
    track_body: str | None,
    fall_height_m: float = 0.35,
    video_width: int = 640,
    video_height: int = 360,
    fps: int = 25,
) -> EvalResult:
    """Run every schedule in SCHEDULES against `checkpoint` and return
    aggregated results."""
    policy = PolicyWrapper.from_config(checkpoint, config, device)
    step = _parse_step(checkpoint) or -1
    rollouts: list[RolloutResult] = []
    for name, schedule in SCHEDULES.items():
        video_path = out_dir / f"eval_{step:08d}_{name}.mp4"
        r = _run_single_rollout(
            sim=sim,
            obs_processor=obs_processor,
            act_processor=act_processor,
            command_provider=command_provider,
            policy=policy,
            schedule_name=name,
            schedule=schedule,
            video_path=video_path,
            fall_height_m=fall_height_m,
            track_body=track_body,
            init_fn=init_fn,
            video_width=video_width,
            video_height=video_height,
            fps=fps,
        )
        rollouts.append(r)
    return EvalResult(step=step, checkpoint=checkpoint.name, rollouts=rollouts)


def _log_wandb(run, result: EvalResult, log_video: bool):
    if run is None:
        return
    import wandb

    payload: dict = {
        "sim_to_sim/iter": result.step,
        "sim_to_sim/fall_rate": result.fall_rate,
        "sim_to_sim/mean_survival_time_s": result.mean_survival_time_s,
        "sim_to_sim/mean_tracking_err_vx": result.mean_tracking_err_vx,
        "sim_to_sim/mean_tracking_err_vy": result.mean_tracking_err_vy,
        "sim_to_sim/mean_tracking_err_wz": result.mean_tracking_err_wz,
    }
    for r in result.rollouts:
        prefix = f"sim_to_sim/{r.name}"
        payload[f"{prefix}/survival_time_s"] = r.survival_time_s
        payload[f"{prefix}/fell"] = int(r.fell)
        payload[f"{prefix}/tracking_err_vx"] = r.tracking_err_vx
        payload[f"{prefix}/tracking_err_vy"] = r.tracking_err_vy
        payload[f"{prefix}/tracking_err_wz"] = r.tracking_err_wz
        if log_video and Path(r.video_path).exists():
            payload[f"{prefix}/video"] = wandb.Video(r.video_path, fps=25, format="mp4")
    # Custom step axis: the run's default `_step` is owned by the training
    # process and advances past our checkpoint's iteration, which would
    # otherwise cause W&B to drop our data as out-of-order.
    run.log(payload)


def _parse_quat(s: str) -> tuple[float, float, float, float]:
    parts = [float(x) for x in s.replace(",", " ").split()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--init-quat expects 4 floats (w x y z)")
    return tuple(parts)  # type: ignore[return-value]


def _load_schedules_yaml(path: Path) -> dict[str, list[tuple[float, float, float, float, float, str]]]:
    """Load command schedules from YAML. Top-level mapping is name -> list of
    [t, vx, vy, wz, height, label] entries. Used to override the built-in
    SCHEDULES so users can add a stand-still schedule, sweep height, etc.
    without editing the script."""
    import yaml  # local — avoid the dependency on the import path when unused

    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"--schedules: expected top-level mapping in {path}")
    out: dict[str, list[tuple[float, float, float, float, float, str]]] = {}
    for name, entries in raw.items():
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"--schedules: '{name}' must be a non-empty list of entries")
        parsed: list[tuple[float, float, float, float, float, str]] = []
        for e in entries:
            if len(e) != 6:
                raise ValueError(f"--schedules: '{name}' entry {e!r} must be [t, vx, vy, wz, height, label]")
            parsed.append((float(e[0]), float(e[1]), float(e[2]), float(e[3]), float(e[4]), str(e[5])))
        out[str(name)] = parsed
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--log-dir", type=Path, help="Directory to watch for model_*.pt")
    mode.add_argument("--once", type=Path, help="Run once against a single checkpoint and exit")

    p.add_argument("--config", type=Path, required=True, help="Policy YAML config (IO descriptor)")
    p.add_argument("--mjcf", type=Path, required=True, help="MJCF scene file")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write mp4s; default: <checkpoint_dir>/sim_to_sim/",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device for the policy (default: cpu — does not compete with training for GPU)",
    )

    p.add_argument("--every", type=int, default=1000, help="Only eval checkpoints whose iter is a multiple of this")
    p.add_argument("--poll-interval", type=float, default=30.0, help="Seconds between checkpoint-dir polls")
    p.add_argument(
        "--fall-height",
        type=float,
        default=0.35,
        help="Base z (m) below which the robot is considered fallen; depends on the robot",
    )
    p.add_argument(
        "--init-quat",
        type=_parse_quat,
        default=(1.0, 0.0, 0.0, 0.0),
        help='Base spawn orientation (w x y z). Default: identity. Example: "0.9946 0.0523 0.0872 0.0"',
    )
    p.add_argument(
        "--track-body",
        type=str,
        default=None,
        help="Body name the camera should track (default: free camera). Ignored if not present in MJCF.",
    )
    p.add_argument(
        "--schedules",
        type=Path,
        default=None,
        help="YAML file overriding the built-in command schedules. Top-level mapping "
        "is name -> [[t, vx, vy, wz, height, label], ...]. Use this to add e.g. a "
        "'stand' schedule (all-zero velocity) or to sweep height.",
    )

    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="Attach to an existing W&B run via resume=allow (recommended: the training run's id)",
    )
    p.add_argument("--no-wandb", action="store_true", help="Don't log to W&B even if a run-id is given")
    p.add_argument("--no-video-upload", action="store_true", help="Save mp4s locally but don't push to W&B")

    args = p.parse_args()

    if args.schedules is not None:
        global SCHEDULES
        SCHEDULES = _load_schedules_yaml(args.schedules)
        print(f"[schedules] loaded {len(SCHEDULES)} from {args.schedules}: {list(SCHEDULES)}")

    device = torch.device(args.device)
    config, sim, obs_proc, act_proc, cmd_prov = build_once(args.config, args.mjcf, device)

    init_fn = partial(_init_base_pose, init_quat_wxyz=args.init_quat)

    wb_run = None
    if not args.no_wandb and args.wandb_run_id:
        import wandb

        # Shared mode lets this watcher log into the training run concurrently
        # with the training process. Without it, W&B silently treats the
        # training as the primary writer and the watcher's sim_to_sim/*
        # history ends up at low `_step` values that the default UI panels
        # don't surface — the data lands on the server but is invisible.
        wb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            id=args.wandb_run_id,
            resume="allow",
            job_type="sim_to_sim_eval",
            settings=wandb.Settings(
                mode="shared",
                x_label="sim2sim_watcher",
                x_primary=False,
            ),
        )
        wb_run.define_metric("sim_to_sim/iter")
        wb_run.define_metric("sim_to_sim/*", step_metric="sim_to_sim/iter")
        print(f"[wandb] resumed run {args.wandb_entity}/{args.wandb_project}/{args.wandb_run_id}")

    def _eval_one(ckpt: Path):
        step = _parse_step(ckpt)
        if step is None:
            return
        out_dir = args.output_dir or (ckpt.parent / "sim_to_sim")
        print(f"[eval] {ckpt.name} …")
        t0 = time.time()
        result = run_eval(
            config=config,
            sim=sim,
            obs_processor=obs_proc,
            act_processor=act_proc,
            command_provider=cmd_prov,
            checkpoint=ckpt,
            out_dir=out_dir,
            device=device,
            init_fn=init_fn,
            track_body=args.track_body,
            fall_height_m=args.fall_height,
        )
        dt = time.time() - t0
        per_rollout = "  ".join(
            f"{r.name}(survived={r.survival_time_s:.1f}s{' FELL' if r.fell else ''})" for r in result.rollouts
        )
        print(
            f"[eval] step={result.step}  fall_rate={result.fall_rate:.2f}  "
            f"mean_survival={result.mean_survival_time_s:.1f}s  {per_rollout}  ({dt:.1f}s wall)"
        )
        _log_wandb(wb_run, result, log_video=not args.no_video_upload)
        return result

    try:
        if args.once is not None:
            _eval_one(args.once)
            return

        watch_dir = args.log_dir.resolve()
        print(f"[watch] {watch_dir}  every {args.every} iters  poll {args.poll_interval:.0f}s")
        processed: set[int] = set()
        existing = sorted(
            (p for p in watch_dir.glob("model_*.pt") if _parse_step(p) is not None),
            key=lambda p: _parse_step(p) or 0,
        )
        if existing:
            anchor = _parse_step(existing[-1]) or 0
            print(f"[watch] anchor: only evaluating checkpoints with step > {anchor}")
            processed = {s for s in (_parse_step(p) for p in existing) if s is not None}

        while True:
            candidates = []
            for ckpt in watch_dir.glob("model_*.pt"):
                step = _parse_step(ckpt)
                if step is None or step in processed:
                    continue
                if step % args.every != 0:
                    processed.add(step)  # mark seen so we don't re-check
                    continue
                # Stability check: the file size must hold across a short wait
                # so we don't race the training process mid-write.
                s1 = ckpt.stat().st_size
                time.sleep(0.5)
                if not ckpt.exists() or ckpt.stat().st_size != s1 or s1 == 0:
                    continue
                candidates.append((step, ckpt))

            for step, ckpt in sorted(candidates):
                _eval_one(ckpt)
                processed.add(step)

            time.sleep(args.poll_interval)
    finally:
        if wb_run is not None:
            wb_run.finish()
        sim.close()


if __name__ == "__main__":
    main()
