# CBF-RL Implementation

This page documents the current Control Barrier Function Reinforcement Learning
(CBF-RL) implementation used by the dodgeball task. The implementation is a
lightweight, observation-driven safety filter integrated into the vendored
RSL-RL PPO rollout path.

The main source files are:

- `agile/rl_env/rsl_rl/cbf/barrier_terms.py`
- `agile/rl_env/rsl_rl/cbf/cbf_filter.py`
- `agile/algorithms/rsl_rl/rsl_rl/runners/on_policy_runner.py`
- `agile/algorithms/rsl_rl/rsl_rl/algorithms/ppo.py`
- `agile/rl_env/tasks/dodgeball/g1/agents/rsl_rl_ppo_cfg.py`

## High-Level Flow

During rollout collection, PPO first samples a nominal action from the policy.
If CBF filtering is enabled, the runner projects that action into a safer action
using relative ball position and velocity observations. The environment is
stepped with the filtered action, and PPO stores/logs the filtered action so the
training data matches what was executed in simulation.

```text
policy(obs) -> nominal_action
nominal_action + ball_relative_state -> CBFActionFilter -> safe_action
env.step(safe_action)
PPO storage/reward/loss use safe_action and CBF stats
```

The current filter is CBF-style rather than a full model-based CBF-QP. It does
not compute robot dynamics, Lie derivatives, or solve a constrained quadratic
program. Instead, it computes a barrier violation from observed ball-relative
kinematics and applies a bounded additive correction to the first action
dimensions.

## 1. CBF Term

The CBF term is implemented in `distance_velocity_barrier()` in
`agile/rl_env/rsl_rl/cbf/barrier_terms.py`.

Inputs:

- `relative_position`: obstacle position relative to the robot root frame.
- `relative_velocity`: obstacle linear velocity relative to the robot root frame.
- `safe_distance`: minimum desired distance from the robot root.
- `reaction_time`: time horizon used to tighten the barrier when the obstacle is
  approaching.

The filter first computes distance:

```text
distance = ||relative_position||
```

Then it computes incoming radial speed:

```text
radial_speed = -(relative_position dot relative_velocity) / distance
approaching_speed = max(radial_speed, 0)
```

Positive `approaching_speed` means the ball is moving toward the robot center.
Receding or tangential motion contributes zero incoming speed.

The barrier value is:

```text
barrier = distance - safe_distance - reaction_time * approaching_speed
```

Interpretation:

- `barrier >= 0`: the current state is considered safe by this barrier.
- `barrier < 0`: the state violates the safety margin.
- `reaction_time * approaching_speed` tightens the distance threshold for fast
  incoming balls, so a ball moving quickly toward the robot triggers correction
  earlier than a slow or stationary ball at the same distance.

The violation scalar used by the action filter is:

```text
violation = max(-barrier, 0)
```

For the dodgeball task, the position and velocity observations come from
`ball_pos_rel_root()` and `ball_vel_rel_root()` in
`agile/rl_env/mdp/observations/dodgeball_observations.py`. These functions
compute the ball state relative to the robot root in world coordinates and then
rotate it into the robot root frame with `quat_apply_inverse()`.

## 2. CBF Reward Implementation

The CBF reward contribution is implemented inside PPO, not as an Isaac Lab
`RewTerm` in the dodgeball environment config.

### Projection-Norm Reward Penalty

In `PPO.process_env_step()`, after the environment reward and termination
adjustments are applied, PPO checks for `cbf_projection_norm` in `infos`:

```text
transition.rewards -= cbf_reward_penalty_coef * cbf_projection_norm
```

This means the reward is reduced when the CBF filter had to move the action.
The penalty is proportional to the L2 norm of:

```text
safe_action - nominal_action
```

The intent is to encourage the learned policy to choose actions that already
satisfy the filter, so the safety layer has less work to do.

In the dodgeball PPO config, the coefficient is:

```python
cbf_reward_penalty_coef = 0.1
```

### Safe-Action Auxiliary Loss

The PPO update also includes an auxiliary supervised term when
`cbf_safe_action_loss_coef > 0`:

```text
cbf_safe_action_loss = mean((policy_mean - stored_action)^2)
loss += cbf_safe_action_loss_coef * cbf_safe_action_loss
```

Because `apply_action_filter()` replaces the transition action with the filtered
action before the environment step is processed, `stored_action` is the CBF
filtered action during CBF-enabled rollouts. This trains the policy mean toward
the action that the safety filter actually executed.

In the dodgeball PPO config, the coefficient is:

```python
cbf_safe_action_loss_coef = 0.05
```

### Task Rewards Are Separate

The dodgeball task also has ordinary shaping rewards such as
`ball_clearance_reward()` and `ball_closing_speed_penalty()` in
`agile/rl_env/mdp/rewards/dodgeball_rewards.py`. These use related ball distance
and velocity signals, but they are not the CBF reward hook described above.

The CBF-specific reward signal is the projection-norm penalty injected by PPO
from `infos["cbf_projection_norm"]`.

## 3. CBF Filtering Implementation

The runtime action filter is `CBFActionFilter` in
`agile/rl_env/rsl_rl/cbf/cbf_filter.py`.

### Configuration

The reusable config class is `RslRlCbfCfg` in `agile/rl_env/rsl_rl/rl_cfg.py`.
The current G1 dodgeball defaults are:

```python
cbf_cfg = RslRlCbfCfg(
    enabled=True,
    observation_pos_key="ball_pos_rel_root",
    observation_vel_key="ball_vel_rel_root",
    safe_distance=0.6,
    reaction_time=0.25,
    projection_gain=1.5,
    max_projection_norm=0.5,
    action_clip=6.0,
)
```

The runner constructs the filter only when `cbf_cfg` exists:

```text
self.cbf_filter = CBFActionFilter(cbf_cfg) if cbf_cfg is not None else None
```

### Observation Extraction

`CBFActionFilter` expects observations to be a `TensorDict` or `dict`. It reads:

- `observation_pos_key`, default `ball_pos_rel_root`
- `observation_vel_key`, default `ball_vel_rel_root`

Only the first three components are used:

```text
rel_pos = rel_pos[..., :3]
rel_vel = rel_vel[..., :3]
```

If the filter is disabled or either observation key is missing, actions pass
through unchanged and all CBF stats are returned as zero tensors.

### Action Projection

After computing `violation`, the filter builds a correction direction from the
relative position vector:

```text
direction = relative_position / ||relative_position||
correction = projection_gain * violation * direction
```

The correction is L2-clamped:

```text
correction = correction * min(max_projection_norm / ||correction||, 1)
```

Then it is added to the leading action dimensions:

```text
safe_action[..., :correction_dims] += correction[..., :correction_dims]
```

where:

```text
correction_dims = min(action_dim, 3)
```

If `action_clip` is configured, the final action is clamped elementwise to:

```text
[-action_clip, action_clip]
```

For the current dodgeball config, this is `[-6.0, 6.0]`.

### Stats Returned by the Filter

The filter returns the safe action and a stats dictionary:

- `cbf_barrier`: raw barrier value.
- `cbf_violation`: `max(-barrier, 0)`.
- `cbf_projection_norm`: L2 norm of `safe_action - nominal_action`.
- `cbf_safe_action_ratio`: per-environment indicator that the action was
  modified. It is averaged by the runner for logging.

### Runner Integration

In `OnPolicyRunner.learn()`, the filter is applied between policy action
sampling and `env.step()`:

```text
actions = alg.act(obs, privileged_obs)
actions, cbf_stats = cbf_filter.filter_actions(obs, actions)
alg.apply_action_filter(actions, cbf_stats)
obs, rewards, dones, infos = env.step(actions)
infos.update(cbf_stats)
alg.process_env_step(rewards, dones, infos)
```

`apply_action_filter()` updates PPO's transition bookkeeping:

- stores the filtered action,
- recomputes the action log probability for that filtered action,
- records the current policy mean and standard deviation.

This is important because the action in rollout storage should match the action
that was actually executed by the simulator.

### Logging

The runner accumulates CBF stats during rollout and writes them to TensorBoard:

- `Train/cbf_violation_rate`
- `Train/cbf_avg_projection_norm`
- `Train/cbf_safe_action_ratio`

The same values are also printed in the runner log output when CBF stats are
available.

## Current Behavior and Limitations

- The barrier is based on the ball relative to the robot root, not per-link
  geometry.
- The filter modifies only the first up to three action dimensions.
- The correction direction is the relative ball direction in root frame; there
  is no learned or dynamics-aware mapping from Cartesian safety correction to
  joint-space control.
- There is no QP solve, no explicit CBF constraint of the form
  `h_dot + alpha(h) >= 0`, and no robot dynamics model in the filter.
- The CBF reward penalty is applied only when the runner has inserted
  `cbf_projection_norm` into `infos`.
- Missing CBF observation keys silently produce pass-through behavior with zero
  stats.

## Test Coverage

Focused tests live in `agile/algorithms/rsl_rl/rsl_rl/tests/test_cbf_filter.py`.
They verify that:

- projection activates for a near, approaching ball;
- projection remains zero for a safe ball;
- disabled filtering passes actions through unchanged;
- filtered actions remain finite.
