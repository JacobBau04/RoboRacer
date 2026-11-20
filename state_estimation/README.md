# state_estimation: switchable state estimation for the MPC

One setting picks what the MPC drives on: **`ESTIMATOR` at the top of `launch/estimation_launch.py`**
(or `estimator:=...` on the command line for one run, given to both launch files).

| estimator | what runs | MPC pose_topic / pose_offset / delay_comp |
|---|---|---|
| `race` | particle filter on raw `/odom` (wheel speed + steering command) + LiDAR, its own map -> laser TF: the race-2 setup | `pf/pose/odom` / 0.27 / 0.20 |
| `1A` | `sensor_prep` -> robot_localization EKF `ekf1` (wheel speed + gyro) -> particle filter in scan mode -> `state_assembler` (rear axle, predicted to now) -> `/state_estimate`; 1B shadow runs alongside (logged only); `viz_tf` publishes map -> odom | `/state_estimate` / 0 / 0.14 |
| `1B` | as 1A, the assembler predicts with the shadow EKF (bicycle yaw model with the steering command) | `/state_estimate` / 0 / 0.14 |
| `1C`, `2A` | not on the car yet (launch stops with a message) | |

Per-estimator settings: `config/estimators/<name>.yaml`. Adding an estimator = one YAML + its nodes in
`estimation_launch.py`.

## Running (all terminals: `export ROS_DOMAIN_ID=12`, source `/opt/ros/jazzy` and `~/f1tenth_ws/install`)

1. Bringup: `ros2 launch f1tenth_stack sick_bringup_launch.py`, then hold LB ~1 s (`/odom` starts after the first
   steering command).
2. State estimation (replaces `particle_filter localize_launch.py`):
   `ros2 launch state_estimation estimation_launch.py` (keep the car still ~2 s: 1A/1B calibrate the gyro).
   Set the pose in RViz/Foxglove (`/initialpose`) or pass `initial_pose:='[x, y, theta]'` (laser pose in map).
3. MPC: `ros2 launch mpc race1_obstacles_launch.py track:=levine_race avoidance:=false note:="..."`
   (race2: `race2_obstacles_launch.py`). It prints the estimator and pose settings it uses.

The recorder stores the estimator in `meta.json`, always records the raw gyro and commands (for offline replays), and
in 1A/1B also `/state_estimate`, `/ekf1/odometry`, `/prep/*`, `/shadow_ekf1/*`.

## Notes
- Map: `particle_filter/config/localize.yaml` `map` (one setting for every estimator).
- Foxglove draws the `/map` grid in the wrong place (TF, scan and poses are right). In 1A/1B, `viz_tf` gives a
  clean map -> odom -> base_link -> laser tree.
- Obstacle avoidance takes its pose from TF map -> laser: the race PF's TF in `race`, `viz_tf` in 1A/1B.
- Car values and their evidence: `~/stage1_data/2026-11-18/session_log.md`, `~/stage1_data/2026-11-19/session_log.md`
  (gyro deg/s and offset, sigma_w 0.03 from driving vibration, cddt, 25 Hz `/state_estimate`).
- Tests (no ROS needed): `python3 -m pytest test -q`.
- History: the stage 1 development workspace `~/stage1_ws` (sim, emulator, VM tools) is kept but not sourced.
