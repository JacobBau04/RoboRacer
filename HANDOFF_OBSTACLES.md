# Obstacle avoidance on race1: handoff (2026-11-03)

For a new Claude Code session continuing this work on the car's Jetson (`/home/nvidia`).
Earlier context: `OBSTACLE_AVOIDANCE_HANDOFF.md` (the laptop session's analysis and plan; it
assumed the Levine loop and a Hokuyo, both wrong, see section 2).

## 1. Task, rules, user preferences

- **Race:** obstacle race on the **race1** track (not Levine), judged **on the real car**.
  Score = **fastest average of 2 laps**. Touching an obstacle only costs time; getting stuck
  costs the race.
- **Obstacles:** boxes 5–30 % of the local track width, tall enough for the LiDAR, placement
  unknown ("probably nothing extremely unfair"), count per lap unknown. The car drives
  continuously; obstacles are moved while the car is far away and cannot see them. race1 is
  walled all round; walls are above the LiDAR, so people outside never show up.
- **Goal:** fastest lap time with a low chance of hitting obstacles.
- **Code location:** `~/f1tenth_ws/src/jacob_opt_obstacles/` = copy of `jacob_opt_car`
  (commit 788a111) **without git**. Do **not** commit to git. The teammates' avoidance code is
  **not** used here (it stays untouched in `ese-6150-lab-7-.../` and `obstacle_jacob_integration/`).
- **Build switch (done, approved):** `COLCON_IGNORE` added to
  `src/ese-6150-lab-7-model-predictive-control-group-12/`, removed from `jacob_opt_obstacles/`,
  so package `mpc` is built from the new folder. The user may undo this later.
  Build: `cd ~/f1tenth_ws && colcon build --symlink-install --packages-select mpc`.
- **MPC corridor limits:** planned for later, after this is tested on the real car. Not now.
- **User preferences:**
  - For "how do I" questions, give commands and don't run them (memory note).
  - Ask before state-changing actions on the car.
  - The user asked to be told about anything that conflicts with "fastest lap + low hit chance".
  - Write short, plain summaries.
- **Car ROS domain 12, sim domain 42.** Never run the sim on 12: the sim's MPC publishes
  `/drive`, which the car's mux would forward.

## 2. Facts measured on this car (they changed the plan)

- **Jetson:** Orin Nano Super, 6 cores, 7.4 GB RAM, ROS 2 Jazzy.
- **LiDAR:** **SICK TiM571**, not a Hokuyo. Bringup is `ros2 launch f1tenth_stack sick_bringup_launch.py`.
  - 811 beams, 270°, 0.333° steps, **15 Hz**, returns seen up to 14 m.
  - `time_increment` 61.7 µs, so a sweep takes 50 ms. The header stamp ≈ arrival time, i.e. the
    **end** of the sweep: the first beam is ~50 ms older than the stamp.
  - About 58 of 811 beams return **0 (no echo)** and must be dropped.
- **TF on the car:** two separate trees.
  - The particle filter publishes `map -> laser`, stamped with the wheel-odometry time.
  - The VESC publishes `odom -> base_link`.
  - Bringup's static `base_link -> laser` gets overridden, so there is **no `map -> base_link`**
    (this is why the teammates' code fails).
  - The particle filter pose = the LiDAR position, 0.27 m ahead of the rear axle. `map -> laser`
    arrives on average 22 ms late (87 ms max). The `/pf/pose/odom` stamp is publish time.
- **OpenBLAS:** pip numpy's OpenBLAS spun 5 idle threads at 100 % in the particle filter
  (566 % CPU).
  - With `OPENBLAS_NUM_THREADS=1`: filter at 100 % CPU, Jetson 24 % busy, filter pose rate
    16.7 → 27.7 Hz (jitter 17 → 0.5 ms).
  - Added to `particle_filter/launch/localize_launch.py` (rebuilt) and the MPC launch files.
    `mpc_node.py` already set it internally.
  - MPC dry run (stationary, `/drive` remapped) with the fix: solve 12.3 ms (was 22.8 on Nov 1),
    callback 16.3 ms (was 31.6), about 38 commands/s (was ~20).
  - **Re-check `delay_comp` (0.26 s) on a clean lap:** latency changed.
- **No joystick connected:** `/odom` stays silent until a steering command is sent. To start it
  without moving anything, publish one message:
  `ros2 topic pub --once /sensors/servo_position_command std_msgs/msg/Float64 "{data: 0.475}"`.
- **race1:** 25.1 m lap; width 1.16–4.19 m (median 2.24); the raceline is ≥ 0.40 m from the walls.
  - Raceline `waypoints/race1.csv`: v_max 3.0, predicted 8.5 s, sim clean lap 8.72 s.
  - Tight sections: s 8.9–12 (hairpin entry, right wall only 0.4–0.5 m from the raceline at
    s 9–11) and s 20–23.
  - **Note:** `race1_launch.py` (what the user last ran on race1) sets no `delay_comp`, so it is 0.

## 3. What is built (all in `jacob_opt_obstacles/mpc/`)

| File | Role |
|---|---|
| `scripts/track_model.py` | Raceline as a closed periodic cubic spline. Track coordinates (s, d), d + = left. Wall distance field (EDT of `maps/race1_cleaned_blocked`; unknown cells count as wall). Wall offsets along normals. Windowed projection (no matches across the inner wall). |
| `scripts/obstacle_tracker.py` | Detection: valid returns → map points with a **per-beam pose**. Keep points inside the corridor and ≥ `wall_margin` (0.15 + 0.01/m of range) from map walls. Cluster neighbouring beams (≥ 3 points). Box in (s, d), back side padded to 0.3 m only when seen from ≥ 1 m ahead. Confirmed after 2 scans; kept until passed; deleted after 4 scans of "beams pass through it"; **hidden-edge flags**: the planning box grows 0.3 m on an edge hidden behind something closer. |
| `scripts/detour_planner.py` | Candidates per side combination: quintic swerves (3 margins × 3 lengths) plus a **corridor QP (OSQP)** when no quintic fits. Exact checks: obstacle margin (heading-aware), steering limit, car-body wall clearance. Speed profile (grip, accel/brake 3 m/s², speed caps by margin). Picks the fastest; keeps the active plan (5 cm margin slack, 1 s side-flip penalty, 0.1 s replan gain, cached while the boxes don't change). Levels: 0 normal (margin 0.12–0.30, walls 0.08, κ ≤ 1.15), 1 tight (0.05/0.03/1.28, 1.2 m/s), 2 crawl (0.02/0.01/1.33, 0.6 m/s). If nothing fits, keep the current plan at crawl speed while it still clears (> 0); else stop 1.5 m before the box. |
| `scripts/avoidance_msgs.py` | Float64MultiArray formats for the plan and obstacles; resample the detour to 0.1 m spacing; **splice** it into the raceline (the MPC assumes uniform spacing from the first segment). |
| `scripts/obstacle_avoidance_node.py` | ROS node. `pose_source` `tf` (car) or `odom` (sim; Python TF cannot keep up with the sim's 959 TF msgs/s). Pose-jump guard. Publishes `/obstacle_avoidance/{plan,obstacles,stats,markers}`. Logs one line per plan change. Parameters `detection.*` and `planner.*` = dataclass fields. |
| `scripts/mpc_node.py` | Changes, all behind `avoidance:=true` (off by default): `pose_topic` parameter; follows the spliced detour; if plans stop, keeps the last detour at ≤ 1.5 m/s; **safety check** of the predicted path vs the seen boxes only (never walls), lowering speed only (steering unchanged), ignoring box points the car already overlaps (otherwise it froze beside boxes); `/mpc/safety`. |
| `launch/race1_obstacles_launch.py` | Car: MPC + node + recorder (bag kept, `/scan` + `/tf` recorded); args `avoidance`, `record`, `delay_comp` (default 0.26), `note`. |
| `launch/race1_sim_launch.py` | Sim: pose `/ego_racecar/odom`, body front 0.46 / rear 0.12 around the rear axle, `delay_comp` 0, scans thinned to 15 Hz, `pose_source odom`, `laser_offset` 0.275. |
| `tools/run_recorder.py` | + avoidance topics, `/scan`, `/tf_static` (car); skips deserializing unused topics; `raw.npz` gets `avoid` and `safety`. |
| `tools/scan_sim.py`, `tools/offline_driveby.py`, `tools/planner_stats.py` | Offline tests without ROS: synthetic SICK scans; drive-by with pose drift (`--case s,w,dc` for verbose); planner-only statistics. |
| `tools/make_obstacle_layouts.py`, `tools/sim_batch.py` | Random race1 layouts for the sim (`~/sim_ws/race1_layouts`, 20 layouts + clean); batch runner (refuses to start if another sim is on domain 42). |
| `tools/replay_detection.py` | Replays a car bag (`/scan` + `map -> laser` TF) through detection; map-fit statistics, beam-offset sweep, plot. |
| `test/test_obstacle_avoidance.py` | 16 unit tests, all passing (`cd mpc && python3 -m pytest test -q -p no:cacheprovider`). |
| `maps/race1_cleaned_blocked.*`, `maps/race1_cleaned_centerline.csv` | Walls and lap-counter centerline (installed to share/mpc/maps). |

**Simulator:** `~/sim_ws` with `f1tenth_gym_ros` (`dev-jazzy`) + `f1tenth_gym` (`dev-jax`) in a venv
`~/sim_ws/.venv`; built with `python -m colcon build --symlink-install` (plain colcon gives the bridge
the system Python and it fails). Its LiDAR already matches the SICK (819 beams, 270°, 25 m),
but scans arrive at 250 Hz and with `time_increment` 0. It runs at ~0.92× real time on the Jetson.
Run our nodes in a shell **without** the venv (the venv has numpy 2; cvxpy in `~/.local` needs 1.26).

## 4. Results so far

- **Offline drive-by** (earlier planner version): 41 relevant random boxes, 0 hits, no false
  obstacles on clean laps. Confirmed a median 4.6 m ahead (5th percentile 2.3 m). Mean time
  lost 0.45 s per box (95th percentile 2.0 s). Min true clearance 0.008 m (5th percentile 0.075).
- **planner_stats.py** (221 random single boxes, car 5 m before): level 0 = 162, level 1 = 24,
  level 2 = 8, raceline = 20, stop = 7 (mostly physically impossible spots at s 9–12 and 21–22).
  Planning 26 ms mean, 71 ms p95, 132 ms max.
- **Sim batch 1** (`~/sim_ws/race1_layouts/batch1/results.csv`, 2 laps each, latest code):
  - clean: 8.72 s, no false detours.
  - **9 of 21 finished**: 004, 006, 007, 009, 011, 013, 015, 016 + clean. Lap times 8.6–9.4 s,
    and 12.0–12.2 s on two hard layouts.
  - **5 hits:** 000, 001, 002, 018 at s 8.9–9.9 on the **right wall** (where it closes in; the
    car's centre was 0.25–0.38 m from the wall, box clearance > 0), and 014 at s 15.7. So these
    are wall contacts while returning from a detour, not box hits.
  - **7 stuck:** 003, 005, 008, 010, 012, 017, 019. Stop plans or a standstill; 012 and 017 were
    never near a box (min clearance 1.19 / 0.61), so these are likely false or oversized boxes or
    safety-check stops.
  - Node logs per layout: `batch1/<layout>_nodes.log`. Grep `obstacle_avoidance\]: s` for the
    plan changes and `predicted path` for safety-check events.
- **Replay tool** tested on a synthetic bag: the obstacle was detected and confirmed at 4.2 m.
  The map-fit / beam-offset statistic is **not validated**: it counts obstacle points too; use a
  clean lap, or exclude clusters.

## 5. What was being done when this stopped

Debugging the sim batch: why the car hits the right wall at s 9–10 after detours, and why it
gets stuck on 7 layouts.

## 6. To do, in order

1. **Wall hits at s 9–10:** rerun one layout with recording
   (`python3 tools/sim_batch.py --layouts ~/sim_ws/race1_layouts --only layout_002 --laps 2 --record true`)
   and compare the planned detour (`/obstacle_avoidance/plan`) with the driven path. Suspects:
   - MPC tracking error on detours;
   - the wall check using 3 centreline points + half width, not the body corners (front 0.46 m in the sim);
   - the return transition too short or too steep where the wall closes in.
   Possible fixes:
   - check the 4 body corners;
   - add a margin to the wall check that grows with curvature or speed;
   - slow down where the wall clearance is small.
2. **Stuck layouts:** for each, find the cause (stop plan, MPC safety brake, false or oversized
   box from union growth, see-through deletion not happening). Consider: a box that never got
   confirmed shouldn't block; unblocking after a stop (re-plan from standstill; crawl); time-out
   of a stop.
3. **Layout fairness:** `make_obstacle_layouts.py` checks the gap only at each box's centre s.
   Also check that the car body fits along the box's length and where the walls close in (as
   001 shows). Otherwise many "failures" are unfair layouts.
4. Run the full batch again plus `--avoidance false` as a baseline. Target: no stuck runs, few
   hits, small time loss.
5. Validate `replay_detection.py` map-fit on a synthetic **clean** bag (and with the beam-time
   effect), so the car data can be tuned.
6. **On the car (user runs, or with permission):**
   - bringup + particle filter (rebuilt, single OpenBLAS thread);
   - `ros2 launch mpc race1_obstacles_launch.py avoidance:=false` for clean laps (re-check `delay_comp`);
   - then `avoidance:=true` clean laps (expect no detours);
   - then boxes of several widths and positions;
   - after each run: `tools/replay_detection.py mpc/data/<run> --sweep-offset` to tune `wall_margin`
     and `beam_time_offset`.
   Real car body: front 0.19 m / rear 0.39 m around the LiDAR pose (assumed from the f1tenth
   dimensions; measure).
7. Tune margins and speeds for the best 2-lap average. Then MPC corridor limits (later, per user).

## 7. Things to tell the user

- The particle filter's TF tree and launch were changed only by adding `OPENBLAS_NUM_THREADS=1`.
- `delay_comp` needs re-measuring; `race1_launch.py` had none.
- Crawl and stop behaviour is a design choice: stop only if no path clears at all.

## 8. Update: race2 (2026-11-03 evening)

New race course **race2** (similar to race1, different frame). Rules now known: obstacles
**only on straights**, **10-20 %** of the track width.

Set up (all done, built):
- **Particle filter:** `particle_filter/maps/race2_remapped_edited.{pgm,yaml}`;
  `config/localize.yaml` map switched to `race2_remapped_edited` (race1 line kept as a
  comment); rebuilt.
- **Openings:** none. The free space is one closed region; no unknown cells touch it.
- **Centerline:** a `race2` entry in `make_centerline.py` `TRACKS`. Seed `race2_seed.csv` =
  race1's centerline moved onto race2 by wall matching (rotation 13.6°). Output
  `race2_centerline.csv` (28.0 m, CCW, width 1.31-2.41 m) and `race2_blocked.{pgm,yaml}`.
- **Raceline:** `min_curv_raceline/config/params_real_race2.yaml`, writing
  `jacob_opt_obstacles/mpc/waypoints/race2.csv`. Result: 26.1 m, predicted 8.97 s, max
  |kappa| 1.06 (race1: 0.90), at least 0.40 m from the walls. Run it with `MPLBACKEND=Agg` on
  the Jetson; otherwise the plot crashes after the waypoints are written.
- **mpc package:**
  - `maps/race2_blocked.*` and `race2_centerline.csv`;
  - `launch/race2_launch.py` (plain MPC), `race2_obstacles_launch.py` (car) and
    `race2_sim_launch.py` (sim) wrap the race1 files with `track:=race2 map_yaml:=race2_blocked.yaml`;
  - the race1 obstacle/sim launches got a `map_yaml` argument;
  - `make_obstacle_layouts.py` has a `--straights` option and names the centerline copy after
    its source; `planner_stats.py` has `--waypoints` and `--map`; `sim_batch.py` has `--launch`.
- **Sim layouts:** `~/sim_ws/race2_straights` (20 + clean, straights only, 10-20 %). race2's
  straights are the top (4.3 m) and bottom (5.7 m).
- **First race2 sim results** (`race2_straights/results_2026-11-03_19-40-20`):
  - clean 9.23 s;
  - layout_000 (3 boxes) 9.73 s, min clearance 0.12;
  - layout_002 9.32 s;
  - **layout_001 stuck:** two boxes 1.5 m apart on the top straight, at
    (s 19.6, w 0.27, dc 0.06) and (s 21.1, w 0.17, dc 0.27). The planner found no feasible
    joint plan, so STOP, and it stays stopped. **Top priority to fix** (realistic case).

`race1_sim_launch.py` refuses ROS_DOMAIN_ID 12. The user's `.bashrc` exports 12, so sim
terminals need `export ROS_DOMAIN_ID=42`.
