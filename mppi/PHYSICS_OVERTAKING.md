# Physics-based LEFT / RIGHT / FOLLOW

This implementation is based on `real_MPPI` at `6fb7fa6`. It uses the existing
kinematic bicycle model and JAX MPPI cost; it does not train or load an RL policy.
The original controller remains available with `physics_overtaking:=false`.

## Read the code in this order

1. `scripts/physics_overtaking.py`: `reference()` builds three track-relative
   references. FOLLOW uses a speed-dependent following gap; LEFT and RIGHT make
   smooth lateral transitions around a slower/nearby opponent.
2. `tracking_seed()` supplies physically motivated starting controls. Random
   steering noise alone often never discovers a pass. Each mode retains its own
   previous plan.
3. `mode_update()` samples 256 plans per mode (768 total with an opponent).
   Invalid trajectories receive exactly zero weight for the entire horizon;
   normalization excludes them and handles zero valid samples without NaNs.
4. `trajectory_valid()` checks conservative circular footprints covering the
   whole vehicle, swept opponent separation between prediction steps, wall
   clearance, acceleration/steering/rate limits, and lateral acceleration.
   Each check includes every scenario of every active opponent, not a mean
   prediction. `side_valid()` keeps LEFT and RIGHT on their respective passing sides.
5. The weighted control sequence is propagated and checked again. If averaging
   invalidates it, the best validated sample is used instead. The behavior
   selector compares progress, charges a switching penalty, and prevents direct
   LEFT-to-RIGHT reversals while passing the same target.
6. `mppi_node.py`: `_physics_control()` aligns incoming predictions, calls the
   planner, publishes `/mppi/behavior` and `/mppi/selected_path`, and sends only
   the first speed/steering command. An invalid plan commands zero speed. A
   checked braking plan is returned when the planner cannot find a candidate.

`mppi_core.py` contains the extracted original bicycle model, RK4 integrator,
cost function and legacy MPPI update. Offline tests use this actual code, not a
separate approximation of the controller. Speed limits now apply inside the
RK4 intermediate evaluations as well: braking at rest no longer predicts a
small backwards movement despite the final speed being clipped to zero.

## Opponent maneuvers: six possible futures

`predict_opponent_scenarios()` creates six alternatives from each source-stamped
opponent prediction. CONTINUE preserves the existing predictor's path. BRAKE
reduces forward speed at `opponent_braking` until stopped, without reversing.
LEFT and RIGHT follow the track at the initial estimated forward speed and
shift sideways smoothly. BRAKE_LEFT and BRAKE_RIGHT combine braking with a
sideways shift that stops when forward motion stops. A stationary opponent does
not slide sideways. These are possible motions, not inferred intentions.

Every MPPI sample, the weighted result, the best-sample fallback, the final
selected plan and the braking fallback are checked against all six futures for
each active opponent. The nominal path remains the source of soft costs and
target identity. FOLLOW uses the BRAKE future for its speed reference, so it
starts slowing before the nominal path requires it. Passing references leave
room for the lateral displacement bound plus both vehicles' footprints and
clearance. A narrower gap can now lead to FOLLOW instead of a nominal-only pass.

The default scenario assumptions, configurable as `overtaking.*` ROS parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `opponent_braking` | 3.0 m/s² | Positive braking magnitude |
| `opponent_lateral_displacement` | 0.65 m | Maximum additional left/right shift |
| `opponent_lateral_speed` | 0.5 m/s | Maximum track-relative sideways speed |
| `opponent_lateral_accel` | 1.0 m/s² | Maximum track-relative sideways acceleration |

Lateral quintic curves are time-scaled to respect those bounds, including the
effect of braking. The sideways speed is also limited to half the initial
forward speed. Directions follow the track normal, not global x/y axes. Source
age and actuation delay advance **every** scenario before its first controller
point, since a maneuver could already have begun during that interval. All six
are checked independently; they are never averaged or assigned probabilities.

## Input contracts and defaults

- Car state: rear-axle position in `map`, heading in radians, speed in m/s.
  The node's `pose_offset` converts a LiDAR pose if needed.
- Opponent `Path` messages: vehicle-centre positions in `map`, including the
  point at `header.stamp`, then points separated by `overtaking.opponent_dt`
  (default 0.1 s). The existing three predictor nodes use this interval.
- Predictions are resampled using their source stamp. A constant-velocity tail
  extension of at most 0.35 s accommodates delivery and actuation delay;
  shorter or stale inputs trigger BRAKE, rather than disappearing obstacles.
- `prediction_start_delay` is **actuation delay only** (0 in sim, 0.14 in the
  real-car launch). `delay_comp` also includes localization delay for the car
  state (0.20 with the original particle filter). They are not interchangeable.
- Default footprint: 0.51 m length, 0.28 m width, geometric centre 0.13 m ahead
  of the rear axle. These are starting assumptions to measure on your car.
  Clearance is 0.08 m plus opponent uncertainty growing with prediction time.
- All `PlannerConfig` fields are ROS parameters under `overtaking.*`.
- A stale known opponent remains a hazard until a fresh valid prediction
  arrives (or the node is restarted). This deliberately conservative baseline
  may stop after losing sight of a car; track retirement/reacquisition needs
  evaluation in VMware before enabling more permissive behavior.
- The pose watchdog demands zero speed after 0.5 s without pose updates. Old or
  future-dated pose messages also demand a stop.
- `wall_map.py` treats unknown and out-of-map space as occupied. It supports
  axis-aligned ROS map YAML files and rejects rotated maps explicitly.

The real-car launch now uses the four-car detector, matching the predictor's
`/opponents/detection_odom_0`, `_1`, `_2` inputs. The new state-estimation branch
has not been merged. To integrate `/state_estimate`, use rear-axle offset zero
and recheck the delay settings; do not apply the LiDAR offset twice.

## Offline tests (no VMware or racing simulator)

From the repository root with Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r mppi/test/requirements.txt
.venv/bin/python -m pytest mppi/test -q
```

The tests cover left/right blocked passages, following a slower car, a second
opponent blocking the pass, no valid candidates, stale/misaligned predictions,
continuous crossing collisions, walls between samples, vehicle limits, lap
wraparound, real waypoint files, unknown map cells, stopping distance, bounded
lateral motion, combined braking/sideways motion, source-time alignment of all
scenarios, nominal-safe paths invalidated by opponent maneuvers, and stop-command behavior
in the ROS adapter with in-memory message substitutes. They do not start ROS,
a simulator, a closed-loop race, or the physical car.

## Later, inside your ROS/VMware workspace

Put **this checkout** in the VM's ROS workspace `src/` (only one copy of each
ROS package), install the dependencies and build:

```sh
colcon build --symlink-install --packages-select mppi mppi_bringup
source install/setup.bash
```

After you start your existing simulator yourself, the existing simulation launch
uses the new planner by default:

```sh
ros2 launch mppi sim_lev_blocked_launch.py
ros2 topic echo /mppi/behavior
```

The `mppi_bringup` launch with a similar name starts the separate `mppi_example`
package and will not exercise this implementation. Use the `mppi` launch above.

The behavior JSON includes the selected mode, validation result, rejection
reason, valid sample count for each evaluated mode and opponent scenario names. Inspect its path in
RViz/Foxglove. First test one slow opponent on a straight, then blocked sides,
braking/lateral movement, corners, and multiple opponents. Compare against the
legacy controller with `physics_overtaking` set to false in the node parameters.

## What remains unverified

This is a finite-horizon physics baseline, not a collision-free guarantee.
The collision model assumes circular vehicle bounds and linear opponent motion
between prediction points. These six hypotheses do not cover every possible
braking strength, maneuver onset time, reversal of lateral direction, acceleration,
or intermediate combination. The lateral bounds are relative to the track;
total tire forces through corners are not modeled for opponents. The speed
estimate comes from the supplied predictor and can itself be wrong. The bounds
are starting assumptions, not measured/certified opponent limits. No unseen maneuver, localization error beyond the
margins, tire-slip behavior, actuator response, or safe continuation beyond the
two-second horizon is certified. A zero-speed request is a best-effort stop,
not proof the car can stop before an unavoidable collision. ROS transport,
detector performance, full laps, runtime on the Jetson and physical-car behavior
still need testing in your VMware/ROS environment. JAX's first call compiles;
measure warmed control latency before any hardware run.
