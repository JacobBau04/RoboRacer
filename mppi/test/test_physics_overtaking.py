"""Offline unit/scenario tests: no ROS, VMware, simulator or vehicle commands."""
from dataclasses import replace
import ast
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mppi_core import MPPIConfig, simulate_jit
from physics_overtaking import (OvertakingPlanner, PlannerConfig, OpponentPrediction,
                                Track, align_prediction, mode_update, trajectory_valid, side_valid,
                                SCENARIO_NAMES, predict_opponent_scenarios)


class Corridor:
    """Analytic walls for deterministic clearance tests, not a racing simulator."""
    def __init__(self, bottom=-1.5, top=1.5):
        self.bottom, self.top = bottom, top

    def sample_wall_distance(self, xy):
        return jnp.minimum(xy[..., 1] - self.bottom, self.top - xy[..., 1])


@pytest.fixture
def track():
    return Track(np.array([[-20., 0.], [20., 0.], [20., 10.], [-20., 10.]]), np.full(4, 2.5))


def predictions(cfg=None, x=2.0, y=0.0, speed=0.8):
    cfg = cfg or MPPIConfig()
    opponents = np.full((3, cfg.H + 1, 2), 1000.0)
    opponents[0, :, 0] = x + np.arange(cfg.H + 1) * cfg.DT * speed
    opponents[0, :, 1] = y
    return opponents, np.array([True, False, False])


def test_rk4_matches_straight_line_physics():
    cfg = MPPIConfig()
    controls = jnp.zeros((cfg.H, 2))
    path = np.asarray(simulate_jit(jnp.array([0., 0., 0., 2.]), controls, cfg))
    np.testing.assert_allclose(path[-1], [4., 0., 0., 2.], atol=1e-5)


def test_braking_at_rest_never_predicts_reverse_motion():
    cfg = MPPIConfig()
    controls = jnp.tile(jnp.array([cfg.MIN_ACCEL, 0.]), (cfg.H, 1))
    path = np.asarray(simulate_jit(jnp.array([0., 0., 0., 0.]), controls, cfg))
    np.testing.assert_allclose(path, 0., atol=1e-6)
    moving = np.asarray(simulate_jit(jnp.array([0., 0., 0., .1]), controls, cfg))
    assert np.all(np.diff(moving[:, 0]) >= -1e-6)


@pytest.mark.parametrize('bottom,top,expected', [(-2.5, 0.6, 'RIGHT'), (-0.6, 2.5, 'LEFT'),
                                              (-0.6, 0.6, 'FOLLOW')])
def test_side_clearance_selects_behavior(track, bottom, top, expected):
    planner = OvertakingPlanner(track, Corridor(bottom, top))
    opponents, active = predictions()
    result = planner.plan([0., 0., 0., 2.], opponents, active)
    assert result.behavior == expected
    assert result.valid
    assert np.isfinite(result.controls).all()
    assert result.valid_samples[expected] > 0
    if expected == 'FOLLOW':
        assert result.trajectory[-1, 3] < 1.2  # actually slows, does not just label itself FOLLOW
    else:
        assert np.sign(result.trajectory[-1, 1]) == (1 if expected == 'LEFT' else -1)
    assert bool(trajectory_valid(jnp.asarray(result.trajectory), jnp.asarray(result.controls),
                                jnp.asarray(opponents), jnp.asarray(active), planner.cfg,
                                planner.safety, planner.wall_env))


def test_clear_track_accelerates_and_does_not_invent_pass(track):
    planner = OvertakingPlanner(track, Corridor())
    result = planner.plan([0., 0., 0., 1.])
    assert result.valid and result.behavior == 'FOLLOW'
    assert result.controls[0, 0] > 0
    assert set(result.valid_samples) == {'FOLLOW'}


def test_complete_pass_rejoins_without_tracking_old_opponent(track):
    planner = OvertakingPlanner(track, Corridor())
    planner.behavior, planner.target = 'LEFT', 0
    opponents, active = predictions(x=-2.)
    ref, leader = planner.reference(np.array([0., 0.8, 0., 2.]), 'FOLLOW', opponents, active)
    assert leader is None
    assert abs(ref[-1, 1]) < 0.1


def test_swept_check_catches_crossing_between_samples():
    cfg = replace(MPPIConfig(), H=1, DT=1.0, MAX_SPEED=5.)
    safety = PlannerConfig()
    X = jnp.array([[-2., 0., 0., 4.], [2., 0., 0., 4.]])
    U = jnp.zeros((1, 2))
    opponents = np.full((3, 2, 2), 1000.)
    opponents[0] = [[0., -2.], [0., 2.]]
    # Endpoints are far apart, but the two segments cross.
    assert not bool(trajectory_valid(X, U, jnp.asarray(opponents), jnp.array([1, 0, 0], bool),
                                     cfg, safety, Corridor(-5, 5)))


def test_swept_wall_check_catches_thin_wall_between_time_steps():
    class ThinWall:
        def sample_wall_distance(self, xy):
            return jnp.abs(xy[..., 0])
    cfg = replace(MPPIConfig(), H=1, DT=1., MAX_SPEED=5.)
    X = jnp.array([[-2., 0., 0., 4.], [2., 0., 0., 4.]])
    assert not bool(trajectory_valid(X, jnp.zeros((1, 2)), jnp.full((3, 2, 2), 1e3),
                                     jnp.zeros(3, bool), cfg, PlannerConfig(), ThinWall()))


def test_all_invalid_update_has_no_nan_and_returns_failure():
    cfg = replace(MPPIConfig(), T=64)
    state = jnp.array([0., 0., 0., 2.])
    controls = jnp.zeros((cfg.H, 2))
    ref = simulate_jit(state, controls, cfg)
    result = mode_update(jax.random.PRNGKey(7), controls, controls, state, ref,
                         jnp.full((3, cfg.H + 1, 2), 1e3), jnp.zeros(3, bool),
                         0., cfg, PlannerConfig(), Corridor(-0.1, 0.1))
    u, path, count, valid = result
    assert int(count) == 0 and not bool(valid)
    assert np.isfinite(u).all() and np.isfinite(path).all()


def test_no_valid_behavior_reports_braking_failure(track):
    planner = OvertakingPlanner(track, Corridor(-0.1, 0.1))
    result = planner.plan([0., 0., 0., 2.])
    assert result.behavior == 'BRAKE' and not result.valid
    assert result.reason == 'no_valid_candidate'
    assert result.controls[0, 0] == planner.cfg.MIN_ACCEL


def test_second_opponent_blocks_otherwise_open_pass(track):
    planner = OvertakingPlanner(track, Corridor(-0.6, 2.5))
    opponents, active = predictions()
    baseline = planner.plan([0., 0., 0., 2.], opponents, active)
    assert baseline.behavior == 'LEFT' and baseline.valid
    planner = OvertakingPlanner(track, Corridor(-0.6, 2.5))
    opponents[1] = opponents[0] + np.array([0., 0.85])
    active[1] = True
    result = planner.plan([0., 0., 0., 2.], opponents, active)
    assert result.behavior == 'FOLLOW' and result.valid


def test_stale_prediction_brakes_even_if_no_collision_is_predicted(track):
    planner = OvertakingPlanner(track, Corridor())
    opponents, active = predictions(x=5.)
    result = planner.plan([0., 0., 0., 2.], opponents, active, stale=True)
    assert result.behavior == 'BRAKE'
    assert result.reason == 'stale_or_short_prediction'
    assert result.controls[0, 0] < 0


def test_alignment_uses_source_stamp_and_controller_delay():
    t = np.arange(21) * 0.1
    prediction = OpponentPrediction(np.column_stack((2*t, np.zeros(21))), 10.)
    path, covered = align_prediction(prediction, 10.05, 20, .1, .2)
    assert covered
    np.testing.assert_allclose(path[[0, -1], 0], [.5, 4.5])
    assert not align_prediction(prediction, 11., 20, .1)[1]
    assert not align_prediction(prediction, 9., 20, .1)[1]


def test_track_wrap_and_left_normal():
    track = Track(np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.]]), np.ones(4))
    s, d = track.project(np.array([5., 1.]))
    assert s == 5. and d == 1.
    assert track.gap(0.2, track.length - 0.1) == pytest.approx(.3)
    np.testing.assert_allclose(track.sample(track.length + 5.)[0], [5., 0.])


def test_invalid_data_is_rejected(track):
    with pytest.raises(ValueError):
        OvertakingPlanner(track, Corridor()).plan([np.nan, 0, 0, 1])
    with pytest.raises(ValueError):
        Track(np.zeros((3, 2)), np.ones(3))


@pytest.mark.parametrize('field,value', [('clearance', -1.), ('wall_substeps', 0),
                                         ('max_lateral_accel', 0.)])
def test_invalid_settings_fail_at_startup(field, value):
    with pytest.raises(ValueError):
        PlannerConfig(**{field: value})


def test_opposite_passing_modes_are_not_mixed():
    reference = jnp.array([[0., 0., 0., 2.], [1., 0., 0., 2.]])
    left = reference.at[:, 1].set(0.9)
    right = reference.at[:, 1].set(-0.9)
    target = jnp.array([[0., 0.], [1., 0.]])
    assert bool(side_valid(left, reference, target, 1.))
    assert not bool(side_valid(right, reference, target, 1.))
    assert bool(side_valid(right, reference, target, -1.))


def test_unsafe_average_falls_back_to_valid_sample(track):
    planner = OvertakingPlanner(track, Corridor(-2.5, 2.5))
    cfg = replace(planner.cfg, T=2, U_STD=(0., 0.))
    state = np.array([0., 0., 0., 2.])
    opponents, active = predictions(cfg)
    left_ref, _ = planner.reference(state, 'LEFT', opponents, active)
    right_ref, _ = planner.reference(state, 'RIGHT', opponents, active)
    left = planner.tracking_seed(state, left_ref, 0.)
    right = planner.tracking_seed(state, right_ref, 0.)
    average = jnp.asarray((left + right) / 2)
    args = (jnp.asarray(opponents), jnp.asarray(active), cfg, planner.safety, planner.wall_env)
    assert not bool(trajectory_valid(simulate_jit(jnp.asarray(state), average, cfg), average, *args))
    # No noise: exactly one left and one right sample, symmetric costs. The
    # arithmetic average collides; the production update must reject it.
    reference = (left_ref + right_ref) / 2
    u, path, count, ok = mode_update(jax.random.PRNGKey(0), jnp.asarray(right), jnp.asarray(left),
                                     jnp.asarray(state), jnp.asarray(reference), args[0], args[1],
                                     0., cfg, planner.safety, planner.wall_env)
    assert bool(ok) and int(count) == 2
    assert bool(trajectory_valid(path, u, *args))
    assert abs(float(path[-1, 1])) > 0.5


def test_committed_pass_does_not_consider_opposite_side(track):
    planner = OvertakingPlanner(track, Corridor())
    planner.behavior, planner.target = 'LEFT', 0
    opponents, active = predictions()
    result = planner.plan([0., 0., 0., 2.], opponents, active)
    assert 'RIGHT' not in result.valid_samples


@pytest.mark.parametrize('violation', ['steering_rate', 'lateral_acceleration', 'nonfinite'])
def test_vehicle_limit_violations_are_rejected(violation):
    cfg, safety = MPPIConfig(), PlannerConfig()
    controls = np.zeros((cfg.H, 2))
    state = np.array([0., 0., 0., 2.5])
    if violation == 'steering_rate':
        controls[0, 1] = cfg.MAX_STEER
    elif violation == 'lateral_acceleration':
        controls[:, 1] = cfg.MAX_STEER
        safety = replace(safety, max_steering_rate=10.)
    else:
        controls[0, 0] = np.nan
    path = simulate_jit(jnp.asarray(state), jnp.asarray(controls), cfg)
    assert not bool(trajectory_valid(path, jnp.asarray(controls), jnp.full((3, cfg.H + 1, 2), 1e3),
                                     jnp.zeros(3, bool), cfg, safety, Corridor(-20, 20)))


def test_shipped_racelines_load_with_closed_track_geometry():
    root = Path(__file__).resolve().parents[2]
    car = np.loadtxt(root / 'mpc/waypoints/levine_race.csv', delimiter=',')
    sim = np.loadtxt(root / 'mppi_bringup/waypoints/old/lev_testing/lev_blocked.csv', delimiter=';')
    assert Track(car[:, :2], car[:, 3]).length > 10
    assert Track(sim[:, 1:3], sim[:, 5]).length > 10


def adapter_methods(*names):
    source = Path(__file__).resolve().parents[1] / 'scripts' / 'mppi_node.py'
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'MPPI')
    methods = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {'np': np, 'jnp': jnp, 'OpponentPrediction': OpponentPrediction,
                 'align_prediction': align_prediction, 'SCENARIO_NAMES': SCENARIO_NAMES,
                 'predict_opponent_scenarios': predict_opponent_scenarios}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def test_ros_receive_keeps_missing_opponent_as_stale():
    # Execute the production adapter method with in-memory messages, without
    # importing/pretending to run ROS or substituting the algorithm itself.
    receive = adapter_methods('_receive_physics_prediction')['_receive_physics_prediction']
    adapter = SimpleNamespace(physics_enabled=True, physics_predictions=[None]*3,
                              physics_invalid=set(), physics_settings=PlannerConfig())
    def message(points, frame='map'):
        return SimpleNamespace(poses=[SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y))) for x, y in points],
                               header=SimpleNamespace(frame_id=frame, stamp=SimpleNamespace(sec=10, nanosec=0)))
    receive(adapter, 0, message([(1., 0.), (2., 0.)]))
    receive(adapter, 0, message([]))
    assert adapter.physics_predictions[0] is not None and 0 in adapter.physics_invalid
    receive(adapter, 0, message([(1., 0.), (2., 0.)]))
    assert 0 not in adapter.physics_invalid
    receive(adapter, 0, message([(1., 0.), (2., 0.)], frame='laser'))
    assert 0 in adapter.physics_invalid


def test_pose_watchdog_commands_stop_without_pose():
    namespace = adapter_methods('_pose_watchdog')
    namespace['time'] = SimpleNamespace(monotonic=lambda: 10.)
    commands = []
    adapter = SimpleNamespace(physics_enabled=True, last_pose_arrival=None,
                              previous_steer=.1, pose_timeout=.5,
                              publish_drive=lambda *cmd: commands.append(cmd))
    namespace['_pose_watchdog'](adapter)
    assert commands == [(0., .1)]
    adapter.last_pose_arrival = 9.9
    namespace['_pose_watchdog'](adapter)
    assert len(commands) == 1
    adapter.last_pose_arrival = 9.
    namespace['_pose_watchdog'](adapter)
    assert len(commands) == 2


def test_ros_adapter_never_sends_acceleration_for_an_invalid_plan(track):
    namespace = adapter_methods('_physics_control')
    # Minimal message objects; the actual adapter and planner are executed.
    class PathMessage:
        def __init__(self):
            self.header = SimpleNamespace(stamp=None, frame_id='')
            self.poses = []
    class PoseMessage:
        def __init__(self):
            self.pose = SimpleNamespace(position=SimpleNamespace(x=0., y=0.),
                                        orientation=SimpleNamespace(z=0., w=1.))
    namespace.update(Path=PathMessage, PoseStamped=PoseMessage,
                     String=lambda **kwargs: SimpleNamespace(**kwargs))
    commands = []
    settings = PlannerConfig()
    adapter = SimpleNamespace(
        physics_predictions=[None]*3, physics_invalid=set(), previous_steer=0.,
        prediction_start_delay=0., physics_settings=settings, last_physics_status=None,
        physics_planner=OvertakingPlanner(track, Corridor(-.1, .1)),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_000_000_000,
                                                                      to_msg=lambda: None)),
        get_logger=lambda: SimpleNamespace(info=lambda _: None),
        behavior_pub=SimpleNamespace(publish=lambda _: None),
        path_pub_=SimpleNamespace(publish=lambda _: None),
        publish_drive=lambda *command: commands.append(command))
    namespace['_physics_control'](adapter, np.array([0., 0., 0., 2.]))
    assert commands == [(0., 0.)]
    assert adapter.last_physics_status == ('BRAKE', False, 'no_valid_candidate')
