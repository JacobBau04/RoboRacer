"""Deterministic offline predictions and controller checks; no simulator/ROS."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mppi_core import MPPIConfig, simulate_jit
from physics_overtaking import (OpponentPrediction, OvertakingPlanner, PlannerConfig,
                                SCENARIO_NAMES, Track, mode_update, opponent_scenario_batch,
                                predict_opponent_scenarios, trajectory_valid)
from test_physics_overtaking import Corridor, adapter_methods, predictions


@pytest.fixture
def straight():
    return Track(np.array([[-20., 0.], [20., 0.], [20., 10.], [-20., 10.]]), np.full(4, 2.5))


def source_prediction(speed=2., dt=.1, horizon=20):
    return OpponentPrediction(np.column_stack((3 + speed*np.arange(horizon + 1)*dt,
                                               np.zeros(horizon + 1))), 10., dt)


def test_braking_stops_at_physical_stopping_distance(straight):
    settings = replace(PlannerConfig(), opponent_braking=2.)
    paths, covered = predict_opponent_scenarios(straight, source_prediction(), 10., 20, .1, settings)
    assert covered and paths.shape == (len(SCENARIO_NAMES), 21, 2)
    assert paths[0, -1, 0] == pytest.approx(7.)
    # v=2, a=-2: stop after 1 second and 1 metre, then remain stopped.
    np.testing.assert_allclose(paths[1, 10:, 0], 4., atol=1e-6)
    assert np.all(np.diff(paths[1, :, 0]) >= -1e-9)
    np.testing.assert_allclose(paths[:, 0], np.tile([3., 0.], (6, 1)))


def test_left_right_are_symmetric_and_bounded(straight):
    p = PlannerConfig()
    dt = .005
    prediction = source_prediction(dt=dt, horizon=800)
    paths, _ = predict_opponent_scenarios(straight, prediction, 10., 800, dt, p)
    left, right = paths[2, :, 1], paths[3, :, 1]
    np.testing.assert_allclose(left, -right, atol=1e-9)
    assert left[-1] > .5
    for i in (2, 3, 4, 5):
        d = paths[i, :, 1]
        assert np.max(np.abs(d)) <= p.opponent_lateral_displacement + 1e-6
        assert np.max(np.abs(np.diff(d)/dt)) <= p.opponent_lateral_speed + 1e-3
        assert np.max(np.abs(np.diff(d, n=2)/dt**2)) <= p.opponent_lateral_accel + 1e-2


def test_stationary_opponent_does_not_slide_sideways(straight):
    paths, _ = predict_opponent_scenarios(straight, source_prediction(speed=0.), 10., 20, .1, PlannerConfig())
    np.testing.assert_allclose(paths, np.broadcast_to([3., 0.], paths.shape), atol=1e-6)


def test_braking_sideways_scenarios_freeze_after_stopping(straight):
    paths, _ = predict_opponent_scenarios(straight, source_prediction(), 10., 20, .1, PlannerConfig())
    for i in (4, 5):
        np.testing.assert_allclose(paths[i, 8:], np.tile(paths[i, 8], (13, 1)), atol=1e-6)
    assert paths[4, -1, 1] > 0 and paths[5, -1, 1] < 0


def test_latency_advances_every_hypothesis_from_source_time(straight):
    p, prediction = PlannerConfig(), source_prediction()
    delayed, covered = predict_opponent_scenarios(straight, prediction, 10.1, 20, .1, p, .1)
    earlier, _ = predict_opponent_scenarios(straight, prediction, 10., 22, .1, p)
    assert covered
    np.testing.assert_allclose(delayed, earlier[:, 2:], atol=1e-6)
    assert delayed[2, 0, 1] > 0  # a lateral move may already have begun
    assert delayed[1, 0, 0] < delayed[0, 0, 0]


def test_lateral_directions_rotate_with_track_and_wrap_lap():
    track = Track(np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.]]), np.ones(4))
    # Travelling north: left is west, not global +y.
    t = np.arange(21)*.1
    prediction = OpponentPrediction(np.column_stack((np.full(21, 10.), 2 + t)), 0.)
    paths, _ = predict_opponent_scenarios(track, prediction, 0., 20, .1, PlannerConfig())
    assert paths[2, -1, 0] < 10. and paths[3, -1, 0] > 10.
    # A new lap must not be mistaken for a large negative speed.
    xy, _, _ = track.sample(track.length - .1 + t)
    paths, _ = predict_opponent_scenarios(track, OpponentPrediction(xy, 0.), 0., 20, .1, PlannerConfig())
    assert np.isfinite(paths).all()
    assert np.linalg.norm(paths[1, -1] - paths[1, 0]) > .05


@pytest.mark.parametrize('field', ['opponent_braking', 'opponent_lateral_speed',
                                  'opponent_lateral_accel', 'opponent_lateral_displacement'])
def test_invalid_prediction_limits_fail_explicitly(field):
    for value in (0., -1., float('nan')):
        with pytest.raises(ValueError):
            PlannerConfig(**{field: value})


def test_plan_safe_against_nominal_is_rejected_for_sudden_braking(straight):
    cfg, p, walls = MPPIConfig(), PlannerConfig(), Corridor()
    opponents, active = predictions(x=2., speed=2.)
    controls = jnp.zeros((cfg.H, 2))
    path = simulate_jit(jnp.array([0., 0., 0., 2.]), controls, cfg)
    assert bool(trajectory_valid(path, controls, jnp.asarray(opponents), jnp.asarray(active), cfg, p, walls))
    scenarios = opponent_scenario_batch(straight, opponents, active, cfg.DT, p)
    checks = jnp.asarray(scenarios.reshape(-1, cfg.H + 1, 2))
    flags = jnp.asarray(np.repeat(active, len(SCENARIO_NAMES)))
    assert not bool(trajectory_valid(path, controls, checks, flags, cfg, p, walls))


@pytest.mark.parametrize('scenario_index,side', [(2, 1.), (3, -1.)])
def test_nominal_clear_pass_conflicts_with_sideways_hypothesis(straight, scenario_index, side):
    cfg, p = MPPIConfig(), PlannerConfig()
    opponents, active = predictions(x=.13, speed=2.)
    # Already alongside with nominally sufficient lateral separation.
    controls = jnp.zeros((cfg.H, 2))
    path = simulate_jit(jnp.array([0., side*.85, 0., 2.]), controls, cfg)
    walls = Corridor(-3., 3.)
    assert bool(trajectory_valid(path, controls, jnp.asarray(opponents), jnp.asarray(active), cfg, p, walls))
    scenarios = opponent_scenario_batch(straight, opponents, active, cfg.DT, p)
    assert not bool(trajectory_valid(path, controls, jnp.asarray(scenarios[0, scenario_index:scenario_index+1]),
                                     jnp.ones(1, bool), cfg, p, walls))


def test_selected_plan_clears_every_scenario_of_every_opponent(straight):
    planner = OvertakingPlanner(straight, Corridor(-2., 2.))
    opponents, active = predictions(x=3.)
    opponents[1] = opponents[0] + [2., -.9]
    active[1] = True
    result = planner.plan([0., 0., 0., 2.], opponents, active)
    assert result.valid
    scenarios = opponent_scenario_batch(straight, opponents, active, planner.cfg.DT, planner.safety)
    assert bool(trajectory_valid(jnp.asarray(result.trajectory), jnp.asarray(result.controls),
                                 jnp.asarray(scenarios.reshape(-1, 21, 2)),
                                 jnp.asarray(np.repeat(active, 6)), planner.cfg, planner.safety, planner.wall_env))


def test_follow_reference_brakes_before_nominal_requires_it(straight):
    planner = OvertakingPlanner(straight, Corridor())
    opponents, active = predictions(x=2., speed=2.)
    scenarios = opponent_scenario_batch(straight, opponents, active, planner.cfg.DT, planner.safety)
    nominal, _ = planner.reference(np.array([0., 0., 0., 2.]), 'FOLLOW', opponents, active)
    robust, _ = planner.reference(np.array([0., 0., 0., 2.]), 'FOLLOW', opponents, active,
                                   follow_opponents=scenarios[:, 1])
    assert robust[0, 3] < nominal[0, 3]
    assert robust[-1, 3] < .5  # slows early; can still creep toward the standstill gap


def test_previous_pass_reassessed_against_side_move(straight):
    planner = OvertakingPlanner(straight, Corridor(-.6, 1.5))
    planner.behavior, planner.target = 'LEFT', 0
    opponents, active = predictions(x=2.)
    result = planner.plan([0., 0., 0., 2.], opponents, active)
    assert result.behavior in ('FOLLOW', 'BRAKE')
    assert 'RIGHT' not in result.valid_samples


@pytest.mark.parametrize('bottom,top', [(-1.5, .6), (-.6, 1.5)])
def test_narrow_pass_follows_with_room_for_opponent_motion(straight, bottom, top):
    planner = OvertakingPlanner(straight, Corridor(bottom, top))
    opponents, active = predictions()
    result = planner.plan([0., 0., 0., 2.], opponents, active)
    assert result.behavior == 'FOLLOW' and result.valid
    assert result.trajectory[-1, 3] < 1.


def test_scenario_collisions_exclude_samples_from_mppi_update(straight):
    cfg = replace(MPPIConfig(), T=2, U_STD=(0., 0.))
    p, walls = PlannerConfig(), Corridor()
    opponents, active = predictions(x=2., speed=2.)
    controls = jnp.zeros((cfg.H, 2))
    state = jnp.array([0., 0., 0., 2.])
    path = simulate_jit(state, controls, cfg)
    scenarios = opponent_scenario_batch(straight, opponents, active, cfg.DT, p)
    _, result, count, ok = mode_update(
        jax.random.PRNGKey(0), controls, controls, state, path,
        jnp.asarray(opponents), jnp.asarray(active), 0., cfg, p, walls,
        scenario_paths=jnp.asarray(scenarios.reshape(-1, 21, 2)),
        scenario_active=jnp.asarray(np.repeat(active, 6)))
    assert int(count) == 0 and not bool(ok)
    assert np.isfinite(result).all()


def test_ros_adapter_passes_timestamp_aligned_scenarios_to_real_planner(straight):
    from types import SimpleNamespace
    namespace = adapter_methods('_physics_control')
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
    planner = OvertakingPlanner(straight, Corridor(-.6, .6))
    real_plan = planner.plan
    captured = {}
    def record(*args, **kwargs):
        captured.update(kwargs)
        return real_plan(*args, **kwargs)
    planner.plan = record
    commands = []
    statuses = []
    adapter = SimpleNamespace(
        physics_predictions=[source_prediction(), None, None], physics_invalid=set(),
        previous_steer=0., prediction_start_delay=.14, physics_settings=planner.safety,
        last_physics_status=None, physics_planner=planner,
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_100_000_000,
                                                                      to_msg=lambda: None)),
        get_logger=lambda: SimpleNamespace(info=lambda _: None),
        behavior_pub=SimpleNamespace(publish=lambda status: statuses.append(status.data)),
        path_pub_=SimpleNamespace(publish=lambda _: None),
        publish_drive=lambda *command: commands.append(command))
    namespace['_physics_control'](adapter, np.array([0., 0., 0., 2.]))
    paths = captured['scenarios']
    assert paths.shape == (3, 6, 21, 2)
    assert paths[0, 1, 0, 0] < paths[0, 0, 0, 0]  # braking during latency was included
    assert paths[0, 2, 0, 1] > 0.
    assert commands and np.isfinite(commands).all()
    assert 'BRAKE_LEFT' in statuses[-1]
