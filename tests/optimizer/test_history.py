import numpy as np
import pytest
import zarr

import clypto as cy
from clypto.native.collection.vectorize.evolutionary_based import DE
from clypto.native.collection.vectorize.swarm_based import PSO

N_DIMS = 5

METRIC_KEYS = (
    "global_best_fit",
    "current_best_fit",
    "current_worst_fit",
    "mean_fit",
    "std_fit",
    "diversity",
    "exploration",
    "exploitation",
    "epoch_time",
    "nfe",
    "epoch_pop_size",
)


def objective(solution):
    return np.sum(solution**2)


def multi_objective(solution):
    return [np.sum(solution**2), np.sum((solution - 1.0) ** 2)]


@pytest.fixture
def problem():
    return cy.Problem(
        obj_func=objective,
        bounds=cy.FloatVar(lb=[-1] * N_DIMS, ub=[1] * N_DIMS),
        minmax="min",
    )


def test_metrics_recorded(problem):
    model = PSO.OriginalPSO(epoch=10, pop_size=8)
    model.solve(problem, seed=1, debug=True)

    tracker = model.tracker
    assert tracker.path is None
    assert tracker.group.attrs["n_epochs"] == 10
    assert tracker.group.attrs["optimizer"] == "OriginalPSO"
    assert tracker.group.attrs["n_dims"] == N_DIMS

    for key in METRIC_KEYS:
        assert tracker[key].shape == (10,)

    # global best is monotonic (min problem) and matches best-so-far
    gbest = tracker["global_best_fit"]
    assert np.all(np.diff(gbest) <= 1e-12)

    # exploration/exploitation derive from diversity
    diversity = tracker["diversity"]
    exploration = tracker["exploration"]
    assert np.isclose(exploration.max(), 100.0)
    assert np.allclose(exploration + tracker["exploitation"], 100.0)
    assert np.allclose(exploration, 100.0 * diversity / diversity.max())


def test_no_tracking_by_default(problem):
    model = PSO.OriginalPSO(epoch=5, pop_size=6)
    model.solve(problem, seed=1)

    assert model.tracker.recording is False
    assert model.tracker.group is None


def test_population_capture_and_nan_padding(problem):
    model = DE.SAP_DE(epoch=40, pop_size=10)
    model.solve(problem, seed=1, track_population=True)

    tracker = model.tracker
    solution, fitness = tracker["solution"], tracker["fitness"]
    counts = tracker["epoch_pop_size"].astype(int)

    assert solution.shape == (40, counts.max(), N_DIMS)
    assert fitness.shape == (40, counts.max())
    assert counts.max() > 10  # SAP_DE grows the population
    assert counts.min() < counts.max()  # and shrinks it again

    for epoch, n_agents in enumerate(counts):
        assert np.isfinite(solution[epoch, :n_agents]).all()
        assert np.isnan(solution[epoch, n_agents:]).all()
        assert np.isfinite(fitness[epoch, :n_agents]).all()
        assert np.isnan(fitness[epoch, n_agents:]).all()


def test_multi_objective_population():
    multi = cy.Problem(
        obj_func=multi_objective,
        bounds=cy.FloatVar(lb=[-1] * N_DIMS, ub=[1] * N_DIMS),
        minmax="min",
    )
    model = PSO.OriginalPSO(epoch=4, pop_size=6)
    model.solve(multi, seed=1, track_population=True)

    objectives = model.tracker["objectives"]
    assert objectives.shape == (4, 6, 2)
    assert np.isfinite(objectives).all()


def test_history_path_persists(problem, tmp_path):
    path = str(tmp_path / "run.zarr")
    model = PSO.OriginalPSO(epoch=4, pop_size=6)
    model.solve(problem, seed=2, debug=True, history_path=path)

    assert model.tracker.path == path
    reopened = zarr.open_group(path, mode="r")
    assert reopened.attrs["n_epochs"] == 4
    assert reopened.attrs["seed"] == 2
    assert reopened["metrics"]["global_best_fit"].shape == (4,)


def test_hooks_assignment_and_order(problem):
    model = PSO.OriginalPSO(epoch=5, pop_size=8)
    calls = []
    model.tracker.before = lambda pop: calls.append(("before", model.tracker.epoch, len(pop)))
    model.tracker.after = lambda pop: calls.append(("after", model.tracker.epoch, len(pop)))
    model.solve(problem, seed=1, debug=True)

    assert len(calls) == 10
    assert [c[0] for c in calls] == ["before", "after"] * 5
    assert [c[1] for c in calls] == [e for e in range(1, 6) for _ in range(2)]
    assert all(c[2] == 8 for c in calls)


def test_hooks_decorator(problem):
    model = PSO.OriginalPSO(epoch=3, pop_size=5)
    seen = []

    @model.tracker.on_before
    def before_iteration(population):
        seen.append(-len(population))

    @model.tracker.on_after
    def after_iteration(population):
        seen.append(len(population))

    model.solve(problem, seed=1, debug=True)
    assert seen == [-5, 5, -5, 5, -5, 5]


def test_hooks_solve_arguments(problem):
    log = []
    PSO.OriginalPSO(epoch=3, pop_size=5).solve(
        problem,
        seed=1,
        debug=True,
        before_iteration=lambda pop: log.append("before"),
        after_iteration=lambda pop: log.append("after"),
    )
    assert log == ["before", "after"] * 3


def test_hooks_do_not_fire_without_tracking(problem):
    model = PSO.OriginalPSO(epoch=3, pop_size=5)
    calls = []
    model.tracker.after = lambda pop: calls.append(1)
    model.solve(problem, seed=1, debug=False)
    assert calls == []


def test_hook_exception_propagates(problem):
    model = PSO.OriginalPSO(epoch=3, pop_size=5)

    def boom(population):
        raise RuntimeError("hook failed")

    model.tracker.after = boom
    with pytest.raises(RuntimeError, match="hook failed"):
        model.solve(problem, seed=1, debug=True)


def test_termination_dict_records_history(problem):
    model = PSO.OriginalPSO(epoch=1000, pop_size=8)
    model.solve(problem, seed=1, termination={"max_epoch": 20}, debug=True)

    assert model.tracker.group.attrs["n_epochs"] <= 20


def test_termination_max_fe(problem):
    model = PSO.OriginalPSO(epoch=1000, pop_size=8)
    model.solve(problem, seed=1, termination={"max_fe": 500}, debug=True)

    assert model.tracker.group.attrs["n_epochs"] < 1000


def test_termination_early_stop_does_not_crash(problem):
    model = PSO.OriginalPSO(epoch=1000, pop_size=8)
    model.solve(
        problem,
        seed=1,
        termination={"max_epoch": 40, "max_early_stop": 20, "epsilon": 1e-10},
    )

    assert model.g_best is not None
