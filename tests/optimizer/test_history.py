import numpy as np
import pytest

import clypto as cy
from clypto.native.collection.evolutionary_based import DE
from clypto.native.collection.swarm_based import PSO

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
        bounds=cy.NumberBounds(float, low=[-1] * N_DIMS, up=[1] * N_DIMS),
        sense="min",
    )


def test_metrics_recorded(problem):
    model = PSO.OriginalPSO(epoch=10, pop_size=8)
    model.solve(problem, seed=1, debug=True)

    tracker = model.tracker
    assert tracker.n_epochs == 10
    assert tracker.optimizer_name == "OriginalPSO"
    assert tracker.n_dims == N_DIMS

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
    assert model.tracker.n_epochs == 0 and model.tracker.optimizer_name == ""
    with pytest.raises(KeyError):
        model.tracker["global_best_fit"]


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
        bounds=cy.NumberBounds(float, low=[-1] * N_DIMS, up=[1] * N_DIMS),
        sense="min",
    )
    model = PSO.OriginalPSO(epoch=4, pop_size=6)
    model.solve(multi, seed=1, track_population=True)

    tracker = model.tracker
    objectives = tracker["objectives"]
    assert objectives.shape == (4, 6, 2)
    assert np.isfinite(objectives).all()
    assert tracker.n_objs == 2


def test_hooks_solve_arguments(problem):
    model = PSO.OriginalPSO(epoch=3, pop_size=5)
    log = []
    model.solve(
        problem,
        seed=1,
        before_iteration=lambda pop, epoch: log.append(("before", epoch, len(pop))),
        after_iteration=lambda pop, epoch: log.append(("after", epoch, len(pop))),
    )

    assert [(kind, epoch) for kind, epoch, _ in log] == [
        ("before", 1), ("after", 1),
        ("before", 2), ("after", 2),
        ("before", 3), ("after", 3),
    ]
    assert all(entry[2] == 5 for entry in log)


def test_hooks_fire_without_tracking(problem):
    # Callbacks are per-solve arguments, not part of the tracker: they run even
    # when debug/track_population are off.
    model = PSO.OriginalPSO(epoch=3, pop_size=5)
    calls = []
    model.solve(problem, seed=1, after_iteration=lambda pop, epoch: calls.append(epoch))

    assert calls == [1, 2, 3]
    assert model.tracker.recording is False


def test_hooks_do_not_leak_between_solves(problem):
    model = PSO.OriginalPSO(epoch=2, pop_size=5)
    calls = []
    model.solve(problem, seed=1, after_iteration=lambda pop, epoch: calls.append(epoch))
    model.solve(problem, seed=2)  # no callback passed this time

    assert calls == [1, 2]


def test_hook_exception_propagates_and_closes_run(problem):
    model = PSO.OriginalPSO(epoch=3, pop_size=5)

    def boom(population, epoch):
        raise RuntimeError("hook failed")

    with pytest.raises(RuntimeError, match="hook failed"):
        model.solve(problem, seed=1, debug=True, after_iteration=boom)

    # The tracker is finalized by the ``finally``, so the run is not left open.
    # The after hook raises before ``record``, so zero epochs were recorded.
    assert model.tracker.recording is False
    assert model.tracker.n_epochs == 0


def test_tracker_reflects_latest_run(problem):
    model = PSO.OriginalPSO(epoch=4, pop_size=6)
    model.solve(problem, seed=1, debug=True)
    assert model.tracker.n_epochs == 4

    model.solve(problem, seed=1, debug=True)
    assert model.tracker.n_epochs == 4
    assert model.tracker["nfe"].shape == (4,)

    model.solve(problem, seed=1)  # untracked run clears the tracker
    assert model.tracker.n_epochs == 0 and model.tracker.optimizer_name == ""


def test_termination_dict_records_history(problem):
    model = PSO.OriginalPSO(epoch=1000, pop_size=8)
    model.solve(problem, seed=1, termination={"max_epoch": 20}, debug=True)

    assert model.tracker.n_epochs <= 20


def test_termination_max_fe(problem):
    model = PSO.OriginalPSO(epoch=1000, pop_size=8)
    model.solve(problem, seed=1, termination={"max_fe": 500}, debug=True)

    assert model.tracker.n_epochs < 1000


def test_termination_early_stop_does_not_crash(problem):
    model = PSO.OriginalPSO(epoch=1000, pop_size=8)
    model.solve(
        problem,
        seed=1,
        termination={"max_epoch": 40, "max_early_stop": 20, "epsilon": 1e-10},
    )

    assert model.g_best is not None
