#!/usr/bin/env python
"""Bounds blocks, ``Bounds`` and the compiled ``Problem``, checked through their decoders."""

import numpy as np
import pytest

import clypto as cy


def test_float_block_is_the_identity_inside_its_range():
    bounds = cy.Bounds(cy.NumberBounds(float, low=[-1.0, 0.0], up=[1.0, 2.0], name="x"))

    assert bounds.dtype is np.float64 and bounds.n_dims == 2
    assert bounds.decode(np.array([0.5, 3.0]))["x"].tolist() == [0.5, 2.0]


def test_scalar_bounds_broadcast_to_n_vars():
    block = cy.NumberBounds(float, low=-2.0, up=2.0, n_vars=4)

    assert block.n_vars == 4 and block.low.tolist() == [-2.0] * 4


@pytest.mark.parametrize("value_type", [int, bool])
def test_integer_cells_have_equal_width(value_type):
    low, up = (0, 2) if value_type is int else (None, None)
    block = cy.NumberBounds(value_type, low=low, up=up, name="k")
    bounds = cy.Bounds(block)
    top = 2 if value_type is int else 1

    assert block.low.tolist() == [-0.5] and block.up.tolist() == [top + 0.5]
    assert bounds.decode(np.array([-0.5]))["k"] == value_type(0)
    assert bounds.decode(np.array([top + 0.5]))["k"] == value_type(top)
    assert bounds.decode(np.array([0.49]))["k"] == value_type(0)
    assert bounds.decode(np.array([0.5]))["k"] == value_type(1)


def test_integer_generate_covers_both_ends():
    block = cy.NumberBounds(int, low=0, up=2, n_vars=3000)
    block.seed = 1

    counts = np.bincount(block.generate(), minlength=3) / 3000
    assert np.allclose(counts, 1 / 3, atol=0.04)


def test_binary_block():
    bounds = cy.Bounds(cy.NumberBounds(int, 0, 1, n_vars=3, name="b"))

    assert bounds.decode(np.array([0.2, 0.8, 1.4]))["b"].tolist() == [0, 1, 1]


@pytest.mark.parametrize("value_type", [int, bool])
def test_transfer_block(value_type):
    block = cy.TransferBounds(value_type, n_vars=6, tf_func="vstf_04", low=-4.0, up=4.0, name="t")
    bounds = cy.Bounds(block)
    bounds.seed = 3

    assert block.low.tolist() == [-4.0] * 6  # low is honored (MEALPY used lb * zeros)
    x = bounds.correct(np.array([-4.0, -2.0, 0.0, 0.0, 2.0, 4.0]))
    assert set(x.tolist()) <= {0.0, 1.0}
    assert bounds.decode(x)["t"].dtype == np.dtype(value_type)


def test_transfer_block_can_forbid_all_zeros():
    block = cy.TransferBounds(int, n_vars=4, all_zeros=False)
    block.seed = 0

    assert block.correct(np.zeros(4)).sum() == 1


def test_string_block_accepts_any_hashable_label():
    bounds = cy.Bounds(cy.StringBounds([("auto", "forward"), (None, 3, "x")], name="s"))

    assert bounds.low.tolist() == [-0.5, -0.5] and bounds.up.tolist() == [1.5, 2.5]
    assert bounds.decode(bounds.encode([["forward", None]]))["s"] == ["forward", None]
    assert bounds.decode(np.array([9.0, 1.2]))["s"] == ["forward", 3]


def test_single_string_set_is_one_variable():
    block = cy.StringBounds(("a", "b", "c"), name="s")

    assert block.n_vars == 1
    assert cy.Bounds(block).decode(np.array([2.0]))["s"] == "c"


def test_sequence_block():
    bounds = cy.Bounds(cy.SequenceBounds([(1,), (2, 3)], return_type=list, name="q"))

    assert bounds.decode(np.array([1.0]))["q"] == [2, 3]
    assert bounds.decode(bounds.encode([[(1,)]]))["q"] == [1]


def test_permutation_block_round_trips():
    bounds = cy.Bounds(cy.PermutationBounds((-10, -4, 10, 6), name="p"))
    bounds.seed = 2

    perm = bounds.blocks[0].generate()
    assert sorted(perm) == [-10, -4, 6, 10]
    assert bounds.decode(bounds.encode([perm]))["p"] == perm


def test_mixed_bounds_concatenate_and_decode_by_name():
    bounds = cy.Bounds(
        cy.NumberBounds(float, low=[0.0, 0.0], up=[1.0, 1.0], name="w"),
        [cy.NumberBounds(int, low=1, up=5, name="n"), cy.StringBounds(("a", "b"), name="kind")],
    )

    assert bounds.n_dims == 4
    decoded = bounds.decode(bounds.correct(np.array([0.3, 2.0, 7.0, 0.6])))
    assert decoded["w"].tolist() == [0.3, 1.0] and decoded["n"] == 5 and decoded["kind"] == "b"


def test_unnamed_blocks_get_positional_names_and_duplicates_raise():
    bounds = cy.Bounds(cy.NumberBounds(float, 0.0, 1.0), cy.NumberBounds(int, 0, 3))

    assert [b.name for b in bounds.blocks] == ["x0", "x1"]
    with pytest.raises(ValueError):
        cy.Bounds(cy.NumberBounds(float, 0.0, 1.0, name="a"), cy.NumberBounds(float, 0.0, 1.0, name="a"))


def test_seed_makes_generation_reproducible():
    def make():
        return cy.Bounds(cy.NumberBounds(float, -1.0, 1.0, n_vars=5), cy.NumberBounds(int, 0, 9, n_vars=5))

    a, b = make(), make()
    a.seed = b.seed = 11

    assert np.array_equal(a.generate(), b.generate())


def test_problem_binds_bounds_sense_and_evaluates_a_matrix():
    problem = cy.Problem(
        bounds=[cy.NumberBounds(float, low=[-1.0] * 3, up=[1.0] * 3)],
        sense="max",
        obj_func=lambda x: [float(np.sum(x)), float(np.sum(x**2))],
        obj_weights=[1.0, 2.0],
        seed=5,
    )
    X = np.array([[0.0, 0.5, 1.0], [1.0, 1.0, 1.0]])

    fitness, objectives = problem.evaluate(X)
    assert problem.sense == "max" and problem.n_dims == 3 and problem.n_objs == 2
    assert objectives.tolist() == [[1.5, 1.25], [3.0, 3.0]]
    assert fitness.tolist() == [4.0, 9.0]
    assert problem.get_target(X[0]).fitness == 4.0


def test_problem_rejects_unknown_sense():
    with pytest.raises(ValueError):
        cy.Problem(bounds=cy.NumberBounds(float, 0.0, 1.0), sense="minimize")


def test_problem_subclass_overrides_obj_func():
    class Shifted(cy.Problem):
        def obj_func(self, x):
            return float(np.sum((x - 0.5) ** 2))

    problem = Shifted(cy.NumberBounds(float, low=[0.0] * 2, up=[1.0] * 2))
    assert problem.get_target(np.array([0.5, 0.5])).fitness == 0.0


def test_dict_problem_is_seeded_by_solve():
    spec = {"bounds": cy.NumberBounds(float, low=[-1.0] * 4, up=[1.0] * 4), "obj_func": lambda x: float(np.sum(x**2))}
    runs = [cy.get_optimizer_by_class("OriginalPSO")(epoch=5, pop_size=10).solve(dict(spec), seed=4) for _ in range(2)]

    assert np.array_equal(runs[0].solution, runs[1].solution)


def test_agent_functions():
    better, worse = cy.LegacyAgent(np.zeros(2), cy.Target(1.0)), cy.LegacyAgent(np.ones(2), cy.Target(2.0))

    from clypto.optimizer.native import agent as fn

    assert fn.compare_fitness(better, worse) == -1 and fn.compare_fitness(better, worse, "max") == 1
    assert fn.get_better_solution(better, worse) is better
    assert fn.get_better_solution(better, worse, "max") is worse
    assert fn.is_better_than(better, worse) and not fn.is_better_than(better, worse, "max")
    twin = cy.LegacyAgent(np.zeros(2), cy.Target(5.0))
    assert fn.sync_if_duplicate(twin, better) and twin.target is better.target


def test_copy_shares_fields_and_copies_the_target():
    from clypto.native.collection.legacy.swarm_based.PSO.OriginalPSO import _OriginalPSOAgent

    class Tagged(cy.LegacyAgent):
        pass

    compiled = _OriginalPSOAgent(np.zeros(2), cy.Target(1.0), velocity=np.ones(2))
    python = Tagged(np.zeros(2), cy.Target(1.0), tag=[1])
    for agent, field in ((compiled, "velocity"), (python, "tag")):
        copy = agent.copy()
        assert type(copy) is type(agent) and getattr(copy, field) is getattr(agent, field)
        assert copy.solution is agent.solution and copy.target is not agent.target
        assert copy.target.fitness == 1.0
    compiled.update(velocity=None, solution=np.ones(2))
    assert compiled.velocity is None and compiled.solution.tolist() == [1.0, 1.0]
