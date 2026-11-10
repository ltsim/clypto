"""The copy protocol of ``Agent``: base fields, subclass extras, and the guards."""
import re
from pathlib import Path

import numpy as np
import pytest

import clypto as cy

from ._agent_subclass import NaughtyAgent, TaggedAgent

PXD = Path(__file__ ).parents[2] / "clypto/optimizer/native/agent.pxd"


def seeded_agent(cls=cy.Agent):
    """An agent whose every declarable state field holds a unique sentinel."""
    agent = cls(np.arange(5.0), objectives=1.5)
    sentinels = {}
    for line in PXD.read_text().splitlines():
        m = re.match(r"\s*cdef (?:public|readonly) (\S+) (\w+)", line)
        if not m:
            continue
        kind, name = m.groups()
        if name in ("solution", "objectives", "weights", "fitness"):
            continue
        if kind == "cnp.ndarray":
            sentinels[name] = np.full(3, float(len(sentinels)))
        elif kind == "double":
            sentinels[name] = 3.5
        elif kind == "object":
            sentinels[name] = object()
        else:
            raise AssertionError(f"unexpected field type {kind} for {name}")
        setattr(agent, name, sentinels[name])
    return agent, sentinels


@pytest.mark.parametrize("cls", [cy.Agent, TaggedAgent])
@pytest.mark.parametrize("how", [cy.duplicate_agent, lambda a: a.copy()])
def test_state_survives_the_copy(cls, how):
    agent, sentinels = seeded_agent(cls)
    copy = how(agent)
    assert type(copy) is type(agent) and copy is not agent
    assert copy.fitness == 1.5 and copy.objectives is agent.objectives
    assert copy.solution is agent.solution
    for name, sentinel in sentinels.items():
        got = getattr(copy, name)
        if isinstance(sentinel, np.ndarray) or type(sentinel) is object:
            assert got is sentinel, name  # shared by reference, like every state field
        else:
            assert got == sentinel, name


def test_snapshot_copy_is_independent_in_the_vector():
    agent = TaggedAgent(np.arange(5.0))
    agent.tag = object()
    snapshot = agent.snapshot()
    assert snapshot.solution is not agent.solution and np.array_equal(snapshot.solution, agent.solution)
    assert snapshot.tag is agent.tag and type(snapshot) is TaggedAgent


def test_tagged_agent_keeps_the_extras_in_both_algorithms():
    agent = TaggedAgent(np.zeros(3), objectives=4.0)
    agent.tag = {"where": "nowhere"}
    agent.salary = 99.0
    agent.history = np.arange(3.0)

    copied = cy.duplicate_agent(agent)
    assert type(copied) is TaggedAgent
    assert copied.tag is agent.tag and copied.salary == 99.0 and copied.history is agent.history
    assert copied.fitness == 4.0


def test_subclass_missing_the_protocol_reports_it():
    class LooseAgent(TaggedAgent):
        pass  # Python subclass: sorts fine, but has no _new_instance of its own

    loose = LooseAgent(np.zeros(2))
    with pytest.raises(TypeError):  # _new_instance returned a TaggedAgent, not a LooseAgent
        cy.duplicate_agent(loose)

    class BareAgent(cy.Agent):
        pass  # no hooks at all

    with pytest.raises(NotImplementedError):
        cy.duplicate_agent(BareAgent(np.zeros(2)))


def test_copy_state_to_must_call_the_base_first():
    naughty = NaughtyAgent(np.zeros(3))
    naughty.tag = object()
    with pytest.raises(RuntimeError):  # the solution went missing: base call skipped
        cy.duplicate_agent(naughty)
