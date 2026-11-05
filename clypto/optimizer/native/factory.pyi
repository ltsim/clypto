import typing

class AgentFactory:
    """The per-algorithm lifecycle of an agent, held by Population.

    ``Population.create_agent`` / ``Population.generate_agent`` delegate to
    ``create`` / ``generate``; algorithms with extra per-agent state subclass
    and override them. See ``tools/scaffold_agent.py`` for Agent subclasses and
    ``docs/custom-optimizers/`` for the authoring guide.
    """

    problem: typing.Any
    generator: typing.Any

    def __repr__(self) -> str: ...
    def bind(self, problem: typing.Any, generator: typing.Any) -> None: ...
    def create(self, solution: typing.Any = ...) -> typing.Any: ...
    def generate(self, solution: typing.Any = ...) -> typing.Any: ...
