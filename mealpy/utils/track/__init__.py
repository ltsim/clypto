#!/usr/bin/env python
# Created by "LTSIM" at 5:28, 25/05/2021 -----------%
#       Email: tsim@cucei.udg.mx                    %
#       Github: https://github.com/ltsim            %
# --------------------------------------------------%
import typing

A = typing.TypeVar('A')


class HistoryProtocol(typing.Protocol[A]):
    global_best: list[A]
    current_best: list[A]
    epoch_time: list[A]
    global_best_fit: list[A]
    current_best_fit: list[A]
    population: list[A]
    diversity: list[A]
    exploitation: list[A]
    exploration: list[A]
    global_worst: list[A]
    current_worst: list[A]
    epoch: int

    def store_initial_best_worst(self, best_agent: A, worst_agent: A) -> None:
        ...

    def get_global_repeated_times(self, epsilon: float) -> int:
        ...
