#!/usr/bin/env python
# Created by "Thieu" at 16:19, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

__version__ = "2026b0"

import functools
import inspect
import sys

from clypto.collection.bio_based import (
    BBO,
    BBOA,
    BMO,
    EOA,
    SBO,
    SMA,
    SOA,
    SOS,
    TPO,
    TSA,
    VCS,
    WHO,
    BCO,
    EAO,
    SFOA,
)
from clypto.collection.bio_based import IWO
from clypto.collection.evolutionary_based import ES, DE, SHADE, MA, EP, GA, BWO, CRO
from clypto.collection.evolutionary_based import FPA
from clypto.collection.game_based import THRO
from clypto.collection.human_based import (
    SSDO,
    CA,
    ICA,
    AFT,
    CHIO,
    SARO,
    BSO,
    HBO,
    CDDO,
    GSKA,
    SPBO,
    HCO,
    QSA,
    WarSO,
    BRO,
    TLO,
    TOA,
    FBIO,
    DOA,
    LCO,
)
from clypto.collection.math_based import (
    AOA,
    CEM,
    CGO,
    GBO,
    HC,
    INFO,
    PSS,
    RUN,
    SCA,
    SHIO,
    TS,
)
from clypto.collection.math_based import CircleSA
from clypto.collection.music_based import HS
from clypto.collection.physics_based import (
    ArchOA,
    EFO,
    EO,
    EVO,
    FLA,
    HGSO,
    MVO,
    NRO,
    SA,
    TWO,
    WDO,
    ESO,
    SOO,
    MSO,
)
from clypto.collection.physics_based import RIME, ASO, CDO
from clypto.collection.sota_based import LSHADEcnEpSin, IMODE
from clypto.collection.swarm_based import (
    ABC,
    ACOR,
    AVOA,
    BES,
    BFO,
    COA,
    DMOA,
    DO,
    FA,
    GJO,
    GWO,
    HBA,
    MPA,
    MSA,
    MShOA,
    NGO,
    OOA,
    PFA,
    SCSO,
    SeaHO,
    ServalOA,
    SHO,
    SRSR,
    SSpiderO,
    STO,
    TDO,
    WaOA,
    ZOA,
    FDO,
)
from clypto.collection.swarm_based import (
    AO,
    JA,
    EHO,
    NMRA,
    PSO,
    CoatiOA,
    BA,
    TSO,
    GOA,
    MFO,
    SLO,
    ARO,
    SquirrelSA,
    FFA,
    POA,
    ESOA,
    GTO,
    CSO,
    SSO,
    FOA,
    HHO,
    SFO,
    WOA,
    ALO,
    CSA,
    MGO,
    FOX,
    FFO,
    MRFO,
    SMO,
    SSpiderA,
    BSA,
    HGS,
    BeesA,
    EPC,
    AGTO,
    SSA,
)
from clypto.collection.system_based import AEO, GCO
from clypto.collection.system_based import WCA
from clypto.optimizer.classic import Optimizer
from clypto.utils import Problem
from clypto.utils.problem import Problem
from clypto.utils.space import (
    IntegerVar,
    FloatVar,
    StringVar,
    BinaryVar,
    BoolVar,
    CategoricalVar,
    SequenceVar,
    PermutationVar,
    TransferBinaryVar,
    TransferBoolVar,
)
from clypto.utils.termination import Termination

__EXCLUDE_MODULES = ["__builtins__", "current_module", "inspect", "sys"]


@functools.cache
def get_all_optimizers(verbose=False):
    """
    Get all available optimizer classes in clypto library

    Args:
        verbose (bool): whether to print the optimizer information

    Returns:
        dict_optimizers (dict): key is the string optimizer class name, value is the actual optimizer class
    """
    cls = {}

    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.ismodule(obj) and (name not in __EXCLUDE_MODULES):
            for cls_name, cls_obj in inspect.getmembers(obj):
                if (
                        inspect.isclass(cls_obj)
                        and issubclass(cls_obj, Optimizer)
                        and cls_obj is not Optimizer
                ):
                    cls[cls_name] = cls_obj

    if verbose:
        for name, optimizer in cls.items():
            print(f"Optimizer: {name} - {optimizer}")

    return cls


def get_optimizer_by_class(class_name: str, verbose=False):
    """
    Get an optimizer class by its class name

    Args:
        class_name (str): the classname of the optimizer (e.g, C_PSO, OriginalGA), don't pass the module name (e.g, PSO, GA)
        verbose (bool): whether to print the optimizer information

    Returns:
        optimizer (Optimizer): the actual optimizer class or None if the classname is not supported
    """
    try:
        all_optimizers = get_all_optimizers(verbose=verbose)
        return all_optimizers[class_name]
    except KeyError:
        print(f"clypto doesn't support optimizer named: {class_name}.\n")
        return None


def get_optimizer_by_name(name: str, verbose=False):
    """
    Get an optimizer class by name

    Args:
        name (str): the classname of the optimizer (e.g, OriginalGA, OriginalWOA), don't pass the module name (e.g, ABC, WOA, GA)
        verbose (bool): whether to print the optimizer information

    Returns:
        dict_optimizers (dict): key is the string optimizer class name, value is the actual optimizer class
    """
    cls = {}
    flag = False

    for module_name, obj in inspect.getmembers(sys.modules[__name__]):
        if (
                inspect.ismodule(obj)
                and (name not in __EXCLUDE_MODULES)
                and (module_name == name)
        ):
            flag = True
            for cls_name, cls_obj in inspect.getmembers(obj):
                if inspect.isclass(cls_obj) and issubclass(cls_obj, Optimizer):
                    cls[cls_name] = cls_obj
    if verbose:
        if not flag:
            print(f"clypto doesn't support optimizer named: {name}.\n")
            return None

        print(f"Found algorithm: {name}, the supported variants are:")

        for name, optimizer in cls.items():
            print(f"Optimizer: {name} - {optimizer} - {optimizer()}")

    return cls


__all__ = [
    "Problem", "Optimizer",
    "IntegerVar",
    "FloatVar",
    "StringVar",
    "BinaryVar",
    "BoolVar",
    "CategoricalVar",
    "SequenceVar",
    "PermutationVar",
    "TransferBinaryVar",
    "TransferBoolVar",
]
