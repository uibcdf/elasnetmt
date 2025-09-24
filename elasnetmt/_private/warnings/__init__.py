from .user_elasnetmt_warning import UserElasNetMTWarning
from .elasnetmt_deprecation_warning import ElasNetMTDeprecationWarning
from .no_experimental_b_factors_warning import NoExperimentalBFactorsWarning

from typing import Iterable, Type
import warnings

__all__ = ['UserElasNetMTWarning',
           'ElasNetMTDeprecationWarning',
           'NoExperimentalBFactorsWarning',
           'warn',
           'warn_once']

def warn(
    message_or_warning: str | Warning,
    category: Type[Warning] | None = None,
    *,
    stacklevel: int = 2,
) -> None:
    if isinstance(message_or_warning, Warning):
        warnings.warn(message_or_warning, stacklevel=stacklevel)
    else:
        warnings.warn(message_or_warning, category or UserElasNetMTWarning, stacklevel=stacklevel)


__WARNED_ONCE_CACHE__: set[tuple[Type[Warning], str]] = set()

def warn_once(
    message_or_warning: str | Warning,
    category: Type[Warning] | None = None,
    *,
    stacklevel: int = 2,
) -> None:
    if isinstance(message_or_warning, Warning):
        msg, cat = str(message_or_warning), type(message_or_warning)
    else:
        msg, cat = message_or_warning, category or UserElasNetMTWarning

    key = (cat, msg)
    if key in __WARNED_ONCE_CACHE__:
        return
    __WARNED_ONCE_CACHE__.add(key)
    warnings.warn(message_or_warning, cat, stacklevel=stacklevel)



