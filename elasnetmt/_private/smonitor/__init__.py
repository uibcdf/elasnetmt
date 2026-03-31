import os
from pathlib import Path
from smonitor.integrations.diagnostic import (
    CatalogException,
    CatalogWarning,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent

class ElasNetMTError(CatalogException):
    """Base error for ElasNetMT."""
    pass

class ElasNetMTWarning(CatalogWarning):
    """Base warning for ElasNetMT."""
    pass

class ArgumentError(ElasNetMTError):
    """Error in function arguments."""
    pass

class InternalAlgorithmError(ElasNetMTError):
    """Error in internal calculation logic."""
    pass

class LibraryNotFoundError(ElasNetMTError):
    """Error when a required library is missing."""
    pass

def warn(message, code=None, **kwargs):
    import warnings
    warnings.warn(message, ElasNetMTWarning)
