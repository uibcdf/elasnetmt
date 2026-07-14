"""
ElastNetMT
A toolkit for Elastic Network Models in the MolSysSuite ecosystem.
"""

# versioningit
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("elastnetmt")
except PackageNotFoundError:
    # Package is not installed
    try:
        from ._version import __version__
    except ImportError:
        __version__ = "0.1.0"

# 1. SMonitor Configuration
from smonitor.integrations import ensure_configured as _ensure_smonitor_configured
from elastnetmt._private.smonitor import PACKAGE_ROOT as _SMONITOR_PACKAGE_ROOT

_ensure_smonitor_configured(_SMONITOR_PACKAGE_ROOT)

# 2. Units (PyUnitWizard)
from ._pyunitwizard import puw as pyunitwizard

# 3. Expose Exceptions and Diagnostics
from ._private.smonitor import (
    ElastNetMTError,
    ElastNetMTWarning,
    ArgumentError,
    InternalAlgorithmError,
    LibraryNotFoundError,
    warn,
)

# 4. Import Core Models
from .model import (
    ElasticNetworkModel,
    GaussianNetworkModel,
    AnisotropicNetworkModel,
)

# 5. Support ArgDigest and DepDigest
from argdigest import arg_digest
from depdigest import dep_digest

__all__ = [
    'pyunitwizard',
    'GaussianNetworkModel',
    'AnisotropicNetworkModel',
    'ElasticNetworkModel',
    'arg_digest',
    'dep_digest',
]
