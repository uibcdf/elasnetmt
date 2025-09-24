"""
ElasNetMT
This must be a short description of the project
"""

# versioningit
from ._version import __version__

def __print_version__():
    print("ElasNetMT version " + __version__)

from . import config
config.setup_logging(level="WARNING", capture_warnings=True, simplify_warning_format=True)

from ._pyunitwizard import pyunitwizard

from . import model

# With the following list sphinx can document de methods in the api section without adding the
# module files names explicitly:

__all__ = []

