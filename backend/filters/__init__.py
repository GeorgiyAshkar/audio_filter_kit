"""Filter modules for the DAS Speech Filter Explorer backend.

This subpackage aggregates all available filter classes.  Filters are
automatically discovered by :func:`backend.core.get_filter_classes`,
which scans this package for subclasses of
:class:`backend.core.FilterBase`.  When adding a new filter module,
ensure that the module defines at least one subclass of
:class:`backend.core.FilterBase` so it can be discovered.

This file marks the directory as a Python package and may also
explicitly import known filters for static analyzers or to avoid
package isolation issues.
"""

# Import built-in and advanced filters so that static importers see
# them as part of this package.  The dynamic discovery mechanism in
# backend.core will still find any filter classes present in these
# modules even if they are not imported here.  However, importing
# them here helps some tools recognize that these modules are part
# of this package.

from .builtins import *  # noqa: F401,F403
from .advanced import *  # noqa: F401,F403

__all__ = []  # names will be populated dynamically by get_filter_classes