"""Public logging facade.

Keeps the historical ``get_logger``/``set_level``/``get_log_level``/
``get_current_level`` surface — 74 callers import ``get_logger`` from here — and
re-exports the plumbing helpers from :mod:`gnomad_meb.utils.logging`, where the
console, handler wiring, JVM interception and hail bridge actually live.

The module-level logger is still built at import for backward compatibility
(``import gnomad_meb.logging; import gnomad_meb.hail`` works without any explicit
setup). :func:`configure_logging` is the setup seam the CLI calls once it knows
the settings.
"""

from __future__ import annotations

import logging
from typing import Literal

from gnomad_meb.utils.logging import (
    configure_logging,  # noqa: F401  (re-exported public surface)
    get_console,  # noqa: F401
    get_default_handler,
    intercept_jvm_output,  # noqa: F401
    HailLogHandler,  # noqa: F401
    attach_hail_logging,  # noqa: F401
)

_logger = logging.getLogger("gnomad_meb")
_logger.setLevel(logging.ERROR)
_logger.addHandler(get_default_handler())
_logger.propagate = False

_log_level = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def get_logger() -> logging.Logger:
    global _logger
    return _logger


def set_level(level: Literal["debug", "info", "warning", "error", "critical"]):
    global _logger
    global _log_level
    _logger.setLevel(_log_level[level])


def get_log_level() -> dict[str, int]:
    global _log_level
    return _log_level


def get_current_level() -> Literal["debug", "info", "warning", "error", "critical"]:
    global _logger
    lvl = _logger.level
    if lvl > 40:
        return "critical"
    elif lvl > 30:
        return "error"
    elif lvl > 20:
        return "warning"
    elif lvl > 10:
        return "info"
    else:
        return "debug"
