"""Constants for the logging plumbing.

Shared between :mod:`gnomad_meb.utils.logging` and its tests: a module's public
surface is its contract, so these live here under public names rather than as
underscore-prefixed globals.
"""

from __future__ import annotations

import logging
import re

#: A Java log **record header** looks like ``<timestamp> <logger>: <LEVEL>:
#: <message>`` (Hail's log4j pattern ``%d{...} %-5p %c{1}: %m%n``). ``%m`` may
#: itself contain newlines (Hail dumps whole IR trees on one record), so the
#: physical line this matches is only the *first* line of the record; the
#: handler groups every following non-matching line onto it. The timestamp is
#: optional for robustness against unframed writers.
HAIL_LOG_LINE_PATTERN = re.compile(
    r"^(?:(?P<timestamp>[0-9-]+\s[0-9:\.]+)\s*)?(?P<logger>[^:]+):\s*(?P<level>TRACE|DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL):\s*(?P<message>.*)$"
)

#: Java level names (as log4j spells them) to stdlib levels.
HAIL_LOG_LEVELS = {
    "TRACE": logging.DEBUG,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "FATAL": logging.CRITICAL,
}
