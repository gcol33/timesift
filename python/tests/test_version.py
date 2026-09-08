"""One version, and `DESCRIPTION` carries it.

`pyproject.toml` reads the version from `DESCRIPTION` at build time and `timesift.__version__`
reads it back off the installed distribution, so a wheel and a tarball cut from one commit carry
the same string. This asserts the two ends of that meet, which a bump that reaches one file and
not the others would otherwise only show on the index.
"""

from __future__ import annotations

import re
from pathlib import Path

import timesift

DESCRIPTION = Path(__file__).resolve().parents[2] / "DESCRIPTION"


def test_the_installed_version_is_the_one_description_declares():
    declared = re.search(r"(?im)^Version: *(\S+)", DESCRIPTION.read_text(encoding="utf-8"))
    assert declared is not None
    assert timesift.__version__ == declared.group(1)

