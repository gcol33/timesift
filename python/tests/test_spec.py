"""The spec's account of what each language carries, held against what this side exports.

The last section of ``inst/spec/representation.md`` records every name on both sides, every name
shaped differently, and every name present in one language only. A public name it does not
record, or one it records on the wrong side, fails here, so the tables cannot fall behind the
exports. The R suite reads the same section against its own NAMESPACE.
"""

from __future__ import annotations

import re
from pathlib import Path

import timesift

SPEC = Path(__file__).resolve().parents[2] / "inst" / "spec" / "representation.md"
NAME = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)(?:\(\))?`")


def section() -> list[str]:
    lines = SPEC.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("## What each language carries"))
    return lines[start:]


def names(lines) -> set[str]:
    return {m for line in lines for m in NAME.findall(line)}


def table_names(lines, header: str, column: int = 0) -> set[str]:
    """The bare names in one column of the table whose header line is ``header``. A name written
    with an argument, such as ``elasticnet(s =)``, names an argument and is left out."""
    start = lines.index(header)
    cells = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells.append(line.lstrip("|").split("|")[column])
    return names(cells)


def test_every_public_name_is_recorded_in_the_spec():
    missing = set(timesift.__all__) - names(section())
    assert not missing, f"public but not in the spec: {sorted(missing)}"


def test_a_name_recorded_on_both_sides_is_public_here():
    shared = table_names(section(), "| concept | the name, on both sides |", column=1)
    shared.discard("n_inner")
    missing = shared - set(timesift.__all__)
    assert not missing, f"recorded on both sides but not public: {sorted(missing)}"


def test_a_name_recorded_on_one_side_is_on_that_side():
    lines = section()
    r_only = table_names(lines, "| in R only | why |")
    leaked = r_only & set(timesift.__all__)
    assert not leaked, f"recorded as R only but public here: {sorted(leaked)}"
    python_only = table_names(lines, "| in Python only | what it is |")
    missing = python_only - set(timesift.__all__)
    assert not missing, f"recorded as Python only but not public: {sorted(missing)}"
