"""A hinge basis as the MARS and discriminant fixtures write it.

A fit's terms are its factors, ``column:direction`` in column order with the column one-based and
the intercept ``1``, and the cuts apart so they are compared as numbers; a fixture's are the same
terms written ``column:direction:cut``, joined by ``*`` within a term and by a space between them.
"""

from __future__ import annotations

import numpy as np


def hinge_terms(fit):
    start = fit["factor_start"]
    out = []
    for t in range(len(start) - 1):
        idx = range(start[t], start[t + 1])
        key = "*".join(f"{fit['factor_column'][i] + 1}:{fit['factor_dir'][i]}" for i in idx)
        out.append((key or "1", [float(fit["factor_cut"][i]) for i in idx]))
    return out


def fixture_hinge_terms(terms):
    out = []
    for term in terms.split(" "):
        if term == "1":
            out.append(("1", []))
            continue
        parts = [p.split(":") for p in term.split("*")]
        out.append(("*".join(f"{p[0]}:{p[1]}" for p in parts), [float(p[2]) for p in parts]))
    return out


def assert_hinge_terms(fit, terms, tolerance):
    got, want = hinge_terms(fit), fixture_hinge_terms(terms)
    assert [k for k, _ in got] == [k for k, _ in want]
    np.testing.assert_allclose([c for _, cs in got for c in cs],
                               [c for _, cs in want for c in cs], rtol=tolerance, atol=tolerance)
