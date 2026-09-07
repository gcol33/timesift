"""The cross-language digest of a representation, defined in ``inst/spec/representation.md``.

Values are traversed in the array's own order, unit fastest, then bin, then channel; each is
formatted to twelve decimal places, joined with a line feed and terminated with one, and the UTF-8
bytes of that are hashed. The line ending is a line feed on every platform, so a digest produced on
one machine and checked on another has to agree.

The array has to be finite. ``.12f`` writes an infinity as ``inf`` here and as ``Inf`` in R, and R
tells a missing value apart from a not-a-number where there is one spelling for both here, so a
digest over such an array would say something about the language rather than about the
representation. A representation never holds one: a reading that is not a finite number is refused
where the record is read.
"""

from __future__ import annotations

import hashlib

import numpy as np


def digest_array(values) -> str:
    """MD5 of the representation, byte-exactly as the spec defines it."""
    if hasattr(values, "values"):
        values = values.values
    flat = np.asarray(values, dtype=np.float64).flatten(order="F")
    bad = int((~np.isfinite(flat)).sum())
    if bad:
        raise ValueError(f"{bad} of {flat.size} values are not finite. A digest is defined for a "
                         "finite array: the two languages spell an infinity and a missing value "
                         "differently, so a digest over one would compare the language rather "
                         "than the representation.")
    body = "".join(f"{v:.12f}\n" for v in flat)
    return hashlib.md5(body.encode("utf-8")).hexdigest()
