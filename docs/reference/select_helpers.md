# Choose columns by name, by prefix or by type

`y`, `x` and `static` in
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
are tidyselect expressions, so a response spread over a hundred columns
is named once, as `y = starts_with("sp_")`, and the value columns of a
series carrying several sensors are named the same way. These are
tidyselect's own helpers, re-exported so that attaching timesift is
enough to reach them and a session attaching both packages still has one
implementation of each.
