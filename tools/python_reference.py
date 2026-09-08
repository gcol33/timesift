"""Write the Python reference pages of the pkgdown site from the Python sources.

Reads ``python/timesift/*.py`` with ``ast`` rather than importing it, so the pages can be
written without the compiled core on the machine, and emits one article per section of
``_pkgdown.yml``'s reference index, so the two languages are read in the same order.

    python tools/python_reference.py            write the pages
    python tools/python_reference.py --check    exit 1 if what is on disk differs
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "python" / "timesift"
DEST = ROOT / "vignettes" / "articles"

UNDOCUMENTED: list[str] = []

BANNER = "<!-- Written by tools/python_reference.py from the Python sources. Do not edit. -->"

# Each section is one article, and carries the symbols of the matching section of the R
# reference index. Every name in `timesift.__all__` belongs to exactly one of them; the script
# stops if one is missing or listed twice.
SECTIONS = (
    dict(
        slug="python-representation",
        title="Python: the representation",
        desc="What a representation is before any record has been read, and the array it becomes.",
        names=("native", "grain", "multigrain", "lookback", "grains", "lookbacks",
               "Representation", "Sift", "as_sift", "expand_sift", "auto_grains",
               "build_representation", "grain_matrix", "lookback_matrix", "coverage",
               "timesift_set", "calendar_channels", "bind_channels", "feature_matrix",
               "TimesiftMatrix", "TimesiftSet", "Coverage", "GRAINS", "STATS",
               "DAY_LEVEL_STATS"),
    ),
    dict(
        slug="python-split",
        title="Python: the split and the cells",
        desc="One fold map read by everything that scores, and the cells a score is defined on, "
             "computed with no model involved.",
        names=("cv", "grouped_cv", "Resampling", "as_resampling", "resolve_folds", "fold_map",
               "scorable_cells", "align_folds", "as_response", "Response", "Folds", "Cells",
               "PRESENCE_ABSENCE", "positive_weights"),
    ),
    dict(
        slug="python-fitting",
        title="Python: fitting",
        desc="The run from targets and series, the combiner over its candidates, and fitting "
             "across a set of grains on its own.",
        names=("timesift", "Timesift", "TimesiftSpec", "n_targets",
               "target_labels", "select_columns", "column_names", "summary", "candidate_table",
               "ensemble_row", "ensemble", "ensemble_fit", "ensemble_combine",
               "ensemble_weights", "EnsembleSpec", "Stack", "grain_ladder", "fit_learner",
               "select_grain", "Ladder", "Fit", "Selection"),
    ),
    dict(
        slug="python-learners",
        title="Python: learners",
        desc="The arms that ship, how they are trained, and the interface a learner of your own "
             "goes through.",
        names=("elasticnet", "stepwise", "forest", "mlp", "cnn", "rescnn", "Learner",
               "train_control", "TrainControl", "flatten"),
    ),
    dict(
        slug="python-scoring",
        title="Python: scoring and comparison",
        desc="The metrics, the paired contrast between two arms on matched cells, the inflation "
             "of a score read at its own best threshold, and what a fitted model read.",
        names=("tss", "roc_auc", "kappa_score", "cohen_kappa", "decision_threshold",
               "model_agreement", "score_predictions", "paired_contrast", "tss_inflation",
               "implied_skill", "occlusion"),
    ),
    dict(
        slug="python-extending",
        title="Python: extending",
        desc="The response head and the metric are registrations, never a fork of the fitting "
             "code.",
        names=("register_learner", "register_metric", "register_response", "learners", "metrics",
               "responses", "get_learner", "resolve_metric"),
    ),
    dict(
        slug="python-artifacts",
        title="Python: what crosses the boundary",
        desc="The three artifacts a split is carried in, and the digest that says two arrays are "
             "the same array.",
        names=("write_folds", "read_folds", "write_response", "read_response", "write_cells",
               "read_cells", "digest_array"),
    ),
)

OVERVIEW = """`timesift()` here and `timesift()` in R take the same two tables and do the same
thing: build every candidate representation, fit the learners that can read each one, score them
all on one set of held-out folds, and stack the out-of-fold predictions. The binning, the
statistics and the array assembly are `src/`, compiled into both languages and answering to
[the representation contract](contract.html), and the fixtures under `inst/spec/fixtures/` hold the
two to the same numbers.

```bash
pip install git+https://github.com/gcol33/timesift
```

```python
import timesift as ts

fit = ts.timesift(plots, logger, y="sp_*", id="plot_id", time="datetime",
                  models=[ts.elasticnet(), ts.forest()],
                  sift=ts.grains("day", "week", "month"))
print(ts.summary(fit))
```

`y`, `x` and `static` are selections over their own table: a column name, a list of names, a glob
such as `"sp_*"`, or a function of a name. The contract's last section says what each language
carries, so a difference between the two is a recorded decision.
"""


def public_symbols(tree: ast.Module) -> list[str]:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return [e.value for e in node.value.elts]
    raise SystemExit("no __all__ in python/timesift/__init__.py")


def annotation(node) -> str:
    return "" if node is None else ": " + ast.unparse(node)


def signature(node: ast.FunctionDef) -> str:
    """The call as written, one argument per line once it outgrows the column."""
    a = node.args
    out, defaults = [], list(a.defaults)
    positional = list(a.posonlyargs) + list(a.args)
    pad = [None] * (len(positional) - len(defaults)) + defaults
    for arg, default in zip(positional, pad):
        piece = arg.arg + annotation(arg.annotation)
        if default is not None:
            piece += ("=" if not arg.annotation else " = ") + ast.unparse(default)
        out.append(piece)
        if a.posonlyargs and arg is a.posonlyargs[-1]:
            out.append("/")
    if a.vararg:
        out.append("*" + a.vararg.arg + annotation(a.vararg.annotation))
    elif a.kwonlyargs:
        out.append("*")
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        piece = arg.arg + annotation(arg.annotation)
        if default is not None:
            piece += ("=" if not arg.annotation else " = ") + ast.unparse(default)
        out.append(piece)
    if a.kwarg:
        out.append("**" + a.kwarg.arg + annotation(a.kwarg.annotation))
    flat = "{}({})".format(node.name, ", ".join(out))
    if len(flat) <= 88:
        return flat
    joined = ",\n    ".join(out)
    return "{}(\n    {},\n)".format(node.name, joined)


def prose(doc: str | None) -> str:
    """The docstring as markdown: reST inline roles to code spans, indentation removed."""
    if not doc:
        return ""
    text = re.sub(r":(?:class|func|meth|attr|mod|data|obj):`~?([^`]+)`", r"`\1`", doc)
    text = text.replace("``", "`")
    lines = [line.rstrip() for line in text.strip("\n").splitlines()]
    body = [line for line in lines[1:] if line.strip()]
    indent = min((len(line) - len(line.lstrip()) for line in body), default=0)
    return "\n".join([lines[0].strip()] + [line[indent:] for line in lines[1:]]).strip()


def fields(node: ast.ClassDef) -> list[str]:
    return [
        "`{}`{}".format(item.target.id, annotation(item.annotation).replace(":", " -", 1))
        for item in node.body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
        and not item.target.id.startswith("_")
    ]


def methods(node: ast.ClassDef) -> list[ast.FunctionDef]:
    return [
        item for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not item.name.startswith("_")
    ]


def decorated(node: ast.ClassDef | ast.FunctionDef) -> str:
    names = [ast.unparse(d).split("(")[0] for d in node.decorator_list]
    return names[0] if names else ""


def collect() -> dict[str, tuple[str, ast.AST]]:
    """Every top-level definition and assignment of the package, by name."""
    found: dict[str, tuple[str, ast.AST]] = {}
    for path in sorted(SOURCE.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                found.setdefault(node.name, (path.name, node))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        found.setdefault(target.id, (path.name, node))
    return found


def render_function(node: ast.FunctionDef, level: str = "##", owner: str = "") -> list[str]:
    doc = ast.get_docstring(node)
    if not doc:
        UNDOCUMENTED.append(owner + node.name)
    if decorated(node) == "property":
        return ["{} `{}`".format(level, node.name), "", prose(doc), ""]
    return ["{} `{}()`".format(level, node.name), "",
            "```python", signature(node), "```", "", prose(doc), ""]


def render_class(node: ast.ClassDef) -> list[str]:
    out = ["## `{}`".format(node.name), ""]
    if not ast.get_docstring(node):
        UNDOCUMENTED.append(node.name)
    if decorated(node) == "dataclass":
        taken = [item.target.id for item in node.body
                 if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)]
        init = "{}({})".format(node.name, ", ".join(taken))
        if len(init) > 88:
            init = "{}(\n    {},\n)".format(node.name, ",\n    ".join(taken))
        out += ["```python", init, "```", ""]
    out += [prose(ast.get_docstring(node)), ""]
    named = fields(node)
    if named:
        out += ["Attributes:", ""] + ["- {}".format(f) for f in named] + [""]
    for item in methods(node):
        out += render_function(item, level="###", owner=node.name + ".")
    return out


def render_value(name: str, node: ast.Assign) -> list[str]:
    return ["## `{}`".format(name), "",
            "```python", "{} = {}".format(name, ast.unparse(node.value)), "```", ""]


def render(section: dict, found: dict) -> str:
    out = ["---", 'title: "{}"'.format(section["title"]), "---", "", BANNER, "",
           section["desc"], ""]
    for name in section["names"]:
        _, node = found[name]
        if isinstance(node, ast.ClassDef):
            out += render_class(node)
        elif isinstance(node, ast.Assign):
            out += render_value(name, node)
        else:
            out += render_function(node)
    return "\n".join(out).rstrip() + "\n"


def overview() -> str:
    out = ["---", 'title: "Python"', "---", "", BANNER, "", OVERVIEW, "## The pages", ""]
    for section in SECTIONS:
        out += ["[{}]({}.html)".format(section["title"].split(": ", 1)[1].capitalize(),
                                       section["slug"]),
                "\n: {}\n".format(section["desc"])]
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    exported = public_symbols(ast.parse((SOURCE / "__init__.py").read_text(encoding="utf-8")))
    listed = [name for section in SECTIONS for name in section["names"]]
    missing = sorted(set(exported) - set(listed))
    extra = sorted(set(listed) - set(exported))
    twice = sorted({name for name in listed if listed.count(name) > 1})
    if missing or extra or twice:
        print("SECTIONS and timesift.__all__ disagree.", file=sys.stderr)
        for label, names in (("not in any section", missing), ("not exported", extra),
                             ("in two sections", twice)):
            if names:
                print("  {}: {}".format(label, ", ".join(names)), file=sys.stderr)
        return 1

    found = collect()
    pages = {"python.Rmd": overview()}
    for section in SECTIONS:
        pages[section["slug"] + ".Rmd"] = render(section, found)

    stale = []
    for name, text in pages.items():
        path = DEST / name
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == text:
            continue
        if args.check:
            stale.append(name)
        else:
            path.write_text(text, encoding="utf-8", newline="\n")
            print("wrote vignettes/articles/{}".format(name))

    if stale:
        print("Out of date, rerun tools/python_reference.py: " + ", ".join(sorted(stale)),
              file=sys.stderr)
        return 1
    if UNDOCUMENTED:
        print("No docstring: " + ", ".join(sorted(set(UNDOCUMENTED))), file=sys.stderr)
    if args.check:
        print("The Python reference matches the sources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
