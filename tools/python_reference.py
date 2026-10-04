"""Write the Python reference pages of the pkgdown site from the Python sources.

Reads ``python/timesift/*.py`` with ``ast`` rather than importing it, so the pages can be
written without the compiled core on the machine. The reference is laid out as the R one is: one
page per section of ``_pkgdown.yml``'s reference index, carrying that section's title and
description, and an index page listing every section and every name, so the two languages are
read in the same order and under the same headings.

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
PKGDOWN = ROOT / "_pkgdown.yml"

UNDOCUMENTED: list[str] = []

BANNER = "<!-- Written by tools/python_reference.py from the Python sources. Do not edit. -->"
INDEX = "python-reference"

# One entry per section of the R reference index, keyed by its title and in its order; the title
# and the description are read from `_pkgdown.yml`, so a section renamed there is renamed here or
# the script stops. Every name in `timesift.__all__` belongs to exactly one section; the script
# stops if one is missing or listed twice.
SECTIONS = (
    ("The one call", "python-one-call",
     ("timesift", "Timesift", "TimesiftSpec", "summary", "candidate_table", "procedure_table",
      "plot", "project", "range_change", "RangeChange", "pseudo_absences", "select_columns",
      "column_names", "n_targets", "target_labels")),
    ("Representations", "python-representations",
     ("native", "grain", "multigrain", "lookback", "grains", "lookbacks", "Representation",
      "Sift", "as_sift", "expand_sift", "auto_grains", "build_representation")),
    ("Learners", "python-learners",
     ("elasticnet", "stepwise", "forest", "tree", "boosting", "maxnet", "envelope", "mars",
      "discriminant", "additive", "hierarchical", "mlp", "cnn", "rescnn", "train_control",
      "TrainControl", "Learner", "flatten", "register_learner", "get_learner", "learners", "tune",
      "Tuned", "register_tuning", "tunings")),
    ("The split and the cells", "python-split",
     ("cv", "grouped_cv", "block_cv", "env_cv", "Resampling", "as_resampling", "resolve_folds",
      "fold_map", "scorable_cells", "align_folds", "Folds", "Cells")),
    ("Combining the candidates", "python-combining",
     ("ensemble", "ensemble_fit", "ensemble_combine", "ensemble_spread", "SPREAD_STATISTICS",
      "ensemble_weights", "EnsembleSpec", "Stack")),
    ("Scoring and comparison", "python-scoring",
     ("tss", "roc_auc", "average_precision", "kappa_score", "table_metric", "boyce_index",
      "regression_metric", "ordinal_metric", "cohen_kappa", "decision_threshold",
      "model_agreement", "score_predictions", "paired_contrast", "grain_contrasts",
      "tss_inflation", "implied_skill", "occlusion", "response_curve", "ResponseCurve")),
    ("The arrays themselves", "python-arrays",
     ("grain_matrix", "lookback_matrix", "coverage", "Coverage", "timesift_set",
      "calendar_channels", "bind_channels", "feature_matrix", "TimesiftMatrix", "TimesiftSet",
      "GRAINS", "STATS", "DAY_LEVEL_STATS")),
    ("One grain at a time", "python-ladder",
     ("grain_ladder", "fit_learner", "Fit", "Ladder", "select_grain", "Selection")),
    ("Extending", "python-extending",
     ("register_response", "responses", "as_response", "Response", "PRESENCE_ABSENCE",
      "CONTINUOUS", "ABUNDANCE", "ORDINAL", "COUNT", "positive_weights", "register_metric",
      "metrics", "resolve_metric")),
    ("What crosses the boundary", "python-artifacts",
     ("write_folds", "read_folds", "write_response", "read_response", "write_cells",
      "read_cells", "digest_array")),
    ("A record to test on", "python-simulate",
     ("simulate_records", "Simulation")),
)

# Sections of the R index that document the R package itself and have no Python counterpart.
R_ONLY = ("The package",)


def r_sections() -> list[tuple[str, str]]:
    """The title and description of every section of `_pkgdown.yml`'s reference index.

    The file is read for the one shape the index is written in: a top-level ``reference:`` list
    whose items open with ``- title:`` and may carry a block-literal ``desc: |``.
    """
    out, inside, desc = [], False, None
    for line in PKGDOWN.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if not line.startswith((" ", "-")):
            inside = line.rstrip() == "reference:"
            desc = None
            continue
        if not inside:
            continue
        if line.startswith("- title:"):
            out.append([line.split(":", 1)[1].strip(), ""])
            desc = None
        elif line.startswith("  desc:"):
            rest = line.split(":", 1)[1].strip()
            desc = [] if rest == "|" else None
            if rest != "|":
                out[-1][1] = rest
        elif desc is not None and line.startswith("    "):
            desc.append(line.strip())
            out[-1][1] = " ".join(desc)
        else:
            desc = None
    return [(title, text) for title, text in out]


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


def first_sentence(doc: str | None) -> str:
    """The docstring's opening paragraph on one line, as the index lists it."""
    text = prose(doc)
    return " ".join(text.split("\n\n", 1)[0].split()) if text else ""


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


def shown_name(name: str, node: ast.AST) -> str:
    """A name as its heading and the index show it: a function called, a class or value bare."""
    return name + "()" if isinstance(node, ast.FunctionDef) else name


def heading(level: str, text: str, anchor: str) -> str:
    """A heading with its anchor written out, so a function and a class differing only in case
    (`timesift()` and `Timesift`) are linked to separately."""
    return "{} `{}` {{#{}}}".format(level, text, anchor)


def render_function(node: ast.FunctionDef, level: str = "##", owner: str = "") -> list[str]:
    doc = ast.get_docstring(node)
    if not doc:
        UNDOCUMENTED.append(owner + node.name)
    anchor = owner + node.name
    if decorated(node) == "property":
        return [heading(level, node.name, anchor), "", prose(doc), ""]
    return [heading(level, node.name + "()", anchor), "",
            "```python", signature(node), "```", "", prose(doc), ""]


def render_class(node: ast.ClassDef) -> list[str]:
    out = [heading("##", node.name, node.name), ""]
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
    return [heading("##", name, name), "",
            "```python", "{} = {}".format(name, ast.unparse(node.value)), "```", ""]


def page_title(title: str) -> str:
    return "Python: " + title[0].lower() + title[1:]


def render(title: str, desc: str, names, found: dict) -> str:
    out = ["---", 'title: "{}"'.format(page_title(title)), "---", "", BANNER, "",
           desc, "", "[All of the Python reference]({}.html)".format(INDEX), ""]
    for name in names:
        _, node = found[name]
        if isinstance(node, ast.ClassDef):
            out += render_class(node)
        elif isinstance(node, ast.Assign):
            out += render_value(name, node)
        else:
            out += render_function(node)
    return "\n".join(out).rstrip() + "\n"


def render_index(sections, found: dict) -> str:
    out = ["---", 'title: "Python reference"', "---", "", BANNER, "",
           "The Python package's functions, classes and values, under the sections of "
           "[the R reference](../reference/index.html) and in its order. "
           "[Get started](python.html) runs them on a simulated record.", ""]
    for title, desc, slug, names in sections:
        out += ["## [{}]({}.html)".format(title, slug), "", desc, ""]
        for name in names:
            _, node = found[name]
            doc = ast.get_docstring(node) if not isinstance(node, ast.Assign) else None
            out += ["[`{}`]({}.html#{})".format(shown_name(name, node), slug, name)]
            out += [": {}".format(first_sentence(doc)) if doc else ": A value.", ""]
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    r = [(t, d) for t, d in r_sections() if t not in R_ONLY]
    ours = [title for title, _, _ in SECTIONS]
    if [t for t, _ in r] != ours:
        print("SECTIONS and the reference index of _pkgdown.yml disagree.\n"
              "  R:      " + " | ".join(t for t, _ in r) + "\n"
              "  Python: " + " | ".join(ours), file=sys.stderr)
        return 1
    missing_desc = [t for t, d in r if not d]
    if missing_desc:
        print("A reference section of _pkgdown.yml carries no desc: " + ", ".join(missing_desc),
              file=sys.stderr)
        return 1

    exported = public_symbols(ast.parse((SOURCE / "__init__.py").read_text(encoding="utf-8")))
    listed = [name for _, _, names in SECTIONS for name in names]
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
    sections = [(title, desc, slug, names)
                for (title, desc), (_, slug, names) in zip(r, SECTIONS)]
    pages = {INDEX + ".Rmd": render_index(sections, found)}
    for title, desc, slug, names in sections:
        pages[slug + ".Rmd"] = render(title, desc, names, found)

    # A page this script wrote under a name it no longer writes is removed with it.
    orphans = [p.name for p in DEST.glob("*.Rmd")
               if p.name not in pages and BANNER in p.read_text(encoding="utf-8")]

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
    for name in orphans:
        if args.check:
            stale.append(name)
        else:
            (DEST / name).unlink()
            print("removed vignettes/articles/{}".format(name))

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
