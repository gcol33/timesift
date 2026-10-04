"""Write the Python articles of the pkgdown site by running them.

An article is written by hand under ``python/articles/`` as markdown with ``python`` code blocks.
Every block is run in one namespace per article, in order, the way a knitted R vignette runs its
chunks, and the article is written to ``vignettes/articles/`` with what each block printed under
every statement: its standard output, and its value where it is an expression that has one.
A figure a block leaves open is saved as SVG beside the article and drawn under the block.

A block's info string carries options after the language, as ``key=value`` pairs:

    ```python error=true          the block must raise, and the error is shown as its output
    ```python fig-alt="..."       the alternative text of the figure the block draws

The written article records the digest of the source it was run from, so ``--check`` says
whether every article is current without running anything, and without the compiled core:

    python tools/python_articles.py            run every article and write it
    python tools/python_articles.py NAME       run python/articles/NAME.md alone
    python tools/python_articles.py --check    exit 1 if an article is older than its source
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import html
import io
import os
import shlex
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "python" / "articles"
DEST = ROOT / "vignettes" / "articles"

BANNER = ("<!-- Written by tools/python_articles.py from python/articles/{name}.md "
          "(md5 {digest}). Do not edit. -->")
PROMPT = "#> "


def digest(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def split_front_matter(text: str) -> tuple[str, str]:
    """The YAML header, delimiters included, and the body after it."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise SystemExit("an article opens with a YAML header between --- lines")
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "".join(lines[:i + 1]), "".join(lines[i + 1:])
    raise SystemExit("the YAML header is not closed")


def parts(body: str) -> list[tuple[str, str, dict]]:
    """The body as ("prose", text, {}) and ("code", source, options) in order.

    A fence opens on a line starting with three backticks and closes on the next line that is
    three backticks alone; only fences whose language is ``python`` are run, every other one is
    carried through as prose.
    """
    out, prose, lines, i = [], [], body.splitlines(keepends=True), 0
    while i < len(lines):
        line = lines[i]
        info = line.strip()[3:].strip() if line.lstrip().startswith("```") else None
        words = shlex.split(info) if info else []
        if words and words[0] == "python":
            options = dict(w.split("=", 1) for w in words[1:])
            end = next((j for j in range(i + 1, len(lines)) if lines[j].strip() == "```"), None)
            if end is None:
                raise SystemExit(f"the python block opening on line {i + 1} is not closed")
            out.append(("prose", "".join(prose), {}))
            out.append(("code", "".join(lines[i + 1:end]), options))
            prose, i = [], end + 1
        else:
            prose.append(line)
            i += 1
    out.append(("prose", "".join(prose), {}))
    return [p for p in out if p[0] == "code" or p[1]]


def run_statement(stmt: ast.stmt, shown_value: bool, namespace: dict) -> None:
    """Run one top-level statement, printing the value of an expression the way a console does."""
    if isinstance(stmt, ast.Expr):
        value = eval(compile(ast.Expression(stmt.value), "<article>", "eval"), namespace)
        if shown_value and value is not None:
            print(value if isinstance(value, str) else repr(value))
    else:
        exec(compile(ast.Module([stmt], type_ignores=[]), "<article>", "exec"), namespace)


def run_block(source: str, namespace: dict, expect_error: bool) -> list[str]:
    """Run one block statement by statement and return its lines with each statement's output
    under it, as a knitted chunk shows them.

    An expression's value is shown unless its line ends in a semicolon. Lines between two
    statements, comments and blank lines, travel with the statement after them.
    """
    lines = source.rstrip("\n").splitlines()
    out, start, raised = [], 0, False
    for stmt in ast.parse(source).body:
        end = stmt.end_lineno
        out += lines[start:end]
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            try:
                run_statement(stmt, not lines[end - 1].rstrip().endswith(";"), namespace)
            except Exception as error:  # noqa: BLE001 -- the error is the output being shown
                if not expect_error:
                    sys.stdout = sys.__stdout__
                    traceback.print_exc()
                    raise SystemExit(f"a block raised:\n{source}")
                print(f"{type(error).__name__}: {error}")
                raised = True
        out += [(PROMPT + line).rstrip() for line in shown.getvalue().rstrip("\n").splitlines()]
        start = end
        if raised:
            break
    if expect_error and not raised:
        raise SystemExit(f"a block marked error=true ran without raising:\n{source}")
    return out + lines[start:]


def take_figures(name: str, count: int, alt: str | None) -> dict[str, str]:
    """Every figure left open, as SVG text keyed by its path beside the article, closed after."""
    if "matplotlib.pyplot" not in sys.modules:
        return {}
    import matplotlib.pyplot as plt
    # The SVG's element ids are drawn from a salt, a fresh random one unless it is set, and a
    # figure that changes on every build would change the site's committed build with it.
    plt.rcParams["svg.hashsalt"] = name
    out = {}
    for number in plt.get_fignums():
        count += 1
        figure = plt.figure(number)
        recolour(figure)
        drawn = io.StringIO()
        figure.savefig(drawn, format="svg", metadata={"Date": None}, transparent=True)
        out[f"{name}_files/figure-{count}.svg"] = drawn.getvalue().replace(INK, "currentColor")
    plt.close("all")
    return out


# The text, spines and ticks of a figure are drawn in this colour and written out as
# `currentColor`, so a figure reads in the page's own text colour under the light and the dark
# theme alike, as the site's R figures do. No figure draws in it on purpose.
INK = "#010203"


def recolour(figure) -> None:
    """Put a figure's text, spines, ticks and legend frame in the page's colour, not black."""
    for text in figure.findobj(lambda a: hasattr(a, "get_text") and hasattr(a, "set_color")):
        if text.get_color() in ("black", "#000000", "k", (0.0, 0.0, 0.0, 1.0)):
            text.set_color(INK)
    for ax in figure.axes:
        for spine in ax.spines.values():
            spine.set_edgecolor(INK)
        ax.tick_params(colors=INK, which="both")
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_alpha(0)


def render(name: str) -> tuple[str, dict[str, str]]:
    """The article and its figures, nothing written until every block has run."""
    text = (SOURCE / f"{name}.md").read_text(encoding="utf-8")
    header, body = split_front_matter(text)

    os.environ.setdefault("MPLBACKEND", "Agg")
    namespace: dict = {"__name__": "__article__"}
    figures: dict[str, str] = {}
    out = []
    for kind, content, options in parts(body):
        if kind == "prose":
            out.append(content)
            continue
        lines = run_block(content, namespace, options.get("error") == "true")
        out.append("```python\n" + "\n".join(lines) + "\n```\n")
        drawn = take_figures(name, len(figures), options.get("fig-alt"))
        figures.update(drawn)
        if drawn:
            # An image alone in a paragraph is a captioned figure to pandoc, and the alternative
            # text is not a caption, so the image is written as HTML, as knitr writes one.
            alt = html.escape(options.get("fig-alt", ""), quote=True)
            out.append("\n" + "\n\n".join(f'<img src="{path}" alt="{alt}" />' for path in drawn)
                       + "\n")

    if figures:
        resources = "resource_files:\n" + "".join(f"  - {path}\n" for path in figures)
        header = header.rstrip("\n")[:-3] + resources + "---\n"
    banner = BANNER.format(name=name, digest=digest(text))
    return header + "\n" + banner + "\n" + "".join(out).rstrip("\n") + "\n", figures


def check_install() -> None:
    """Stop unless the timesift being imported is the one in this tree.

    The articles show what the package prints, so they are run against the package as it stands
    here, never against an older install: the version must be the one `DESCRIPTION` carries, and
    every Python source of the install must be the file in `python/timesift/`.
    """
    try:
        import timesift
    except ImportError as error:
        raise SystemExit(f"{sys.executable} cannot import timesift ({error}). Install this tree "
                         "with `pip install .`, or name an interpreter that has it.")
    installed = Path(timesift.__file__).parent
    wanted = next(line.split(":", 1)[1].strip()
                  for line in (ROOT / "DESCRIPTION").read_text(encoding="utf-8").splitlines()
                  if line.startswith("Version:"))
    differ = [p.name for p in sorted((ROOT / "python" / "timesift").glob("*.py"))
              if not (installed / p.name).exists()
              or (installed / p.name).read_bytes() != p.read_bytes()]
    if timesift.__version__ != wanted or differ:
        raise SystemExit(f"the timesift {sys.executable} imports is {timesift.__version__} at "
                         f"{installed}, and this tree is {wanted}"
                         + (f" with {', '.join(differ)} changed since" if differ else "")
                         + ". Install it with `pip install .` first.")

def recorded_digest(path: Path) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("<!-- Written by tools/python_articles.py"):
            return line.split("(md5 ", 1)[1].split(")", 1)[0]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    names = args.names or sorted(p.stem for p in SOURCE.glob("*.md"))
    if args.check:
        stale = [n for n in names
                 if recorded_digest(DEST / f"{n}.Rmd")
                 != digest((SOURCE / f"{n}.md").read_text(encoding="utf-8"))]
        if stale:
            print("Older than its source, rerun tools/python_articles.py: " + ", ".join(stale),
                  file=sys.stderr)
            return 1
        print("The Python articles match their sources.")
        return 0

    check_install()
    for name in names:
        article, figures = render(name)
        for old in (DEST / f"{name}_files").glob("figure-*.svg"):
            if f"{name}_files/{old.name}" not in figures:
                old.unlink()
        for path, svg in figures.items():
            (DEST / path).parent.mkdir(parents=True, exist_ok=True)
            (DEST / path).write_text(svg, encoding="utf-8", newline="\n")
        (DEST / f"{name}.Rmd").write_text(article, encoding="utf-8", newline="\n")
        print(f"wrote vignettes/articles/{name}.Rmd")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
