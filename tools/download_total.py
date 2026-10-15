"""Write the shields.io endpoint file behind the README's total-downloads badge.

CRAN's lifetime total comes from the cranlogs API. PyPI keeps only the last 180 days of daily
counts, so the per-day series is merged into a history file that is never trimmed, and the PyPI
total is the sum of that history (mirrors excluded).

usage: python tools/download_total.py DIR
DIR holds pypi_daily.json (read if present, rewritten), downloads.json (rewritten) and README.md
(rewritten), the page the badge links to.
"""
import json
import sys
import urllib.request
from pathlib import Path

PACKAGE = "timesift"
CRAN_URL = f"https://cranlogs.r-pkg.org/downloads/total/1970-01-01:2100-01-01/{PACKAGE}"
PYPI_URL = f"https://pypistats.org/api/packages/{PACKAGE}/overall?mirrors=false"


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": f"{PACKAGE}-download-badge"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def short(n):
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k".replace(".0k", "k")
    return f"{n / 1_000_000:.1f}M".replace(".0M", "M")


def write(path, payload):
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main(directory):
    directory = Path(directory)
    history_path = directory / "pypi_daily.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    for row in fetch(PYPI_URL)["data"]:
        history[row["date"]] = row["downloads"]

    cran = sum(row["downloads"] for row in fetch(CRAN_URL))
    total = cran + sum(history.values())

    write(history_path, history)
    write(directory / "downloads.json",
          {"schemaVersion": 1, "label": "downloads", "message": short(total), "color": "blue"})
    pypi = sum(history.values())
    page = (f"# {PACKAGE} downloads\n\n"
            "| source | downloads |\n|---|---|\n"
            f"| [CRAN](https://cran.r-project.org/package={PACKAGE}) | {cran:,} |\n"
            f"| [PyPI](https://pypi.org/project/{PACKAGE}/) ([pepy](https://pepy.tech/project/{PACKAGE})) | {pypi:,} |\n"
            f"| total | {total:,} |\n")
    (directory / "README.md").write_text(page, encoding="utf-8", newline="\n")
    print(f"cran {cran}, pypi {pypi}, total {total}")


if __name__ == "__main__":
    main(sys.argv[1])
