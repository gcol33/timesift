"""Write the shields.io endpoint files behind the README's downloads badges.

CRAN counts come from the cranlogs API. PyPI keeps only the last 180 days of daily counts, so the
per-day series is merged into a history file that is never trimmed (mirrors excluded). The total
badge sums all of CRAN and the whole PyPI history; the monthly badge sums CRAN's last-month window
and the PyPI days inside the same start and end dates.

usage: python tools/download_badges.py DIR
DIR holds pypi_daily.json (read if present, rewritten), downloads.json and downloads_month.json
(rewritten) and README.md (rewritten), the page the badges link to.
"""
import json
import sys
import urllib.request
from pathlib import Path

PACKAGE = "timesift"
CRAN_TOTAL_URL = f"https://cranlogs.r-pkg.org/downloads/total/1970-01-01:2100-01-01/{PACKAGE}"
CRAN_MONTH_URL = f"https://cranlogs.r-pkg.org/downloads/total/last-month/{PACKAGE}"
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


def badge(message):
    return {"schemaVersion": 1, "label": "downloads", "message": message, "color": "blue"}


def main(directory):
    directory = Path(directory)
    history_path = directory / "pypi_daily.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    for row in fetch(PYPI_URL)["data"]:
        history[row["date"]] = row["downloads"]

    cran = sum(row["downloads"] for row in fetch(CRAN_TOTAL_URL))
    pypi = sum(history.values())
    total = cran + pypi

    month = fetch(CRAN_MONTH_URL)[0]
    start, end = month["start"], month["end"]
    cran_month = month["downloads"]
    pypi_month = sum(n for day, n in history.items() if start <= day <= end)
    total_month = cran_month + pypi_month

    write(history_path, history)
    write(directory / "downloads.json", badge(short(total)))
    write(directory / "downloads_month.json", badge(f"{short(total_month)}/month"))
    page = (f"# {PACKAGE} downloads\n\n"
            f"| source | total | {start} to {end} |\n|---|---|---|\n"
            f"| [CRAN](https://cran.r-project.org/package={PACKAGE}) | {cran:,} | {cran_month:,} |\n"
            f"| [PyPI](https://pypi.org/project/{PACKAGE}/) ([pepy](https://pepy.tech/project/{PACKAGE})) "
            f"| {pypi:,} | {pypi_month:,} |\n"
            f"| total | {total:,} | {total_month:,} |\n")
    (directory / "README.md").write_text(page, encoding="utf-8", newline="\n")
    print(f"total: cran {cran}, pypi {pypi}, sum {total}")
    print(f"{start} to {end}: cran {cran_month}, pypi {pypi_month}, sum {total_month}")


if __name__ == "__main__":
    main(sys.argv[1])
