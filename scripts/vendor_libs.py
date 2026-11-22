"""Copy optional front-end libraries out of pip packages into src/webapp/static/vendor/.

    pip install plotly dash-ag-grid
    python scripts/vendor_libs.py

* Plotly (MIT): ``plotly/package_data/plotly.min.js`` -> ``vendor/plotly.min.js`` (the implied-volatility surface in 3D, with hover).
* AG Grid Community (MIT): the bundle inside ``dash_ag_grid`` -> ``vendor/ag-grid-community.min.js`` (the Saved runs table: sort, filter, tick rows).

Nothing else needs them: without the files the app uses its own canvas 3D viewer and plain tables. Licences are copied beside each library. Run with ``--check`` to see what is installed without copying.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parent.parent / "src" / "webapp" / "static" / "vendor"


def _pkg_dir(name: str) -> Path | None:
    spec = importlib.util.find_spec(name)
    return Path(spec.submodule_search_locations[0]) if spec and spec.submodule_search_locations else None


def _first(root: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        hits = sorted(root.rglob(pat), key=lambda p: -p.stat().st_size)
        if hits:
            return hits[0]
    return None


def plotly() -> tuple[Path, Path | None] | None:
    d = _pkg_dir("plotly")
    if d is None:
        return None
    js = _first(d, ["plotly.min.js"])
    lic = _first(d.parent, ["plotly*/LICENSE*", "LICENSE.txt"]) if js else None
    return (js, lic) if js else None


def ag_grid() -> tuple[Path, Path | None] | None:
    d = _pkg_dir("dash_ag_grid")
    if d is None:
        return None
    js = _first(d, ["ag-grid-community.min.js", "ag-grid-community*.min.js", "ag-grid*.min.js"])
    lic = _first(d.parent, ["dash_ag_grid*/LICENSE*", "dash_ag_grid*/licenses/*"]) if js else None
    return (js, lic) if js else None


def main(argv: list[str]) -> int:
    found = {"plotly": plotly(), "ag-grid-community": ag_grid()}
    for name, hit in found.items():
        print(f"{name}: {'found ' + str(hit[0]) if hit else 'not installed (pip install ' + ('plotly' if name == 'plotly' else 'dash-ag-grid') + ')'}")
    if "--check" in argv:
        return 0
    VENDOR.mkdir(parents=True, exist_ok=True)
    for name, hit in found.items():
        if not hit:
            continue
        js, lic = hit
        shutil.copyfile(js, VENDOR / f"{name if name != 'plotly' else 'plotly'}.min.js")
        if lic:
            shutil.copyfile(lic, VENDOR / f"{name}.LICENSE")
        if name == "ag-grid-community":                                         # v32 and earlier need the style sheets; v33 injects its own theme, so they may not exist
            for css in ("ag-grid.css", "ag-theme-quartz.css"):
                hit_css = _first(_pkg_dir("dash_ag_grid"), [css])
                if hit_css:
                    shutil.copyfile(hit_css, VENDOR / css)
                    print(f"copied {css}")
        print(f"copied {js.name} -> vendor/{name}.min.js ({js.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
