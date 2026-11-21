# Charts: the SVG kit and Apache ECharts

Every chart on the dashboard is drawn through one small API, `window.Charts` (`C.line`, `C.columns`, `C.hbars`, `C.histogram`, `C.scatter`, `C.matrix` ...), which the pages call with plain options (`dates`, `series`,
`yFormat`, `regions`, `tipExtra` ...). Two engines sit behind it:

* **Apache ECharts** (`static/vendor/echarts.min.js`, version 6.1.0, Apache-2.0; its `LICENSE` and `NOTICE` are beside it), driven by the adapter `static/charts_ec.js`. It draws with its **SVG renderer**, so the
  root `<svg>` carries the class `chart` and "Save as SVG / PNG", the Table view and the CSV export work as before.
* **The platform's own SVG kit** (`static/charts.js`). It stays in charge of everything the adapter does not cover, and of every chart if ECharts cannot load.

The adapter changes *how* a chart is drawn, never *what*: it reads the same options and the same numbers, and a chart type or an option it does not cover (for instance a shaded sign-fill, or point labels) falls
straight through to the SVG kit, so no chart loses a feature by being ported.

## What is on ECharts

| Chart | Covers |
| --- | --- |
| `line` (about 120 call sites) | a time or numeric axis, any number of series (dash, width, dots, gaps), area, baseline, regions, vertical references, bands, a log axis, fixed or padded limits, tooltips with `tipFormat`, `tipTitle` and `tipExtra`, a legend whose entries hide and show series (the last visible one stays), synchronized cursors (`sync`), an opening window (`window`) |
| `columns`, `hbars` (waterfalls included), `histogram` | the options their pages use, with the same hover text |

Still on the SVG kit: `scatter`, `matrix`, `heatmap`, `ranges`, `stack`, `mirror`, the trade panel (`tradechart.js`, which is one drawing for the price, the trades and the position), and any `line` that uses `signFill`,
`events`, `pointLabel`, or `areaBase` with `area`.

## What the reader gets

A zoom **strip** under every long series, **two-finger pinch** on a touch screen, a mouse **drag across the plot** to zoom to the stretch (double-click or *Reset* undoes it; Ctrl with the wheel zooms around the pointer;
shift-drag pans), range buttons (1M ... All), a readout of how much is shown, a crosshair that snaps to the data, tooltips kept inside the screen (on a touch screen a tap shows one and it stays until the next tap),
line **sampling** for long series, and the keyboard (`+`, `-`, `0` on a focused chart). The old plus, minus and arrow buttons are gone: the strip and the pinch do their work. A theme change repaints every chart.

## Switching back

`localStorage.setItem("ql.charts", "svg")` and a reload draws every chart with the SVG kit; `localStorage.removeItem("ql.charts")` returns to ECharts.

## Why it is vendored

The app's content-security policy allows scripts only from its own origin (`default-src 'self'`), so a library cannot be loaded from a CDN. `src/webapp/static/vendor/` is part of the repository, listed in the server's
file allow-list (`server.FILES`) and in the package data (`pyproject.toml`). To update it: download the `echarts` package from npm, copy `dist/echarts.min.js`, `LICENSE` and `NOTICE` over the three files, and run
the browser tests. The full build is 1.1 MB (359 KB gzipped).

## For test authors

A chart's hover area is no longer an SVG rectangle. Read a chart's plot area, its zoom window or its series from the ECharts instance (`echarts.getInstanceByDom(el)`), and use `:is(.ec-chart, svg.chart rect[tabindex])`
where a test must work with either engine. The tooltip element carries the class `tip` and the attribute `hidden` while it is not shown, as in the SVG kit.
