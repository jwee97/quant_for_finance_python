"""Helpers for browser tests that look at a chart's contents. Charts are drawn by ECharts (see docs/charts.md): count and read the data from the chart instance instead of counting SVG elements."""

from __future__ import annotations


def chart_points(scope) -> int:
    """How many data points (bars, cells, markers) the ECharts charts inside ``scope`` (a Playwright locator, or a page for the whole document) hold: the sum over their series, counting a custom series' items."""
    script = """root => [...(root.querySelectorAll ? root : document).querySelectorAll('.ec-chart')].reduce((n, el) => n + echarts.getInstanceByDom(el).getOption().series.reduce((m, s) => m + (s.data ? s.data.length : 0), 0), 0)"""
    if hasattr(scope, "evaluate") and not hasattr(scope, "goto"):
        return scope.evaluate(script)                                                           # a locator: its element is the root
    return scope.evaluate("() => (%s)(document)" % script)                                      # a page
