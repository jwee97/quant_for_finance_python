"""``quant serve``: a local dashboard to backtest strategies on your own tickers, compare runs, build formula strategies and read the guides.

``universe``  tickers -> a ``MarketBundle`` (downloads cached under ``data/user``)
``payload``   a ``PipelineResult`` -> the JSON the charts draw
``api``       the application: catalogue, jobs, documents
``server``    a small threaded HTTP server bound to localhost with a per-launch token
"""
