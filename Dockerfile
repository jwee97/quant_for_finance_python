# Reproducible environment for the research platform (Generation 4).
#
#   docker build -t multi-asset-research .
#   docker run --rm multi-asset-research                       # the test suite
#   docker run --rm -v "$PWD/reports:/app/reports" -v "$PWD/data:/app/data" \
#       multi-asset-research python -m experiments.run_all --generation 4
#
# Python 3.11, CPU-only PyTorch. The image holds the code and the committed raw data; derived artefacts
# (data/processed, the research database) are rebuilt by the pipeline, not baked in.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# System libraries: none beyond the slim image are needed for the pinned wheels; git is for the data manifest checks.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Dependencies first, so a code change does not invalidate the layer that holds them.
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2" \
 && pip install -r requirements.txt

# The test suite reads the documentation, the example specifications, the notebooks and the mkdocs configuration, so they belong in the image.
COPY pyproject.toml README.md mkdocs.yml ./
COPY src ./src
COPY config ./config
COPY experiments ./experiments
COPY tests ./tests
COPY docs ./docs
COPY examples ./examples
COPY notebooks ./notebooks
COPY data ./data
COPY reports ./reports
RUN pip install --no-deps -e .

# Unprivileged user; the mounted volumes need to be writable by it.
RUN useradd --create-home researcher && chown -R researcher /app
USER researcher

CMD ["python", "-m", "pytest", "-q"]
