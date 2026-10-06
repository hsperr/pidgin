# The Brill Seat Robot API with three bots, for serving locally:
#   docker build -f server/Dockerfile -t pidgin-brill .   # from the repo root
#   docker run --rm -p 8080:8080 pidgin-brill
#   curl 'localhost:8080/apis/brill/'          -> models: PidginV1, BRL, PidginV2
# Pick a bot with model=<id> (or model_id=<id>) on /apis/brill/{bid,lead,play}; without one,
# BRILL_DEFAULT_MODEL answers. The rest of the site (the table at /) runs too.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/mpl BRILL_DEFAULT_MODEL=PidginV2 \
    PLAY_SEARCH_DEFENCE_FROM=0
WORKDIR /app

# torch from the CPU-only index: the default wheel drags in ~2GB of CUDA.
COPY server/requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY training training
COPY server/emergent emergent
COPY server/models models
RUN mkdir -p data/bench && useradd --system --no-create-home bridge && chown -R bridge data
USER bridge

EXPOSE 8080
# One worker: games live in the process and the search is serialised inside it.
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "300", "--bind", "0.0.0.0:8080", \
     "--access-logfile", "-", "emergent.bidserver:create_app()"]
