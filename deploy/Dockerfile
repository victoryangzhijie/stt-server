# CPU image: mock + sherpa-onnx + funasr + silero backends.
#
# Multi-stage so the final image only carries the resolved virtualenv plus
# application source — no compiler toolchain, no uv binary, no build cache.
#
# NOTE on size: `funasr` pulls in `torch`/`torchaudio` (CPU wheels), so this
# image is large (multiple GB) regardless of stage hygiene. That is an
# accepted tradeoff for a single image supporting all CPU backends; see
# deploy/README notes / task-8 report for measured/estimated size.

FROM python:3.12-slim AS builder

# uv is copied in as a static binary from its official distroless image
# rather than pip-installed, so it never touches the runtime venv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy only the dependency manifests first so the resolved venv layer is
# cached across source-only changes.
COPY pyproject.toml uv.lock ./

# --frozen: never touch/regenerate uv.lock during the image build (the lock
# is committed and already contains resolved hashes for all extras).
# --no-install-project: install dependencies only in this layer; the
# project itself is installed in a second `uv sync` after source is copied,
# so editing application code doesn't bust the dependency layer cache.
RUN uv sync --frozen --no-install-project \
    --extra sherpa --extra funasr --extra silero

COPY src ./src
COPY README.md ./README.md

RUN uv sync --frozen --extra sherpa --extra funasr --extra silero

FROM python:3.12-slim AS runtime

RUN groupadd --system stt && useradd --system --gid stt --create-home stt

WORKDIR /app

COPY --from=builder --chown=stt:stt /app/.venv /app/.venv
COPY --from=builder --chown=stt:stt /app/src /app/src
COPY --chown=stt:stt configs ./configs

ENV PATH="/app/.venv/bin:${PATH}"
# STT__-prefixed environment variables override the mounted YAML config at
# startup (see src/stt_server/config/settings.py); e.g.
# STT__AUTH__TOKENS='["secret"]' or STT__SERVER__PORT=9000.

VOLUME ["/app/models"]
EXPOSE 8000

USER stt

ENTRYPOINT ["stt-server"]
CMD ["--config", "/app/configs/mock.yaml", "--host", "0.0.0.0"]
