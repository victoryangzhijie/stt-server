from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from config import settings
from observability.logging import setup_logging
from transport.http import router as http_router
from transport.ws import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    # 预加载模型 (仅 qwen3 后端)
    if settings.backend == "qwen3":
        from backends.qwen3 import preload_model

        preload_model()  # 阻塞式加载模型
    elif settings.backend == "nemo":
        from backends.nemo import preload_model

        preload_model()

    yield


app = FastAPI(title="realtime-stt-ws", lifespan=lifespan)
app.include_router(ws_router)
app.include_router(http_router)

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/health")
async def health():
    return {"status": "ok", "backend": settings.backend}
