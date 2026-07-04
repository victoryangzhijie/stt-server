"""CLI entry point: `stt-server --config configs/mock.yaml`."""

from __future__ import annotations

import argparse

import uvicorn

from stt_server.api.app import create_app
from stt_server.config.settings import ConfigError, Settings, load_settings
from stt_server.logging import configure_logging, uvicorn_log_config


def build_settings(argv: list[str] | None = None) -> Settings:
    parser = argparse.ArgumentParser(prog="stt-server")
    parser.add_argument("--config", default=None, help="path to config YAML")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    if args.host is not None:
        settings.server.host = args.host
    if args.port is not None:
        settings.server.port = args.port
    return settings


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    try:
        settings = build_settings(argv)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from None
    uvicorn.run(
        create_app(settings),
        host=settings.server.host,
        port=settings.server.port,
        log_config=uvicorn_log_config(),
    )


if __name__ == "__main__":
    main()
