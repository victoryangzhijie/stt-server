from stt_server.logging import uvicorn_log_config


def test_uvicorn_log_config_shape_is_internally_consistent():
    cfg = uvicorn_log_config()

    assert cfg["version"] == 1
    formatters = cfg["formatters"]
    handlers = cfg["handlers"]
    loggers = cfg["loggers"]

    # Every handler's formatter must reference a formatter that exists.
    for name, handler in handlers.items():
        assert handler["formatter"] in formatters, (
            f"handler {name!r} references undefined formatter {handler['formatter']!r}"
        )

    # Every logger's handlers must reference handlers that exist.
    for name, logger_cfg in loggers.items():
        for handler_name in logger_cfg["handlers"]:
            assert handler_name in handlers, (
                f"logger {name!r} references undefined handler {handler_name!r}"
            )

    # The three uvicorn loggers this exists to route must all be configured.
    assert {"uvicorn", "uvicorn.error", "uvicorn.access"} <= loggers.keys()

    # Every formatter must route through structlog's JSON renderer.
    for name, formatter in formatters.items():
        assert formatter["()"].__name__ == "ProcessorFormatter", (
            f"formatter {name!r} does not use structlog.stdlib.ProcessorFormatter"
        )


def test_uvicorn_log_config_returns_a_fresh_dict_each_call():
    # uvicorn.run(log_config=...) may mutate the dict it's handed (dictConfig
    # semantics); callers must not share mutable state across invocations.
    # The processor objects inside don't implement value equality, so compare
    # structure (keys) rather than the whole dict.
    a = uvicorn_log_config()
    b = uvicorn_log_config()
    assert a is not b
    assert a.keys() == b.keys()
    assert a["handlers"] is not b["handlers"]
    assert a["loggers"] is not b["loggers"]
