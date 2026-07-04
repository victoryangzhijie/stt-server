from stt_server.__main__ import build_settings, main


def test_cli_overrides_config(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("server:\n  port: 9001\n")
    s = build_settings(["--config", str(cfg), "--port", "9100"])
    assert s.server.port == 9100

    s = build_settings(["--config", str(cfg)])
    assert s.server.port == 9001


def test_main_passes_uvicorn_log_config(tmp_path, monkeypatch):
    # main() must never boot a real server in a test — capture uvicorn.run's
    # kwargs instead of letting it run.
    calls = []
    monkeypatch.setattr(
        "stt_server.__main__.uvicorn.run", lambda app, **kw: calls.append(kw)
    )
    cfg = tmp_path / "c.yaml"
    cfg.write_text("server:\n  port: 9001\n")

    main(["--config", str(cfg)])

    assert len(calls) == 1
    log_config = calls[0]["log_config"]
    assert {"uvicorn", "uvicorn.error", "uvicorn.access"} <= log_config["loggers"].keys()
