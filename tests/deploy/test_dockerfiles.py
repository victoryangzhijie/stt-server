"""Static checks over `deploy/` — CI has no Docker daemon, so these tests
parse the Dockerfiles/compose/prometheus/grafana files as text/YAML/JSON
rather than actually building or running anything.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEPLOY_DIR = REPO_ROOT / "deploy"
METRICS_SOURCE = (
    REPO_ROOT / "src" / "stt_server" / "metrics" / "registry.py"
).read_text()


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


# --------------------------------------------------------------------------
# Dockerfile (CPU)
# --------------------------------------------------------------------------


class TestCpuDockerfile:
    @property
    def text(self) -> str:
        return (DEPLOY_DIR / "Dockerfile").read_text()

    def test_multi_stage_python_slim_base(self):
        assert re.search(r"^FROM python:3\.12-slim AS builder", self.text, re.M)
        assert re.search(r"^FROM python:3\.12-slim AS runtime", self.text, re.M)

    def test_copies_uv_from_official_image(self):
        assert "ghcr.io/astral-sh/uv" in self.text

    def test_uv_sync_frozen_with_expected_extras(self):
        assert re.search(r"uv sync --frozen[^\n]*--extra sherpa", self.text)
        assert "--extra funasr" in self.text
        assert "--extra silero" in self.text

    def test_non_root_user(self):
        assert re.search(r"^USER stt\b", self.text, re.M)
        # user must actually be created, not just referenced
        assert re.search(r"useradd\b.*\bstt\b", self.text)

    def test_declares_models_volume(self):
        assert re.search(r"^VOLUME \[?\"?/app/models", self.text, re.M)

    def test_exposes_8000(self):
        assert re.search(r"^EXPOSE 8000\b", self.text, re.M)

    def test_entrypoint_boots_stt_server(self):
        assert "ENTRYPOINT" in self.text
        assert "stt-server" in self.text
        assert "--host" in self.text and "0.0.0.0" in self.text

    def test_final_stage_has_no_build_tools_layer_copy(self):
        # the runtime stage should COPY --from=builder the venv/src, not
        # re-run `uv sync`/apt-get build-essential itself
        runtime_stage = self.text.split("FROM python:3.12-slim AS runtime", 1)[1]
        assert "build-essential" not in runtime_stage
        assert "COPY --from=builder" in runtime_stage


# --------------------------------------------------------------------------
# Dockerfile.gpu
# --------------------------------------------------------------------------


class TestGpuDockerfile:
    @property
    def text(self) -> str:
        return (DEPLOY_DIR / "Dockerfile.gpu").read_text()

    def test_cuda_runtime_base(self):
        assert re.search(
            r"^FROM nvidia/cuda:12\.4\.1-runtime-ubuntu22\.04", self.text, re.M
        )

    def test_uv_provisions_python(self):
        assert "uv python install" in self.text

    def test_uv_sync_frozen_with_qwen3asr_extra(self):
        assert re.search(r"uv sync --frozen[^\n]*--extra qwen3asr", self.text)

    def test_non_root_user(self):
        assert re.search(r"^USER stt\b", self.text, re.M)
        assert re.search(r"useradd\b.*\bstt\b", self.text)

    def test_declares_models_volume(self):
        assert re.search(r"^VOLUME \[?\"?/app/models", self.text, re.M)

    def test_exposes_8000(self):
        assert re.search(r"^EXPOSE 8000\b", self.text, re.M)

    def test_entrypoint_boots_stt_server(self):
        assert "ENTRYPOINT" in self.text
        assert "stt-server" in self.text


# --------------------------------------------------------------------------
# docker-compose.yaml
# --------------------------------------------------------------------------


class TestCompose:
    @property
    def doc(self) -> dict:
        return yaml.safe_load((DEPLOY_DIR / "docker-compose.yaml").read_text())

    def test_parses(self):
        assert isinstance(self.doc, dict)
        assert "services" in self.doc

    def test_declares_expected_services(self):
        services = self.doc["services"]
        assert {"stt-server", "stt-server-gpu", "prometheus", "grafana"} <= set(
            services
        )

    def test_service_profiles(self):
        services = self.doc["services"]
        assert services["stt-server"]["profiles"] == ["cpu"]
        assert services["stt-server-gpu"]["profiles"] == ["gpu"]
        assert services["prometheus"]["profiles"] == ["observability"]
        assert services["grafana"]["profiles"] == ["observability"]

    def test_gpu_service_reserves_nvidia_device(self):
        gpu = self.doc["services"]["stt-server-gpu"]
        devices = gpu["deploy"]["resources"]["reservations"]["devices"]
        assert any(d.get("driver") == "nvidia" for d in devices)
        assert any("gpu" in d.get("capabilities", []) for d in devices)

    def test_stt_server_publishes_8000_and_mounts_models(self):
        stt = self.doc["services"]["stt-server"]
        assert any("8000:8000" in str(p) for p in stt["ports"])
        assert any("models:/app/models" in v for v in stt["volumes"])

    def test_models_mount_is_a_host_bind_not_a_named_volume(self):
        """scripts/download_models.py writes weights to the repo's models/
        directory on the HOST; a named volume would silently give the
        container an empty /app/models. The mount source must therefore be
        a relative host path (compose resolves it against the compose
        file's directory, deploy/), not a bare volume name."""
        for service in ("stt-server", "stt-server-gpu"):
            volumes = self.doc["services"][service]["volumes"]
            (models_mount,) = [v for v in volumes if v.endswith(":/app/models")]
            source = models_mount.rsplit(":/app/models", 1)[0]
            assert source.startswith((".", "..", "/")), (
                f"{service}: models mount source {source!r} looks like a "
                "named volume, not a host path bind mount"
            )
        # ...and no top-level named `models` volume should exist to tempt
        # anyone back into the broken pattern.
        assert "models" not in self.doc.get("volumes", {})

    def test_stt_services_pass_through_stt_env_overrides(self):
        """README documents STT__ env-var overrides via compose; without an
        `environment:` passthrough entry, shell vars never reach the
        container. Bare-name (list) form is required: it omits unset vars
        entirely instead of injecting empty strings the settings parser
        would choke on."""
        for service in ("stt-server", "stt-server-gpu"):
            env = self.doc["services"][service]["environment"]
            assert isinstance(env, list), (
                f"{service}: use bare-name list form for STT__ passthrough"
            )
            assert "STT__AUTH__TOKENS" in env

    def test_prometheus_is_not_published_to_host(self):
        # /metrics on stt-server is unauthenticated; prometheus must reach
        # it only over the internal compose network, and prometheus itself
        # must not be given a host-published port either.
        prometheus = self.doc["services"]["prometheus"]
        assert "ports" not in prometheus

    def test_prometheus_mounts_config(self):
        prometheus = self.doc["services"]["prometheus"]
        assert any("prometheus.yml" in v for v in prometheus["volumes"])

    def test_grafana_provisioned_from_deploy_grafana(self):
        grafana = self.doc["services"]["grafana"]
        volume_str = " ".join(grafana["volumes"])
        assert "datasource.yml" in volume_str
        assert "dashboards.yml" in volume_str
        assert "dashboard.json" in volume_str

    def test_services_share_internal_network(self):
        services = self.doc["services"]
        for name in ("stt-server", "prometheus", "grafana"):
            assert "internal" in services[name]["networks"]


# --------------------------------------------------------------------------
# prometheus.yml
# --------------------------------------------------------------------------


class TestPrometheusConfig:
    @property
    def doc(self) -> dict:
        return yaml.safe_load((DEPLOY_DIR / "prometheus.yml").read_text())

    def test_parses(self):
        assert isinstance(self.doc, dict)

    def test_scrapes_stt_server_on_8000(self):
        jobs = self.doc["scrape_configs"]
        targets = [
            t
            for job in jobs
            for sc in job.get("static_configs", [])
            for t in sc.get("targets", [])
        ]
        assert "stt-server:8000" in targets

    def test_five_second_interval(self):
        jobs = self.doc["scrape_configs"]
        stt_job = next(j for j in jobs if j["job_name"] == "stt-server")
        interval = stt_job.get("scrape_interval") or self.doc["global"].get(
            "scrape_interval"
        )
        assert interval == "5s"


# --------------------------------------------------------------------------
# Grafana provisioning + dashboard
# --------------------------------------------------------------------------


class TestGrafanaProvisioning:
    def test_datasource_yaml_parses_and_points_at_prometheus(self):
        doc = yaml.safe_load((DEPLOY_DIR / "grafana" / "datasource.yml").read_text())
        (ds,) = doc["datasources"]
        assert ds["type"] == "prometheus"

    def test_dashboards_yaml_parses_and_points_at_provider_dir(self):
        doc = yaml.safe_load((DEPLOY_DIR / "grafana" / "dashboards.yml").read_text())
        (provider,) = doc["providers"]
        assert provider["options"]["path"] == "/etc/grafana/dashboards"

    def test_dashboard_json_parses(self):
        doc = json.loads((DEPLOY_DIR / "grafana" / "dashboard.json").read_text())
        assert doc["panels"]
        assert len(doc["panels"]) >= 6

    def test_dashboard_references_only_real_metric_names(self):
        """Every `stt_*` identifier used in a panel expr must be an actual
        metric family defined in the metrics registry — otherwise the panel
        is silently dead (empty graph) once deployed."""
        doc = json.loads((DEPLOY_DIR / "grafana" / "dashboard.json").read_text())
        exprs = " ".join(
            target["expr"]
            for panel in doc["panels"]
            for target in panel.get("targets", [])
        )
        used_metrics = set(re.findall(r"stt_[a-zA-Z0-9_]*", exprs))
        assert used_metrics, "expected the dashboard to reference stt_* metrics"

        for name in used_metrics:
            # histograms are queried via their `_bucket` series
            base_name = re.sub(r"_bucket$", "", name)
            assert re.search(
                rf'"{re.escape(base_name)}"', METRICS_SOURCE
            ), f"dashboard references unknown metric: {name}"

    def test_dashboard_covers_required_panel_topics(self):
        doc = json.loads((DEPLOY_DIR / "grafana" / "dashboard.json").read_text())
        titles = " ".join(p["title"].lower() for p in doc["panels"])
        for topic in (
            "active session",
            "first-partial",
            "final latency",
            "utterance",
            "rejection",
            "error",
            "audio seconds",
        ):
            assert topic in titles


# --------------------------------------------------------------------------
# .dockerignore
# --------------------------------------------------------------------------


class TestDockerignore:
    @property
    def text(self) -> str:
        return (REPO_ROOT / ".dockerignore").read_text()

    def test_excludes_git_and_models_and_scratch(self):
        assert ".git" in self.text
        assert "models/" in self.text
        assert ".superpowers" in self.text
        assert ".venv" in self.text
