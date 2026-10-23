"""Contract between universal inventory and upstream host facts API."""

from __future__ import annotations

import subprocess

from hermes_platform.host import facts
from hermes_cli.universal_models.discovery import hardware_summary
from hermes_cli.universal_models import ModelService


def test_hardware_snapshot_reports_only_canonical_host_facts(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("hardware snapshot must never launch GPU probing subprocesses")

    monkeypatch.setattr(subprocess, "run", forbidden)
    result = hardware_summary()

    assert result["os_family"] == facts.os_family()
    assert result["native_arch"] == facts.native_arch()
    assert result["cpu_model"] == facts.cpu_model()
    assert result["gpu_class"] == facts.gpu_class()
    assert [row["name"] for row in result["gpu"]] == [gpu.name for gpu in facts.nvidia_gpus()]
    assert result["gpu_discovery"] == ("identity_only" if result["gpu"] else "unavailable")
    assert result["gpu_vram_discovery"] == "not_measured"
    assert result["cpu_logical_cores"] is None
    assert all("memory_free_mib" not in row and "memory_total_mib" not in row for row in result["gpu"])
    ram = facts.ram_total_bytes()
    if ram is not None:
        assert result["ram_total_mib"] == ram // (1024 * 1024)
    else:
        assert result["ram_total_mib"] is None


def test_refresh_preserves_hardware_provenance_across_gateway_instances(tmp_path):
    first = ModelService(home=tmp_path, config={}, local_endpoints=[])
    assert first.status()["hardware"]["gpu_discovery"] == "not_measured"

    measured = first.refresh()["hardware"]
    assert measured["ram_source"] in {"hermes_platform.host.facts", "unavailable"}
    assert measured["gpu_vram_discovery"] == "not_measured"

    second = ModelService(home=tmp_path, config={}, local_endpoints=[])
    assert second.status()["hardware"] == measured
