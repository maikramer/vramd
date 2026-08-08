"""Testes de overrides de env para constantes do protocolo UMS."""

from __future__ import annotations

import importlib

import pytest


class TestProtocolEnvOverrides:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VRAMD_MAX_AFFINITY_CUTS", raising=False)
        monkeypatch.delenv("VRAMD_MAX_QUEUE_DEPTH", raising=False)
        monkeypatch.delenv("VRAMD_MAX_INFLIGHT", raising=False)
        monkeypatch.delenv("VRAMD_STARVATION_TIMEOUT_SEC", raising=False)
        import vramd.protocol as proto

        importlib.reload(proto)
        assert proto.MAX_AFFINITY_CUTS == 3
        assert proto.MAX_QUEUE_DEPTH == 32
        assert proto.MAX_INFLIGHT == 1
        assert proto.STARVATION_TIMEOUT_SEC == 0.0

    def test_valid_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VRAMD_MAX_AFFINITY_CUTS", "5")
        monkeypatch.setenv("VRAMD_MAX_QUEUE_DEPTH", "10")
        monkeypatch.setenv("VRAMD_MAX_INFLIGHT", "2")
        monkeypatch.setenv("VRAMD_STARVATION_TIMEOUT_SEC", "120")
        import vramd.protocol as proto

        importlib.reload(proto)
        assert proto.MAX_AFFINITY_CUTS == 5
        assert proto.MAX_QUEUE_DEPTH == 10
        assert proto.MAX_INFLIGHT == 2
        assert proto.STARVATION_TIMEOUT_SEC == 120.0

    def test_invalid_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VRAMD_MAX_AFFINITY_CUTS", "abc")
        import vramd.protocol as proto

        importlib.reload(proto)
        assert proto.MAX_AFFINITY_CUTS == 3

    def teardown_method(self) -> None:
        # Restaurar defaults para não contaminar outros testes no mesmo processo.
        import vramd.protocol as proto

        importlib.reload(proto)
