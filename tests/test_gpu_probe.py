"""Diagnóstico do probe de VRAM (learn/status) e do scrub sem torch."""

from __future__ import annotations

from vramd import gpu


class TestProbeProcessVram:
    def _probe(self, monkeypatch, apps, diag=("nvml", None), pid: int = 42):
        monkeypatch.setattr(gpu, "list_nvidia_compute_apps", lambda: apps)
        monkeypatch.setattr(gpu, "compute_apps_diagnostics", lambda: diag)
        return gpu.probe_process_vram_mib(pid)

    def test_ok_soma_entradas_do_pid(self, monkeypatch) -> None:
        probe = self._probe(monkeypatch, [(42, "python", 100), (42, "python", 50), (7, "x", 1)])
        assert probe.reason == "ok"
        assert probe.mib == 150
        assert probe.apps == 3

    def test_pid_ausente(self, monkeypatch) -> None:
        probe = self._probe(monkeypatch, [(7, "outro", 100)])
        assert probe.reason == "pid_not_listed"
        assert probe.mib is None
        assert probe.apps == 1

    def test_lista_vazia_com_erro_do_driver(self, monkeypatch) -> None:
        probe = self._probe(monkeypatch, [], diag=("nvidia-smi", "rc=1: driver wedged"))
        assert probe.reason == "driver_error"
        assert probe.error == "rc=1: driver wedged"

    def test_lista_vazia_sem_erro(self, monkeypatch) -> None:
        probe = self._probe(monkeypatch, [], diag=("nvml", None))
        assert probe.reason == "no_compute_apps"
        assert probe.error is None

    def test_entrada_sem_memoria(self, monkeypatch) -> None:
        probe = self._probe(monkeypatch, [(42, "python", None)])
        assert probe.reason == "used_mib_unavailable"
        assert probe.mib is None


class TestTorchAbsent:
    def test_torch_available_detecta_ausencia(self, monkeypatch) -> None:
        monkeypatch.setattr(gpu, "_torch_available_cache", None)
        monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
        assert gpu.torch_available() is False

    def test_evicao_sem_torch_nao_chama_cuda(self, monkeypatch) -> None:
        """Supervisor subprocesso não tem torch: não deve avisar nem tentar scrub."""
        from vramd.backend_manager import BackendManager
        from vramd.registry import Registry

        monkeypatch.setattr(gpu, "torch_available", lambda: False)
        called: list[bool] = []
        monkeypatch.setattr(gpu, "clear_cuda_memory", lambda *a, **k: called.append(True))
        mgr = BackendManager(Registry(descriptors={}), query_free_mib=lambda: 99999)
        mgr._clear_cache()
        assert called == []
        assert mgr._torch_absent_logged is True
