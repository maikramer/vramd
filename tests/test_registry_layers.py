"""Testes das camadas de configuração v2: ``runtime:``, merge e key sets por backend.

O que estas camadas compram: instalar um descriptor calibrado (ou registar um
backend externo) **sem tocar no package**. Os testes cobrem a precedência do
merge, a resolução do comando do worker e a queda para o comportamento legado.
"""

from __future__ import annotations

import os

import pytest
import yaml

from vramd.registry import (
    ENV_BACKENDS_DIR,
    ENV_BACKENDS_FILE,
    BackendDescriptor,
    Registry,
    RuntimeSpec,
    descriptor_sources,
    load_descriptors,
    merge_entries,
)

BASE_ENTRY = {
    "name": "demo",
    "adapter": "vramd.adapters.text3d",
    "vram_mib": 1000,
    "priority": 10,
    "tool": "demo",
}


def write_yaml(path, entries):
    path.write_text(yaml.safe_dump({"version": 2, "backends": entries}), encoding="utf-8")
    return str(path)


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    """Sem camadas do utilizador a interferir (a máquina pode ter ~/.config/ums)."""
    monkeypatch.delenv(ENV_BACKENDS_FILE, raising=False)
    monkeypatch.setenv(ENV_BACKENDS_DIR, str(tmp_path / "vazio"))
    return tmp_path


class TestMergeEntries:
    def test_later_source_overrides_field_by_field(self):
        merged = merge_entries([[BASE_ENTRY], [{"name": "demo", "vram_mib": 5632}]])
        assert merged["demo"]["vram_mib"] == 5632
        # O resto sobrevive: é isto que torna um override parcial utilizável.
        assert merged["demo"]["adapter"] == "vramd.adapters.text3d"
        assert merged["demo"]["priority"] == 10

    def test_new_backend_from_later_source_is_added(self):
        merged = merge_entries([[BASE_ENTRY], [{"name": "outro", "adapter": "x", "vram_mib": 1}]])
        assert set(merged) == {"demo", "outro"}

    def test_empty_sources_give_empty_result(self):
        assert merge_entries([]) == {}

    def test_first_source_wins_order(self):
        merged = merge_entries([[BASE_ENTRY, {"name": "b", "adapter": "x", "vram_mib": 1}], []])
        assert list(merged) == ["demo", "b"]


class TestDescriptorSources:
    def test_explicit_path_is_the_only_source(self, tmp_path, isolated_env):
        path = write_yaml(tmp_path / "só.yaml", [BASE_ENTRY])
        assert descriptor_sources(path) == [path]

    def test_env_file_is_appended_after_the_package(self, tmp_path, monkeypatch, isolated_env):
        override = write_yaml(tmp_path / "override.yaml", [{"name": "demo", "vram_mib": 2}])
        monkeypatch.setenv(ENV_BACKENDS_FILE, override)
        sources = descriptor_sources()
        assert sources[-1] == override
        assert len(sources) >= 2

    def test_env_file_accepts_multiple_paths(self, tmp_path, monkeypatch, isolated_env):
        a = write_yaml(tmp_path / "a.yaml", [{"name": "demo", "vram_mib": 2}])
        b = write_yaml(tmp_path / "b.yaml", [{"name": "demo", "vram_mib": 3}])
        monkeypatch.setenv(ENV_BACKENDS_FILE, os.pathsep.join([a, b]))
        assert descriptor_sources()[-2:] == [a, b]

    def test_missing_env_file_is_skipped(self, monkeypatch, isolated_env):
        monkeypatch.setenv(ENV_BACKENDS_FILE, "/nao/existe.yaml")
        assert "/nao/existe.yaml" not in descriptor_sources()

    def test_config_dir_files_are_sorted(self, tmp_path, monkeypatch):
        conf = tmp_path / "backends.d"
        conf.mkdir()
        write_yaml(conf / "20-b.yaml", [{"name": "demo", "vram_mib": 3}])
        write_yaml(conf / "10-a.yaml", [{"name": "demo", "vram_mib": 2}])
        monkeypatch.delenv(ENV_BACKENDS_FILE, raising=False)
        monkeypatch.setenv(ENV_BACKENDS_DIR, str(conf))
        tail = descriptor_sources()[-2:]
        assert [os.path.basename(p) for p in tail] == ["10-a.yaml", "20-b.yaml"]

    def test_non_yaml_files_in_config_dir_are_ignored(self, tmp_path, monkeypatch):
        conf = tmp_path / "backends.d"
        conf.mkdir()
        (conf / "notas.txt").write_text("nada", encoding="utf-8")
        monkeypatch.delenv(ENV_BACKENDS_FILE, raising=False)
        monkeypatch.setenv(ENV_BACKENDS_DIR, str(conf))
        assert not any(p.endswith(".txt") for p in descriptor_sources())


class TestLoadDescriptors:
    def test_partial_override_keeps_the_rest(self, tmp_path, monkeypatch, isolated_env):
        base = write_yaml(tmp_path / "base.yaml", [BASE_ENTRY])
        override = write_yaml(tmp_path / "over.yaml", [{"name": "demo", "vram_mib": 5632}])
        monkeypatch.setenv(ENV_BACKENDS_FILE, os.pathsep.join([base, override]))
        # Sem yaml_path explícito o package entra também; filtra-se pelo nome.
        descs = load_descriptors()
        assert descs["demo"].vram_mib == 5632
        assert descs["demo"].priority == 10

    def test_packaged_backend_can_be_recalibrated_by_a_user_file(self, tmp_path, monkeypatch, isolated_env):
        override = write_yaml(tmp_path / "cal.yaml", [{"name": "example-diffusion", "vram_mib": 5632}])
        monkeypatch.setenv(ENV_BACKENDS_FILE, override)
        descs = load_descriptors()
        assert descs["example-diffusion"].vram_mib == 5632
        assert descs["example-diffusion"].tool == "my_diffusion"  # herdado do package

    def test_missing_required_field_after_merge_raises(self, tmp_path, isolated_env):
        path = write_yaml(tmp_path / "mau.yaml", [{"name": "x", "vram_mib": 1}])
        with pytest.raises(ValueError, match="falta adapter"):
            load_descriptors(path)

    def test_entry_without_name_raises(self, tmp_path, isolated_env):
        path = write_yaml(tmp_path / "mau.yaml", [{"adapter": "x", "vram_mib": 1}])
        with pytest.raises(ValueError, match="sem 'name'"):
            load_descriptors(path)

    def test_explicit_missing_path_raises_file_not_found(self, isolated_env):
        with pytest.raises(FileNotFoundError):
            load_descriptors("/nao/existe/backends.yaml")

    def test_calibrated_descriptor_loads(self, tmp_path, isolated_env):
        """O ficheiro que o `ums calibrate --out` escreve tem de entrar tal e qual."""
        entry = {
            **BASE_ENTRY,
            "runtime": {"monorepo_tool": "demo"},
            "vram": {"weights_gib": 4.0, "activation_gib": 1.4, "peak_mib": 5610},
            "peak_profile": {"quant_mode": "none", "unload_frees_vram": False},
            "measured": {"confidence": "high"},
        }
        descs = load_descriptors(write_yaml(tmp_path / "cal.yaml", [entry]))
        desc = descs["demo"]
        assert desc.vram["peak_mib"] == 5610
        assert desc.unload_frees_vram is False
        assert desc.runtime is not None


class TestRuntimeSpec:
    def test_none_for_empty_block(self):
        assert RuntimeSpec.from_dict(None) is None
        assert RuntimeSpec.from_dict({}) is None

    def test_string_command_becomes_a_single_token(self):
        assert RuntimeSpec.from_dict({"command": "meu-worker"}).command == ("meu-worker",)

    def test_explicit_command_is_used_verbatim(self):
        spec = RuntimeSpec.from_dict({"command": ["python", "-m", "meu.worker"]})
        assert spec.resolve_command() == ["python", "-m", "meu.worker"]

    def test_env_reference_is_expanded(self, monkeypatch):
        monkeypatch.setenv("MEU_PY", "/opt/py")
        spec = RuntimeSpec.from_dict({"command": ["${env:MEU_PY}", "-m", "w"]})
        assert spec.resolve_command() == ["/opt/py", "-m", "w"]

    def test_unresolved_env_reference_gives_none(self, monkeypatch):
        monkeypatch.delenv("NAO_DEFINIDA", raising=False)
        spec = RuntimeSpec.from_dict({"command": ["${env:NAO_DEFINIDA}", "-m", "w"]})
        # None em vez de um argv com "${env:...}" literal: erro acionável.
        assert spec.resolve_command() is None

    def test_monorepo_tool_derives_the_canonical_command(self, monkeypatch):
        monkeypatch.setattr("vramd.registry._monorepo_tool_python", lambda tool: f"/venv/{tool}/python")
        spec = RuntimeSpec.from_dict({"monorepo_tool": "text3d"})
        assert spec.resolve_command() == ["/venv/text3d/python", "-m", "text3d", "serve", "--ums-worker"]

    def test_monorepo_reference_inside_command(self, monkeypatch):
        monkeypatch.setattr("vramd.registry._monorepo_tool_python", lambda tool: f"/venv/{tool}/python")
        spec = RuntimeSpec.from_dict({"command": ["${monorepo:paint3d}", "-m", "x"]})
        assert spec.resolve_command() == ["/venv/paint3d/python", "-m", "x"]

    def test_missing_venv_gives_none(self, monkeypatch):
        monkeypatch.setattr("vramd.registry._monorepo_tool_python", lambda tool: None)
        assert RuntimeSpec.from_dict({"monorepo_tool": "fantasma"}).resolve_command() is None

    def test_env_block_is_expanded_and_user_path_resolved(self, monkeypatch):
        monkeypatch.setenv("CACHE_RAIZ", "/data/hf")
        spec = RuntimeSpec.from_dict({"env": {"HF_HOME": "${env:CACHE_RAIZ}", "TMP": "~/tmp"}})
        resolved = spec.resolve_env()
        assert resolved["HF_HOME"] == "/data/hf"
        assert not resolved["TMP"].startswith("~")

    def test_non_mapping_env_raises(self):
        with pytest.raises(ValueError, match=r"runtime\.env"):
            RuntimeSpec.from_dict({"env": ["a=b"]})

    def test_cwd_expands_user(self):
        assert not RuntimeSpec.from_dict({"cwd": "~/modelos"}).resolve_cwd().startswith("~")

    def test_cwd_none_when_absent(self):
        assert RuntimeSpec.from_dict({"command": ["x"]}).resolve_cwd() is None

    def test_timeouts_are_floats(self):
        spec = RuntimeSpec.from_dict({"command": ["x"], "load_timeout_sec": 300, "event_timeout_sec": 60})
        assert spec.load_timeout_sec == 300.0
        assert spec.event_timeout_sec == 60.0


class TestDescriptorHelpers:
    def test_worker_command_falls_back_to_tool(self, monkeypatch):
        monkeypatch.setattr("vramd.registry._monorepo_tool_python", lambda tool: f"/venv/{tool}/python")
        desc = BackendDescriptor(name="d", adapter="a", vram_mib=1, priority=0, tool="text3d")
        assert desc.worker_command()[0] == "/venv/text3d/python"

    def test_worker_command_none_without_tool_or_runtime(self):
        desc = BackendDescriptor(name="d", adapter="a", vram_mib=1, priority=0)
        assert desc.worker_command() is None

    def test_runtime_wins_over_tool(self, monkeypatch):
        desc = BackendDescriptor(
            name="d",
            adapter="a",
            vram_mib=1,
            priority=0,
            tool="text3d",
            runtime=RuntimeSpec(command=("/bin/echo",)),
        )
        assert desc.worker_command() == ["/bin/echo"]

    def test_unload_frees_vram_defaults_true(self):
        assert BackendDescriptor(name="d", adapter="a", vram_mib=1, priority=0).unload_frees_vram is True

    def test_unload_frees_vram_reads_peak_profile(self):
        desc = BackendDescriptor(
            name="d", adapter="a", vram_mib=1, priority=0, peak_profile={"unload_frees_vram": False}
        )
        assert desc.unload_frees_vram is False


class TestRegistryWithLayers:
    def test_registry_uses_merged_descriptors(self, tmp_path, monkeypatch, isolated_env):
        override = write_yaml(tmp_path / "cal.yaml", [{"name": "example-diffusion", "vram_mib": 5760}])
        monkeypatch.setenv(ENV_BACKENDS_FILE, override)
        assert Registry().descriptor("example-diffusion").vram_mib == 5760

    def test_external_backend_registers_without_touching_the_package(self, tmp_path, monkeypatch, isolated_env):
        entry = {
            "name": "whisper-large-v3",
            "adapter": "meu.pacote.adapter",
            "vram_mib": 4200,
            "priority": 20,
            "runtime": {"command": ["/opt/whisper/venv/bin/python", "-m", "meu_whisper.worker"]},
            "load_keys": ["device", "compute_type", "beam_size"],
            "shape_keys": ["device", "compute_type"],
        }
        monkeypatch.setenv(ENV_BACKENDS_FILE, write_yaml(tmp_path / "whisper.yaml", [entry]))
        desc = Registry().descriptor("whisper-large-v3")
        assert desc.worker_command() == ["/opt/whisper/venv/bin/python", "-m", "meu_whisper.worker"]
        assert desc.load_keys == frozenset({"device", "compute_type", "beam_size"})
        assert desc.shape_keys == frozenset({"device", "compute_type"})
