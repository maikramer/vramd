"""Cache HF: specs, pins/GC e migração do cache existente."""

from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

import pytest

from vramd import protocol as P
from vramd.models import ModelError, ModelManager, ModelSpec, move_hf_cache


class TestModelSpec:
    def test_from_dict(self) -> None:
        spec = ModelSpec.from_dict(
            {"repo": "org/model", "revision": "v1", "allow": ["*.safetensors"], "ignore": ["*.md"]}
        )
        assert spec.repo == "org/model"
        assert spec.revision == "v1"
        assert spec.allow == ("*.safetensors",)
        assert spec.ignore == ("*.md",)
        assert spec.key == "org/model@v1"

    @pytest.mark.parametrize("bad", ["", "../etc", "/abs/path", None])
    def test_rejects_bad_repo(self, bad) -> None:
        with pytest.raises(ModelError):
            ModelSpec.from_dict({"repo": bad})


class TestEnsure:
    def _fake_hub(self, monkeypatch, *, cached: bool, download_path: str = "/cache/model") -> types.ModuleType:
        module = types.ModuleType("huggingface_hub")
        calls: list[dict] = []

        def snapshot_download(**kwargs):
            calls.append(kwargs)
            if kwargs.get("local_files_only"):
                if cached:
                    return download_path
                raise FileNotFoundError("not cached")
            return download_path

        module.snapshot_download = snapshot_download
        module.calls = calls
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)
        return module

    def test_cached_is_pinned(self, monkeypatch, tmp_path: Path) -> None:
        module = self._fake_hub(monkeypatch, cached=True)
        manager = ModelManager(home=tmp_path, policy="off")
        result = manager.ensure([ModelSpec("org/model")])
        assert result[0]["path"] == "/cache/model"
        assert manager.pinned() == {"org/model@main"}
        assert module.calls[0]["local_files_only"] is True
        manager.unpin(ModelSpec("org/model"))
        assert manager.pinned() == set()

    def test_missing_with_policy_off(self, monkeypatch, tmp_path: Path) -> None:
        self._fake_hub(monkeypatch, cached=False)
        manager = ModelManager(home=tmp_path, policy="off")
        with pytest.raises(ModelError) as exc:
            manager.ensure([ModelSpec("org/model")])
        assert exc.value.code == P.ERR_MODEL_MISSING

    def test_downloads_with_token(self, monkeypatch, tmp_path: Path) -> None:
        module = self._fake_hub(monkeypatch, cached=False, download_path="/downloaded")
        manager = ModelManager(home=tmp_path, policy="hub", token="hf_secret")
        result = manager.ensure([ModelSpec("org/model", revision="v2")])
        assert result[0]["path"] == "/downloaded"
        assert any(call.get("token") == "hf_secret" and not call.get("local_files_only") for call in module.calls)


class TestGC:
    def _info(self):
        class Revision:
            def __init__(self, commit, refs, size, accessed):
                self.commit_hash = commit
                self.refs = refs
                self.size_on_disk = size
                self.last_accessed = accessed

        class Repo:
            def __init__(self, repo_id, revisions):
                self.repo_id = repo_id
                self.repo_type = "model"
                self.revisions = revisions
                self.size_on_disk = sum(r.size_on_disk for r in revisions)

        class Info:
            def __init__(self, repos):
                self.repos = repos
                self.cache_dir = Path("/cache/hf")
                self.size_on_disk = sum(r.size_on_disk for r in repos)
                self.deleted: list[str] = []

            def delete_revisions(self, *hashes):
                info = self

                class Strategy:
                    def execute(self):
                        info.deleted.extend(hashes)

                return Strategy()

        repo_a = Repo(
            "org/a",
            [
                Revision("aaaa", ["main"], 100, 10),
                Revision("bbbb", ["old"], 100, 1),
            ],
        )
        repo_b = Repo("org/b", [Revision("cccc", ["main"], 100, 5)])
        return Info([repo_a, repo_b])

    def test_gc_keeps_pinned_revision(self, monkeypatch, tmp_path: Path) -> None:
        info = self._info()
        module = types.ModuleType("huggingface_hub")
        seen: dict[str, object] = {}

        def _scan(cache_dir=None):
            seen["cache_dir"] = cache_dir
            return info

        module.scan_cache_dir = _scan
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)
        manager = ModelManager(home=tmp_path, max_gib=0)
        # Pina a revision main de org/a; a revision `old` fica elegível.
        manager.pin(ModelSpec("org/a", revision="main"))
        result = manager.gc()
        assert "bbbb" in info.deleted
        assert "aaaa" not in info.deleted
        assert result["removed"] >= 1
        # O `home` do config manda, não o HF_HOME global do processo.
        assert seen["cache_dir"] == str(tmp_path / "hub")

    def test_gc_keeps_recent_revision(self, monkeypatch, tmp_path: Path) -> None:
        """O tecto não apaga um modelo descarregado há minutos."""
        info = self._info()
        # `cccc` foi tocado agora; as revisions antigas (epoch) são elegíveis.
        for repo in info.repos:
            if repo.repo_id == "org/b":
                repo.revisions[0].last_accessed = time.time()
        module = types.ModuleType("huggingface_hub")
        module.scan_cache_dir = lambda **_kwargs: info
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)
        result = ModelManager(home=tmp_path, max_gib=0).gc()
        assert "cccc" not in info.deleted
        assert "bbbb" in info.deleted
        assert result["removed"] >= 1

    def test_list_tolerates_hub_2_4_last_modified(self, monkeypatch, tmp_path: Path) -> None:
        """huggingface_hub 2.4 removeu ``last_accessed`` do CachedRevisionInfo."""
        info = self._info()
        for repo in info.repos:
            for revision in repo.revisions:
                del revision.last_accessed
                revision.last_modified = 123.0
        module = types.ModuleType("huggingface_hub")
        module.scan_cache_dir = lambda **_kwargs: info
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)
        result = ModelManager(home=tmp_path).list()
        assert result["repos"][0]["revisions"][0]["last_accessed"] == 123.0


class TestMoveCache:
    def _source(self, tmp_path: Path) -> Path:
        source = tmp_path / "old-hf"
        (source / "hub" / "models--org--model" / "snapshots" / "abc").mkdir(parents=True)
        (source / "hub" / "models--org--model" / "blobs").mkdir(parents=True)
        (source / "hub" / "models--org--model" / "blobs" / "weight").write_bytes(b"W" * 1000)
        os.symlink(
            "../../blobs/weight",
            source / "hub" / "models--org--model" / "snapshots" / "abc" / "weight",
        )
        (source / "token").write_text("hf_x", encoding="utf-8")
        return source

    @pytest.mark.skipif(os.name == "nt", reason="symlink relativo do cache HF: semântica POSIX")
    def test_move_same_filesystem(self, tmp_path: Path) -> None:
        source = self._source(tmp_path)
        dest = tmp_path / "new-hf"
        result = move_hf_cache(source, dest)
        assert result["moved"] is True
        assert result["same_filesystem"] is True
        assert not source.exists()
        assert (dest / "token").read_text(encoding="utf-8") == "hf_x"
        link = dest / "hub" / "models--org--model" / "snapshots" / "abc" / "weight"
        assert link.is_symlink()
        assert os.readlink(link) == "../../blobs/weight"
        assert link.read_bytes() == b"W" * 1000

    def test_keep_source_copies(self, tmp_path: Path) -> None:
        source = self._source(tmp_path)
        dest = tmp_path / "new-hf"
        result = move_hf_cache(source, dest, delete_source=False)
        assert result["moved"] is True
        assert source.exists()
        assert (dest / "token").read_text(encoding="utf-8") == "hf_x"
        assert (source / "token").exists()

    def test_dry_run_does_not_touch(self, tmp_path: Path) -> None:
        source = self._source(tmp_path)
        dest = tmp_path / "new-hf"
        result = move_hf_cache(source, dest, dry_run=True)
        assert result["dry_run"] is True
        assert result["files"] == 2  # token + blob
        assert result["symlinks"] == 1
        assert source.exists() and not dest.exists()

    def test_missing_source(self, tmp_path: Path) -> None:
        with pytest.raises(ModelError):
            move_hf_cache(tmp_path / "nope", tmp_path / "dest")
