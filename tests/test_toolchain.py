"""Testes da descoberta do interpretador das tools (``VRAMD_TOOLS_ROOT``).

Regressão: pastas com camelCase após um dígito (Text2Icon, Paint3D,
Skymap2D — o layout do AiGameKit) não eram encontradas porque
``str.capitalize()`` só capitaliza a primeira letra.
"""

from __future__ import annotations

from pathlib import Path

from vramd.toolchain import _camel_title, _candidate_dirs, resolve_tool_python


class TestCamelTitle:
    def test_camel_case_after_digit(self) -> None:
        # O layout do AiGameKit: pasta capitalizada por segmento.
        assert _camel_title("text2icon") == "Text2Icon"
        assert _camel_title("skymap2d") == "Skymap2D"
        assert _camel_title("paint3d") == "Paint3D"
        assert _camel_title("text2sound") == "Text2Sound"
        assert _camel_title("motion3d") == "Motion3D"

    def test_simple_names_unchanged_in_spirit(self) -> None:
        assert _camel_title("whisper") == "Whisper"
        assert _camel_title("my_tool") == "My_tool"


class TestCandidateDirs:
    def test_includes_camel_title(self) -> None:
        dirs = _candidate_dirs("text2icon")
        assert "Text2Icon" in dirs  # o que o capitalize() não dava
        assert "Text2icon" in dirs
        assert "text2icon" in dirs

    def test_no_duplicates(self) -> None:
        assert len(_candidate_dirs("text2icon")) == len(set(_candidate_dirs("text2icon")))


class TestResolveToolPython:
    def test_finds_camel_case_folder(self, tmp_path: Path, monkeypatch) -> None:
        venv = tmp_path / "Text2Icon" / ".venv" / "bin"
        venv.mkdir(parents=True)
        python = venv / "python"
        python.write_text("#!/bin/sh\n")
        python.chmod(0o755)
        monkeypatch.setenv("VRAMD_TOOLS_ROOT", str(tmp_path))
        assert resolve_tool_python("text2icon") == str(python)

    def test_plain_name_still_works(self, tmp_path: Path, monkeypatch) -> None:
        venv = tmp_path / "whisper" / ".venv" / "bin"
        venv.mkdir(parents=True)
        python = venv / "python"
        python.write_text("#!/bin/sh\n")
        python.chmod(0o755)
        monkeypatch.setenv("VRAMD_TOOLS_ROOT", str(tmp_path))
        assert resolve_tool_python("whisper") == str(python)

    def test_no_env_means_no_discovery(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("VRAMD_TOOLS_ROOT", raising=False)
        assert resolve_tool_python("text2icon") is None
