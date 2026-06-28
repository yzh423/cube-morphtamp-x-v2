from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_update_readme_module():
    path = ROOT / "tools" / "update_readme.py"
    spec = importlib.util.spec_from_file_location("update_readme", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readme_is_generated_from_current_code():
    module = load_update_readme_module()

    expected = module.build_readme()
    actual = (ROOT / "README.md").read_text(encoding="utf-8")

    assert actual == expected


def test_readme_lists_every_cli_subcommand():
    from morphtamp_x_v2.cli import parser

    module = load_update_readme_module()
    readme = module.build_readme()
    subparsers = next(
        action for action in parser()._actions
        if getattr(action, "dest", None) == "command"
    )

    assert "## CLI command reference" in readme
    for command in sorted(subparsers.choices):
        assert f"| `{command}` |" in readme


def test_update_readme_check_mode_reports_fresh_file(capsys):
    module = load_update_readme_module()

    assert module.main(["--check"]) == 0

    captured = capsys.readouterr()
    assert "README.md is up to date" in captured.out
