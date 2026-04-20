from __future__ import annotations

import builtins
import importlib
import sys


def test_dump_torch_devices_lists_vulkan_compute_target(monkeypatch, capsys) -> None:
    from lada.cli import utils
    from lada.extensions import vulkan  # noqa: F401

    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(utils.torch.vulkan, "is_available", lambda: True)
    monkeypatch.setattr(utils.torch.vulkan, "get_device_name", lambda _index: "Vulkan (NCNN)")
    monkeypatch.setattr(utils, "_", lambda text: text, raising=False)

    utils.dump_torch_devices()
    output = capsys.readouterr().out

    assert "vulkan:0" in output
    assert "Vulkan (NCNN)" in output


def test_cli_parser_exposes_fp16_switches() -> None:
    from lada.cli.main import setup_argparser

    parser = setup_argparser()

    assert "--fp16" in parser._option_string_actions
    assert "--no-fp16" in parser._option_string_actions


def test_cli_vulkan_path_uses_torch_vulkan_availability(monkeypatch, capsys) -> None:
    main = importlib.import_module("lada.cli.main")
    main = importlib.reload(main)
    monkeypatch.setattr(main.torch.vulkan, "is_available", lambda: True)
    monkeypatch.setattr(builtins, "_", lambda text: text, raising=False)
    monkeypatch.setattr(sys, "argv", ["lada", "--input", "missing.mp4", "--device", "vulkan:0"])

    try:
        main.main()
    except SystemExit as exc:
        assert exc.code == 1

    output = capsys.readouterr().out
    assert "Invalid input. No file or directory at missing.mp4" in output
