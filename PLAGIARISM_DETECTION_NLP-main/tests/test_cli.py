import json
import sys

from plagiarism_detection.cli import main


def test_cli_compares_two_utf8_files(tmp_path, monkeypatch, capsys) -> None:
    text = "Reliable services include monitoring, testing, and clear ownership."
    source = tmp_path / "source.txt"
    candidate = tmp_path / "candidate.txt"
    source.write_text(text, encoding="utf-8")
    candidate.write_text(text, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["sourcelens", str(source), str(candidate)])

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "high_overlap"
    assert payload["similarity_score"] == 1.0
