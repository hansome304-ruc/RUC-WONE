from __future__ import annotations

from medicine_agentic.act_dataset import discover_ready


def test_discovers_ready_episodes_inside_classification_directories(tmp_path) -> None:
    first = tmp_path / "act1_xx" / "episode_1"
    second = tmp_path / "act1_xxx" / "episode_2"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "READY").write_text("{}\n", encoding="utf-8")
    (second / "READY").write_text("{}\n", encoding="utf-8")

    assert discover_ready(tmp_path) == [first, second]
