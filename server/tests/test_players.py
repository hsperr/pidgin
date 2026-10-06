"""The player record: Elo against the bots, names, forfeits, the champion.

    python3 -m pytest -q tests/test_players.py
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emergent import players  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(players, "DATA_DIR", tmp_path)
    monkeypatch.setattr(players._LOCAL, "conn", None, raising=False)
    yield


def test_a_board_is_rated_once():
    assert players.board_started("a" * 32, "deal1")
    x = players.board_finished("a" * 32, "deal1", 5)
    assert x == {"before": 1500, "after": 1512}
    assert players.board_finished("a" * 32, "deal1", 5) is None
    assert not players.board_started("a" * 32, "deal1")      # the same deal again is not rated
    assert players.me("a" * 32)["boards"] == 1


def test_a_tie_against_equal_bots_changes_nothing():
    players.board_started("a" * 32, "d")
    assert players.board_finished("a" * 32, "d", 0) == {"before": 1500, "after": 1500}


def test_an_unfinished_board_is_a_loss_later():
    players.board_started("a" * 32, "d")
    players.forfeit_open("a" * 32)
    m = players.me("a" * 32)
    assert (m["rating"], m["boards"]) == (1488, 1)
    assert players.board_finished("a" * 32, "d", 30) is None   # too late to finish it


def test_names_are_checked_and_unique():
    assert players.set_name("a" * 32, "x")
    assert players.set_name("a" * 32, "Ann") is None
    assert players.set_name("b" * 32, "ann") == "that name is taken"


def test_the_champion_needs_a_name_and_a_record():
    for k in range(players.CHAMPION_MIN_BOARDS):
        players.board_started("a" * 32, f"d{k}")
        players.board_finished("a" * 32, f"d{k}", 3)
    assert not players.me("a" * 32)["champion"]                # no name, not on the board
    players.set_name("a" * 32, "Ann")
    assert players.me("a" * 32)["champion"]
    assert [r["name"] for r in players.leaderboard()] == ["Ann"]
