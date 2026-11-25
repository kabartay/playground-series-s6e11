"""Command-line checks of train_gbdt.py: impossible option combinations fail before any data is read."""

from __future__ import annotations

import pytest

from common import ConfigError
from train_gbdt import check_args, parse_args

BASE = ["--name", "x"]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--model", "lgb", "--offset", "glm"], "needs --orig 0"),
        (["--model", "glm", "--offset", "glm", "--orig", "0", "--te", "single"], "needs a booster"),
        (["--model", "realmlp", "--orig", "1"], "realmlp needs --orig 0"),
        (["--model", "lgb", "--hier", "20", "--te", "single"], "--hier needs"),
        (["--model", "lgb", "--key-counts"], "--key-counts needs"),
        (["--model", "lgb", "--neg-frac", "0"], "--neg-frac"),
        (["--model", "lgb", "--folds", "6"], "--folds"),
        (["--model", "lgb", "--params", "{bad"], "not valid JSON"),
        (["--model", "lgb", "--params", "[1]"], "JSON object"),
    ],
)
def test_check_args_refuses(argv: list[str], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        check_args(parse_args(BASE + argv))


def test_check_args_accepts_the_round_eleven_recipe() -> None:
    argv = "--model lgb --offset glm --fe art --orig 0 --te single --keys full --te-smooth 5,20,100"
    argv += " --joint glm --key-counts --es inner --seeds 3"
    check_args(parse_args(BASE + argv.split()))


def test_main_turns_config_errors_into_exit_status_2(capsys: pytest.CaptureFixture[str]) -> None:
    from train_gbdt import main

    assert main([*BASE, "--model", "lgb", "--folds", "0"]) == 2
    assert "error: --folds" in capsys.readouterr().err
