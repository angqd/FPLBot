"""CLI state parsing, offline loading and profile selection. No network."""
import gzip
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fplbot import cli, config
from fplbot.data import store


def _team(limit=2, made=0, status="cost", picks=None):
    picks = picks if picks is not None else [
        {"element": 1, "position": 1, "selling_price": 45, "purchase_price": 45},
        {"element": 2, "position": 2, "selling_price": 59, "purchase_price": 55},
    ]
    return {"picks": picks,
            "transfers": {"cost": 4, "status": status, "limit": limit, "made": made, "bank": 41, "value": 992}}


# ---- parse_my_team ---------------------------------------------------------

def test_parse_normal_two_free_transfers():
    picks, bank, ft, unlimited = cli.parse_my_team(_team(limit=2, made=0))
    assert picks == {1: 4.5, 2: 5.9}  # selling_price, not purchase/now_cost
    assert bank == pytest.approx(4.1)
    assert (ft, unlimited) == (2, False)


def test_parse_used_transfer_is_zero_not_one():
    _, _, ft, unlimited = cli.parse_my_team(_team(limit=1, made=1))
    assert (ft, unlimited) == (0, False)


def test_parse_over_limit_floors_at_zero():
    assert cli.parse_my_team(_team(limit=1, made=3))[2] == 0


@pytest.mark.parametrize("limit,status", [(None, "cost"), (None, "unlimited"), (2, "unlimited")])
def test_parse_unlimited(limit, status):
    _, _, ft, unlimited = cli.parse_my_team(_team(limit=limit, status=status))
    assert unlimited and ft is None


def test_parse_missing_selling_price_raises():
    bad = [{"element": 1, "selling_price": 45}, {"element": 2, "purchase_price": 55, "now_cost": 60}]
    with pytest.raises(ValueError, match="selling_price"):
        cli.parse_my_team(_team(picks=bad))


def test_state_from_api_unlimited_uses_cap():
    class FakeClient:
        def my_team(self, entry):
            return _team(limit=None, status="unlimited")

    s = cli._state_from_api(FakeClient(), 1, max_free_transfers=5)
    assert s.free_transfers == 5 and s.bank == pytest.approx(4.1)


# ---- offline loading -------------------------------------------------------

BS = {"events": [{"id": 7, "is_next": True}], "elements": []}
FX = [{"id": 1}]


def _write_gz(d: Path, bs=BS, fx=FX):
    d.mkdir(parents=True)
    for name, obj in (("bootstrap", bs), ("fixtures", fx)):
        with gzip.open(d / f"{name}.json.gz", "wt") as f:
            json.dump(obj, f)


def test_offline_gz_snapshot_dir(tmp_path):
    snap = tmp_path / "2026-11-25T100000Z"
    _write_gz(snap)
    assert cli._load(snap) == (BS, FX)


def test_offline_raw_dir_picks_newest(tmp_path):
    _write_gz(tmp_path / "2026-11-24T100000Z", bs={"old": True})
    _write_gz(tmp_path / "2026-11-25T100000Z")
    assert cli._load(tmp_path)[0] == BS


def test_offline_uncompressed_still_works(tmp_path):
    (tmp_path / "bootstrap.json").write_text(json.dumps(BS))
    (tmp_path / "fixtures.json").write_text(json.dumps(FX))
    assert cli._load(tmp_path) == (BS, FX)


def test_offline_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        cli._load(Path("latest"))
    _write_gz(tmp_path / "raw" / "2026-11-25T100000Z")
    assert cli._load(Path("latest")) == (BS, FX)


def test_offline_missing_files_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        cli._load(tmp_path)


# ---- profiles / env file selection ------------------------------------------

@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Point config at temp files; restore the real profile afterwards."""
    for k in ("FPLBOT_ENV_FILE", "FPLBOT_PROFILE", "FPL_REFRESH_TOKEN", "FPL_TOKEN_URL", "FPL_CLIENT_ID"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FPLBOT_DATA_DIR", str(tmp_path / "data"))
    yield monkeypatch
    monkeypatch.undo()
    config.activate_profile()


def test_resolve_env_file():
    root = config.ROOT
    assert config.resolve_env_file(None, environ={}) == root / ".env"
    assert config.resolve_env_file("bot", environ={}) == root / ".env.bot"
    assert config.resolve_env_file("bot", environ={"FPLBOT_ENV_FILE": "/x/.env.custom"}) == Path("/x/.env.custom")
    with pytest.raises(ValueError):
        config.resolve_env_file("../evil", environ={})


def test_env_file_selects_env_state_dir_and_no_leak(tmp_path, isolated_env):
    a, b = tmp_path / ".env.a", tmp_path / ".env.b"
    a.write_text("FPL_REFRESH_TOKEN=token-a\nFPL_ENTRY_ID=111\n")
    b.write_text("FPL_ENTRY_ID=222\n")
    isolated_env.setenv("FPLBOT_ENV_FILE", str(a))
    assert config.activate_profile("a") == a
    assert config.ENV_FILE == a and config.load_config().entry_id == 111
    assert config.STATE_DIR == tmp_path / "data" / "a"
    assert store._state_db_path() == tmp_path / "data" / "a" / "fplbot.sqlite"

    isolated_env.setenv("FPLBOT_ENV_FILE", str(b))
    config.activate_profile("b")
    import os
    assert os.environ.get("FPL_REFRESH_TOKEN") is None  # profile a's token must not leak into b
    assert config.load_config().entry_id == 222


def test_profile_decisions_do_not_collide(tmp_path, isolated_env):
    for name in ("a", "b"):
        (tmp_path / f".env.{name}").write_text("")
    isolated_env.setenv("FPLBOT_ENV_FILE", str(tmp_path / ".env.a"))
    config.activate_profile("a")
    store.record_decision(12, {"x": 1}, executed=True)
    assert store.already_executed(12)
    isolated_env.setenv("FPLBOT_ENV_FILE", str(tmp_path / ".env.b"))
    config.activate_profile("b")
    assert not store.already_executed(12)


def test_auth_writes_rotated_token_to_selected_env(tmp_path, isolated_env):
    from fplbot.data import auth

    env = tmp_path / ".env.bot"
    env.write_text("FPL_REFRESH_TOKEN=old\nFPL_TOKEN_URL=https://example.invalid/token\nFPL_CLIENT_ID=cid\n")
    isolated_env.setenv("FPLBOT_ENV_FILE", str(env))
    config.activate_profile("bot")

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"access_token": "acc", "expires_in": 3600, "refresh_token": "new"}

    isolated_env.setattr(auth.httpx, "post", lambda *a, **k: Resp())
    assert auth.OIDCAuth().access_token() == "acc"
    assert "FPL_REFRESH_TOKEN='new'" in env.read_text() or "FPL_REFRESH_TOKEN=new" in env.read_text()


def test_cli_profile_missing_env_file_errors(isolated_env):
    r = CliRunner().invoke(cli.app, ["--profile", "definitely-not-a-profile-xyz", "draft", "--offline", "tests/fixtures"])
    assert r.exit_code == 2 and "does not exist" in r.output
