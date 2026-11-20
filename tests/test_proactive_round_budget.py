"""Keep browsing rounds, watched videos and comment budgets in their own units."""
import asyncio
from datetime import datetime
import types
from unittest.mock import AsyncMock

import pytest

from test_autonomous_ranges import _install_astrbot_stub

_install_astrbot_stub()

from core.behavior_budget import BehaviorBudget
from core.features import proactive_browse
from core.lifecycle.schedule_rules import ScheduleRules
from core.schedule_files import ScheduleFiles
from core.settings import Settings


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 11, 19, 12, 0, 0, tzinfo=tz)


BASE_CONFIG = {
    "ENABLE_PROACTIVE": True,
    "ENABLE_AUTONOMOUS_DAILY_PLAN": False,
    "PROACTIVE_TIMES_COUNT": 6,
    "AUTONOMOUS_PROACTIVE_DAILY_MAX": 6,
    "PROACTIVE_VIDEO_COUNT": 5,
    "PROACTIVE_DAILY_LIMIT": 0,
    "PROACTIVE_COMMENT_DAILY_LIMIT": 30,
    "FIXED_PROACTIVE_WINDOWS": [
        "10:00-11:30", "14:00-15:00", "17:00-18:00",
        "19:30-21:00", "22:00-23:00",
    ],
}


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr(proactive_browse, "datetime", FixedDatetime)


def host(overrides=None):
    owner = types.SimpleNamespace(settings=Settings({**BASE_CONFIG, **(overrides or {})}))
    owner.schedule_rules = ScheduleRules(lambda: owner.settings)
    return owner


def watch_records(count, day="2026-11-19"):
    sources = ("manual", "llm", "follow", "search", "pool", "special_follow")
    return [
        {"time": f"{day} 10:00:00", "source": sources[index % len(sources)],
         "bvid": f"seen{index}", "error": "evaluation_failed" if index == 0 else ""}
        for index in range(count)
    ]


@pytest.mark.parametrize("already_watched", [0, 5, 6, 10, 20])
def test_unlimited_video_day_keeps_full_round(already_watched):
    browse = proactive_browse.ProactiveBrowse(host())
    assert browse._proactive_round_budget(watch_records(already_watched)) == (5, 30)


@pytest.mark.parametrize("autonomous", [False, True])
@pytest.mark.parametrize("round_cap", [1, 6, 12])
def test_round_cap_does_not_become_a_video_cap(autonomous, round_cap):
    browse = proactive_browse.ProactiveBrowse(host({
        "ENABLE_AUTONOMOUS_DAILY_PLAN": autonomous,
        "AUTONOMOUS_PROACTIVE_DAILY_MAX": round_cap,
    }))
    assert browse._proactive_round_budget(watch_records(6)) == (5, 30)


@pytest.mark.parametrize("limit,watched,expected", [
    (12, 0, 5), (12, 5, 5), (12, 10, 2), (12, 12, None),
    (3, 0, 3), (3, 3, None),
])
def test_explicit_video_day_limit_is_respected(limit, watched, expected):
    browse = proactive_browse.ProactiveBrowse(host({"PROACTIVE_DAILY_LIMIT": limit}))
    assert browse._proactive_round_budget(watch_records(watched)) == (expected, 30)


@pytest.mark.parametrize("watched,requested,comments,expected", [
    (10, 3, 2, 2), (10, 8, 0, 2), (12, 1, 7, None),
])
def test_manual_request_keeps_remaining_video_and_comment_limits(watched, requested, comments, expected):
    browse = proactive_browse.ProactiveBrowse(host({"PROACTIVE_DAILY_LIMIT": 12}))
    assert browse._proactive_round_budget(
        watch_records(watched), max_watch=requested, max_comment=comments,
    ) == (expected, comments)


def test_failed_records_and_all_sources_count_toward_explicit_video_limit():
    browse = proactive_browse.ProactiveBrowse(host({"PROACTIVE_DAILY_LIMIT": 7}))
    assert browse._proactive_round_budget(watch_records(6)) == (1, 30)


def test_previous_day_records_do_not_consume_today_video_limit():
    browse = proactive_browse.ProactiveBrowse(host({"PROACTIVE_DAILY_LIMIT": 3}))
    assert browse._proactive_round_budget(watch_records(10, "2026-11-18")) == (3, 30)


@pytest.mark.parametrize("round_cap,expected_rounds", [(6, 5), (2, 2), (0, 0)])
def test_fixed_schedule_still_respects_round_cap(round_cap, expected_rounds):
    owner = host({"PROACTIVE_TIMES_COUNT": round_cap, "AUTONOMOUS_PROACTIVE_DAILY_MAX": round_cap})
    saved = []
    owner.daily_plan = types.SimpleNamespace(_autonomous_plan_for_today=lambda: {})
    owner.states = types.SimpleNamespace(schedule=types.SimpleNamespace(write=saved.append))
    times, triggered = ScheduleFiles(owner)._generate_daily_schedule()
    assert len(times) == len(saved[-1]["proactive_windows"]) == expected_rounds
    assert triggered == set()


def test_autonomous_schedule_keeps_plan_time_pairs():
    owner = host({"ENABLE_AUTONOMOUS_DAILY_PLAN": True})
    windows = [{"start_time": "11:00", "end_time": "12:00", "scheduled_time": "11:15"}]
    owner.daily_plan = types.SimpleNamespace(
        _autonomous_plan_for_today=lambda: {"proactive_windows": windows},
        _plan_time_pairs=lambda _key: [(11, 15)],
    )
    owner.states = types.SimpleNamespace(schedule=types.SimpleNamespace(write=lambda _value: None))
    assert ScheduleFiles(owner)._generate_daily_schedule()[0] == [(11, 15)]


@pytest.mark.parametrize("kind,video_limit,expected", [
    ("proactive_watch", 0, None), ("proactive_watch", 12, 12),
    ("proactive_comment", 0, 30),
])
def test_action_budget_uses_video_and_comment_units(kind, video_limit, expected):
    config = {**BASE_CONFIG, "PROACTIVE_DAILY_LIMIT": video_limit}
    rules = BehaviorBudget(lambda key, default=None: config.get(key, default)).rules_for(
        types.SimpleNamespace(kind=kind), 0,
    )
    limits = {bucket: value for bucket, _window, value in rules}
    assert limits.get(f"behavior:{kind}:day") == expected


@pytest.mark.parametrize("video_limit,expected_round_counts", [
    (0, [5, 5, 5, 5, 5]), (12, [5, 5, 2, 0, 0]),
])
def test_five_browse_rounds_use_the_configured_video_quota(monkeypatch, video_limit, expected_round_counts):
    """Run the actual browse loop while every download, model and outward action is fake."""
    owner = host({"PROACTIVE_DAILY_LIMIT": video_limit})
    ledger, watched, comment_limits = [], [], []
    owner.environment = types.SimpleNamespace(
        _get_environment_status=lambda: {"features": {"proactive_video_media": True}},
    )
    owner.activity_logs = types.SimpleNamespace(
        watch=types.SimpleNamespace(recent=lambda: ledger),
        proactive=types.SimpleNamespace(recent=lambda: []),
    )
    owner.owner_recommend = types.SimpleNamespace(_load_owner_recommendation_state=lambda _log: ({}, 0))
    owner.memory_docs = types.SimpleNamespace(external_memory=types.SimpleNamespace(read=lambda _default: {}))
    owner.processed_keys = types.SimpleNamespace(commented_video=types.SimpleNamespace(all=lambda: set()))
    owner.seen_videos = types.SimpleNamespace(bvids=AsyncMock(return_value=set()))
    owner.taste = types.SimpleNamespace(_get_taste_tids=lambda: [17])
    owner.video_discovery = types.SimpleNamespace(
        _should_watch_video_before_download=AsyncMock(return_value=(True, "")),
    )
    owner.now_doing = types.SimpleNamespace(report=lambda *_args, **_kwargs: None)

    async def watch(round_state, video):
        round_state.watch_count += 1
        watched.append(video["bvid"])
        ledger.append({"time": "2026-11-19 12:00:00", "bvid": video["bvid"], "source": "pool"})
        comment_limits.append(round_state.daily_comment_limit)
        return True

    owner.watch_video = types.SimpleNamespace(
        _proactive_per_video_comment=lambda: 1,
        _today_proactive_comment_count=lambda *_args: 0,
        _reserve_proactive_watch=AsyncMock(return_value=types.SimpleNamespace(success=True)),
        _watch_one_video=watch,
    )
    browse = proactive_browse.ProactiveBrowse(owner)

    async def candidates(_log, _seen, _budget, _focus=None):
        return [
            {"bvid": f"round{len(watched)}video{index}", "title": "offline", "up_mid": "123", "_source": "pool"}
            for index in range(8)
        ]

    browse._collect_proactive_candidates = candidates
    browse._post_video_dynamic = AsyncMock()
    monkeypatch.setattr(proactive_browse.asyncio, "sleep", AsyncMock())

    async def run_rounds():
        actual = []
        for _round in range(5):
            previous_count = len(watched)
            await browse._run_proactive_inner()
            actual.append(len(watched) - previous_count)
        return actual

    assert asyncio.run(run_rounds()) == expected_round_counts
    expected_total = sum(expected_round_counts)
    assert owner.watch_video._reserve_proactive_watch.await_count == expected_total
    assert len(set(watched)) == expected_total
    assert comment_limits == [30] * expected_total
