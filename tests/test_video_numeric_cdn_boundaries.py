"""视频时长溢出、备用 CDN 超时和畸形元数据；命令与网络全部替身。"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.infra.video_files import VideoFiles


def service(tmp_path):
    process = AsyncMock()
    host = SimpleNamespace(config={}, settings=SimpleNamespace(get=lambda _: False),
                           processes=SimpleNamespace(_run_process=process))
    return VideoFiles(host), process


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"),
                                  "1e9999", 10 ** 1000, True, -1, None, "bad"])
def test_bad_download_duration_uses_safe_default(tmp_path, value):
    engine, _ = service(tmp_path)
    assert engine._video_download_timeout(value) == 120


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-1", "1e9999", "N/A"])
async def test_ffprobe_invalid_duration_is_unknown(tmp_path, value):
    engine, process = service(tmp_path)
    process.return_value = (0, value, "")
    assert await engine._get_video_duration("fixture.mp4") == 0


@pytest.mark.asyncio
async def test_frames_never_pass_nonfinite_timestamp_to_ffmpeg(tmp_path):
    engine, process = service(tmp_path)
    process.side_effect = [(0, "inf", ""), (1, "", ""), (1, "", "")]
    await engine._extract_video_frames(str(tmp_path / "clip.mp4"), count=2)
    seeks = [call.args[call.args.index("-ss") + 1]
             for call in process.await_args_list if "-ss" in call.args]
    assert seeks == ["10.00", "20.00"]


@pytest.mark.asyncio
async def test_long_cdn_timeout_stops_format_retries(tmp_path):
    engine, process = service(tmp_path)
    metadata = {"formats": [{"url": "https://original.bilivideo.com/path?sign=fixture",
                              "fragments": [None, {"url": "https://original.bilivideo.com/chunk"}]}]}
    process.side_effect = [(0, json.dumps(metadata), ""), (124, "", "timeout")]
    engine._reachable_cdn_host = AsyncMock(side_effect=["", engine._CDN_MIRROR_HOSTS[0]])
    result = await engine._download_via_reachable_cdn(
        "BV1TEST", 480, "fixture-cookie.txt", output_dir=str(tmp_path), duration=3600)
    assert result is None
    assert [call.kwargs["timeout"] for call in process.await_args_list] == [60, 1800]
    assert not (tmp_path / "_cdn_BV1TEST.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [[], None, {"formats": "bad"}])
async def test_invalid_cdn_metadata_falls_back(tmp_path, metadata):
    engine, process = service(tmp_path)
    process.return_value = (0, json.dumps(metadata), "")
    assert await engine._download_via_reachable_cdn(
        "BV1TEST", 480, "fixture-cookie.txt", output_dir=str(tmp_path)) is None
    process.assert_awaited_once()
