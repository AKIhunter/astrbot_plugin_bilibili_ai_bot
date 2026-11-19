"""分镜 OCR 原文、采样秒数绑定和非法输出的确定性验收。"""
import json
from unittest.mock import patch

import pytest

from core.server_ext.media import storyboard_text as ocr
from core.video_timeline import VideoTimelineError, parse_storyboard_watch


def raw(entries):
    return json.dumps({"timeline": entries, "impression": "积木拼装过程",
                       "audio": "有人讲话", "sensitive": False, "comment_hint": "聊拼装"},
                      ensure_ascii=False)


def board(text="看似一碰就散，实则环环相扣，非常牢固", status="ok"):
    return {"selected_timestamps": [1, 11, 23, 34], "ocr": {"frames": [
        {"frame_id": i, "texts": [text] if i == 3 and text else [],
         "status": status if i == 3 else "no_text"} for i in range(1, 5)]}}


def record(text, score=.99, box=(10, 10, 150, 25)):
    return {"text": text, "score": score, "box": list(box)}


def test_ocr_preserves_original_text_and_frame_time():
    result = parse_storyboard_watch(raw([{"frame_id": 4, "visual": "坦克转动炮塔"}]), board(), 300, 40)
    assert (323, "画面文字（本地OCR）：「看似一碰就散，实则环环相扣，非常牢固」") in result["timeline"]
    assert (334, "坦克转动炮塔") in result["timeline"]
    assert result["audio"] == "听不到"


def test_guessed_subtitle_is_removed_from_visual():
    result = parse_storyboard_watch(raw([{"frame_id": 4, "visual": "双手拼装绿色战车；字幕写着一键就散"}]),
                                    board(), length=40)
    assert all("一键" not in text for _, text in result["timeline"])
    assert any(at == 23 and "一碰" in text for at, text in result["timeline"])
    assert (34, "双手拼装绿色战车") in result["timeline"]


@pytest.mark.parametrize("frame_id", [True, False, 0, 5, -1, 1.0, "1", None])
def test_invalid_frame_id_rejected(frame_id):
    with pytest.raises(VideoTimelineError, match="storyboard_frame_invalid"):
        parse_storyboard_watch(raw([{"frame_id": frame_id, "visual": "物体"}]), board())


def test_duplicate_frame_and_free_seconds_rejected():
    with pytest.raises(VideoTimelineError, match="duplicate"):
        parse_storyboard_watch(raw([{"frame_id": 1, "visual": "前"}] * 2), board())
    with pytest.raises(VideoTimelineError, match="entry_mismatch"):
        parse_storyboard_watch(raw([{"at": 34, "text": "一键"}]), board())


@pytest.mark.parametrize("status", ["unavailable", "partial", "source_missing", "failed"])
def test_failed_ocr_is_explicit_abstention(status):
    result = parse_storyboard_watch(raw([{"frame_id": 1, "visual": "物体"}]), board("", status))
    assert (23, "画面文字：无法可靠辨认") in result["timeline"]


@pytest.mark.parametrize("time", [float("nan"), float("inf"), -1, True, 45])
def test_bad_sample_time_rejected(time):
    data = board()
    data["selected_timestamps"][0] = time
    with pytest.raises(VideoTimelineError, match="times_invalid"):
        parse_storyboard_watch(raw([{"frame_id": 1, "visual": "物体"}]), data, length=40)


def test_consensus_requires_matching_confident_text_and_position():
    assert ocr.consensus_records([record("余额小鲸鱼")], [record("余额小鲸鱼")])
    assert not ocr.consensus_records([record("余额小鲸鱼")], [record("余额小鱼")])
    assert not ocr.consensus_records([record("玻璃箱", .94)], [record("玻璃箱")])
    assert not ocr.consensus_records([record("玻璃箱")], [record("玻璃箱", float("nan"))])
    assert not ocr.consensus_records([record("字")], [record("字", box=(200, 100, 300, 125))])


def test_crop_requires_support_from_whole_frame():
    from PIL import Image
    source = Image.new("RGB", (360, 202))
    text = "看似一碰就散，实则环环相扣，非常牢固"
    engine = lambda *args, **kwargs: ([[text, .98]], None)
    result = ocr.confirmed_records(source, [record(text)], [record(text.replace("环环", "环不"))], engine)
    assert result[0]["text"] == text
    assert not ocr.confirmed_records(source, [record("一键")], [record("一键")], engine)


def test_missing_engine_returns_abstention():
    with patch.dict("sys.modules", {"rapidocr_onnxruntime": None}), patch.object(ocr, "_engine", None):
        data = ocr.read_storyboard_text(["missing.jpg"])
    assert data["engine"] == "unavailable"
    assert data["frames"] == [{"frame_id": 1, "texts": [], "status": "unavailable"}]
