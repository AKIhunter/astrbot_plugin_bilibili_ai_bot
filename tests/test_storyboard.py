"""原帧裁切、检测不可用、文字预算与模型输入映射；无外部调用。"""
import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from core.domain.video_watch import VideoWatch
from core.server_ext.media import storyboard


def frame(tmp_path, name="frame.png", size=(360, 200)):
    path = tmp_path / name
    Image.new("RGB", size, (80, 110, 140)).save(path)
    return str(path)


def test_compact_frame_and_unavailable_detector(tmp_path):
    source = frame(tmp_path)
    with patch.object(storyboard, "_text_regions", return_value=[]):
        result = storyboard._render_storyboard([source], 360, 200, 3, 3, str(tmp_path))
    assert result["grid_size"] == [720, 400]
    assert result["resize_reason"] == "compact"
    Path(result["grid_path"]).unlink()
    with patch.object(storyboard, "np", None):
        result = storyboard._render_storyboard([source], 360, 200, 3, 3, str(tmp_path))
    assert result["grid_size"] == [1080, 600]
    assert result["resize_reason"] == "text_detection_unavailable"


def test_text_crop_uses_original_frame_pixels(tmp_path):
    source = frame(tmp_path, size=(1200, 800))
    region = {"box": [400, 500, 900, 540], "glyph_height": 20}
    with patch.object(storyboard, "_text_regions", return_value=[region]):
        result = storyboard._render_storyboard([source], 360, 200, 3, 3, str(tmp_path))
    with Image.open(result["text_detail"]["path"]) as crop:
        assert crop.size == (500 + storyboard.TEXT_DETAIL_LABEL_W, 40)
        # 交付为 JPEG，允许编码量化造成的少量颜色差异。
        pixel = crop.getpixel((storyboard.TEXT_DETAIL_LABEL_W + 10, 10))
        assert max(abs(a - b) for a, b in zip(pixel, (80, 110, 140))) <= 3
    assert "宫格[1]" in result["text_detail"]["manifest"]


def test_text_budget_falls_back_to_full_grid(tmp_path):
    source = frame(tmp_path)
    region = {"box": [30, 150, 330, 180], "glyph_height": 8}
    with patch.object(storyboard, "_text_regions", return_value=[region]):
        result = storyboard._render_storyboard([source] * 9, 360, 200, 3, 3, str(tmp_path))
    assert result["resize_reason"] == "text_detail_budget"
    assert result["grid_size"] == [1080, 600]
    assert result["text_detail"] is None


def test_model_sees_crop_mapping_and_cleanup_preserves_video(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"0" * 2048)
    grid, detail = frame(tmp_path, "grid.jpg"), frame(tmp_path, "detail.jpg", (300, 30))
    board = {"grid_path": grid, "frames": [], "manifest": "[1] 0:03",
             "text_detail": {"path": detail, "manifest": "文字裁切[1] 对应宫格[1]"},
             "ocr": {"frames": [{"frame_id": 1, "texts": [], "status": "unavailable"}]}}
    watcher = SimpleNamespace(settings=SimpleNamespace(bool=lambda _: True),
                              _cleanup_storyboard=VideoWatch._cleanup_storyboard)
    with patch.object(storyboard, "build_storyboard", return_value=board):
        content, returned = asyncio.run(VideoWatch._storyboard_content(watcher, str(video), "看片"))
    assert returned is board
    assert [part["type"] for part in content] == ["image_url", "image_url", "text"]
    assert base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1]) == Path(detail).read_bytes()
    assert "文字无法可靠辨认" in content[-1]["text"]
    assert "对应宫格[1]" in content[-1]["text"]
    VideoWatch._cleanup_storyboard(board)
    assert not Path(grid).exists() and not Path(detail).exists()
    assert video.exists()
