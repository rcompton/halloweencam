import os
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import pytest
from ghoulfluids.config import AppConfig
from ghoulfluids.debug import DebugOverlay

# Mock freetype before importing DebugOverlay if it's imported at the module level
# In this case, it's fine since we're mocking it in the test functions.

@pytest.fixture
def mock_ctx():
    """Provides a mocked ModernGL context."""
    ctx = MagicMock()
    ctx.program.return_value = MagicMock()
    ctx.buffer.return_value = MagicMock()
    ctx.vertex_array.return_value = MagicMock()
    ctx.sampler.return_value = MagicMock()
    ctx.texture.return_value = MagicMock()
    return ctx

@pytest.fixture
def config():
    """Provides a default AppConfig."""
    return AppConfig()

@patch('ghoulfluids.debug.freetype')
@patch('ghoulfluids.debug.os.path.exists', return_value=True)
def test_debug_overlay_init(mock_exists, mock_freetype, mock_ctx, config):
    """Test the initialization of the DebugOverlay."""
    # Mock the freetype face
    mock_face = MagicMock()
    mock_glyph = MagicMock()
    mock_glyph.advance.x = 16 << 6  # Example advance value
    mock_glyph.bitmap.rows = 16
    mock_glyph.bitmap.width = 16
    mock_glyph.bitmap.buffer = [0] * (16*16)
    mock_face.glyph = mock_glyph
    mock_face.load_char.return_value = MagicMock(glyph=mock_glyph)
    mock_freetype.Face.return_value = mock_face

    overlay = DebugOverlay(mock_ctx, config)
    assert overlay.font_loaded
    mock_ctx.program.assert_called_once()
    mock_ctx.buffer.assert_called_once()
    mock_ctx.vertex_array.assert_called_once()

@patch('ghoulfluids.debug.os.path.exists')
def test_find_font_path(mock_exists, mock_ctx, config):
    """Test the _find_font_path method."""
    # Test case where a font is found
    mock_exists.side_effect = lambda p: "FiraCode-SemiBold.ttf" in p
    overlay = DebugOverlay(mock_ctx, config)
    assert "FiraCode-SemiBold.ttf" in overlay._find_font_path()

    # Test case where no font is found
    mock_exists.side_effect = lambda p: False
    overlay = DebugOverlay(mock_ctx, config)
    assert overlay._find_font_path() is None

@patch('ghoulfluids.debug.freetype')
@patch('ghoulfluids.debug.os.path.exists', return_value=True)
def test_render(mock_exists, mock_freetype, mock_ctx, config):
    """Test the render method."""
    # Setup mocks
    mock_face = MagicMock()
    mock_glyph = MagicMock()
    mock_glyph.advance.x = 16 << 6
    mock_glyph.bitmap.rows = 16
    mock_glyph.bitmap.width = 16
    mock_glyph.bitmap_left = 0
    mock_glyph.bitmap_top = 16
    mock_glyph.bitmap.buffer = [0] * (16*16)
    mock_face.glyph = mock_glyph
    mock_face.load_char.return_value = MagicMock(glyph=mock_glyph)
    mock_freetype.Face.return_value = mock_face

    # Mock the texture's width attribute used in _add_char_quad
    mock_ctx.texture.return_value.width = 512
    mock_ctx.texture.return_value.height = 512

    overlay = DebugOverlay(mock_ctx, config)
    overlay.font_loaded = True # Assume font loaded successfully

    # Call render
    lines = ["line 1", "line 2"]
    overlay.render(lines, 10, 100)

    # Assertions
    assert overlay.char_count == sum(len(line) for line in lines)
    overlay.vbo.write.assert_called_once()
    overlay.vao.render.assert_called_once()
