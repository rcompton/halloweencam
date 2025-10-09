from __future__ import annotations
import os
import freetype
import moderngl
import numpy as np

from . import shaders as S
from .config import AppConfig
from .logging import get_logger


class DebugOverlay:
    def __init__(self, ctx: moderngl.Context, cfg: AppConfig):
        self.ctx = ctx
        self.cfg = cfg
        self.logger = get_logger(__name__)
        self.prog = self.ctx.program(vertex_shader=S.VS_DEBUGOVERLAY, fragment_shader=S.FS_DEBUGOVERLAY)

        self.sampler = self.ctx.sampler(
            filter=(moderngl.LINEAR, moderngl.LINEAR),
            repeat_x=False,
            repeat_y=False,
        )

        self.max_chars = 1024
        # 6 vertices per char, 4 floats per vertex (x, y, u, v)
        self.vertices = np.zeros((self.max_chars * 6, 4), dtype="f4")
        self.vbo = self.ctx.buffer(self.vertices.tobytes(), dynamic=True)
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "2f 2f", "in_vert", "in_uv")]
        )
        self.char_count = 0

        self.font_map = {}
        self.font_loaded = False
        font_path = self._find_font_path()

        if font_path:
            try:
                self._load_font(font_path)
                self.font_loaded = True
                self.logger.info(f"Loaded font: {font_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load font '{font_path}': {e}")
        else:
            self.logger.warning(
                "Could not find a suitable mono-spaced font for the debug overlay."
            )

    def _find_font_path(self) -> str | None:
        # Common paths for different OSes
        font_paths = [
            # Linux
            "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            # macOS
            "/System/Library/Fonts/Menlo.ttc",
            # Windows
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/Consolas.ttf",
        ]
        for path in font_paths:
            if os.path.exists(path):
                return path
        return None

    def _load_font(self, font_path: str, size: int = 16):
        face = freetype.Face(font_path)
        face.set_pixel_sizes(0, size)

        # Create a texture atlas for the font
        width = sum(face.load_char(c).glyph.advance.x >> 6 for c in range(128))
        height = max(face.load_char(c).glyph.bitmap.rows for c in range(128))

        texture_data = np.zeros((height, width), dtype="u1")
        x = 0
        for i in range(128):
            face.load_char(chr(i))
            bitmap = face.glyph.bitmap
            w, h = bitmap.width, bitmap.rows

            if w > 0 and h > 0:
                texture_data[0:h, x:x+w] = np.array(bitmap.buffer, dtype="u1").reshape((h, w))

            self.font_map[chr(i)] = {
                "size": (w, h),
                "bearing": (face.glyph.bitmap_left, face.glyph.bitmap_top),
                "advance": face.glyph.advance.x >> 6,
                "uv_offset": x / width
            }
            x += w

        self.font_texture = self.ctx.texture((width, height), 1, texture_data.tobytes(), dtype="f1")
        self.sampler.use(location=0)

    def render(self, lines: list[str], x: int, y: int, color=(1.0, 1.0, 1.0)):
        self.char_count = 0
        cursor_x, cursor_y = x, y

        for line in lines:
            for char in line:
                if char in self.font_map and self.char_count < self.max_chars:
                    self._add_char_quad(char, cursor_x, cursor_y)
                    char_info = self.font_map[char]
                    cursor_x += char_info["advance"]

            cursor_y -= 18  # Line height
            cursor_x = x

        if self.char_count > 0:
            self.vbo.write(self.vertices[:self.char_count * 6].tobytes())
            self.prog["textColor"].value = color
            self.font_texture.use(location=0)
            self.vao.render(moderngl.TRIANGLES, vertices=self.char_count * 6)

    def _add_char_quad(self, char, x, y):
        char_info = self.font_map[char]
        w, h = char_info["size"]

        xpos = x + char_info["bearing"][0]
        ypos = y - (h - char_info["bearing"][1])

        u = char_info["uv_offset"]
        u_w = w / self.font_texture.width

        # Convert to normalized device coordinates
        px = (xpos / self.cfg.width) * 2.0 - 1.0
        py = (ypos / self.cfg.height) * 2.0 - 1.0
        pw = (w / self.cfg.width) * 2.0
        ph = (h / self.cfg.height) * 2.0

        v_idx = self.char_count * 6
        self.vertices[v_idx]     = (px,      py + ph, u,       0.0)
        self.vertices[v_idx + 1] = (px,      py,      u,       h / self.font_texture.height)
        self.vertices[v_idx + 2] = (px + pw, py,      u + u_w, h / self.font_texture.height)
        self.vertices[v_idx + 3] = (px,      py + ph, u,       0.0)
        self.vertices[v_idx + 4] = (px + pw, py,      u + u_w, h / self.font_texture.height)
        self.vertices[v_idx + 5] = (px + pw, py + ph, u + u_w, 0.0)

        self.char_count += 1
