"""Tray icon artwork, drawn at runtime so the app ships no image assets."""

from __future__ import annotations

from PIL import Image, ImageDraw

from .pipeline import State

SIZE = 64

# Colour per state, so the tray tells you what the app is doing at a glance.
_COLORS: dict[State, tuple[int, int, int]] = {
    State.STOPPED: (110, 116, 128),    # grey
    State.LOADING: (222, 158, 54),     # amber
    State.IDLE: (86, 148, 220),        # blue
    State.LISTENING: (72, 186, 108),   # green
    State.THINKING: (222, 158, 54),    # amber
    State.REVIEWING: (150, 110, 200),  # purple: waiting on you, not on the machine
    State.SPEAKING: (206, 88, 96),     # red
}


def make_icon(state: State = State.STOPPED) -> Image.Image:
    color = _COLORS.get(state, _COLORS[State.STOPPED])
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    d.ellipse((2, 2, SIZE - 2, SIZE - 2), fill=color)

    white = (255, 255, 255, 235)
    # Microphone capsule.
    d.rounded_rectangle((26, 14, 38, 36), radius=6, fill=white)
    # Cradle arc plus stem and base.
    d.arc((20, 22, 44, 44), start=0, end=180, fill=white, width=4)
    d.rectangle((30, 44, 34, 50), fill=white)
    d.rectangle((24, 50, 40, 54), fill=white)

    if state is State.STOPPED:
        d.line((14, 50, 50, 14), fill=(240, 240, 240, 255), width=6)
    return img
