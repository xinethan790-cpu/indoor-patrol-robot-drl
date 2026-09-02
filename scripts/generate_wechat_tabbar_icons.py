#!/usr/bin/env python3
"""Generate the deterministic selected/unselected WeChat tab-bar icon set."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 81
SCALE = 4
OUTPUT_DIRECTORY = Path(__file__).absolute().parents[1] / "wechat-miniprogram/assets/tabbar"

INK = "#748497"
ACTIVE = "#0878F9"
ACTIVE_ACCENT = "#20C8E8"
ACTIVE_SURFACE = "#EAF4FF"
WHITE = "#FFFFFF"


def _scaled(values: tuple[float, ...]) -> tuple[int, ...]:
    return tuple(round(value * SCALE) for value in values)


def _canvas(selected: bool) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", (SIZE * SCALE, SIZE * SCALE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    if selected:
        draw.ellipse(_scaled((7, 7, 74, 74)), fill=ACTIVE_SURFACE)
    return image, draw


def _line(
    draw: ImageDraw.ImageDraw,
    points: tuple[tuple[float, float], ...],
    *,
    fill: str,
    width: float = 4,
) -> None:
    draw.line(
        [(_scaled((x,))[0], _scaled((y,))[0]) for x, y in points],
        fill=fill,
        width=round(width * SCALE),
        joint="curve",
    )


def _robot(selected: bool) -> Image.Image:
    image, draw = _canvas(selected)
    glyph = ACTIVE if selected else INK
    _line(draw, ((40.5, 27), (40.5, 20)), fill=glyph, width=3.5)
    draw.ellipse(_scaled((37.5, 16.5, 43.5, 22.5)), fill=ACTIVE_ACCENT if selected else glyph)
    head = _scaled((20, 27, 61, 57))
    if selected:
        draw.rounded_rectangle(head, radius=round(9 * SCALE), fill=ACTIVE)
        eye = WHITE
        mouth = WHITE
    else:
        draw.rounded_rectangle(
            head,
            radius=round(9 * SCALE),
            outline=INK,
            width=round(4 * SCALE),
        )
        eye = INK
        mouth = INK
    draw.ellipse(_scaled((28, 37, 34, 43)), fill=eye)
    draw.ellipse(_scaled((47, 37, 53, 43)), fill=eye)
    _line(draw, ((32, 49), (49, 49)), fill=mouth, width=3)
    draw.rounded_rectangle(_scaled((24, 59, 34, 64)), radius=round(2.5 * SCALE), fill=glyph)
    draw.rounded_rectangle(_scaled((47, 59, 57, 64)), radius=round(2.5 * SCALE), fill=glyph)
    return image


def _alarm(selected: bool) -> Image.Image:
    image, draw = _canvas(selected)
    glyph = ACTIVE if selected else INK
    body = (
        (22, 53),
        (27, 47),
        (27, 37),
        (29, 31),
        (34, 26),
        (40.5, 24),
        (47, 26),
        (52, 31),
        (54, 37),
        (54, 47),
        (59, 53),
    )
    if selected:
        draw.polygon([_scaled(point) for point in body], fill=ACTIVE)
        draw.rounded_rectangle(_scaled((21, 50, 60, 56)), radius=round(3 * SCALE), fill=ACTIVE)
        _line(draw, ((34, 32), (37, 29), (41, 28)), fill=WHITE, width=2.5)
    else:
        _line(draw, body, fill=INK, width=4)
        _line(draw, ((21, 53), (60, 53)), fill=INK, width=4)
    draw.ellipse(_scaled((36.5, 58, 44.5, 66)), fill=glyph)
    return image


def _history(selected: bool) -> Image.Image:
    image, draw = _canvas(selected)
    clock = _scaled((19, 19, 62, 62))
    if selected:
        draw.ellipse(clock, fill=ACTIVE)
        hand = WHITE
    else:
        draw.ellipse(clock, outline=INK, width=round(4 * SCALE))
        hand = INK
    _line(draw, ((40.5, 28), (40.5, 41), (50, 47)), fill=hand, width=4)
    draw.ellipse(_scaled((38, 38.5, 43, 43.5)), fill=hand)
    return image


def _project(selected: bool) -> Image.Image:
    image, draw = _canvas(selected)
    boxes = (
        (20, 20, 37, 37),
        (44, 20, 61, 37),
        (20, 44, 37, 61),
        (44, 44, 61, 61),
    )
    for index, box in enumerate(boxes):
        if selected:
            fill = ACTIVE_ACCENT if index == 1 else ACTIVE
            draw.rounded_rectangle(_scaled(box), radius=round(5 * SCALE), fill=fill)
        else:
            draw.rounded_rectangle(
                _scaled(box),
                radius=round(5 * SCALE),
                outline=INK,
                width=round(4 * SCALE),
            )
    return image


def _save(name: str, renderer: Callable[[bool], Image.Image]) -> None:
    for selected in (False, True):
        suffix = "-selected" if selected else ""
        image = renderer(selected).resize((SIZE, SIZE), Image.Resampling.LANCZOS)
        image.save(
            OUTPUT_DIRECTORY / f"{name}{suffix}.png",
            format="PNG",
            optimize=True,
            compress_level=9,
        )


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name, renderer in (
        ("patrol", _robot),
        ("alarms", _alarm),
        ("history", _history),
        ("project", _project),
    ):
        _save(name, renderer)
    print(f"Generated 8 tab-bar icons in {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
