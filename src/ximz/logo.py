"""The Ximz logo, drawn as vectors so it is crisp at any size.

Shape (normalised 0..1, y down): a thick "U" arc on top, a rounded horizontal
bar through the middle, and the mirrored arc underneath. Used as a template
image for the menu bar (macOS tints it for light/dark mode).
"""

from __future__ import annotations

from AppKit import NSBezierPath, NSColor, NSImage, NSMakeRect, NSMakeSize

BRAND_PURPLE = (0.29, 0.0, 0.68)   # #4A00AD, logo background

# Geometry, as fractions of the icon size.
_ARM_END_X = 0.265     # where the arms end, left (mirrored on the right)
_ARM_END_Y = 0.255     # top arc arm end height (mirrored for the bottom arc)
_ARC_LOW_Y = 0.395     # lowest point of the top arc's centreline (meets the bar)
_ARC_STROKE = 0.12     # thickness of the arcs
_BAR = (0.19, 0.44, 0.62, 0.11)  # x, y, w, h of the middle bar
_BAR_RADIUS = 0.015


def _arc(size: float, flip: bool) -> NSBezierPath:
    def pt(x: float, y: float):  # type: ignore[no-untyped-def]
        return (x * size, (1 - y if flip else y) * size)

    path = NSBezierPath.bezierPath()
    path.moveToPoint_(pt(_ARM_END_X, _ARM_END_Y))
    path.curveToPoint_controlPoint1_controlPoint2_(pt(0.5, _ARC_LOW_Y), pt(0.40, 0.385), pt(0.44, _ARC_LOW_Y))
    path.curveToPoint_controlPoint1_controlPoint2_(
        pt(1 - _ARM_END_X, _ARM_END_Y), pt(0.56, _ARC_LOW_Y), pt(0.60, 0.385)
    )
    path.setLineWidth_(_ARC_STROKE * size)
    return path


def draw_logo(size: float, color: NSColor) -> None:
    """Draw the glyph into the current graphics context (flipped, top-left origin)."""
    color.set()
    for flip in (False, True):
        _arc(size, flip).stroke()
    x, y, w, h = (v * size for v in _BAR)
    NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
        NSMakeRect(x, y, w, h), _BAR_RADIUS * size, _BAR_RADIUS * size
    ).fill()


def logo_image(points: float, template: bool = True, glyph: NSColor | None = None,
               background: bool = False) -> NSImage:
    """The glyph as an NSImage.

    - template=True: black-on-clear, tinted by macOS (menu bar).
    - otherwise: `glyph` colour (default white), on brand purple if `background`.
    """

    def handler(rect) -> bool:  # type: ignore[no-untyped-def]
        size = rect.size.width
        if template:
            draw_logo(size, NSColor.blackColor())
            return True
        if background:
            NSColor.colorWithSRGBRed_green_blue_alpha_(*BRAND_PURPLE, 1.0).set()
            NSBezierPath.fillRect_(rect)
        draw_logo(size, glyph or NSColor.whiteColor())
        return True

    image = NSImage.imageWithSize_flipped_drawingHandler_(NSMakeSize(points, points), True, handler)
    image.setTemplate_(template)
    return image
