"""Floating sound-wave overlay + menu-bar icon that show what Ximz is doing.

No text: just animated wave lines whose colour, height and speed change with
the state (listening / thinking / speaking). AppKit must own the main thread, so
when the GUI is enabled the voice loop runs on a background thread and state
events are marshalled onto the main thread with `AppHelper.callAfter`. The
overlay only listens to the EventBus; a GUI failure can't crash the assistant.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass

from AppKit import (
    NSAnimationContext,
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSColor,
    NSFont,
    NSFontWeightHeavy,
    NSImage,
    NSImageScaleProportionallyUpOrDown,
    NSImageView,
    NSMakeRect,
    NSMenu,
    NSMenuItem,
    NSPanel,
    NSScreen,
    NSStatusBar,
    NSStatusWindowLevel,
    NSTextAlignmentCenter,
    NSTextField,
    NSVariableStatusItemLength,
    NSView,
    NSVisualEffectBlendingModeBehindWindow,
    NSVisualEffectMaterialHUDWindow,
    NSVisualEffectStateActive,
    NSVisualEffectView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from PyObjCTools import AppHelper
from Quartz import (
    CABasicAnimation,
    CAKeyframeAnimation,
    CAMediaTimingFunction,
    CAShapeLayer,
    CATransform3DMakeScale,
    CGPathAddLineToPoint,
    CGPathCreateMutable,
    CGPathMoveToPoint,
    kCAMediaTimingFunctionEaseInEaseOut,
)

from ximz.config import GuiConfig
from ximz.events import EventBus, State, StateEvent
from ximz.logo import logo_image

log = logging.getLogger(__name__)

# Layout / animation constants.
PADDING = 22
CORNER_RADIUS = 28
WAVE_SAMPLES = 90          # points per line
WAVE_KEYFRAMES = 32        # phase steps per animation cycle
WAVE_BREATHE_MIN = 0.55    # amplitude "breathing" lower bound (scale.y)
AMPLITUDE_ANIM_S = 0.35    # how fast the wave grows/shrinks between states
NAME_TEXT = "ximz"
NAME_FONT_SIZE = 40
NAME_GAP = 6               # space between the name and the waves
LOGO_SCALE = 0.9           # background logo size, fraction of panel height
LOGO_ALPHA = 0.10          # how faint the background logo is
MENU_ICON_PT = 18


@dataclass(frozen=True)
class WaveLine:
    cycles: float      # sine periods across the width
    amplitude: float   # fraction of half the wave height
    opacity: float
    period_s: float    # time for one full phase cycle (negative = drift left)
    line_width: float
    breathe_s: float   # amplitude breathing period


WAVE_LINES = (
    WaveLine(cycles=1.6, amplitude=1.0, opacity=1.0, period_s=1.3, line_width=3.0, breathe_s=0.9),
    WaveLine(cycles=2.3, amplitude=0.75, opacity=0.6, period_s=-1.0, line_width=2.2, breathe_s=0.7),
    WaveLine(cycles=3.1, amplitude=0.5, opacity=0.35, period_s=0.8, line_width=1.6, breathe_s=0.55),
)


@dataclass(frozen=True)
class StateStyle:
    label: str                      # menu-bar menu only; never drawn in the overlay
    color: Callable[[], NSColor]
    symbol: str                     # SF Symbol for the menu-bar icon
    amplitude: float                # wave height, 0..1
    speed: float                    # wave motion speed multiplier


STYLES: dict[State, StateStyle] = {
    State.IDLE: StateStyle("Idle", NSColor.systemGrayColor, "waveform", 0.2, 0.5),
    State.LISTENING: StateStyle("Listening", NSColor.systemBlueColor, "ear", 0.35, 0.6),
    State.TRANSCRIBING: StateStyle("Transcribing", NSColor.systemIndigoColor, "text.bubble", 0.5, 0.9),
    State.THINKING: StateStyle("Thinking", NSColor.systemPurpleColor, "sparkles", 0.55, 1.5),
    State.SPEAKING: StateStyle("Speaking", NSColor.systemGreenColor, "speaker.wave.2.fill", 1.0, 1.0),
    State.ERROR: StateStyle("Error", NSColor.systemRedColor, "exclamationmark.triangle", 0.25, 0.4),
}


def wave_points(
    width: float, height: float, cycles: float, amplitude: float, phase: float, samples: int
) -> list[tuple[float, float]]:
    """One sine line, tapered to flat at both ends by a sin² envelope."""
    mid = height / 2
    half_room = mid * amplitude * 0.92  # leave room for the stroke width
    points = []
    for i in range(samples):
        t = i / (samples - 1)
        envelope = math.sin(math.pi * t) ** 2
        y = mid + half_room * envelope * math.sin(2 * math.pi * cycles * t + phase)
        points.append((t * width, y))
    return points


def _path(points: list[tuple[float, float]]):  # type: ignore[no-untyped-def]
    path = CGPathCreateMutable()
    CGPathMoveToPoint(path, None, *points[0])
    for x, y in points[1:]:
        CGPathAddLineToPoint(path, None, x, y)
    return path


class Overlay:
    """The floating wave panel and menu-bar item. All methods run on the main
    thread except `post`, which is the thread-safe EventBus listener."""

    def __init__(self, cfg: GuiConfig) -> None:
        self._cfg = cfg
        self._hide_generation = 0
        self._visible = False
        self._state: State | None = None
        self._build_panel()
        self._status_item = self._build_status_item() if cfg.menu_bar_icon else None

    # -- construction -----------------------------------------------------

    def _build_panel(self) -> None:
        w, h = self._cfg.width, self._cfg.height
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, w, h),
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered,
            False,
        )
        panel.setLevel_(NSStatusWindowLevel)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setIgnoresMouseEvents_(True)  # never steals clicks or focus
        panel.setHidesOnDeactivate_(False)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        panel.setAlphaValue_(0.0)

        blur = NSVisualEffectView.alloc().initWithFrame_(NSMakeRect(0, 0, w, h))
        blur.setMaterial_(NSVisualEffectMaterialHUDWindow)
        blur.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        blur.setState_(NSVisualEffectStateActive)
        blur.setWantsLayer_(True)
        blur.layer().setCornerRadius_(CORNER_RADIUS)
        blur.layer().setMasksToBounds_(True)
        panel.setContentView_(blur)

        # Faded logo in the background, centred.
        logo_size = h * LOGO_SCALE
        logo = NSImageView.alloc().initWithFrame_(
            NSMakeRect((w - logo_size) / 2, (h - logo_size) / 2, logo_size, logo_size)
        )
        logo.setImage_(logo_image(logo_size, template=False))
        logo.setImageScaling_(NSImageScaleProportionallyUpOrDown)
        logo.setAlphaValue_(LOGO_ALPHA)
        blur.addSubview_(logo)

        # Bold name at the top (AppKit's y axis points up).
        name = NSTextField.labelWithString_(NAME_TEXT)
        name.setFont_(NSFont.systemFontOfSize_weight_(NAME_FONT_SIZE, NSFontWeightHeavy))
        name.setTextColor_(NSColor.whiteColor())
        name.setAlignment_(NSTextAlignmentCenter)
        name_h = name.fittingSize().height
        name.setFrame_(NSMakeRect(0, h - PADDING - name_h, w, name_h))
        blur.addSubview_(name)

        # Waves fill the space under the name.
        self._wave_w = w - 2 * PADDING
        self._wave_h = h - 2 * PADDING - name_h - NAME_GAP
        wave = NSView.alloc().initWithFrame_(NSMakeRect(PADDING, PADDING, self._wave_w, self._wave_h))
        wave.setWantsLayer_(True)
        blur.addSubview_(wave)
        self._panel, self._wave = panel, wave
        self._build_lines()
        self._position()

    def _build_lines(self) -> None:
        self._wave_lines: list[tuple[CAShapeLayer, WaveLine, list]] = []
        for spec in WAVE_LINES:
            # Precompute one cycle of phase-shifted paths; the last equals the
            # first, so the keyframe loop is seamless.
            frames = [
                _path(wave_points(self._wave_w, self._wave_h, spec.cycles, spec.amplitude,
                                  2 * math.pi * k / WAVE_KEYFRAMES, WAVE_SAMPLES))
                for k in range(WAVE_KEYFRAMES + 1)
            ]
            if spec.period_s < 0:
                frames.reverse()
            line = CAShapeLayer.layer()
            line.setFrame_(NSMakeRect(0, 0, self._wave_w, self._wave_h))
            line.setPath_(frames[0])
            line.setFillColor_(None)
            line.setLineWidth_(spec.line_width)
            line.setLineCap_("round")
            line.setLineJoin_("round")
            line.setOpacity_(spec.opacity)
            self._wave.layer().addSublayer_(line)
            self._wave_lines.append((line, spec, frames))

    def _build_status_item(self):  # type: ignore[no-untyped-def]
        item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        self._menu_logo = logo_image(MENU_ICON_PT, template=True)
        item.button().setImage_(self._menu_logo)
        menu = NSMenu.alloc().init()
        self._menu_state = menu.addItemWithTitle_action_keyEquivalent_("Idle", None, "")
        self._menu_state.setEnabled_(False)
        menu.addItem_(NSMenuItem.separatorItem())
        menu.addItemWithTitle_action_keyEquivalent_("Quit Ximz", "terminate:", "q")
        item.setMenu_(menu)
        return item

    def _position(self) -> None:
        w, h = self._cfg.width, self._cfg.height
        screen = (NSScreen.mainScreen() or NSScreen.screens()[0]).visibleFrame()
        x = screen.origin.x + (screen.size.width - w) / 2
        if self._cfg.position == "top":
            y = screen.origin.y + screen.size.height - h - self._cfg.margin
        elif self._cfg.position == "bottom":
            y = screen.origin.y + self._cfg.margin
        else:  # center
            y = screen.origin.y + (screen.size.height - h) / 2
        self._panel.setFrame_display_(NSMakeRect(x, y, w, h), True)

    # -- event handling ---------------------------------------------------

    def post(self, event: StateEvent) -> None:
        """EventBus listener; safe to call from any thread."""
        AppHelper.callAfter(self._apply, event)

    def _apply(self, event: StateEvent) -> None:
        try:
            self._render(event.state)
        except Exception:
            log.exception("overlay render failed")

    def _render(self, state: State) -> None:
        style = STYLES[state]
        self._update_status_item(state, style)
        if state == State.IDLE:
            self._schedule_hide()
            return
        if state != self._state:
            self._animate_wave(style)
        self._state = state
        self._position()  # follow the active screen
        self._show()
        if state == State.ERROR:
            self._schedule_hide()

    def _update_status_item(self, state: State, style: StateStyle) -> None:
        if self._status_item is None:
            return
        # The Ximz logo when idle; a state symbol while active.
        if state == State.IDLE:
            image = self._menu_logo
        else:
            image = NSImage.imageWithSystemSymbolName_accessibilityDescription_(style.symbol, style.label)
            if image is not None:
                image.setTemplate_(True)
        self._status_item.button().setImage_(image or self._menu_logo)
        self._menu_state.setTitle_(style.label)

    # -- animation ----------------------------------------------------------

    def _animate_wave(self, style: StateStyle) -> None:
        # Grow/shrink smoothly to the state's amplitude. Animate the scalar key
        # path with plain floats: Core Animation needs boxed values, and a raw
        # CATransform3D struct as from/toValue crashes the app at commit time.
        layer = self._wave.layer()
        current = (layer.presentationLayer() or layer).valueForKeyPath_("sublayerTransform.scale.y")
        grow = CABasicAnimation.animationWithKeyPath_("sublayerTransform.scale.y")
        grow.setFromValue_(float(current) if current is not None else style.amplitude)
        grow.setToValue_(style.amplitude)
        grow.setDuration_(AMPLITUDE_ANIM_S)
        layer.setSublayerTransform_(CATransform3DMakeScale(1.0, style.amplitude, 1.0))
        layer.addAnimation_forKey_(grow, "amplitude")

        color = style.color().CGColor()
        ease = CAMediaTimingFunction.functionWithName_(kCAMediaTimingFunctionEaseInEaseOut)
        for line, spec, frames in self._wave_lines:
            line.removeAllAnimations()
            line.setStrokeColor_(color)

            drift = CAKeyframeAnimation.animationWithKeyPath_("path")
            drift.setValues_(frames)
            drift.setDuration_(abs(spec.period_s) / style.speed)
            drift.setRepeatCount_(float("inf"))
            line.addAnimation_forKey_(drift, "drift")

            breathe = CABasicAnimation.animationWithKeyPath_("transform.scale.y")
            breathe.setFromValue_(WAVE_BREATHE_MIN)
            breathe.setToValue_(1.0)
            breathe.setDuration_(spec.breathe_s / style.speed)
            breathe.setAutoreverses_(True)
            breathe.setRepeatCount_(float("inf"))
            breathe.setTimingFunction_(ease)
            line.addAnimation_forKey_(breathe, "breathe")

    def _stop_wave(self) -> None:
        for line, _, _ in self._wave_lines:
            line.removeAllAnimations()
        self._state = None

    def _show(self) -> None:
        self._hide_generation += 1  # cancel any pending hide
        if not self._visible:
            self._visible = True
            self._panel.orderFrontRegardless()
        self._fade_to(1.0)

    def _schedule_hide(self) -> None:
        self._hide_generation += 1
        generation = self._hide_generation
        AppHelper.callLater(self._cfg.hide_delay_s, self._hide, generation)

    def _hide(self, generation: int) -> None:
        if generation != self._hide_generation or not self._visible:
            return

        def done() -> None:
            if generation == self._hide_generation:
                self._visible = False
                self._panel.orderOut_(None)
                self._stop_wave()  # no animation work while hidden

        self._fade_to(0.0, done)

    def _fade_to(self, alpha: float, done: Callable[[], None] | None = None) -> None:
        def animate(ctx) -> None:  # type: ignore[no-untyped-def]
            ctx.setDuration_(self._cfg.fade_s)
            self._panel.animator().setAlphaValue_(alpha)

        NSAnimationContext.runAnimationGroup_completionHandler_(animate, done or (lambda: None))


def run_with_gui(cfg: GuiConfig, bus: EventBus, worker: Callable[[], None]) -> None:
    """Run `worker` (the voice loop) on a background thread and AppKit on this one.
    Returns when the worker finishes, the user quits from the menu, or Ctrl+C."""
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)  # no Dock icon

    overlay = Overlay(cfg)
    bus.subscribe(overlay.post)

    def target() -> None:
        try:
            worker()
        except Exception:
            log.exception("voice loop crashed")
        finally:
            AppHelper.callAfter(AppHelper.stopEventLoop)

    threading.Thread(target=target, name="ximz-voice", daemon=True).start()
    AppHelper.runEventLoop(installInterrupt=True)
