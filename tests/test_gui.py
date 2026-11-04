import math

from ximz.events import State
from ximz.gui import STYLES, wave_points


def test_every_state_has_a_style() -> None:
    assert set(STYLES) == set(State)
    assert all(0 < s.amplitude <= 1 and s.speed > 0 for s in STYLES.values())
    assert STYLES[State.SPEAKING].amplitude > STYLES[State.LISTENING].amplitude


def test_wave_line_tapers_and_stays_in_bounds() -> None:
    pts = wave_points(400, 60, cycles=2, amplitude=1.0, phase=0.7, samples=50)
    assert len(pts) == 50
    assert pts[0] == (0, 30) and math.isclose(pts[-1][1], 30, abs_tol=1e-9)
    assert all(0 <= y <= 60 for _, y in pts)
    assert max(abs(y - 30) for _, y in pts) > 15  # it actually waves


def test_wave_loop_is_seamless() -> None:
    a = wave_points(400, 60, 2, 1.0, 0.0, 30)
    b = wave_points(400, 60, 2, 1.0, 2 * math.pi, 30)
    assert all(math.isclose(ya, yb, abs_tol=1e-9) for (_, ya), (_, yb) in zip(a, b))
