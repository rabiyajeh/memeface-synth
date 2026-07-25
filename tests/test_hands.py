import time

from src.game_modes import HandMayhemGame, HandTarget
from src.hand_tracker import HandState, HandTracker


def make_hand(scale=1.0):
    points = [(0.5, 0.8)] * 21
    points[0] = (0.5, 0.8)
    points[9] = (0.5, 0.6)
    points[4] = (0.45, 0.40)
    points[8] = (0.46, 0.40)
    for tip, pip in ((8,6),(12,10),(16,14),(20,18)):
        points[pip] = (0.5, 0.55)
        points[tip] = (0.5, 0.35)
    points[4] = (0.49, 0.35)
    if scale != 1:
        origin = points[0]
        points = [(origin[0] + (x-origin[0])*scale,
                   origin[1] + (y-origin[1])*scale) for x,y in points]
    return points


def test_pinch_measurement_is_scale_independent():
    a = HandTracker.recognize(make_hand(1.0))
    b = HandTracker.recognize(make_hand(.5))
    assert a.pinch == pytest.approx(b.pinch)


def test_short_landmark_list_is_rejected():
    assert HandTracker.recognize([(0, 0)]).confidence == 0


def test_pinching_orb_increases_score():
    game = HandMayhemGame()
    game.start()
    game.target = HandTarget(.5, .5, .1, "orb", time.monotonic(), 10)
    hand = HandState(cursor=(.5,.5), pinch=.1)
    events = game.update([hand])
    assert "hand_collect" in events
    assert game.score == 10
    assert game.collected == 1


import pytest
