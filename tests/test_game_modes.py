from src.game_modes import StayCalmGame
from src.gesture_detector import GestureState


def test_game_score_is_nonnegative():
    game = StayCalmGame()
    game.start()
    game.calmness = 72
    game.best_streak = 8
    assert game.score == 760


def test_remaining_never_negative():
    game = StayCalmGame(duration=0)
    game.start()
    assert game.remaining == 0


def test_strong_reaction_reduces_calmness():
    game = StayCalmGame()
    game.start()
    game.last_reaction -= 10
    state = GestureState(mouth=.1, smile=.6, active={"mouth": True, "smile": True})
    before = game.calmness
    game.update(state)
    assert game.calmness < before
    assert game.reactions == 1
