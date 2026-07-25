"""Free play, gesture training, and the 60-second Stay Calm challenge."""
from __future__ import annotations
import random
import time
from dataclasses import dataclass


class StayCalmGame:
    def __init__(self, duration=60):
        self.duration = duration
        self.running = False
        self.finished = False

    def start(self):
        self.running, self.finished = True, False
        self.started = time.monotonic()
        self.calmness, self.reactions, self.best_streak = 100.0, 0, 0.0
        self.streak_started = self.started
        self.last_reaction = self.started - 3
        self.next_event = self.started + random.uniform(2, 4)
        self.event = ""
        self.event_until = 0.0
        self.previous_energy = 0.0

    def update(self, state):
        if not self.running:
            return
        now = time.monotonic()
        if now - self.started >= self.duration:
            self.running, self.finished = False, True
            self.best_streak = max(self.best_streak, now - self.streak_started)
            return
        energy = state.mouth*7 + max(0, state.smile-.36)*4 + state.left_brow + state.right_brow
        reacting = (state.active.get("smile") or state.active.get("mouth") or
                    abs(energy-self.previous_energy) > .22)
        if reacting and now - self.last_reaction > .65:
            self.calmness = max(0, self.calmness - 4.5)
            self.reactions += 1
            self.best_streak = max(self.best_streak, now - self.streak_started)
            self.streak_started, self.last_reaction = now, now
        else:
            self.calmness = min(100, self.calmness + .012)
        self.previous_energy = energy
        if now >= self.next_event:
            self.event = random.choice(["DO NOT SMILE!", "SERIOUS POTATO", "YOUR NOSE KNOWS", "CALM.exe"])
            self.event_until = now + 1.8
            self.next_event = now + random.uniform(2.5, 5.5)

    @property
    def remaining(self):
        return max(0, self.duration - (time.monotonic() - self.started)) if self.running else 0

    @property
    def score(self):
        return int(self.calmness * 10 + self.best_streak * 5)


@dataclass
class HandTarget:
    x: float
    y: float
    radius: float
    kind: str
    born: float
    lifetime: float


class HandMayhemGame:
    """Arcade game: point with an index finger and pinch targets to collect."""

    def __init__(self, duration=45):
        self.duration = duration
        self.running = self.finished = False
        self.target: HandTarget | None = None
        self.pinched_last_frame = False

    def start(self):
        now = time.monotonic()
        self.running, self.finished = True, False
        self.started = now
        self.score, self.lives, self.combo = 0, 3, 0
        self.best_combo, self.collected = 0, 0
        self.message, self.message_until = "PINCH THE ORBS!", now + 1.5
        self.pinched_last_frame = False
        self._spawn(now)

    def _spawn(self, now):
        roll = random.random()
        kind = "bomb" if roll < .20 else ("bonus" if roll < .38 else "orb")
        self.target = HandTarget(random.uniform(.36,.90), random.uniform(.16,.82),
                                 .045 if kind != "bonus" else .055, kind, now,
                                 2.4 if kind == "bomb" else 3.1)

    def update(self, hands):
        if not self.running:
            return []
        now = time.monotonic()
        events = []
        if now - self.started >= self.duration or self.lives <= 0:
            self.running, self.finished = False, True
            self.target = None
            return ["hand_game_over"]
        if not self.target:
            self._spawn(now)
        target = self.target
        assert target is not None
        if now - target.born > target.lifetime:
            if target.kind != "bomb":
                self.lives -= 1
                self.combo = 0
                self.message, self.message_until = "TOO SLOW!", now + .8
                events.append("hand_miss")
            self._spawn(now)
            return events
        pinching = any(h.pinch < .34 for h in hands)
        shield = any(h.fist for h in hands)
        for hand in hands:
            dx, dy = hand.cursor[0] - target.x, hand.cursor[1] - target.y
            touching = dx*dx + dy*dy <= target.radius*target.radius
            collect = touching and pinching and not self.pinched_last_frame
            bonus_collect = touching and target.kind == "bonus" and hand.palm_open
            if collect or bonus_collect:
                if target.kind == "bomb":
                    if shield:
                        self.score += 50
                        self.message = "FIST SHIELD!"
                        events.append("hand_shield")
                    else:
                        self.lives -= 1
                        self.combo = 0
                        self.message = "KABOOM!"
                        events.append("hand_bomb")
                else:
                    points = 25 if target.kind == "bonus" else 10
                    self.combo += 1
                    self.best_combo = max(self.best_combo, self.combo)
                    self.score += points * min(self.combo, 5)
                    self.collected += 1
                    self.message = f"+{points * min(self.combo, 5)}  COMBO x{self.combo}"
                    events.append("hand_collect")
                self.message_until = now + .8
                self._spawn(now)
                break
        for hand in hands:
            if hand.swipe:
                self.score += 5
                self.message, self.message_until = f"SWIPE {hand.swipe.upper()} +5", now + .6
                events.append("hand_swipe")
        self.pinched_last_frame = pinching
        return events

    @property
    def remaining(self):
        return max(0, self.duration - (time.monotonic() - self.started)) if self.running else 0


class ModeController:
    MODES = ("FREE PLAY", "STAY CALM", "GESTURE TRAINING", "HAND MAYHEM")
    TRAIN = ("mouth", "smile", "left_brow", "right_brow", "left_wink", "cheek_puff")
    INSTRUCTIONS = {
        "mouth": "Open and close your jaw", "smile": "Show us that suspicious grin",
        "left_brow": "Raise your left eyebrow", "right_brow": "Raise your right eyebrow",
        "left_wink": "Close one eye while keeping the other open",
        "cheek_puff": "Puff cheeks (fallback: closed lips + wide cheeks)",
    }
    def __init__(self):
        self.index, self.training_index = 0, 0
        self.game = StayCalmGame()
        self.hand_game = HandMayhemGame()

    @property
    def name(self): return self.MODES[self.index]
    def cycle(self):
        self.index = (self.index + 1) % len(self.MODES)
        if self.index == 1: self.game.start()
        if self.index == 3: self.hand_game.start()
    def update(self, state, hands=()):
        if self.index == 1: self.game.update(state)
        if self.index == 3: return self.hand_game.update(hands)
        return []
