"""Landmark-anchored comic overlays and particle effects."""
from __future__ import annotations
from dataclasses import dataclass
import random
import time
import cv2
import numpy as np


@dataclass
class Particle:
    x: float; y: float; vx: float; vy: float; life: float; color: tuple[int, int, int]


class VisualEffects:
    FACE_EDGES = ((10, 152), (234, 454), (33, 133), (362, 263), (61, 291), (13, 14))
    HAND_EDGES = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                  (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
                  (15,16),(13,17),(17,18),(18,19),(19,20),(0,17))
    def __init__(self):
        self.intensity = 1.0
        self.particles: list[Particle] = []
        self.shockwaves: list[tuple[float, float, float, float]] = []
        self.messages: list[tuple[str, float, tuple[int,int,int]]] = []
        self.shake_until = 0.0

    def trigger(self, name, anchor):
        now = time.monotonic()
        text, color = {
            "left_wink": ("WAAAH!", (0,220,255)), "right_wink": ("BONK!", (255,80,255)),
            "cheek_puff": ("BOOM!", (0,80,255)), "both_closed": ("MUTED!", (180,180,180)),
            "hand_collect": ("GOT IT!", (50,255,100)), "hand_bomb": ("HAND BOOM!", (0,40,255)),
            "hand_shield": ("BLOCKED!", (255,220,30)), "hand_swipe": ("WHOOSH!", (255,80,220)),
            "hand_miss": ("OOPS!", (0,150,255)), "hand_game_over": ("TIME!", (255,255,255)),
        }.get(name, ("CHAOS!", (0,255,150)))
        self.messages.append((text, now+1.0, color))
        if name in ("cheek_puff", "hand_bomb"):
            self.shockwaves.append((anchor[0], anchor[1], 10, now+1.0))
            self.shake_until = now + .5

    def _pt(self, lm, i, w, h): return int(lm[i][0]*w), int(lm[i][1]*h)

    def render(self, frame, lm, state, mesh=False):
        h, w = frame.shape[:2]
        now = time.monotonic()
        if lm:
            mouth = self._pt(lm, 13, w, h)
            face_px = max(40, int(state.face_scale*w))
            if mesh:
                for a,b in self.FACE_EDGES:
                    cv2.line(frame, self._pt(lm,a,w,h), self._pt(lm,b,w,h), (100,255,210), 1, cv2.LINE_AA)
                for i in range(0, 468, 5):
                    cv2.circle(frame, self._pt(lm,i,w,h), 1, (255,120,240), -1)
            if state.active.get("turbo"):
                for eye in (468, 473):
                    if eye < len(lm):
                        p = self._pt(lm, eye, w, h)
                        cv2.circle(frame, p, max(5, face_px//30), (20,20,255), -1)
                        cv2.circle(frame, p, max(9, face_px//18), (20,20,255), 2)
            if state.active.get("mouth"):
                end = (w, int(mouth[1] + random.randint(-3,3)*self.intensity))
                cv2.line(frame, mouth, end, (30,30,255), max(2,int(5*self.intensity)))
                cv2.line(frame, mouth, end, (230,230,255), 1)
            if state.active.get("smile") and random.random() < .65:
                for _ in range(max(1, int(3*self.intensity))):
                    self.particles.append(Particle(mouth[0], mouth[1], random.uniform(-3,3),
                        random.uniform(-5,-1), 1.0, random.choice([(0,0,255),(0,255,255),(0,255,0),(255,0,255)])))
        live = []
        for p in self.particles:
            p.x += p.vx; p.y += p.vy; p.vy += .08; p.life -= .035
            if p.life > 0:
                cv2.circle(frame, (int(p.x),int(p.y)), max(1,int(5*p.life)), p.color, -1)
                live.append(p)
        self.particles = live[-400:]
        waves = []
        for x,y,r,until in self.shockwaves:
            if now < until:
                r += 18*self.intensity
                cv2.circle(frame, (int(x),int(y)), int(r), (0,180,255), max(1,int(8*(until-now))))
                waves.append((x,y,r,until))
        self.shockwaves = waves
        self.messages = [m for m in self.messages if now < m[1]]
        for i,(text,until,color) in enumerate(self.messages[-3:]):
            scale = 1.4 + .35*np.sin(now*18)
            cv2.putText(frame, text, (max(20,w//2-130), 100+i*55), cv2.FONT_HERSHEY_DUPLEX,
                        scale, (0,0,0), 7, cv2.LINE_AA)
            cv2.putText(frame, text, (max(20,w//2-130), 100+i*55), cv2.FONT_HERSHEY_DUPLEX,
                        scale, color, 3, cv2.LINE_AA)
        if now < self.shake_until:
            dx,dy = random.randint(-8,8),random.randint(-8,8)
            frame[:] = np.roll(np.roll(frame, dx, axis=1), dy, axis=0)
        return frame

    def render_hands(self, frame, hands):
        """Draw a lightweight neon skeleton and gesture label."""
        h, w = frame.shape[:2]
        for hand in hands:
            pts = [(int(x*w), int(y*h)) for x,y in hand.landmarks]
            color = (255, 190, 40) if hand.handedness == "Left" else (255, 70, 220)
            for a,b in self.HAND_EDGES:
                cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(frame, p, 3, (240,240,255), -1)
            cursor = pts[8]
            cv2.circle(frame, cursor, 13, (20,255,255), 2)
            gesture = ("PINCH" if hand.pinch < .34 else "FIST" if hand.fist else
                       "OPEN PALM" if hand.palm_open else "PEACE" if hand.peace else "POINT")
            cv2.putText(frame, f"{hand.handedness}: {gesture}", (max(5,cursor[0]-65),max(20,cursor[1]-20)),
                        cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
        return frame
