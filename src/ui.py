"""OpenCV game HUD and clickable controls."""
from __future__ import annotations
import cv2
import numpy as np


class GameUI:
    def __init__(self):
        self.help = False
        self.pending = None
        self.buttons = {}

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            for action, (x1,y1,x2,y2) in self.buttons.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.pending = action

    def pop_action(self):
        value, self.pending = self.pending, None
        return value

    def _bar(self, frame, label, value, y, color):
        cv2.putText(frame, label, (18,y), cv2.FONT_HERSHEY_SIMPLEX, .48, (230,230,240), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (120,y-12), (265,y), (50,50,60), -1)
        cv2.rectangle(frame, (120,y-12), (120+int(145*np.clip(value,0,1)),y), color, -1)

    def draw(self, frame, state, preset, mode, fps, recording, calibration_prompt="",
             audio=True, thresholds=None):
        h,w = frame.shape[:2]
        self.buttons = {}
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (285,h), (18,13,30), -1)
        cv2.addWeighted(overlay,.82,frame,.18,0,frame)
        cv2.putText(frame, "MEMEFACE SYNTH", (18,34), cv2.FONT_HERSHEY_DUPLEX, .72, (255,80,230), 2)
        cv2.putText(frame, "AI PUPPET INSTRUMENT", (18,55), cv2.FONT_HERSHEY_SIMPLEX, .38, (80,240,255), 1)
        self._bar(frame,"MOUTH", state.mouth*12, 92, (30,50,255))
        self._bar(frame,"L BROW", (state.left_brow-.05)*10, 122, (255,180,20))
        self._bar(frame,"R BROW", (state.right_brow-.05)*10, 152, (255,80,200))
        self._bar(frame,"SMILE", (state.smile-.30)*4, 182, (40,255,120))
        chaos = np.clip((state.smile-.34)*4 + state.mouth*6,0,1)
        self._bar(frame,"CHAOS", chaos, 212, (0,140,255))
        cv2.putText(frame, f"MODE  {mode.name}", (18,250), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
        cv2.putText(frame, f"SOUND {preset}", (18,275), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
        cv2.putText(frame, f"{fps:4.1f} FPS  {'AUDIO' if audio else 'MUTED'}", (18,300), cv2.FONT_HERSHEY_SIMPLEX,.45,(170,180,200),1)
        if mode.index == 1:
            game = mode.game
            cv2.putText(frame, f"CALM {game.calmness:3.0f}   TIME {game.remaining:04.1f}", (18,330),
                        cv2.FONT_HERSHEY_SIMPLEX,.52,(80,255,255),2)
            if game.running and game.event_until > __import__("time").monotonic():
                cv2.putText(frame, game.event, (w//2-140, h//2), cv2.FONT_HERSHEY_DUPLEX, 1.4,(0,255,255),3)
            if game.finished:
                cv2.rectangle(frame, (w//2-245, h//2-65), (w//2+245, h//2+90), (20,12,35), -1)
                cv2.putText(frame, f"SCORE {game.score}  REACTIONS {game.reactions}", (w//2-210,h//2-20),
                            cv2.FONT_HERSHEY_DUPLEX,.8,(0,255,255),2)
                cv2.putText(frame, f"BEST CALM STREAK {game.best_streak:.1f}s", (w//2-170,h//2+15),
                            cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                replay=(w//2-75,h//2+35,w//2+75,h//2+70)
                self.buttons["replay"]=replay
                cv2.rectangle(frame,(replay[0],replay[1]),(replay[2],replay[3]),(180,55,180),-1)
                cv2.putText(frame,"REPLAY",(replay[0]+31,replay[1]+24),cv2.FONT_HERSHEY_DUPLEX,.55,(255,255,255),1)
        if mode.index == 2:
            key = mode.TRAIN[mode.training_index]
            val = getattr(state, key, min(state.left_ear,state.right_ear)) if key != "left_wink" else min(state.left_ear,state.right_ear)
            threshold_key = "wink" if key == "left_wink" else key
            threshold = (thresholds or {}).get(threshold_key, 0.0)
            active = (val < threshold) if key == "left_wink" else state.active.get(key, False)
            cv2.putText(frame, f"TRAIN: {key.upper()}  value={val:.3f}", (310,35),cv2.FONT_HERSHEY_DUPLEX,.7,(0,255,255),2)
            cv2.putText(frame, mode.INSTRUCTIONS[key]+"  [T next]", (310,62),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            status_color = (40,255,100) if active else (80,160,255)
            cv2.putText(frame, f"THRESHOLD {threshold:.3f}   {'ACTIVE!' if active else 'waiting...'}",
                        (310,88),cv2.FONT_HERSHEY_SIMPLEX,.55,status_color,2)
        if mode.index == 3:
            game = mode.hand_game
            if game.target:
                target = game.target
                center = (int(target.x*w), int(target.y*h))
                radius = int(target.radius*min(w,h))
                colors = {"orb": (255,80,230), "bonus": (0,255,255), "bomb": (0,40,255)}
                color = colors[target.kind]
                cv2.circle(frame, center, radius+7, color, 2, cv2.LINE_AA)
                cv2.circle(frame, center, radius, tuple(int(c*.35) for c in color), -1)
                icon = "!" if target.kind == "bomb" else ("+" if target.kind == "bonus" else "*")
                cv2.putText(frame, icon, (center[0]-10,center[1]+10),cv2.FONT_HERSHEY_DUPLEX,1,color,2)
            cv2.putText(frame, f"SCORE {game.score}   LIVES {game.lives}   COMBO x{game.combo}   {game.remaining:04.1f}s",
                        (305,34),cv2.FONT_HERSHEY_DUPLEX,.62,(255,255,255),2)
            if game.message_until > __import__("time").monotonic():
                cv2.putText(frame,game.message,(w//2-130,75),cv2.FONT_HERSHEY_DUPLEX,.8,(0,255,255),2)
            if game.finished:
                cv2.rectangle(frame,(w//2-230,h//2-70),(w//2+230,h//2+90),(18,12,32),-1)
                cv2.putText(frame,f"HAND SCORE {game.score}",(w//2-150,h//2-25),
                            cv2.FONT_HERSHEY_DUPLEX,.9,(0,255,255),2)
                cv2.putText(frame,f"COLLECTED {game.collected}   BEST COMBO x{game.best_combo}",
                            (w//2-190,h//2+10),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                replay=(w//2-75,h//2+35,w//2+75,h//2+70)
                self.buttons["replay_hand"]=replay
                cv2.rectangle(frame,(replay[0],replay[1]),(replay[2],replay[3]),(180,55,180),-1)
                cv2.putText(frame,"REPLAY",(replay[0]+31,replay[1]+24),cv2.FONT_HERSHEY_DUPLEX,.55,(255,255,255),1)
        labels = [("camera","START/STOP"),("mode","MODE [G]"),("preset","PRESET"),("calibrate","CALIBRATE [C]"),
                  ("mesh","MESH [F]"),("audio","AUDIO [M]"),("record","REC [R]"),("help","HELP [H]"),("quit","QUIT [Q]")]
        for i,(action,label) in enumerate(labels):
            col,row=i%2,i//2; x1=18+col*126; y1=h-145+row*27
            rect=(x1,y1,x1+116,y1+21); self.buttons[action]=rect
            cv2.rectangle(frame,(rect[0],rect[1]),(rect[2],rect[3]),(70,45,90),-1)
            cv2.putText(frame,label,(x1+5,y1+15),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1)
        if recording:
            cv2.circle(frame,(w-28,26),9,(0,0,255),-1)
            cv2.putText(frame,"REC",(w-75,32),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        if state.confidence < .5:
            cv2.putText(frame,"FACE NOT DETECTED",(w//2-170,h//2),cv2.FONT_HERSHEY_DUPLEX,1,(0,120,255),3)
        if calibration_prompt:
            cv2.rectangle(frame,(285,h-55),(w,h),(10,10,10),-1)
            cv2.putText(frame,calibration_prompt,(305,h-20),cv2.FONT_HERSHEY_DUPLEX,.7,(0,255,255),2)
        if self.help:
            cv2.rectangle(frame,(300,85),(w-25,h-25),(12,10,20),-1)
            lines=["FACE CONTROLS","Jaw = wah filter / laser","Left brow = pitch | Right brow = vibrato",
                   "Left wink = horn | Right wink = impact","Both eyes closed = mute | Puff cheeks = BOOM",
                   "Hand Mayhem: point + pinch, open palm bonus, fist shield",
                   "1-5 presets   M mute   F mesh   C calibrate","R record   G mode   T training target   Q quit",
                   "All video stays on this computer. Nothing is uploaded."]
            for i,line in enumerate(lines):
                cv2.putText(frame,line,(330,125+i*35),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA)
        return frame
