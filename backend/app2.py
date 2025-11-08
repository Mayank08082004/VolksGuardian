"""
Driver Wellness Monitoring Prototype — Phase 10.8 (Stable + Fixed Fatigue Scoring)
-----------------------------------------------------------------------------------
- Correct fatigue scoring using EAR + PERCLOS + Blink/min + CNN
- Adaptive threshold with head-motion compensation
- Hysteresis fatigue trigger with cooldown
- Async TTS and Ollama (non-blocking)
- Full HUD display (EAR, Blink/min, PERCLOS, CNN, Fatigue, HR, FPS)
- CSV logging for post-analysis
"""

import os, cv2, time, random, threading, subprocess, numpy as np, csv, queue
from collections import deque
import tensorflow as tf
from playsound import playsound
import pyttsx3
import mediapipe as mp

# ==========================================================
# ---------------------- CONFIG ----------------------------
# ==========================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

USE_OLLAMA = False
ALERT_SOUND = "alert.wav"
IMG_SIZE = 224
LOG_PATH = "fatigue_log.csv"


# Automatically locate model files in the same directory as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = [
    os.path.join(BASE_DIR, "driver_fatigue_detector_v1.h5"),
    os.path.join(BASE_DIR, "driver_fatigue_detector_v1_backup.h5")  # optional backup
]

TFLITE_PATH = os.path.join(BASE_DIR, "driver_fatigue_detector_v1.tflite")




# ==========================================================
# --------------------- MODEL LOADING ----------------------
# ==========================================================
def load_model():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            print(f"✅ Loading {p}")
            try:
                m = tf.keras.models.load_model(p, compile=False)
                m.trainable = False
                return m, False
            except Exception as e:
                print(f"⚠️ Error loading {p}: {e}")
    if os.path.exists(TFLITE_PATH):
        print("✅ Loading TFLite model")
        inter = tf.lite.Interpreter(model_path=TFLITE_PATH)
        inter.allocate_tensors()
        return inter, True
    raise FileNotFoundError("❌ No model found!")

model, IS_TFLITE = load_model()

def predict_model(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img.astype("float32") / 255.0, axis=0)
    if IS_TFLITE:
        inp = model.get_input_details()
        out = model.get_output_details()
        model.set_tensor(inp[0]['index'], img)
        model.invoke()
        res = model.get_tensor(out[0]['index'])
        prob = float(res[0][0])
    else:
        prob = float(model.predict(img, verbose=0)[0][0])
    return float(np.clip(prob, 0.0, 1.0))

# ==========================================================
# ---------------------- TTS THREAD ------------------------
# ==========================================================
class TTSWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        self.tts = pyttsx3.init()
        self.tts.setProperty("rate", 175)
        self.tts.setProperty("volume", 1.0)
    def run(self):
        while True:
            text = self.q.get()
            try:
                self.tts.say(text)
                self.tts.runAndWait()
            except Exception as e:
                print("[TTS Error]", e)
    def speak(self, text): 
        self.q.put(text)

tts_worker = TTSWorker()
tts_worker.start()

# ==========================================================
# ---------------------- AUDIO -----------------------------
# ==========================================================
class AudioController:
    def __init__(self): 
        self.last_alert = 0
    def play_alert(self):
        if time.time() - self.last_alert < 5: 
            return
        self.last_alert = time.time()
        threading.Thread(target=self._sound, daemon=True).start()
    def _sound(self):
        try:
            if os.path.exists(ALERT_SOUND): 
                playsound(ALERT_SOUND, block=False)
            else: 
                print("\a")
        except Exception as e: 
            print("[Alert Error]", e)
    def speak(self, txt): 
        tts_worker.speak(txt)

# ==========================================================
# ------------------- OLLAMA WORKER ------------------------
# ==========================================================
class LLMWorker(threading.Thread):
    def __init__(self, audio):
        super().__init__(daemon=True)
        self.audio, self.q, self.last_message = audio, queue.Queue(), None
    def enqueue(self, a, f): 
        self.q.put((a, f))
    def run(self):
        while True:
            try:
                a, f = self.q.get()
                base = {
                    "beep_alert": "Wake up and focus.",
                    "speak_break": "Take a short rest.",
                    "speak_breathing": "Let’s breathe slowly.",
                    "play_music": "Playing calm sounds."
                }[a]
                if USE_OLLAMA:
                    proc = subprocess.Popen(
                        ["ollama", "run", "llama3",
                         f"Driver fatigue {f:.2f}. Short caring message like: {base}"],
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
                    )
                    out, _ = proc.communicate(timeout=6)
                    msg = (out or "").strip() or base
                else:
                    msg = base
                self.last_message = msg
                self.audio.speak(msg)
            except Exception as e:
                print("[LLM ERROR]", e)
            time.sleep(0.05)

# ==========================================================
# ------------------ SENSOR (Sim) --------------------------
# ==========================================================
class HeartSource:
    def __init__(self): 
        self.hr = 78
        self.hrv = 40
    def read(self):
        self.hr += random.choice([-1, 0, 1])
        self.hrv += random.choice([-2, -1, 0, 1])
        return {"hr": int(np.clip(self.hr, 60, 110)), "hrv": int(np.clip(self.hrv, 15, 80))}

# ==========================================================
# ------------------ VISION MODULE -------------------------
# ==========================================================
class VisionModule:
    LEFT = [33,160,158,133,153,144]
    RIGHT = [362,385,387,263,373,380]
    HEAD_POINTS = [1, 33, 263]

    def __init__(self, warmup_s=10.0, perclos_horizon_s=30.0):
        self.mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.draw = mp.solutions.drawing_utils
        self.ear_hist = deque(maxlen=15)
        self.closed_samples = deque()
        self.blink_times = deque(maxlen=240)
        self.baseline_ear = None
        self.frames_closed = 0
        self.EAR_T = 0.23
        self.warmup_s = warmup_s
        self.start_time = time.time()
        self.perclos_horizon_s = perclos_horizon_s
        self.fatigue = 0.0
        self.last_head_pos = None
        self.motion_tolerance = 0.015
        # new state
        self.chin_hist = deque(maxlen=30)
        self.yawn_start = None
        self.offroad_start = None

        # angle-based nod detection state
        self.angle_hist = deque(maxlen=30)   # stores recent pitch angles (degrees)
        self.last_nod_time = None
        



    def _ear(self, lm, idx, w, h):
        def p(i): return np.array([lm[idx[i]].x*w, lm[idx[i]].y*h])
        p1,p2,p3,p4,p5,p6 = [p(i) for i in range(6)]
        vert = (np.linalg.norm(p2-p6)+np.linalg.norm(p3-p5))/2
        horiz = np.linalg.norm(p1-p4)+1e-6
        return vert/horiz

    def _head_motion(self, lm, w, h):
        pts = [np.array([lm[i].x*w, lm[i].y*h]) for i in self.HEAD_POINTS]
        center = np.mean(pts, axis=0)
        if self.last_head_pos is None:
            self.last_head_pos = center
            return 0.0
        dist = np.linalg.norm(center - self.last_head_pos) / w
        self.last_head_pos = center
        return dist

    def _update_perclos(self, is_closed):
        now = time.time()
        self.closed_samples.append((now, 1 if is_closed else 0))
        while self.closed_samples and now - self.closed_samples[0][0] > self.perclos_horizon_s:
            self.closed_samples.popleft()
        if not self.closed_samples: return 0.0
        vals = [c for _, c in self.closed_samples]
        return sum(vals) / len(vals)

    def _blink_rate_per_min(self):
        now = time.time()
        while self.blink_times and now - self.blink_times[0] > 60:
            self.blink_times.popleft()
        return len(self.blink_times)

    # ---------- Yawning Detection (MAR) ----------
    def _mar(self, lm, w, h):
        # Using mediapipe indices for upper/lower lips and mouth corners
        top = np.array([lm[13].x*w, lm[13].y*h])
        bottom = np.array([lm[14].x*w, lm[14].y*h])
        # corners 78 and 308 are common mouth corners in MediaPipe
        left = np.array([lm[78].x*w, lm[78].y*h])
        right = np.array([lm[308].x*w, lm[308].y*h])
        mar = np.linalg.norm(top - bottom) / (np.linalg.norm(left - right) + 1e-6)
        return mar

    # ---------- Head Nod Detection ----------
        # ---------- Pitch angle (forehead->chin) ----------
    def _pitch_angle(self, lm):
        """
        Compute pitch angle (degrees) of vector from forehead/top to chin/bottom.
        Uses MediaPipe landmarks: forehead ~10, chin ~152 (normalized coords).
        Returns angle in degrees where positive angle implies chin below forehead.
        """
        top = np.array([lm[10].x, lm[10].y])
        bottom = np.array([lm[152].x, lm[152].y])
        vec = bottom - top  # normalized vector
        # angle relative to horizontal; convert to degrees
        ang = np.degrees(np.arctan2(vec[1], vec[0]))
        return ang

    # ---------- Head nod detection using pitch angle ----------
    def _head_nod_angle(self, lm, w, h):
        """
        Detect a quick down-then-up motion in pitch angle.
        Uses angle_hist to track recent pitch angles in degrees.
        Returns True when a nod pattern is detected.
        """
        ang = self._pitch_angle(lm)
        self.angle_hist.append(ang)

        # need enough history to compute differences
        if len(self.angle_hist) < 8:
            return False

        # compute recent differences (deg/frame)
        a = np.array(self.angle_hist)
        vel = np.diff(a)  # angular velocity

        # simple pattern: detect a positive peak (down) then negative (up)
        # thresholds (degrees/frame) — tune these for your camera/frame-rate
        DOWN_VEL_TH = 2.5    # downward angular velocity threshold
        UP_VEL_TH   = -2.5   # upward angular velocity threshold

        # require at least one downwards spike and one upwards spike in buffer
        if np.any(vel > DOWN_VEL_TH) and np.any(vel < UP_VEL_TH):
            # enforce minimal time between nods to avoid duplicates
            now = time.time()
            if self.last_nod_time is None or (now - self.last_nod_time) > 1.2:
                self.last_nod_time = now
                # clear history to avoid immediate retrigger
                self.angle_hist.clear()
                return True

        return False


    # ---------- Gaze Off-road Detection ----------
    def _gaze_direction(self, lm, w, h):
        # Use iris landmarks (468..473) — compute average iris x and compare with eye center
        # Note: indices are normalized (0..1), we return normalized deviation
        try:
            left_iris_x = lm[468].x
            right_iris_x = lm[473].x
            eye_center = (lm[33].x + lm[263].x) / 2.0
            deviation = abs(((left_iris_x + right_iris_x) / 2.0) - eye_center)
        except Exception:
            deviation = 0.0
        return deviation

    def step(self, frame, cnn_prob):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        res = self.mesh.process(rgb)
        ear = 0.0; blink_rate = 0.0; perclos = 0.0
        mar = 0.0; gaze_dev = 0.0; nod = False

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            l = self._ear(lm, self.LEFT, w, h)
            r = self._ear(lm, self.RIGHT, w, h)
            ear = (l + r) / 2.0
            self.ear_hist.append(ear)
            smooth_ear = np.mean(self.ear_hist)

            # dynamic baseline EAR threshold
            if self.baseline_ear is None:
                self.baseline_ear = smooth_ear
            elif time.time() - self.start_time < self.warmup_s:
                self.baseline_ear = 0.95 * self.baseline_ear + 0.05 * smooth_ear
                self.EAR_T = 0.7 * self.baseline_ear

            head_motion = self._head_motion(lm, w, h)
            is_closed = smooth_ear < self.EAR_T if head_motion < self.motion_tolerance else False

            if is_closed:
                self.frames_closed += 1
            else:
                if self.frames_closed > 5:
                    self.blink_times.append(time.time())
                self.frames_closed = 0

            blink_rate = self._blink_rate_per_min()
            perclos = self._update_perclos(is_closed)

            # new feature computations
            mar = self._mar(lm, w, h)
            gaze_dev = self._gaze_direction(lm, w, h)
            # replace chin-based nod with angle-based nod
            nod = self._head_nod_angle(lm, w, h)

            ear_deficit = np.clip((self.EAR_T - smooth_ear) / (self.EAR_T * 0.6), 0, 1)
            blink_penalty = np.clip((15.0 - blink_rate) / 15.0, 0, 1)
            visual = 0.6 * perclos + 0.25 * ear_deficit + 0.15 * blink_penalty
            raw_fatigue = 0.6 * visual + 0.4 * cnn_prob

            α_up, α_down = 0.25, 0.1
            self.fatigue += (raw_fatigue - self.fatigue) * (α_up if raw_fatigue > self.fatigue else α_down)

            # ---------- EVENT DETECTIONS ----------
            # Yawning (MAR)
            if mar > 0.6:
                if self.yawn_start is None:
                    self.yawn_start = time.time()
                elif time.time() - self.yawn_start > 1.0:
                    # yawn confirmed
                    print("[Yawn detected]")
                    self.fatigue = min(1.0, self.fatigue + 0.05)
            else:
                self.yawn_start = None

            # Head nod
            if nod:
                print("[Head nod detected]")
                self.fatigue = min(1.0, self.fatigue + 0.08)

            # Gaze off-road
            if gaze_dev > 0.4:
                if self.offroad_start is None:
                    self.offroad_start = time.time()
                elif time.time() - self.offroad_start > 1.0:
                    print("[Gaze off-road detected]")
                    self.fatigue = min(1.0, self.fatigue + 0.05)
            else:
                self.offroad_start = None

            self.draw.draw_landmarks(frame, res.multi_face_landmarks[0],
                                     mp.solutions.face_mesh.FACEMESH_CONTOURS)

        # return expanded metrics so main loop can display/log them
        return {
            "ear": ear,
            "blink_per_min": blink_rate,
            "perclos_30s": perclos,
            "fatigue": float(np.clip(self.fatigue, 0, 1)),
            "mar": float(mar),
            "gaze_dev": float(gaze_dev),
            "head_nod": bool(nod)
        }, frame


# ==========================================================
# ---------------------- LOGGING ---------------------------
# ==========================================================
def log_row(ts, ear, blinkpm, perclos, cnn, fatigue, action, mar=None, gaze_dev=None, head_nod=False):
    header = ["time", "ear", "blink_per_min", "perclos_30s", "cnn", "fatigue", "action", "mar", "gaze_dev", "head_nod"]
    exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([ts, ear, blinkpm, perclos, cnn, fatigue, action or "none",
                    (None if mar is None else f"{mar:.3f}"),
                    (None if gaze_dev is None else f"{gaze_dev:.3f}"),
                    int(bool(head_nod))])


# ==========================================================
# ------------------------ MAIN ----------------------------
# ==========================================================
def main():
    vision = VisionModule(warmup_s=10.0, perclos_horizon_s=30.0)
    audio = AudioController()
    llm = LLMWorker(audio)
    llm.start()
    heart = HeartSource()

    cap = cv2.VideoCapture(0)
    last_action = None
    last_trigger_t = 0.0
    FATIGUE_HIGH = 0.5
    FATIGUE_LOW  = 0.45
    COOLDOWN_S   = 15
    prev = time.time(); fps = 0.0

    print("🚗 Running Phase 10.8 (Stable + Fixed Fatigue Scoring)")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break

        cnn_prob = predict_model(frame)
        vis, overlay = vision.step(frame, cnn_prob)

        f = vis["fatigue"]; ear = vis["ear"]; br = vis["blink_per_min"]
        pcl = vis["perclos_30s"]; hr = heart.read()["hr"]
        mar = vis.get("mar", 0.0)
        gaze_dev = vis.get("gaze_dev", 0.0)
        head_nod = vis.get("head_nod", False)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-3, (now - prev)))
        prev = now

        can_trigger = (now - last_trigger_t) > COOLDOWN_S
        if can_trigger and f >= FATIGUE_HIGH:
            a = random.choice(["beep_alert", "speak_break", "speak_breathing"])
            last_action = a
            last_trigger_t = now
            print(f"[Trigger] {a} | fatigue={f:.2f}")
            if a == "beep_alert": audio.play_alert()
            llm.enqueue(a, f)

        if f <= FATIGUE_LOW and (now - last_trigger_t) > 3.0:
            last_action = None

        cv2.rectangle(overlay, (0,0), (overlay.shape[1], 110), (25,25,25), -1)
        hud = (f"EAR:{ear:.3f} | Blink/min:{br:.0f} | PERCLOS30:{pcl:.2f} | "
       f"CNN:{cnn_prob:.2f} | Fatigue:{f:.2f} | HR:{hr} | FPS:{fps:.1f} | "
       f"MAR:{mar:.2f} | Gaze:{gaze_dev:.2f}")
        cv2.putText(overlay, hud, (10,40), 0, 0.6, (255,255,255), 2)

        bar_w = 320
        x0, y0 = 10, 70
        cv2.rectangle(overlay, (x0, y0), (x0 + bar_w, y0 + 20), (60,60,60), 1)
        cv2.rectangle(overlay, (x0+1, y0+1), (x0+1 + int(bar_w * float(np.clip(f,0,1))), y0+19),
                      (0,200,0) if f < FATIGUE_HIGH else (0,0,255), -1)

        if llm.last_message:
            cv2.putText(overlay, llm.last_message[:90], (10, overlay.shape[0]-15),
                        0, 0.6, (200,255,200), 2)

        log_row(time.time(), ear, br, pcl, cnn_prob, f, last_action, mar=mar, gaze_dev=gaze_dev, head_nod=head_nod)

        cv2.imshow("Driver Wellness (10.8 Fixed Fatigue Scoring + HUD)", overlay)
        if cv2.waitKey(1) & 0xFF == 27: break
        time.sleep(0.005)

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Session Ended.")

# ==========================================================
if __name__ == "__main__":
    main()
