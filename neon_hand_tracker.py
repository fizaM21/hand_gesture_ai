import cv2
import mediapipe as mp
import numpy as np
import time
import math
import random

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

# ── Fingertip + palm landmark indices ────────────────────────────────────────
FINGERTIPS   = [4, 8, 12, 16, 20]   # thumb → pinky tips
PALM_CENTER  = 9                      # middle-finger MCP ≈ palm center
ADJACENT_PAIRS = [(4,8),(8,12),(12,16),(16,20)]  # fingertip threads

# ── Particle system ───────────────────────────────────────────────────────────
class Particle:
    def __init__(self, x, y, color):
        self.x = float(x)
        self.y = float(y)
        self.color = color
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = random.uniform(0.4, 1.0)   # 0→1
        self.decay = random.uniform(0.02, 0.06)
        self.size = random.randint(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.08          # gentle gravity
        self.vx *= 0.97
        self.life -= self.decay
        return self.life > 0

    def draw(self, canvas):
        alpha = max(0.0, self.life)
        c = tuple(int(ch * alpha) for ch in self.color)
        cv2.circle(canvas, (int(self.x), int(self.y)), self.size, c, -1)


particles: list[Particle] = []

# ── Glow helper ───────────────────────────────────────────────────────────────
def draw_glowing_line(canvas, p1, p2, color, thickness=2, glow_radius=18):
    """Draw a line with additive multi-layer glow."""
    for r, alpha in [(glow_radius, 0.08), (glow_radius//2, 0.18), (3, 0.6), (thickness, 1.0)]:
        c = tuple(int(ch * alpha) for ch in color)
        cv2.line(canvas, p1, p2, c, max(1, r))
    cv2.line(canvas, p1, p2, color, thickness)


def draw_glowing_circle(canvas, center, radius, color, thickness=2, glow_radius=20):
    for r, alpha in [(glow_radius, 0.08), (glow_radius//2, 0.2), (4, 0.5), (thickness, 1.0)]:
        c = tuple(int(ch * alpha) for ch in color)
        cv2.circle(canvas, center, radius + r, c, max(1, r))
    cv2.circle(canvas, center, radius, color, thickness)


# ── Color cycling (HSV → BGR) ──────────────────────────────────────────────
def hue_to_bgr(hue_deg):
    """hue 0-360 → bright neon BGR"""
    hsv = np.uint8([[[int(hue_deg / 2) % 180, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


# ── Previous positions for speed calc ────────────────────────────────────────
prev_positions: dict[int, tuple] = {}   # landmark_id → (x,y,t)

# ── Main loop ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

start_time = time.time()

print("Neon Hand Tracker running — press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Pure black canvas
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    elapsed = time.time() - start_time

    # Slow color cycle — one full rotation every ~12 seconds per finger
    base_hues = [
        (elapsed * 30 + i * 72) % 360 for i in range(5)
    ]

    # MediaPipe detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    now = time.time()

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            lm = hand_lms.landmark

            def pt(idx):
                return (int(lm[idx].x * w), int(lm[idx].y * h))

            palm = pt(PALM_CENTER)

            # ── Per-fingertip: string + particles + speed glow ────────────
            for i, tip_idx in enumerate(FINGERTIPS):
                tip = pt(tip_idx)
                hue = base_hues[i]
                color = hue_to_bgr(hue)

                # Speed calculation
                key = tip_idx
                speed = 0.0
                if key in prev_positions:
                    px, py, pt_time = prev_positions[key]
                    dt = max(now - pt_time, 1e-5)
                    dist = math.hypot(tip[0]-px, tip[1]-py)
                    speed = dist / dt  # pixels/sec

                prev_positions[key] = (tip[0], tip[1], now)

                # Glow radius scales with speed (faster = brighter/wider)
                glow = int(np.clip(10 + speed * 0.04, 10, 50))
                brightness = np.clip(0.5 + speed * 0.001, 0.5, 1.0)
                boosted_color = tuple(int(ch * brightness) for ch in color)

                # String from palm to fingertip
                draw_glowing_line(canvas, palm, tip, boosted_color,
                                  thickness=2, glow_radius=glow)

                # Glowing dot at fingertip
                draw_glowing_circle(canvas, tip, 6, boosted_color,
                                    thickness=-1, glow_radius=glow)

                # Spawn particles at fingertip (more if moving fast)
                n_particles = int(np.clip(speed * 0.05, 1, 8))
                for _ in range(n_particles):
                    particles.append(Particle(tip[0], tip[1], boosted_color))

            # ── Thin threads between adjacent fingertips ──────────────────
            for (a, b) in ADJACENT_PAIRS:
                pa, pb = pt(a), pt(b)
                # blend hues of the two fingers
                hi = FINGERTIPS.index(a)
                thread_hue = (base_hues[hi] + 20) % 360
                thread_color = hue_to_bgr(thread_hue)
                thin_c = tuple(int(ch * 0.5) for ch in thread_color)
                cv2.line(canvas, pa, pb, thin_c, 1)

            # ── Glowing palm dot ──────────────────────────────────────────
            palm_hue = (elapsed * 25) % 360
            draw_glowing_circle(canvas, palm, 8,
                                hue_to_bgr(palm_hue),
                                thickness=-1, glow_radius=22)

    # ── Update + draw particles ───────────────────────────────────────────
    particles[:] = [p for p in particles if p.update()]
    for p in particles:
        p.draw(canvas)

    # Limit particle count
    if len(particles) > 1200:
        particles[:] = particles[-1200:]

    cv2.imshow("Neon Hand Tracker", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
