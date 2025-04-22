import cv2
import numpy as np
import sounddevice as sd
import queue
import time
import random
import os
import json

# ===============================
# Constants / Settings
# ===============================
WINDOW_TITLE = "Clap Bounce – Precision Reflex Challenge"
WIDTH, HEIGHT = 900, 600
BALL_RADIUS = 20
BALL_SPEED = 400  # pixels per second
FPS = 60
FRAME_TIME = 1.0 / FPS

# Clap detection
AUDIO_SAMPLE_RATE = 44100
AUDIO_BLOCK_SIZE = 1024
# Clap detection sensitivity
CLAP_AMPLITUDE_THRESHOLD = 0.55  # restored previous stable value
# Cooldown between claps to avoid double‑counting. Spec mandates ±50 ms window around a
# waveform burst is considered the *same* clap, so we use 50 ms here.
CLAP_COOLDOWN_SEC = 0.05

# Scoring
PERFECT_SCORE = 100       # maximum points per clap
# Remove unused gaussian scoring remnants – spec defines discrete tiers only

# Game flow
BOUNCES_PER_ROUND = 20  # how many wall hits before round ends

# Persistence files
HIGHSCORE_FILE = "highscore.txt"  # kept for backward compatibility (single high score)
LEADERBOARD_FILE = "leaderboard.json"  # stores list of dicts {initials, score}

# Clap timing window (seconds) – claps outside this after an impact are ignored
# Max time a clap can still be attached to the last impact (spec: until next bounce
# but we conservatively allow 0.6 s).
CLAP_WINDOW_SEC = 0.6

# UI
HUD_HEIGHT = 80  # top bar height to display score & messages without overlap

# Colors (BGR for OpenCV)
BLACK = (0, 0, 0)
NEON_GREEN = (0, 255, 0)
NEON_BLUE = (255, 127, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)

# Feedback display
FEEDBACK_DURATION = 1.2  # seconds message stays on‑screen


# ===============================
# Audio stream & clap detection
# ===============================
class ClapDetector:
    """Continuously listens to microphone and pushes clap timestamps into a queue."""

    def __init__(self, threshold: float = CLAP_AMPLITUDE_THRESHOLD):
        self.threshold = threshold
        self._queue: "queue.Queue[float]" = queue.Queue()
        self._last_clap_time = 0.0
        # For debug overlay we expose the most‑recent peak amplitude detected
        self.current_volume = 0.0
        self.stream = sd.InputStream(
            channels=1,
            samplerate=AUDIO_SAMPLE_RATE,
            blocksize=AUDIO_BLOCK_SIZE,
            callback=self._audio_callback,
        )

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            # In practice we ignore status flags but could log
            pass
        # Peak amplitude in buffer (audio is float32 in [-1, 1])
        volume = float(np.max(np.abs(indata)))
        # Keep latest volume for debug overlay
        self.current_volume = volume
        now = time.time()
        if volume > self.threshold and (now - self._last_clap_time) > CLAP_COOLDOWN_SEC:
            self._last_clap_time = now
            self._queue.put(now)

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()

    def has_clap(self):
        return not self._queue.empty()

    def pop_clap(self):
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None


# ===============================
# Helper functions
# ===============================

def load_highscore() -> int:
    if os.path.exists(HIGHSCORE_FILE):
        try:
            with open(HIGHSCORE_FILE, "r") as f:
                return int(f.read().strip())
        except Exception:
            pass
    return 0


def save_highscore(score: int):
    try:
        with open(HIGHSCORE_FILE, "w") as f:
            f.write(str(score))
    except Exception:
        pass


def load_leaderboard():
    if os.path.exists(LEADERBOARD_FILE):
        try:
            with open(LEADERBOARD_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_leaderboard(entries):
    try:
        with open(LEADERBOARD_FILE, "w") as f:
            json.dump(entries, f)
    except Exception:
        pass


def compute_score(delta_ms: float) -> int:
    """Step‑wise scoring based on clap timing proximity.

    The score is symmetric around the impact (pre/post) and awarded as follows
    (|delta_ms| is the absolute difference between the clap and impact time):
        0‑5   ms  -> 100 points
        5‑10  ms  -> 90 points
        10‑20 ms  -> 70 points
        20‑30 ms  -> 50 points
        30‑50 ms  -> 30 points
        50‑100 ms -> 10 points
        >100 ms   -> 0 points
    """
    abs_delta = abs(delta_ms)

    if abs_delta <= 5:
        return 100
    elif abs_delta <= 10:
        return 90
    elif abs_delta <= 20:
        return 70
    elif abs_delta <= 30:
        return 50
    elif abs_delta <= 50:
        return 30
    else:
        return 0


# ===============================
# Main game class
# ===============================
class ClapBounceGame:
    def __init__(self):
        self.ball_x = WIDTH // 2
        self.ball_y = HEIGHT // 2
        angle = random.uniform(0, 2 * np.pi)
        self.ball_vx = BALL_SPEED * np.cos(angle)
        self.ball_vy = BALL_SPEED * np.sin(angle)
        self.last_update_time = time.time()

        self.score = 0
        self.bounce_count = 0

        # Leaderboard and high score
        self.leaderboard = load_leaderboard()
        self.highscore = self.leaderboard[0]['score'] if self.leaderboard else 0

        # Initials input state
        self.initial_input_active = False
        self.initials = ""
        # When finishing initials entry we need to swallow the last key press to
        # avoid unintentionally triggering a new round (e.g. if the final
        # letter was "s").
        self.just_finished_initials = False

        self.state = "start"  # start, play, gameover

        # Audio
        self.clap_detector = ClapDetector()
        self.clap_detector.start()

        # Feedback message (e.g., Good / Late / Missed)
        self.feedback_text = ""
        self.feedback_expires = 0.0

        # Debug / instrumentation
        self.debug_enabled = False
        self.last_delta_ms = None  # Δt for last scored clap
        self.last_points = None
        self.event_log = []  # Keep (delta_ms, points)

    # ---------------------------
    # Physics & game logic
    # ---------------------------
    def _update_ball(self):
        # Before moving the ball, check for missed clap on the previous impact
        current_time = time.time()
        if (self.last_impact_time is not None and not self.clap_registered_for_impact and
                (current_time - self.last_impact_time) > CLAP_WINDOW_SEC):
            # Player failed to clap within the allowed window – mark as missed
            self.clap_registered_for_impact = True
            self._mark_missed()

        # Continue with physics update
        current_time = time.time()
        dt = current_time - self.last_update_time
        if dt <= 0:
            return  # no progression
        self.last_update_time = current_time

        self.ball_x += self.ball_vx * dt
        self.ball_y += self.ball_vy * dt

        impact_occurred = False
        # Check collisions with walls
        if self.ball_x <= BALL_RADIUS:
            self.ball_x = BALL_RADIUS
            self.ball_vx = abs(self.ball_vx)
            impact_occurred = True
        elif self.ball_x >= WIDTH - BALL_RADIUS:
            self.ball_x = WIDTH - BALL_RADIUS
            self.ball_vx = -abs(self.ball_vx)
            impact_occurred = True

        if self.ball_y <= HUD_HEIGHT + BALL_RADIUS:
            self.ball_y = HUD_HEIGHT + BALL_RADIUS
            self.ball_vy = abs(self.ball_vy)
            impact_occurred = True
        elif self.ball_y >= HEIGHT - BALL_RADIUS:
            self.ball_y = HEIGHT - BALL_RADIUS
            self.ball_vy = -abs(self.ball_vy)
            impact_occurred = True

        if impact_occurred:
            # If previous impact never received a clap, mark it as missed before moving on
            if not self.clap_registered_for_impact:
                self._mark_missed()

            self.bounce_count += 1
            self.last_impact_time = current_time
            self.clap_registered_for_impact = False  # await clap

    def _process_claps(self):
        while self.clap_detector.has_clap():
            clap_time = self.clap_detector.pop_clap()
            if clap_time is None:
                break
            if self.last_impact_time is None:
                continue  # no impact yet
            if self.clap_registered_for_impact:
                continue  # already scored for this impact

            delta_sec = clap_time - self.last_impact_time
            # Ignore claps that are way too early (prior to window)
            if abs(delta_sec) > CLAP_WINDOW_SEC:
                continue

            delta_ms = delta_sec * 1000.0
            points = compute_score(delta_ms)
            self.last_delta_ms = delta_ms
            self.last_points = points

            # Update total & log
            self.score += points
            self.event_log.append({"delta_ms": delta_ms, "points": points})

            # Feedback mapping per spec
            if points == 100:
                self.feedback_text = "Perfect Clap +100"
            elif points == 90:
                self.feedback_text = "Close +90"
            elif points == 70:
                self.feedback_text = "Good +70"
            elif points == 50:
                self.feedback_text = "Early/Late +50"
            elif points == 30:
                self.feedback_text = "Barely +30"
            else:  # 0 points – too early or late
                if delta_ms < 0:
                    self.feedback_text = "Too Early – 0"
                else:
                    self.feedback_text = "Too Late – 0"

            self.feedback_expires = time.time() + FEEDBACK_DURATION

            # Regardless of scoring, mark this impact as handled so the clap cannot
            # be (mis)applied to a future bounce.
            self.clap_registered_for_impact = True

    # ---------------------------
    # Drawing helpers
    # ---------------------------
    def _draw_start_screen(self, frame):
        cv2.putText(frame, "Clap Bounce", (WIDTH // 2 - 170, HEIGHT // 3), cv2.FONT_HERSHEY_TRIPLEX, 1.8, NEON_GREEN, 2)
        cv2.putText(frame, "Press 's' to start", (WIDTH // 2 - 180, HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 2)
        cv2.putText(frame, "Press 'q' to quit", (WIDTH // 2 - 170, HEIGHT // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 2)

    def _draw_gameover_screen(self, frame):
        # Title & current score
        cv2.putText(frame, "GAME OVER", (WIDTH // 2 - 160, HEIGHT // 6), cv2.FONT_HERSHEY_TRIPLEX, 1.8, NEON_GREEN, 2)
        cv2.putText(frame, f"Score: {self.score}", (WIDTH // 2 - 120, HEIGHT // 4), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2)

        # Leaderboard
        cv2.putText(frame, "Top 5:", (WIDTH // 2 - 60, HEIGHT // 4 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
        for idx, entry in enumerate(self.leaderboard[:5]):
            text = f"{idx + 1}. {entry['initials']}  {entry['score']}"
            cv2.putText(frame, text, (WIDTH // 2 - 60, HEIGHT // 4 + 80 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)

        # Initials prompt or instructions
        if self.initial_input_active:
            prompt = f"Enter Initials: {self.initials}{'_' if len(self.initials) < 3 else ''}"
            cv2.putText(frame, prompt, (WIDTH // 2 - 180, HEIGHT - 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
        else:
            cv2.putText(frame, "Press 's' to play again", (WIDTH // 2 - 200, HEIGHT - 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
            cv2.putText(frame, "Press 'q' to quit", (WIDTH // 2 - 150, HEIGHT - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)

    def _draw_play_screen(self, frame):
        # Playfield Border (exclude HUD)
        cv2.rectangle(frame, (0, HUD_HEIGHT), (WIDTH - 1, HEIGHT - 1), NEON_GREEN, 2)
        # Ball
        cv2.circle(frame, (int(self.ball_x), int(self.ball_y)), BALL_RADIUS, NEON_BLUE, -1)
        # HUD background
        cv2.rectangle(frame, (0, 0), (WIDTH, HUD_HEIGHT), (0, 0, 0), -1)

        # Scoreboard inside HUD
        cv2.putText(frame, f"Score: {self.score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
        cv2.putText(frame, f"High: {self.highscore}", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
        cv2.putText(frame, f"Bounces: {self.bounce_count}/{BOUNCES_PER_ROUND}", (440, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)

        # Feedback message (center HUD)
        if self.feedback_text and time.time() < self.feedback_expires:
            text_size = cv2.getTextSize(self.feedback_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            x = WIDTH // 2 - text_size[0] // 2
            cv2.putText(frame, self.feedback_text, (x, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, NEON_BLUE, 2)

        # Debug overlay (toggle with 'd')
        if self.debug_enabled:
            debug_y = 40  # inside HUD (HUD_HEIGHT=80)
            cv2.putText(frame, f"MicVol: {self.clap_detector.current_volume:.3f}", (10, debug_y), cv2.FONT_HERSHEY_PLAIN, 1.0, RED, 1)
            debug_y += 15
            cv2.putText(frame, f"Impact ts: {self.last_impact_time:.3f}", (10, debug_y), cv2.FONT_HERSHEY_PLAIN, 1.0, RED, 1)
            debug_y += 15
            cv2.putText(frame, f"Last clap Δt: {self.last_delta_ms if self.last_delta_ms is not None else '—'} ms", (10, debug_y), cv2.FONT_HERSHEY_PLAIN, 1.0, RED, 1)
            debug_y += 15
            cv2.putText(frame, f"Last pts: {self.last_points if self.last_points is not None else '—'}", (10, debug_y), cv2.FONT_HERSHEY_PLAIN, 1.0, RED, 1)

    # ---------------------------
    # Game state transitions
    # ---------------------------
    def _reset_round(self):
        # Reset ball position and velocity
        self.ball_x = WIDTH // 2
        self.ball_y = HEIGHT // 2
        angle = random.uniform(0, 2 * np.pi)
        self.ball_vx = BALL_SPEED * np.cos(angle)
        self.ball_vy = BALL_SPEED * np.sin(angle)
        self.last_update_time = time.time()

        self.score = 0
        self.bounce_count = 0
        self.last_impact_time = None
        self.clap_registered_for_impact = True

    # ---------------------------
    # Leaderboard helpers
    # ---------------------------
    def _handle_initials_input(self, key_code):
        """Collect up to 3 alphabetic characters for player initials."""
        if key_code == 0xFF:
            return  # no key pressed
        # Backspace handling (key code 8 on some systems, 127 on others)
        if key_code in (8, 127):
            self.initials = self.initials[:-1]
            return

        if len(self.initials) >= 3:
            return

        ch = chr(key_code)
        if ch.isalpha():
            self.initials += ch.upper()

        # Auto‑finalize when 3 chars entered
        if len(self.initials) == 3:
            # Add to leaderboard and persist
            self.leaderboard.append({"initials": self.initials, "score": self.score})
            self.leaderboard.sort(key=lambda e: e["score"], reverse=True)
            self.leaderboard = self.leaderboard[:5]
            save_leaderboard(self.leaderboard)

            self.highscore = self.leaderboard[0]["score"]
            self.initial_input_active = False
            # Mark that we have just finished initials entry so the main loop
            # can ignore this frame's key press.
            self.just_finished_initials = True
            # Return to start screen after saving initials
            self.state = "start"

    # ---------------------------
    # Main render & update loop
    # ---------------------------
    def run(self):
        cv2.namedWindow(WINDOW_TITLE)
        while True:
            t_loop_start = time.time()
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

            # State‑specific logic
            if self.state == "start":
                self._draw_start_screen(frame)
            elif self.state == "play":
                self._update_ball()
                self._process_claps()
                self._draw_play_screen(frame)
                if self.bounce_count >= BOUNCES_PER_ROUND:
                    # End round – transition to gameover and prepare initials entry
                    self.state = "gameover"
                    self.initial_input_active = True
                    self.initials = ""
            elif self.state == "gameover":
                self._draw_gameover_screen(frame)
            else:
                raise ValueError("Unknown state")

            cv2.imshow(WINDOW_TITLE, frame)

            # Handle key input
            key = cv2.waitKey(1) & 0xFF
            if self.state == "gameover" and self.initial_input_active:
                self._handle_initials_input(key)

            # If initials input was just completed, ignore this frame's key to
            # prevent accidental "s" from starting a new round immediately.
            if self.just_finished_initials:
                self.just_finished_initials = False
                key = 0xFF

            if key == ord('q'):
                break

            if key == ord('s'):
                if self.state == "start" or (self.state == "gameover" and not self.initial_input_active):
                    self._reset_round()
                    self.state = "play"

            # Global debug toggle
            if key == ord('d'):
                self.debug_enabled = not self.debug_enabled

            # Maintain target FPS
            elapsed = time.time() - t_loop_start
            if elapsed < FRAME_TIME:
                time.sleep(FRAME_TIME - elapsed)

        # Clean up
        self.clap_detector.stop()
        cv2.destroyAllWindows()

    def _mark_missed(self):
        """Handle case where no clap was registered in time."""
        self.feedback_text = "MISSED!"
        self.feedback_expires = time.time() + FEEDBACK_DURATION


# ===============================
# Entrypoint
# ===============================
if __name__ == "__main__":
    game = ClapBounceGame()
    game.run() 