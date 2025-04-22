import cv2
import numpy as np
import mediapipe as mp
import random
import time

# ===== Retro Sci‑Fi Visual Palette =====
NEON_CYAN = (255, 255, 0)      # BGR
NEON_MAGENTA = (255, 0, 255)   # BGR
NEON_GREEN = (0, 255, 0)       # BGR  (true green for border / HUD)
HUD_GREY = (50, 50, 50)        # Grid lines
WHITE = (255, 255, 255)

grid_overlay = None  # Initialized on first frame

# Game Settings
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
PADDLE_WIDTH = 20
PADDLE_HEIGHT = 150
BALL_RADIUS = 15
PADDLE_MARGIN = 40

BALL_SPEED = 14  # initial speed, pixels per frame (slightly faster)
SPEED_INCREMENT = 0.2  # reduced speed growth per paddle hit
MAX_SCORE = 10

mp_hands = mp.solutions.hands

class Paddle:
    def __init__(self, is_left: bool):
        self.is_left = is_left
        self.x = PADDLE_MARGIN if is_left else FRAME_WIDTH - PADDLE_MARGIN - PADDLE_WIDTH
        self.y = FRAME_HEIGHT // 2 - PADDLE_HEIGHT // 2  # top-left corner y
        self.height = PADDLE_HEIGHT
        self.width = PADDLE_WIDTH

    @property
    def rect(self):
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def update(self, hand_y: int):
        """Update paddle position based on hand y coordinate. Map full camera height to full play area."""
        mapped_y = int((hand_y / FRAME_HEIGHT) * (FRAME_HEIGHT - self.height))
        self.y = max(0, min(FRAME_HEIGHT - self.height, mapped_y))

    def draw(self, frame):
        color = NEON_CYAN if self.is_left else NEON_MAGENTA
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.width, self.y + self.height), color, -1)

class Ball:
    def __init__(self):
        self.reset(random.choice([-1, 1]))

    def reset(self, direction: int):
        self.x = FRAME_WIDTH // 2
        self.y = FRAME_HEIGHT // 2
        angle = random.uniform(-0.25, 0.25) * np.pi  # random small angle
        speed = BALL_SPEED
        self.vx = int(direction * speed * np.cos(angle))
        self.vy = int(speed * np.sin(angle))

    def update(self):
        self.x += self.vx
        self.y += self.vy
        # top and bottom bounce
        if self.y <= BALL_RADIUS or self.y >= FRAME_HEIGHT - BALL_RADIUS:
            self.vy = -self.vy

    def draw(self, frame):
        cv2.circle(frame, (self.x, self.y), BALL_RADIUS, NEON_GREEN, -1)

    def increase_speed(self):
        """Increase ball's speed by a constant factor while preserving direction."""
        factor = 1 + SPEED_INCREMENT
        self.vx = int(self.vx * factor) if self.vx != 0 else int(np.sign(random.choice([-1, 1])) * BALL_SPEED)
        self.vy = int(self.vy * factor) if self.vy != 0 else int(np.sign(random.choice([-1, 1])) * BALL_SPEED * 0.5)


def detect_hands_positions(frame, hands_detector):
    """Detect hands and return a dict {'left': y_center, 'right': y_center}."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)
    positions = {}
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Use wrist landmark (0) as reference
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            x_px = int(wrist.x * FRAME_WIDTH)
            y_px = int(wrist.y * FRAME_HEIGHT)
            # Determine side by x position to be more robust
            if x_px < FRAME_WIDTH // 2:
                positions['left'] = y_px
            else:
                positions['right'] = y_px
            # Optionally draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return positions


def main():
    global grid_overlay
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    left_paddle = Paddle(is_left=True)
    right_paddle = Paddle(is_left=False)
    ball = Ball()

    scores = [0, 0]  # left, right

    # Mediapipe Hands
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=4) as hands:
        prev_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)  # Mirror view

            positions = detect_hands_positions(frame, hands)
            if 'left' in positions:
                left_paddle.update(positions['left'])
            if 'right' in positions:
                right_paddle.update(positions['right'])

            # Update ball
            ball.update()

            # Collision with paddles (improved edge tolerance)
            # Left paddle collision
            if (ball.x - BALL_RADIUS <= left_paddle.x + left_paddle.width and
                left_paddle.x <= ball.x <= left_paddle.x + left_paddle.width + BALL_RADIUS and
                left_paddle.y - BALL_RADIUS <= ball.y <= left_paddle.y + left_paddle.height + BALL_RADIUS):
                ball.vx = abs(ball.vx)
                ball.increase_speed()
            # Right paddle collision
            if (ball.x + BALL_RADIUS >= right_paddle.x and
                right_paddle.x - BALL_RADIUS <= ball.x <= right_paddle.x + BALL_RADIUS + right_paddle.width and
                right_paddle.y - BALL_RADIUS <= ball.y <= right_paddle.y + right_paddle.height + BALL_RADIUS):
                ball.vx = -abs(ball.vx)
                ball.increase_speed()

            # Scoring
            if ball.x < 0:
                scores[1] += 1
                ball.reset(direction=1)
            elif ball.x > FRAME_WIDTH:
                scores[0] += 1
                ball.reset(direction=-1)

            # Draw entities
            left_paddle.draw(frame)
            right_paddle.draw(frame)
            ball.draw(frame)

            # Retro grid & border (draw AFTER entities for neon overlap)
            # Create grid overlay once (lazy init)
            if grid_overlay is None:
                grid_overlay = np.zeros_like(frame)
                step = 80
                for x in range(0, FRAME_WIDTH, step):
                    cv2.line(grid_overlay, (x, 0), (x, FRAME_HEIGHT), HUD_GREY, 1)
                for y in range(0, FRAME_HEIGHT, step):
                    cv2.line(grid_overlay, (0, y), (FRAME_WIDTH, y), HUD_GREY, 1)

            # Blend grid overlay (semi‑transparent)
            cv2.addWeighted(grid_overlay, 0.35, frame, 0.65, 0, frame)
            # Border rectangle (neon green)
            cv2.rectangle(frame, (0, 0), (FRAME_WIDTH - 1, FRAME_HEIGHT - 1), NEON_GREEN, 2)

            # Draw scores
            cv2.putText(frame, str(scores[0]), (FRAME_WIDTH // 4, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.4, NEON_CYAN, 4)
            cv2.putText(frame, str(scores[1]), (3 * FRAME_WIDTH // 4, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.4, NEON_MAGENTA, 4)

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, FRAME_HEIGHT - 20), cv2.FONT_HERSHEY_PLAIN, 1.4, NEON_GREEN, 2)

            cv2.imshow('Webcam Pong', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Check for winner
            if max(scores) >= MAX_SCORE:
                winner_text = 'Left Player Wins!' if scores[0] > scores[1] else 'Right Player Wins!'
                print(winner_text)
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main() 