import cv2
import sys
import time
from collections import deque
from typing import List, Tuple
import numpy as np
import pygame
import mediapipe as mp

# ---------------------------
# ====== CONSTANTS ==========
# ---------------------------
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BLOCK_SIZE = 30  # pixels (placeholder; will be recalculated at runtime)
FPS = 60
DROP_INTERVAL = 0.7  # seconds between automatic drops

# Eye‑blink thresholds (tuned empirically; may need adjustment per user)
EAR_THRESHOLD = 0.18
CONSEC_FRAMES_BLINK = 2

# Colors in RGB
COLORS = {
    "I": (0, 240, 240),
    "J": (0, 0, 240),
    "L": (240, 160, 0),
    "O": (240, 240, 0),
    "S": (0, 240, 0),
    "T": (160, 0, 240),
    "Z": (240, 0, 0),
    "grid": (40, 40, 40),
    "background": (10, 10, 10),
    "border": (180, 180, 180),
}

SHAPES = {
    "I": [[1, 1, 1, 1]],
    "J": [[1, 0, 0], [1, 1, 1]],
    "L": [[0, 0, 1], [1, 1, 1]],
    "O": [[1, 1], [1, 1]],
    "S": [[0, 1, 1], [1, 1, 0]],
    "T": [[0, 1, 0], [1, 1, 1]],
    "Z": [[1, 1, 0], [0, 1, 1]],
}

# Mediapipe landmark indices for blink detection and gaze
LEFT_CORNER_LEFT_EYE = 263  # user left eye outer corner
RIGHT_CORNER_LEFT_EYE = 362  # user left eye inner corner (towards nose)
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374

LEFT_CORNER_RIGHT_EYE = 33  # user right eye inner corner (towards nose)
RIGHT_CORNER_RIGHT_EYE = 133  # user right eye outer corner
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145

# Outer eye corners used for approximate head yaw calculation
LEFT_OUTER_EYE = 33
RIGHT_OUTER_EYE = 263

# Additional landmarks for head orientation
NOSE_TIP = 1
CHIN = 152

LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# Eye corner landmark pairs [(outer, inner)] for each eye
LEFT_EYE_CORNERS = (LEFT_CORNER_LEFT_EYE, RIGHT_CORNER_LEFT_EYE)
RIGHT_EYE_CORNERS = (LEFT_CORNER_RIGHT_EYE, RIGHT_CORNER_RIGHT_EYE)

# ---------------------------
#        TETRIS LOGIC
# ---------------------------

class Piece:
    def __init__(self, typ: str):
        self.typ = typ
        self.rotation = 0
        self.shape = np.array(SHAPES[typ])
        self.x = BOARD_WIDTH // 2 - len(self.shape[0]) // 2
        self.y = 0

    def rotated(self, direction: int):
        # direction +1 clockwise, -1 counter‑clockwise
        k = -direction  # numpy.rot90 rotates CCW for positive k
        new_shape = np.rot90(self.shape, k)
        p = Piece(self.typ)
        p.shape = new_shape
        p.x, p.y = self.x, self.y
        return p

class Board:
    def __init__(self):
        self.grid = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype="U1")  # empty strings
        self.current = self.new_piece()
        self.game_over = False
        self.last_drop = time.time()

    def new_piece(self) -> Piece:
        typ = np.random.choice(list(SHAPES.keys()))
        return Piece(typ)

    def can_place(self, piece: Piece) -> bool:
        for dy, row in enumerate(piece.shape):
            for dx, val in enumerate(row):
                if val:
                    x = piece.x + dx
                    y = piece.y + dy
                    if x < 0 or x >= BOARD_WIDTH or y >= BOARD_HEIGHT:
                        return False
                    if y >= 0 and self.grid[y, x]:
                        return False
        return True

    def lock_piece(self):
        p = self.current
        for dy, row in enumerate(p.shape):
            for dx, val in enumerate(row):
                if val:
                    x = p.x + dx
                    y = p.y + dy
                    if 0 <= y < BOARD_HEIGHT:
                        self.grid[y, x] = p.typ
        self.clear_lines()
        self.current = self.new_piece()
        if not self.can_place(self.current):
            self.game_over = True

    def clear_lines(self):
        to_clear = [i for i in range(BOARD_HEIGHT) if all(self.grid[i, :])]
        if to_clear:
            self.grid = np.delete(self.grid, to_clear, axis=0)
            new_rows = np.zeros((len(to_clear), BOARD_WIDTH), dtype="U1")
            self.grid = np.vstack((new_rows, self.grid))

    # ------------- PUBLIC ACTIONS --------------
    def move_horizontal(self, column: int):
        if self.game_over:
            return
        p = self.current
        new_x = max(0, min(column, BOARD_WIDTH - p.shape.shape[1]))
        tentative = Piece(p.typ)
        tentative.shape = p.shape.copy()
        tentative.x = new_x
        tentative.y = p.y
        if self.can_place(tentative):
            self.current.x = new_x

    def rotate(self, direction: int):
        if self.game_over:
            return
        rotated = self.current.rotated(direction)
        if self.can_place(rotated):
            self.current = rotated

    def drop(self, rows: int = 1):
        if self.game_over:
            return
        self.current.y += rows
        if not self.can_place(self.current):
            self.current.y -= rows
            if rows == 1:
                # piece has landed
                self.lock_piece()
            else:
                # if fast drop of 2 rows, step row by row
                for _ in range(rows):
                    self.drop(1)

    def update(self):
        if self.game_over:
            return
        now = time.time()
        if now - self.last_drop >= DROP_INTERVAL:
            self.drop()
            self.last_drop = now

# ---------------------------
#      EYE TRACKING UTILS
# ---------------------------

def eye_aspect_ratio(landmarks, top_id, bottom_id, left_id, right_id):
    top = np.array([landmarks[top_id].x, landmarks[top_id].y])
    bottom = np.array([landmarks[bottom_id].x, landmarks[bottom_id].y])
    left = np.array([landmarks[left_id].x, landmarks[left_id].y])
    right = np.array([landmarks[right_id].x, landmarks[right_id].y])
    # vertical distance
    vert = np.linalg.norm(top - bottom)
    horiz = np.linalg.norm(left - right)
    if horiz == 0:
        return 0
    return vert / horiz

class EyeControl:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.left_blink_counter = 0
        self.right_blink_counter = 0
        self.both_blink_cooldown = 0
        self._calibrating = False
        self._calib_samples: list[float] = []
        self._yaw_samples: list[float] = []
        self._pitch_samples: list[float] = []
        self._roll_samples: list[float] = []
        self._vert_samples: list[float] = []
        self.neutral_ratio = 0.5  # overwritten after calibration
        self.neutral_yaw = 0.0
        self.neutral_pitch = 0.0
        self.neutral_roll = 0.0
        self.neutral_vertical = 0.5
        # extremes collected during manual calibration
        self.yaw_left = None
        self.yaw_right = None
        self.pitch_up = None
        self.pitch_down = None
        self.roll_left = None
        self.roll_right = None
        self.gaze_left = None
        self.gaze_right = None
        self.gaze_up = None
        self.gaze_down = None

    def metrics(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        face_landmarks = results.multi_face_landmarks[0].landmark
        yaw_val = self._yaw_value(face_landmarks)
        pitch_val = self._pitch_value(face_landmarks)
        roll_val = self._roll_value(face_landmarks)

        ratio_left = self._eye_gaze_ratio(face_landmarks, LEFT_IRIS_CENTER, LEFT_EYE_CORNERS)
        ratio_right = self._eye_gaze_ratio(face_landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_CORNERS)
        gaze_ratio = (ratio_left + ratio_right) / 2

        v_left = self._eye_vertical_ratio(face_landmarks, LEFT_IRIS_CENTER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM)
        v_right = self._eye_vertical_ratio(face_landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM)
        gaze_vertical = (v_left + v_right) / 2

        return yaw_val, pitch_val, roll_val, gaze_ratio, gaze_vertical, face_landmarks

    def _eye_gaze_ratio(self, landmarks, iris_id: int, corner_ids: Tuple[int, int]):
        outer_id, inner_id = corner_ids  # outer = temple side, inner = nose side
        iris_x = landmarks[iris_id].x
        outer_x = landmarks[outer_id].x
        inner_x = landmarks[inner_id].x
        denom = inner_x - outer_x
        if denom == 0:
            return 0.5  # avoid div-by-zero
        return (iris_x - outer_x) / denom  # 0 (look outer) .. 1 (look inner / nose)

    def _eye_vertical_ratio(self, landmarks, iris_id: int, top_id: int, bottom_id: int):
        iris_y = landmarks[iris_id].y
        top_y = landmarks[top_id].y
        bottom_y = landmarks[bottom_id].y
        denom = bottom_y - top_y
        if denom == 0:
            return 0.5
        return (iris_y - top_y) / denom

    def _yaw_value(self, landmarks):
        left = landmarks[LEFT_OUTER_EYE].x
        right = landmarks[RIGHT_OUTER_EYE].x
        return right - left

    def _pitch_value(self, landmarks):
        nose = landmarks[NOSE_TIP]
        chin = landmarks[CHIN]
        dy = chin.y - nose.y
        dz = chin.z - nose.z
        return np.degrees(np.arctan2(dz, dy))

    def _roll_value(self, landmarks):
        left = landmarks[LEFT_OUTER_EYE]
        right = landmarks[RIGHT_OUTER_EYE]
        dy = right.y - left.y
        dx = right.x - left.x
        return np.degrees(np.arctan2(dy, dx))

    def process(self, frame: np.ndarray):
        metrics = self.metrics(frame)
        gaze_ratio = None
        gaze_vertical = None
        yaw_val = None
        pitch_val = None
        roll_val = None
        blink_left = False
        blink_right = False
        blink_both = False
        if metrics:
            yaw_val, pitch_val, roll_val, gaze_ratio, gaze_vertical, face_landmarks = metrics
            # EAR for eyes
            ear_left = eye_aspect_ratio(
                face_landmarks,
                LEFT_EYE_TOP,
                LEFT_EYE_BOTTOM,
                LEFT_CORNER_LEFT_EYE,
                RIGHT_CORNER_LEFT_EYE,
            )
            ear_right = eye_aspect_ratio(
                face_landmarks,
                RIGHT_EYE_TOP,
                RIGHT_EYE_BOTTOM,
                LEFT_CORNER_RIGHT_EYE,
                RIGHT_CORNER_RIGHT_EYE,
            )
            # Blink detection
            if ear_left < EAR_THRESHOLD:
                self.left_blink_counter += 1
                if self.left_blink_counter == CONSEC_FRAMES_BLINK:
                    blink_left = True
            else:
                self.left_blink_counter = 0

            if ear_right < EAR_THRESHOLD:
                self.right_blink_counter += 1
                if self.right_blink_counter == CONSEC_FRAMES_BLINK:
                    blink_right = True
            else:
                self.right_blink_counter = 0

            # Determine both blink (simultaneous)
            if blink_left and blink_right:
                blink_both = True
                blink_left = False
                blink_right = False

            # Auto-calibration during first 50 valid frames
            if self._calibrating:
                self._calib_samples.append(gaze_ratio)
                self._yaw_samples.append(yaw_val)
                self._pitch_samples.append(pitch_val)
                self._roll_samples.append(roll_val)
                self._vert_samples.append(gaze_vertical)
                if len(self._calib_samples) >= 50:
                    self.neutral_ratio = float(np.mean(self._calib_samples))
                    self.neutral_yaw = float(np.mean(self._yaw_samples))
                    self.neutral_pitch = float(np.mean(self._pitch_samples))
                    self.neutral_roll = float(np.mean(self._roll_samples))
                    self.neutral_vertical = float(np.mean(self._vert_samples))
                    self._calibrating = False

            # Adjust gaze using calibration and compensate for head orientation
            gaze_ratio -= (self.neutral_ratio - 0.5)
            gaze_ratio += (self.neutral_yaw - yaw_val)
            gaze_ratio += (self.neutral_roll - roll_val) * 0.5
            gaze_ratio = np.clip(gaze_ratio, 0.0, 1.0)

        return gaze_ratio, blink_left, blink_right, blink_both

# ---------------------------
#   CALIBRATION ROUTINE
# ---------------------------

def run_calibration(screen, cap, eye_ctl: EyeControl, clock):
    font = pygame.font.SysFont("Arial", 32)
    info_text = font.render(
        "Press SPACE to begin calibration", True, (255, 255, 255)
    )
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                break
        else:
            screen.fill((0, 0, 0))
            rect = info_text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
            screen.blit(info_text, rect)
            pygame.display.flip()
            clock.tick(30)
            continue
        break

    steps = [
        ("Hold head straight, look forward", "neutral"),
        ("Move head left", "yaw_left"),
        ("Move head right", "yaw_right"),
        ("Tilt head up", "pitch_up"),
        ("Tilt head down", "pitch_down"),
        ("Roll head left", "roll_left"),
        ("Roll head right", "roll_right"),
        ("Look left", "gaze_left"),
        ("Look right", "gaze_right"),
        ("Look up", "gaze_up"),
        ("Look down", "gaze_down"),
    ]

    for text, attr in steps:
        instruction = font.render(text, True, (255, 255, 255))
        samples = []
        for i in range(30):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
            ret, frame = cap.read()
            if not ret:
                continue
            metrics = eye_ctl.metrics(frame)
            if metrics:
                yaw, pitch, roll, gaze, vert, _ = metrics
                samples.append((yaw, pitch, roll, gaze, vert))

            screen.fill((0, 0, 0))
            rect = instruction.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
            screen.blit(instruction, rect)
            pygame.display.flip()
            clock.tick(30)

        if not samples:
            continue
        avg = np.mean(samples, axis=0)
        if attr == "neutral":
            eye_ctl.neutral_yaw = avg[0]
            eye_ctl.neutral_pitch = avg[1]
            eye_ctl.neutral_roll = avg[2]
            eye_ctl.neutral_ratio = avg[3]
            eye_ctl.neutral_vertical = avg[4]
        elif attr == "yaw_left":
            eye_ctl.yaw_left = avg[0]
        elif attr == "yaw_right":
            eye_ctl.yaw_right = avg[0]
        elif attr == "pitch_up":
            eye_ctl.pitch_up = avg[1]
        elif attr == "pitch_down":
            eye_ctl.pitch_down = avg[1]
        elif attr == "roll_left":
            eye_ctl.roll_left = avg[2]
        elif attr == "roll_right":
            eye_ctl.roll_right = avg[2]
        elif attr == "gaze_left":
            eye_ctl.gaze_left = avg[3]
        elif attr == "gaze_right":
            eye_ctl.gaze_right = avg[3]
        elif attr == "gaze_up":
            eye_ctl.gaze_up = avg[4]
        elif attr == "gaze_down":
            eye_ctl.gaze_down = avg[4]

    eye_ctl._calibrating = False

# ---------------------------
#      RENDERING HELPERS
# ---------------------------

def draw_board(screen, board: Board):
    screen.fill(COLORS["background"])
    # Draw existing blocks
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            typ = board.grid[y, x]
            if typ:
                pygame.draw.rect(
                    screen,
                    COLORS[typ],
                    pygame.Rect(x * BLOCK_W, y * BLOCK_H, BLOCK_W, BLOCK_H),
                )
    # Draw current piece
    p = board.current
    for dy, row in enumerate(p.shape):
        for dx, val in enumerate(row):
            if val:
                x = p.x + dx
                y = p.y + dy
                if y >= 0:
                    pygame.draw.rect(
                        screen,
                        COLORS[p.typ],
                        pygame.Rect(x * BLOCK_W, y * BLOCK_H, BLOCK_W, BLOCK_H),
                    )
    # Grid lines
    for x in range(BOARD_WIDTH + 1):
        pygame.draw.line(
            screen, COLORS["grid"], (x * BLOCK_W, 0), (x * BLOCK_W, BOARD_HEIGHT * BLOCK_H)
        )
    for y in range(BOARD_HEIGHT + 1):
        pygame.draw.line(
            screen,
            COLORS["grid"],
            (0, y * BLOCK_H),
            (BOARD_WIDTH * BLOCK_W, y * BLOCK_H),
        )

# ---------------------------
#              MAIN
# ---------------------------

def main():
    pygame.init()
    info = pygame.display.Info()
    scr_h = info.current_h - 40  # leave a margin so bottom isn't cut
    global BLOCK_SIZE
    BLOCK_SIZE = scr_h // BOARD_HEIGHT
    # Make cells a bit wider than tall (1.2 aspect) for easier eye movement
    global BLOCK_W, BLOCK_H
    BLOCK_H = BLOCK_SIZE
    BLOCK_W = int(BLOCK_H * 1.2)

    screen_width = BOARD_WIDTH * BLOCK_W
    screen = pygame.display.set_mode((screen_width, scr_h))
    pygame.display.set_caption("Eye‑controlled Tetris (fullscreen height)")
    clock = pygame.time.Clock()

    board = Board()
    eye_ctl = EyeControl()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        sys.exit(1)

    gaze_history: deque = deque(maxlen=5)

    run_calibration(screen, cap, eye_ctl, clock)

    try:
        while True:
            # Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and board.game_over:
                    # restart game
                    board = Board()
                    gaze_history.clear()
                    run_calibration(screen, cap, eye_ctl, clock)

            # Capture frame
            ret, frame = cap.read()
            if not ret:
                break

            gaze_ratio, blink_left, blink_right, blink_both = eye_ctl.process(frame)

            if gaze_ratio is not None:
                gaze_history.append(gaze_ratio)
                smooth_ratio = np.mean(gaze_history)
                column = int(smooth_ratio * BOARD_WIDTH)
                board.move_horizontal(column)

            if blink_both:
                board.drop(rows=2)
            elif blink_left:
                board.rotate(+1)
            elif blink_right:
                board.rotate(-1)

            board.update()

            draw_board(screen, board)

            if board.game_over:
                font = pygame.font.SysFont("Arial", 36)
                txt = font.render("GAME OVER", True, (255, 0, 0))
                rect = txt.get_rect(center=(BOARD_WIDTH * BLOCK_W // 2, BOARD_HEIGHT * BLOCK_H // 2))
                screen.blit(txt, rect)

            pygame.display.flip()
            clock.tick(FPS)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        pygame.quit()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 