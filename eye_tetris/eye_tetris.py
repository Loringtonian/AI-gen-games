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
        self._calibrating = True
        self._calib_samples: list[float] = []
        self.neutral_ratio = 0.5  # will be overwritten after calibration

    def _eye_gaze_ratio(self, landmarks, iris_id: int, corner_ids: Tuple[int, int]):
        outer_id, inner_id = corner_ids  # outer = temple side, inner = nose side
        iris_x = landmarks[iris_id].x
        outer_x = landmarks[outer_id].x
        inner_x = landmarks[inner_id].x
        denom = inner_x - outer_x
        if denom == 0:
            return 0.5  # avoid div‑by‑zero
        return (iris_x - outer_x) / denom  # 0 (look outer) .. 1 (look inner / nose)

    def process(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        gaze_ratio = None
        blink_left = False
        blink_right = False
        blink_both = False
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            # face bounding box for head‑movement compensation
            xs = [lm.x for lm in face_landmarks]
            x_min, x_max = min(xs), max(xs)
            # Iris center average
            iris_x = (
                face_landmarks[LEFT_IRIS_CENTER].x + face_landmarks[RIGHT_IRIS_CENTER].x
            ) / 2.0
            gaze_ratio = (iris_x - x_min) / (x_max - x_min)
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

            # --- Improved gaze ratio ---
            ratio_left = self._eye_gaze_ratio(
                face_landmarks, LEFT_IRIS_CENTER, LEFT_EYE_CORNERS
            )
            ratio_right = self._eye_gaze_ratio(
                face_landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_CORNERS
            )
            gaze_ratio = (ratio_left + ratio_right) / 2  # 0 = looking towards temples (left), 1 = towards nose/right

            # Auto‑calibration during first 50 valid frames
            if self._calibrating:
                self._calib_samples.append(gaze_ratio)
                if len(self._calib_samples) >= 50:
                    self.neutral_ratio = float(np.mean(self._calib_samples))
                    self._calibrating = False
            # Shift so neutral gaze centres piece
            gaze_ratio = np.clip((gaze_ratio - (self.neutral_ratio - 0.5)), 0.0, 1.0)

        return gaze_ratio, blink_left, blink_right, blink_both

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

    try:
        while True:
            # Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and board.game_over:
                    # restart game
                    board = Board()
                    eye_ctl._calibrating = True
                    eye_ctl._calib_samples.clear()
                    gaze_history.clear()

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