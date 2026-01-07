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
EAR_THRESHOLD = 0.21  # Slightly higher for fewer false positives
CONSEC_FRAMES_BLINK = 3  # Require more frames for reliable blink detection
BLINK_COOLDOWN_FRAMES = 15  # Prevent rapid-fire blinks

# Gaze control parameters
GAZE_HISTORY_SIZE = 15  # More frames for smoother tracking
GAZE_DEAD_ZONE = 0.15  # Center dead zone (no movement when gaze is within this range of center)
MOVEMENT_COOLDOWN_FRAMES = 8  # Only move piece every N frames
GAZE_SENSITIVITY = 1.2  # Amplify gaze movement outside dead zone

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
        self.blink_cooldown = 0  # Cooldown counter for all blinks
        self._calibrating = True
        self._calib_samples: list[float] = []
        self.neutral_ratio = 0.5  # will be overwritten after calibration
        # Track raw gaze for debugging overlay
        self.raw_gaze = 0.5
        self.ear_left = 0.0
        self.ear_right = 0.0

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

        # Decrement blink cooldown
        if self.blink_cooldown > 0:
            self.blink_cooldown -= 1

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
            self.ear_left = eye_aspect_ratio(
                face_landmarks,
                LEFT_EYE_TOP,
                LEFT_EYE_BOTTOM,
                LEFT_CORNER_LEFT_EYE,
                RIGHT_CORNER_LEFT_EYE,
            )
            self.ear_right = eye_aspect_ratio(
                face_landmarks,
                RIGHT_EYE_TOP,
                RIGHT_EYE_BOTTOM,
                LEFT_CORNER_RIGHT_EYE,
                RIGHT_CORNER_RIGHT_EYE,
            )
            # Blink detection (only if not in cooldown)
            if self.blink_cooldown == 0:
                if self.ear_left < EAR_THRESHOLD:
                    self.left_blink_counter += 1
                    if self.left_blink_counter == CONSEC_FRAMES_BLINK:
                        blink_left = True
                else:
                    self.left_blink_counter = 0

                if self.ear_right < EAR_THRESHOLD:
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

                # Set cooldown if any blink detected
                if blink_left or blink_right or blink_both:
                    self.blink_cooldown = BLINK_COOLDOWN_FRAMES
                    self.left_blink_counter = 0
                    self.right_blink_counter = 0

            # --- Improved gaze ratio ---
            ratio_left = self._eye_gaze_ratio(
                face_landmarks, LEFT_IRIS_CENTER, LEFT_EYE_CORNERS
            )
            ratio_right = self._eye_gaze_ratio(
                face_landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_CORNERS
            )
            gaze_ratio = (ratio_left + ratio_right) / 2  # 0 = looking towards temples (left), 1 = towards nose/right
            self.raw_gaze = gaze_ratio  # Store for debug overlay

            # Auto‑calibration during first 60 valid frames (longer for stability)
            if self._calibrating:
                self._calib_samples.append(gaze_ratio)
                if len(self._calib_samples) >= 60:
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

    gaze_history: deque = deque(maxlen=GAZE_HISTORY_SIZE)
    movement_cooldown = 0
    show_debug = False  # Toggle with 'D' key
    last_column = BOARD_WIDTH // 2  # Track last position for smoother movement

    try:
        while True:
            # Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and board.game_over:
                        # restart game
                        board = Board()
                        eye_ctl._calibrating = True
                        eye_ctl._calib_samples.clear()
                        gaze_history.clear()
                        last_column = BOARD_WIDTH // 2
                    elif event.key == pygame.K_d:
                        show_debug = not show_debug
                    elif event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt

            # Capture frame
            ret, frame = cap.read()
            if not ret:
                break

            gaze_ratio, blink_left, blink_right, blink_both = eye_ctl.process(frame)

            # Decrement movement cooldown
            if movement_cooldown > 0:
                movement_cooldown -= 1

            if gaze_ratio is not None:
                gaze_history.append(gaze_ratio)
                smooth_ratio = np.mean(gaze_history)

                # Apply dead zone: only move if gaze is outside center region
                center_offset = smooth_ratio - 0.5  # -0.5 to +0.5

                if abs(center_offset) > GAZE_DEAD_ZONE and movement_cooldown == 0:
                    # Scale movement outside dead zone
                    if center_offset > 0:
                        adjusted = (center_offset - GAZE_DEAD_ZONE) / (0.5 - GAZE_DEAD_ZONE)
                    else:
                        adjusted = (center_offset + GAZE_DEAD_ZONE) / (0.5 - GAZE_DEAD_ZONE)

                    # Apply sensitivity and convert to column
                    adjusted = np.clip(adjusted * GAZE_SENSITIVITY, -1.0, 1.0)
                    target_column = int((adjusted + 1) / 2 * BOARD_WIDTH)
                    target_column = max(0, min(BOARD_WIDTH - 1, target_column))

                    # Move one step toward target for smoother control
                    if target_column > last_column:
                        last_column += 1
                    elif target_column < last_column:
                        last_column -= 1

                    board.move_horizontal(last_column)
                    movement_cooldown = MOVEMENT_COOLDOWN_FRAMES

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

            # Debug overlay (toggle with 'D' key)
            if show_debug:
                debug_font = pygame.font.SysFont("Arial", 14)
                debug_y = 10
                debug_lines = [
                    f"Gaze: {eye_ctl.raw_gaze:.2f} (neutral: {eye_ctl.neutral_ratio:.2f})",
                    f"Smooth: {np.mean(gaze_history) if gaze_history else 0:.2f}",
                    f"EAR L/R: {eye_ctl.ear_left:.2f}/{eye_ctl.ear_right:.2f}",
                    f"Column: {last_column}",
                    f"Calibrating: {eye_ctl._calibrating}",
                    f"[D] toggle debug | [ESC] quit",
                ]
                for line in debug_lines:
                    txt = debug_font.render(line, True, (0, 255, 0))
                    screen.blit(txt, (5, debug_y))
                    debug_y += 16

                # Draw gaze indicator bar at top
                bar_width = BOARD_WIDTH * BLOCK_W
                gaze_x = int(np.mean(gaze_history) * bar_width) if gaze_history else bar_width // 2
                pygame.draw.rect(screen, (50, 50, 50), (0, 0, bar_width, 6))
                # Draw dead zone
                dead_left = int((0.5 - GAZE_DEAD_ZONE) * bar_width)
                dead_right = int((0.5 + GAZE_DEAD_ZONE) * bar_width)
                pygame.draw.rect(screen, (80, 80, 0), (dead_left, 0, dead_right - dead_left, 6))
                # Draw gaze position
                pygame.draw.rect(screen, (0, 255, 0), (gaze_x - 3, 0, 6, 6))

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