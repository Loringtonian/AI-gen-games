import cv2
import numpy as np
import random
import time

"""
Puzzle Piece Speed Challenge
--------------------------------
A simple single‑player game:
    • Left side shows a 512×512 image with one rectangular section removed.
    • Right side shows five candidate pieces; only one fits the missing area.
    • Press keys 1‒5 to choose a piece. Wrong guesses require you to try again.
    • Solve 5 randomly generated puzzles as fast as possible.

The entire game is self‑contained – images are generated procedurally, so no
external assets are required.
"""

# ------------------------ Visual Constants -------------------------
LEFT_IMG_SIZE = 512                # Square image dimensions (px)
PATCH_SIZE_MIN = 60                # Minimum puzzle piece edge (px)
PATCH_SIZE_MAX = 150               # Maximum puzzle piece edge (px)
CANDIDATE_SIZE = 120               # Render size of candidate thumbnails (px)
MARGIN = 20                        # Spacing between UI elements (px)
NUM_CANDIDATES = 5
NUM_PUZZLES = 10

# Compute window/canvas size for new layout (candidates underneath the puzzle)
candidate_row_w = NUM_CANDIDATES * CANDIDATE_SIZE + (NUM_CANDIDATES + 1) * MARGIN
canvas_w = max(LEFT_IMG_SIZE + 2 * MARGIN, candidate_row_w)
canvas_h = LEFT_IMG_SIZE + 3 * MARGIN + CANDIDATE_SIZE  # top margin, puzzle, gap, candidates, bottom margin

FONT = cv2.FONT_HERSHEY_SIMPLEX


class Puzzle:
    """Represents a single puzzle – stores base image with blank, candidate pieces, and correct index."""

    def __init__(self):
        self.base_img = self._generate_base_image()
        (self.h, self.w) = self.base_img.shape[:2]
        self.patch_coords, self.correct_piece = self._remove_random_patch()
        self.candidates = self._generate_candidates()

    @staticmethod
    def _generate_base_image(size: int = LEFT_IMG_SIZE) -> np.ndarray:
        """Generate base image of one of multiple visual styles for variety."""
        style = random.choice(["gradient", "shapes", "stripes"])

        if style == "gradient":
            # Smooth colour gradient
            c1 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
            c2 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
            orientation = random.choice([0, 1, 2])  # h, v, radial
            if orientation == 0:  # horizontal
                grad = np.tile(np.linspace(0, 1, size, dtype=np.float32), (size, 1))[:, :, None]
            elif orientation == 1:  # vertical
                grad = np.tile(np.linspace(0, 1, size, dtype=np.float32), (size, 1)).T[:, :, None]
            else:  # radial
                yy, xx = np.mgrid[0:size, 0:size]
                centre = size / 2.0
                dist = np.sqrt((xx - centre) ** 2 + (yy - centre) ** 2)
                grad = np.clip(dist / dist.max(), 0, 1)[:, :, None]
            img = (c1 * (1 - grad) + c2 * grad).astype(np.uint8)
            # Add mild noise for texture
            noise = np.random.randint(0, 30, (size, size, 3), dtype=np.uint8)
            img = cv2.addWeighted(img, 0.9, noise, 0.1, 0)

        elif style == "shapes":
            # Random geometric shapes on plain background
            img = np.ones((size, size, 3), dtype=np.uint8) * 255
            num_objs = random.randint(8, 15)
            for _ in range(num_objs):
                shape = random.choice(["rect", "circle", "line"])
                color = tuple(int(c) for c in np.random.randint(0, 255, 3))
                thickness = random.randint(2, 8)
                if shape == "rect":
                    pt1 = (random.randint(0, size - 1), random.randint(0, size - 1))
                    pt2 = (random.randint(0, size - 1), random.randint(0, size - 1))
                    cv2.rectangle(img, pt1, pt2, color, thickness)
                elif shape == "circle":
                    center = (random.randint(0, size - 1), random.randint(0, size - 1))
                    radius = random.randint(20, size // 3)
                    cv2.circle(img, center, radius, color, thickness)
                else:  # line
                    pt1 = (random.randint(0, size - 1), random.randint(0, size - 1))
                    pt2 = (random.randint(0, size - 1), random.randint(0, size - 1))
                    cv2.line(img, pt1, pt2, color, thickness)
            img = cv2.GaussianBlur(img, (7, 7), 0)

        else:  # stripes
            img = np.zeros((size, size, 3), dtype=np.uint8)
            stripe_w = random.randint(20, 60)
            horizontal = random.choice([True, False])
            for i in range(0, size, stripe_w):
                color = tuple(int(c) for c in np.random.randint(0, 255, 3))
                if horizontal:
                    cv2.rectangle(img, (0, i), (size, i + stripe_w), color, -1)
                else:
                    cv2.rectangle(img, (i, 0), (i + stripe_w, size), color, -1)
            img = cv2.GaussianBlur(img, (5, 5), 0)

        return img

    def _remove_random_patch(self):
        """Remove a random rectangular patch and return its coordinates + the patch."""
        h, w = self.base_img.shape[:2]
        pw = random.randint(PATCH_SIZE_MIN, PATCH_SIZE_MAX)
        ph = random.randint(PATCH_SIZE_MIN, PATCH_SIZE_MAX)
        x = random.randint(0, w - pw - 1)
        y = random.randint(0, h - ph - 1)
        patch = self.base_img[y : y + ph, x : x + pw].copy()
        # Blank the area on base image (white border for clarity)
        self.base_img[y : y + ph, x : x + pw] = (255, 255, 255)
        cv2.rectangle(self.base_img, (x, y), (x + pw, y + ph), (0, 0, 0), 2)
        return (x, y, pw, ph), patch

    def _generate_candidates(self):
        """Create 5 candidate pieces (1 correct, 4 incorrect from random regions)."""
        candidates = [self.correct_piece]
        h, w = self.base_img.shape[:2]
        while len(candidates) < NUM_CANDIDATES:
            pw = self.patch_coords[2]
            ph = self.patch_coords[3]
            x = random.randint(0, w - pw - 1)
            y = random.randint(0, h - ph - 1)
            cand = self.base_img[y : y + ph, x : x + pw].copy()
            # Ensure candidate isn't identical to correct piece (simple mean check)
            if not np.allclose(cand.mean(), candidates[0].mean(), atol=1):
                candidates.append(cand)
        random.shuffle(candidates)
        # Determine index of the correct piece using array comparison (list.index cannot be used for numpy arrays)
        for idx, piece in enumerate(candidates):
            if np.array_equal(piece, self.correct_piece):
                self.correct_index = idx
                break
        return candidates


class Game:
    def __init__(self):
        self.puzzles = [Puzzle() for _ in range(NUM_PUZZLES)]
        self.current = 0
        self.start_time = None
        self.finished_time = None
        self.wrong_guesses = 0  # count of incorrect selections

    def run(self):
        cv2.namedWindow("Puzzle Challenge")

        # -------- Start screen --------
        self._show_start_screen()
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord(' '):  # space to start
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return

        # -------- Gameplay starts --------
        self.start_time = time.time()
        while self.current < NUM_PUZZLES:
            puzzle = self.puzzles[self.current]
            correct = self._play_round(puzzle, self.current + 1)
            if correct:
                self.current += 1
        self.finished_time = time.time() - self.start_time
        self.penalty = self.wrong_guesses * 2
        self.total_time = self.finished_time + self.penalty
        self._show_completion_screen()
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == ord(' '):  # space bar to play again
            self.__init__()  # reset game state
            self.run()
        # any other key exits

    def _play_round(self, puzzle: Puzzle, round_no: int):
        message = ""
        while True:
            canvas = self._render(puzzle, message, round_no)
            cv2.imshow("Puzzle Challenge", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord(str(i)) for i in range(1, NUM_CANDIDATES + 1)]:
                choice = key - ord("1")
                if choice == puzzle.correct_index:
                    return True
                else:
                    self.wrong_guesses += 1
                    message = "Wrong piece! Try again."
            elif key == ord("q"):
                exit(0)

    def _render(self, puzzle: Puzzle, feedback: str, round_no: int) -> np.ndarray:
        # Start with blank canvas
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 30  # dark background

        # Resize and place left image
        left_img = cv2.resize(puzzle.base_img, (LEFT_IMG_SIZE, LEFT_IMG_SIZE), interpolation=cv2.INTER_AREA)
        canvas[MARGIN : MARGIN + LEFT_IMG_SIZE, MARGIN : MARGIN + LEFT_IMG_SIZE] = left_img

        # Draw candidate pieces in a horizontal row below the puzzle image
        for idx, patch in enumerate(puzzle.candidates):
            thumb = cv2.resize(patch, (CANDIDATE_SIZE, CANDIDATE_SIZE), interpolation=cv2.INTER_AREA)
            x_off = MARGIN + idx * (CANDIDATE_SIZE + MARGIN)
            y_off = LEFT_IMG_SIZE + 2 * MARGIN
            canvas[y_off : y_off + CANDIDATE_SIZE, x_off : x_off + CANDIDATE_SIZE] = thumb
            # Border & numeric label near top‑left inside the thumbnail
            cv2.rectangle(canvas, (x_off, y_off), (x_off + CANDIDATE_SIZE, y_off + CANDIDATE_SIZE), (255, 255, 255), 2)
            cv2.putText(canvas, str(idx + 1), (x_off + 5, y_off + 20), FONT, 0.7, (0, 255, 255), 2)

        # UI text – round and timer
        elapsed = time.time() - self.start_time if self.start_time else 0
        penalty = self.wrong_guesses * 2
        total = elapsed + penalty
        cv2.putText(canvas, f"Round {round_no}/{NUM_PUZZLES}", (MARGIN, canvas_h - 60), FONT, 0.8, (0, 255, 0), 2)
        cv2.putText(canvas, f"Wrong guesses: {self.wrong_guesses}", (MARGIN, canvas_h - 40), FONT, 0.8, (0, 255, 0), 2)
        cv2.putText(canvas, f"Time: {total:.1f}s (+{penalty}s)", (MARGIN, canvas_h - 10), FONT, 0.8, (0, 255, 0), 2)

        if feedback:
            (text_w, text_h), _ = cv2.getTextSize(feedback, FONT, 0.9, 2)
            cv2.putText(canvas, feedback, (canvas_w // 2 - text_w // 2, 40), FONT, 0.9, (0, 0, 255), 2)
        return canvas

    def _show_completion_screen(self):
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 30
        msg1 = f"Base time: {self.finished_time:.2f}s"
        msg2 = f"Wrong guesses: {self.wrong_guesses}  Penalty: +{self.penalty}s"
        msg3 = f"Final time: {self.total_time:.2f}s"
        msg4 = "Press SPACE to play again or any key to quit"
        # Display messages centred
        (w1, _), _ = cv2.getTextSize(msg1, FONT, 1.0, 2)
        (w2, _), _ = cv2.getTextSize(msg2, FONT, 1.0, 2)
        (w3, _), _ = cv2.getTextSize(msg3, FONT, 1.0, 2)
        (w4, _), _ = cv2.getTextSize(msg4, FONT, 0.8, 2)
        x_center = canvas_w // 2
        y_center = canvas_h // 2
        cv2.putText(canvas, msg1, (x_center - w1 // 2, y_center - 30), FONT, 1.0, (0, 255, 255), 3)
        cv2.putText(canvas, msg2, (x_center - w2 // 2, y_center), FONT, 1.0, (0, 255, 255), 3)
        cv2.putText(canvas, msg3, (x_center - w3 // 2, y_center + 30), FONT, 1.0, (0, 255, 255), 3)
        cv2.putText(canvas, msg4, (x_center - w4 // 2, y_center + 70), FONT, 0.8, (0, 255, 0), 2)
        cv2.imshow("Puzzle Challenge", canvas)

    def _show_start_screen(self):
        """Display welcome / ready screen."""
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 40
        title = "Puzzle Challenge"
        prompt = "Press SPACE to start"
        info = "Pick pieces with keys 1‑5, Q to quit"

        (wt, _), _ = cv2.getTextSize(title, FONT, 1.2, 3)
        (wp, _), _ = cv2.getTextSize(prompt, FONT, 1.0, 2)
        (wi, _), _ = cv2.getTextSize(info, FONT, 0.8, 1)
        x_center = canvas_w // 2
        y_center = canvas_h // 2
        cv2.putText(canvas, title, (x_center - wt // 2, y_center - 40), FONT, 1.2, (0, 255, 255), 3)
        cv2.putText(canvas, prompt, (x_center - wp // 2, y_center + 10), FONT, 1.0, (0, 255, 0), 2)
        cv2.putText(canvas, info, (x_center - wi // 2, y_center + 50), FONT, 0.8, (200, 200, 200), 1)
        cv2.imshow("Puzzle Challenge", canvas)


if __name__ == "__main__":
    Game().run() 