# AI Generated Games Collection

This repository contains a collection of simple games generated with the assistance of AI, utilizing computer vision and audio input for novel control schemes.

## General Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url> # Replace with actual URL if applicable
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Some libraries like `sounddevice` (specifically `PortAudio`) and `opencv-python` might have system-level dependencies. Consult their respective documentation for OS-specific installation instructions if you encounter issues.*

## Games Included

Below are the games included in this collection:

---

### 1. Clap Bounce

*   **Folder:** `clap_bounce/`
*   **Description:** A rhythm game where the objective is to clap exactly when a bouncing ball hits any edge of the screen.
*   **How to Run:**
    ```bash
    python clap_bounce/microphone_bounce_game.py
    ```
*   **Gameplay:** Watch the ball bounce. Use your microphone to clap precisely as the ball makes contact with a window border. Your score is based on the timing accuracy of your clap relative to the impact. The game keeps track of the high score and a leaderboard.
*   **Controls:**
    *   **Clap:** Use your microphone.
    *   **Toggle Debug Overlay:** `D` key.
    *   **Quit:** `ESC` key.
*   **Configuration:** Game parameters like sensitivity, scoring windows, and ball speed might be adjustable in `clap_bounce/config.py` (verify usage).
*   **Testing:** Scoring logic can be tested via `python clap_bounce/test_scoring.py`.

---

### 2. Eye Tetris

*   **Folder:** `eye_tetris/`
*   **Description:** A version of the classic Tetris game controlled using eye gaze tracked via your webcam.
*   **How to Run:**
    ```bash
    python eye_tetris/eye_tetris.py
    ```
*   **Gameplay:** Play Tetris by looking at different regions of the game screen. Your gaze direction determines how the falling pieces move and rotate. Requires a webcam.
*   **Controls:**
    *   **Movement/Rotation:** Eye Gaze (look left/right/down/rotate zones/wink one eye or the other to rotate).
    *   **Quit:** Likely `ESC` key or closing the window.

---

### 3. Puzzle Challenge

*   **Folder:** `puzzle_challenge/`
*   **Description:** A sliding puzzle game where the puzzle image is taken from a procedurally generated image.
*   **How to Run:**
    ```bash
    python puzzle_challenge/puzzle_game.py
    ```
*   **Gameplay:** The game creates an image, scrambles it into tiles, leaving one empty space. Click adjacent tiles to slide them into the empty space. The objective is to rearrange the tiles to form the original image.
*   **Controls:**
    *   **Move Tile:** Mouse click on a tile adjacent to the empty space.
    *   **Quit:** `ESC` key.

---

### 4. Webcam Pong

*   **Folder:** `webcam_pong/`
*   **Description:** The classic game of Pong, where the paddles are controlled by tracking your hand movement using a webcam.
*   **How to Run:**
    ```bash
    python webcam_pong/webcam_pong.py
    ```
*   **Gameplay:** Control the vertical position of your paddle (typically the right one) by moving your hand up and down in view of the webcam. Compete against an AI-controlled opponent paddle on the left. Requires a webcam.
*   **Controls:**
    *   **Paddle Movement:** Hand position (vertical).
    *   **Quit:** `ESC` key.

--- 