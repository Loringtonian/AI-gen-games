# Configuration for Precision Clap Timing Game

# Audio Settings
CLAP_THRESHOLD = 0.5  # Minimum peak amplitude to register as a clap (adjust based on mic sensitivity)
AUDIO_CHUNK_SIZE = 1024 # Samples per buffer
AUDIO_FORMAT = 'paInt16' # Sample format
AUDIO_CHANNELS = 1      # Mono audio
AUDIO_RATE = 44100    # Samples per second (standard audio CD quality)
PEAK_WINDOW_MS = 50   # Time window (+/- ms) around a peak to ignore sub-peaks
SCORING_WINDOW_MS = 50 # Time window (+/- ms) around impact to score a clap

# Scoring Breakpoints (absolute delta time in milliseconds)
# Format: {max_ms: (points, feedback_message)}
SCORING_TIERS = {
    5: (100, "Perfect Clap"),
    10: (90, "Close"),
    20: (70, "Good"),
    30: (50, "Early/Late"),
    SCORING_WINDOW_MS: (30, "Barely"),
    float('inf'): (0, "Missed Clap") # Catch-all for > SCORING_WINDOW_MS or no clap
}

# UI Settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
BACKGROUND_COLOR = (0, 0, 0)
BALL_COLOR = (255, 255, 255)
TEXT_COLOR = (200, 200, 200)
FEEDBACK_COLOR = (255, 255, 0) # Yellow for feedback
DEBUG_COLOR = (0, 255, 0)     # Green for debug text

FONT_SIZE_NORMAL = 36
FONT_SIZE_SMALL = 24
FONT_SIZE_FEEDBACK = 48

HUD_Y_POS = 10             # Y position for score/main HUD elements
FEEDBACK_Y_POS = 50        # Y position for per-clap feedback
FEEDBACK_DURATION = 1.0    # How long feedback text stays on screen (seconds)
DEBUG_Y_START = SCREEN_HEIGHT - 100 # Starting Y for debug overlay lines

# Gameplay Settings
BALL_RADIUS = 15
BALL_SPEED_X = 300  # pixels per second
BALL_SPEED_Y = 250  # pixels per second 