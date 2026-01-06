"""Unit tests for the scoring logic in microphone_bounce_game.py.

Run with: python clap_bounce/test_scoring.py

This test file includes a copy of the compute_score function to allow
testing without requiring all game dependencies (pygame, cv2, etc.).
"""

import sys


def compute_score(delta_ms: float) -> int:
    """Step-wise scoring based on clap timing proximity.

    The score is symmetric around the impact (pre/post) and awarded as follows
    (|delta_ms| is the absolute difference between the clap and impact time):
        0-5   ms  -> 100 points
        5-10  ms  -> 90 points
        10-20 ms  -> 70 points
        20-30 ms  -> 50 points
        30-50 ms  -> 30 points
        >50 ms    -> 0 points
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


def test_perfect_timing():
    """Test perfect claps (0-5 ms) score 100 points."""
    assert compute_score(0) == 100, "0ms should score 100"
    assert compute_score(2.5) == 100, "2.5ms should score 100"
    assert compute_score(5) == 100, "5ms should score 100"
    assert compute_score(-3) == 100, "Early clap at -3ms should score 100"
    assert compute_score(-5) == 100, "Early clap at -5ms should score 100"


def test_close_timing():
    """Test close claps (5-10 ms) score 90 points."""
    assert compute_score(6) == 90, "6ms should score 90"
    assert compute_score(10) == 90, "10ms should score 90"
    assert compute_score(-7) == 90, "Early clap at -7ms should score 90"


def test_good_timing():
    """Test good claps (10-20 ms) score 70 points."""
    assert compute_score(11) == 70, "11ms should score 70"
    assert compute_score(15) == 70, "15ms should score 70"
    assert compute_score(20) == 70, "20ms should score 70"
    assert compute_score(-15) == 70, "Early clap at -15ms should score 70"


def test_early_late_timing():
    """Test early/late claps (20-30 ms) score 50 points."""
    assert compute_score(21) == 50, "21ms should score 50"
    assert compute_score(25) == 50, "25ms should score 50"
    assert compute_score(30) == 50, "30ms should score 50"
    assert compute_score(-25) == 50, "Early clap at -25ms should score 50"


def test_barely_timing():
    """Test barely-in-time claps (30-50 ms) score 30 points."""
    assert compute_score(31) == 30, "31ms should score 30"
    assert compute_score(40) == 30, "40ms should score 30"
    assert compute_score(50) == 30, "50ms should score 30"
    assert compute_score(-45) == 30, "Early clap at -45ms should score 30"


def test_miss():
    """Test missed claps (>50 ms) score 0 points."""
    assert compute_score(51) == 0, "51ms should score 0"
    assert compute_score(75) == 0, "75ms should score 0"
    assert compute_score(100) == 0, "100ms should score 0"
    assert compute_score(150) == 0, "150ms should score 0"
    assert compute_score(-100) == 0, "Early clap at -100ms should score 0"


def test_boundary_values():
    """Test exact boundary values to ensure correct tier assignment."""
    # Boundary between perfect and close
    assert compute_score(5) == 100, "5ms is perfect (100)"
    assert compute_score(5.001) == 90, "5.001ms is close (90)"

    # Boundary between close and good
    assert compute_score(10) == 90, "10ms is close (90)"
    assert compute_score(10.001) == 70, "10.001ms is good (70)"

    # Boundary between good and early/late
    assert compute_score(20) == 70, "20ms is good (70)"
    assert compute_score(20.001) == 50, "20.001ms is early/late (50)"

    # Boundary between early/late and barely
    assert compute_score(30) == 50, "30ms is early/late (50)"
    assert compute_score(30.001) == 30, "30.001ms is barely (30)"

    # Boundary between barely and miss
    assert compute_score(50) == 30, "50ms is barely (30)"
    assert compute_score(50.001) == 0, "50.001ms is miss (0)"


def test_symmetry():
    """Test that positive and negative deltas yield the same score."""
    test_values = [0, 3, 7, 15, 25, 40, 75]
    for val in test_values:
        assert compute_score(val) == compute_score(-val), \
            f"Score should be symmetric: {val}ms vs {-val}ms"


def run_all_tests():
    """Run all test functions and report results."""
    test_functions = [
        test_perfect_timing,
        test_close_timing,
        test_good_timing,
        test_early_late_timing,
        test_barely_timing,
        test_miss,
        test_boundary_values,
        test_symmetry,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            print(f"  PASS: {test_func.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test_func.__name__} - {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test_func.__name__} - {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*40}")

    return failed == 0


if __name__ == "__main__":
    print("Running scoring logic tests...")
    print("=" * 40)
    success = run_all_tests()
    sys.exit(0 if success else 1)
