"""
AIRTALK – Capture session (same drawing as draw.py)
---------------------------------------------------
Before the window opens, you are prompted in the terminal for a word/label.
Captures are saved under captures/<label>/ when you press S as capture_NNNN_<timestamp>.png
(NNNN is a running counter; resumes if files already exist in that folder).

Serial format (100 Hz):
  rotX*100, rotY*100, linX*100, linY*100, penDown, accelBiasZ*100
  rotX, rotY, linX, linY = motion (same scaling as before; m/s² and rad/s * 100)
  penDown = 1 (button pressed) | 0 (pen lifted)
  accelBiasZ = averaged raw accel Z at calibration (m/s² * 100), used on the Python side

Controls
--------
  Button (pen)  – hold to draw, release to lift pen
  C             – clear the canvas
  R             – reset cursor to centre
  +/-           – increase / decrease sensitivity
  ESC / Q       – quit
"""

import sys
import re
import glob
import math
from datetime import datetime, timezone
from pathlib import Path
import pygame


def _load_serial_api():
    """Bind Serial + SerialException from pyserial.

    Handles (1) wrong PyPI package named `serial`, (2) broken installs where
    `serial/__init__.py` is missing so `import serial` is a namespace with no Serial.
    """
    try:
        from serial import Serial as _S, SerialException as _E
        return _S, _E
    except (ImportError, AttributeError):
        pass
    try:
        from serial.serialutil import SerialException as _E

        if sys.platform.startswith("win"):
            from serial.serialwin32 import Serial as _S
        else:
            from serial.serialposix import Serial as _S
        return _S, _E
    except ImportError as exc:
        sys.exit(
            "Could not load pyserial.\n\n"
            "Fix:\n"
            "  pip3 uninstall serial -y\n"
            "  pip3 install --force-reinstall pyserial\n\n"
            f"ImportError: {exc}"
        )


Serial, SerialException = _load_serial_api()

# ── Serial auto-detection ────────────────────────────────────────────────────

def find_port() -> str:
    """Return the first likely Arduino serial port on macOS/Linux/Windows."""
    candidates = (
        glob.glob("/dev/cu.usbmodem*")
        + glob.glob("/dev/cu.usbserial*")
        + glob.glob("/dev/tty.usbmodem*")
        + glob.glob("/dev/tty.usbserial*")
        + glob.glob("/dev/ttyUSB*")
        + glob.glob("/dev/ttyACM*")
        + glob.glob("COM[0-9]*")
    )
    if not candidates:
        raise RuntimeError(
            "No serial port found. Connect your Arduino and try again,\n"
            "or set SERIAL_PORT in draw.py, or run:  python3 draw.py /dev/cu.usbmodemXXX"
        )
    return candidates[0]


# ── Constants ────────────────────────────────────────────────────────────────

# macOS: use /dev/cu.* for apps (not /dev/tty.*). Set to "" to auto-detect instead.
SERIAL_PORT = "/dev/cu.usbmodem1101"

BAUD        = 115200
WIDTH, HEIGHT = 900, 650
BG_COLOR    = (15,  15,  20)
INK_COLOR   = (255, 255, 255)
CURSOR_DOWN = (0,   220, 100)   # green  = drawing
CURSOR_UP   = (200,  50,  50)   # red    = lifted
LINE_WIDTH  = 3
DT          = 0.01              # matches Arduino 10 ms loop
DEAD_ZONE   = 0.03             # rad/s – ignore tiny gyro noise

# Gyro drives rotation-based drawing; accel is integrated into a damped
# velocity so sustained hand translation also moves the cursor.
GYRO_WEIGHT = 1.0

# Accel-velocity integration: lin_{x,y} (m/s²) → velocity (px/s) → position.
# Damping kills drift when the hand is still; gain converts m/s into pixels/s.
VEL_DAMPING       = 0.90     # per sample (~100 Hz). 1.0 = no damping, 0 = off.
ACCEL_GAIN        = 220.0    # pixels per (m/s). Tune with [ and ] keys.
ACCEL_DEAD_ZONE   = 0.15     # m/s² – ignore tiny accel noise at rest
# High-pass filter on lin_{x,y}: a slow low-pass tracks the *current* gravity
# leakage in the body frame (which changes whenever the pen is tilted away
# from the calibration pose), and we subtract it.  Only the fast transients
# from real hand translation survive and get integrated into velocity.
#   α near 1 = slow adaption (big time constant, τ ≈ DT / (1-α))
#   α = 0.985 @ 100Hz → τ ≈ 0.67 s
GRAVITY           = 9.81    # m/s² — normalises accel relative to calibration
GRAV_LPF_ALPHA    = 0.985
# While the pen is rotating, gravity leakage is changing rapidly.  Snap the
# low-pass to the current sample so the high-pass output stays ~0 and no
# phantom velocity builds up; also bleed any existing velocity.
ROT_GATE          = 0.25     # rad/s – above this we consider the pen "rotating"
ROT_GATE_DAMPING  = 0.70     # velocity decay while rotating

CAPTURES_ROOT = Path(__file__).resolve().parent / "captures"


def _sanitize_session_folder(name: str) -> str:
    """Make a single path segment safe on common filesystems."""
    s = name.strip()
    if not s:
        return "untitled"
    bad = '<>:"/\\|?*\x00'
    out = []
    for ch in s:
        if ch in bad or ord(ch) < 32:
            out.append("_")
        else:
            out.append(ch)
    folder = "".join(out).strip(" .")
    return folder or "untitled"


_CAPTURE_NUM = re.compile(r"^capture_(\d{4})_")


def _next_capture_index(captures_dir: Path) -> int:
    """Next 4-digit capture index (1-based), continuing from existing capture_NNNN_*.png files."""
    best = 0
    for p in captures_dir.glob("capture_*.png"):
        m = _CAPTURE_NUM.match(p.stem)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


# Pixels per (rad/s · second) of movement – tune with +/- keys
DEFAULT_SCALE = 280

# Independent per-axis sensitivity multipliers.  X defaults to 1.0 (no
# extra scaling); Y compensates for screen aspect ratio and any physical
# axis difference.  Raise X if left/right strokes feel too short.
X_SENSITIVITY = 1.5    # tune independently of Y
Y_SENSITIVITY = 1.38   # ≈ WIDTH / HEIGHT (900/650) as a starting point

# Echo serial to the terminal (watch for READY, then IMU lines). 1 = every sample (~100/s).
PRINT_SERIAL_DATA = True
SERIAL_PRINT_EVERY = 10


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Help prints show up when launched from an IDE (not a real TTY).
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    # CLI arg overrides SERIAL_PORT; if SERIAL_PORT is "", auto-detect.
    if len(sys.argv) > 1:
        port = sys.argv[1]
    elif SERIAL_PORT:
        port = SERIAL_PORT
    else:
        port = find_port()

    try:
        label = input("Enter word / label for this capture folder: ").strip()
    except EOFError:
        label = ""
    folder = _sanitize_session_folder(label)
    session_captures_dir = CAPTURES_ROOT / folder
    session_captures_dir.mkdir(parents=True, exist_ok=True)
    capture_counter = _next_capture_index(session_captures_dir)
    print(f"Captures will save to: {session_captures_dir}", flush=True)
    print(f"Next capture number: {capture_counter:04d}", flush=True)

    print(f"Connecting to {port} @ {BAUD} baud …", flush=True)

    try:
        # timeout=0: readline() returns immediately so we can poll every frame
        # (do not gate on in_waiting — on macOS /dev/cu.* it often stays 0 until a read).
        ser = Serial(port, BAUD, timeout=0)
    except SerialException as e:
        sys.exit(f"Cannot open port: {e}")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"✈ AIRTALK – Capture: {folder}")
    font_big  = pygame.font.SysFont("monospace", 18, bold=True)
    font_hint = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()

    # Persistent drawing surface (survives every frame redraw)
    canvas = pygame.Surface((WIDTH, HEIGHT))
    canvas.fill(BG_COLOR)

    cx, cy = WIDTH // 2, HEIGHT // 2   # cursor position
    px, py = cx, cy                    # previous cursor position
    vx, vy = 0.0, 0.0                  # accel-integrated velocity (px/s)
    grav_lp_x, grav_lp_y = 0.0, 0.0    # low-pass est. of gravity leakage into lin
    pen_down = False
    prev_pen_down = False
    scale = DEFAULT_SCALE
    accel_gain = ACCEL_GAIN
    status_msg = "Waiting for READY or first IMU line …"

    # Do not reset_input_buffer(): if the board already sent READY, flushing would
    # leave us stuck forever with no "READY" substring in new data.
    ready = False
    _serial_print_count = 0
    # Updated from serial when the 6th field is present (~gravity at rest during cal).
    accel_bias_z = 9.81

    running = True
    while running:
        # ── Events ──────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_c:
                    canvas.fill(BG_COLOR)
                    status_msg = "Canvas cleared"
                elif event.key == pygame.K_r:
                    cx, cy = WIDTH // 2, HEIGHT // 2
                    px, py = cx, cy
                    vx, vy = 0.0, 0.0
                    grav_lp_x, grav_lp_y = 0.0, 0.0
                    ready = False
                    ser.write(b'R')
                    status_msg = "Calibrating… hold pen still and flat!"
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    scale = min(scale + 20, 600)
                    status_msg = f"Sensitivity: {scale}"
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    scale = max(scale - 20, 40)
                    status_msg = f"Sensitivity: {scale}"
                elif event.key == pygame.K_RIGHTBRACKET:
                    accel_gain = min(accel_gain + 20, 1000)
                    status_msg = f"Translation gain: {accel_gain:.0f}"
                elif event.key == pygame.K_LEFTBRACKET:
                    accel_gain = max(accel_gain - 20, 0)
                    status_msg = f"Translation gain: {accel_gain:.0f}"
                elif event.key == pygame.K_s:
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S%fZ")[:-3]
                    n = capture_counter
                    save_path = session_captures_dir / f"capture_{n:04d}_{stamp}.png"
                    pygame.image.save(canvas, str(save_path))
                    capture_counter += 1
                    canvas.fill(BG_COLOR)
                    rel = save_path.relative_to(CAPTURES_ROOT.parent)
                    status_msg = f"Saved #{n:04d} + cleared: {rel}"
                    print(f"[capture] #{n:04d} → {save_path}", flush=True)

        # ── Read serial ──────────────────────────────────────────────────────
        # Drain every full line already in the OS buffer (non-blocking).
        while True:
            try:
                raw_bytes = ser.readline()
            except Exception:
                break
            if not raw_bytes:
                break
            try:
                raw = raw_bytes.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue

            if not ready:
                if raw and PRINT_SERIAL_DATA:
                    print(f"[serial] {raw}", flush=True)
                if "CALIBRATING" in raw:
                    status_msg = "Calibrating… hold pen still and flat!"
                    continue
                if "READY" in raw:
                    ready = True
                    status_msg = "PEN UP  |  Press button to draw"
                    if PRINT_SERIAL_DATA:
                        print("[serial] READY — IMU CSV lines should follow", flush=True)
                    continue
                # Missed READY (e.g. opened serial after boot) — accept first CSV row.
                parts_probe = [p.strip() for p in raw.split(",")]
                if len(parts_probe) in (5, 6):
                    try:
                        float(parts_probe[0])
                        float(parts_probe[1])
                        float(parts_probe[2])
                        float(parts_probe[3])
                        _ = int(parts_probe[4])
                        if len(parts_probe) == 6:
                            float(parts_probe[5])
                    except ValueError:
                        continue
                    ready = True
                    status_msg = "PEN UP  |  Press button to draw"
                    if PRINT_SERIAL_DATA:
                        print("[serial] implicit READY (CSV line) — drawing enabled", flush=True)
                else:
                    continue

            parts = [p.strip() for p in raw.split(",")]
            if len(parts) not in (5, 6):
                if PRINT_SERIAL_DATA and raw:
                    print(f"[serial skip] bad line ({len(parts)} fields): {raw!r}", flush=True)
                continue
            try:
                s = 100.0  # Arduino sends rad/s and m/s² scaled by 100
                rot_x = -float(parts[0]) / s
                rot_y = -float(parts[1]) / s
                lin_x = float(parts[2]) / 100.0
                lin_y = float(parts[3]) / 100.0
                pen_down = int(parts[4]) == 1
                if pen_down and not prev_pen_down:
                    canvas.fill(BG_COLOR)
                    cx, cy = WIDTH // 2, HEIGHT // 2
                    px, py = cx, cy
                    vx, vy = 0.0, 0.0
                    grav_lp_x, grav_lp_y = 0.0, 0.0
                prev_pen_down = pen_down
                if len(parts) == 6:
                    accel_bias_z = float(parts[5]) / s
            except ValueError:
                if PRINT_SERIAL_DATA and raw:
                    print(f"[serial skip] parse error: {raw!r}", flush=True)
                continue

            if PRINT_SERIAL_DATA:
                _serial_print_count += 1
                every = max(1, SERIAL_PRINT_EVERY)
                if _serial_print_count % every == 0:
                    print(
                        f"rot=({rot_x:+.4f},{rot_y:+.4f}) "
                        f"lin=({lin_x:+.4f},{lin_y:+.4f}) "
                        f"pen={int(pen_down)} bias_z={accel_bias_z:+.3f}",
                        flush=True,
                    )

            # Dead zone on gyro to suppress idle drift
            if abs(rot_x) < DEAD_ZONE:
                rot_x = 0.0
            if abs(rot_y) < DEAD_ZONE:
                rot_y = 0.0

            # Rotation normalization: prevent diagonal moves from compounding to
            # sqrt(2)x the speed of a single-axis move.  Scale the vector so its
            # magnitude equals the dominant (peak) component — pure horizontal or
            # vertical strokes are unchanged; a 45° diagonal is reduced by 1/sqrt(2).
            rot_mag = math.hypot(rot_x, rot_y)
            if rot_mag > 1e-6:
                peak = max(abs(rot_x), abs(rot_y))
                rot_x = rot_x * peak / rot_mag
                rot_y = rot_y * peak / rot_mag

            # Use calibration Z bias (~|g| when pen was flat) to normalize accel contribution.
            grav_ref = max(abs(accel_bias_z), 0.5)
            accel_norm = GRAVITY / grav_ref

            # Normalise accel scaling relative to calibration gravity.
            raw_ax = lin_x * accel_norm
            raw_ay = lin_y * accel_norm

            # --- Gravity-leakage estimator (complementary / high-pass) --------
            # lin_{x,y} is NOT truly linear accel once the pen is tilted; it
            # contains a slowly-varying DC term = (R(θ) - R(0)) g .  Track that
            # DC with a low-pass filter and subtract it to recover real motion.
            rotating = max(abs(rot_x), abs(rot_y)) > ROT_GATE
            if rotating:
                # Pose is changing fast → snap the low-pass to "now" so the
                # high-pass output is ~0 and no phantom velocity is produced.
                grav_lp_x = raw_ax
                grav_lp_y = raw_ay
            else:
                grav_lp_x = GRAV_LPF_ALPHA * grav_lp_x + (1 - GRAV_LPF_ALPHA) * raw_ax
                grav_lp_y = GRAV_LPF_ALPHA * grav_lp_y + (1 - GRAV_LPF_ALPHA) * raw_ay

            ax = raw_ax - grav_lp_x   # true linear accel in body X
            ay = raw_ay - grav_lp_y   # true linear accel in body Y

            # Dead-zone the residual so idle noise doesn't integrate into drift.
            if abs(ax) < ACCEL_DEAD_ZONE: ax = 0.0
            if abs(ay) < ACCEL_DEAD_ZONE: ay = 0.0

            if rotating:
                # While rotating, don't add new velocity — let existing bleed off.
                vx *= ROT_GATE_DAMPING
                vy *= ROT_GATE_DAMPING
            else:
                # a (m/s²) * DT (s) = Δv (m/s); scale by ACCEL_GAIN (px per m/s).
                vx = vx * VEL_DAMPING + ax * DT * accel_gain
                vy = vy * VEL_DAMPING + ay * DT * accel_gain

            # Snap to zero when the hand is essentially still.
            if ax == 0.0 and abs(vx) < 0.5: vx = 0.0
            if ay == 0.0 and abs(vy) < 0.5: vy = 0.0

            # Gyro rotation drives curves; integrated velocity drives translation.
            move_x = (rot_x * GYRO_WEIGHT * scale * DT + vx * DT) * X_SENSITIVITY
            move_y = (rot_y * GYRO_WEIGHT * scale * DT + vy * DT) * Y_SENSITIVITY

            px, py = cx, cy
            cx = max(0, min(WIDTH  - 1, cx + move_x))
            cy = max(0, min(HEIGHT - 1, cy + move_y))   # screen Y is flipped

            if pen_down and ready:
                pygame.draw.line(canvas, INK_COLOR,
                                 (int(px), int(py)), (int(cx), int(cy)),
                                 LINE_WIDTH)
                status_msg = "PEN DOWN – drawing"
            else:
                if ready:
                    status_msg = "PEN UP  |  Press button to draw"

        # ── Render ───────────────────────────────────────────────────────────
        screen.blit(canvas, (0, 0))

        # Cursor dot
        cursor_color = CURSOR_DOWN if pen_down else CURSOR_UP
        pygame.draw.circle(screen, cursor_color, (int(cx), int(cy)), 8)
        pygame.draw.circle(screen, (255, 255, 255), (int(cx), int(cy)), 8, 1)

        # HUD bar at top
        pygame.draw.rect(screen, (30, 30, 40), (0, 0, WIDTH, 36))
        hud = font_big.render(status_msg, True, (220, 220, 220))
        screen.blit(hud, (10, 8))
        hints = font_hint.render(
            f"C=clear  R=reset  S=save #{capture_counter:04d}→{folder}/  +/-=rot({scale})  [/]=trans({accel_gain:.0f})  ESC=quit",
            True, (120, 120, 140)
        )
        screen.blit(hints, (WIDTH - hints.get_width() - 10, 10))

        pygame.display.flip()
        clock.tick(120)

    ser.close()
    pygame.quit()
    print("Bye!")


if __name__ == "__main__":
    main()
