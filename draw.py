"""
AIRTALK – Air Writing Visualizer
---------------------------------
Reads IMU + button data from the Arduino over serial and draws on screen.

Serial format (100 Hz):  rotX*100, rotY*100, linX*100, linY*100, penDown
  rotX = gyro Z  (left/right rotation, rad/s * 100)
  rotY = gyro X  (up/down   rotation, rad/s * 100)
  linX = accel X (left/right linear,  m/s² * 100)
  linY = accel Y (up/down   linear,   m/s² * 100)
  penDown = 1 (button pressed) | 0 (pen lifted)

Controls
--------
  Button (pen)  – hold to draw, release to lift pen
  C             – clear the canvas
  R             – reset cursor to centre
  +/-           – increase / decrease sensitivity
  ESC / Q       – quit
"""

import sys
import glob
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

# Gyro contributes this fraction; linear accel contributes the rest.
GYRO_WEIGHT = 0.85
ACCEL_WEIGHT = 1.0 - GYRO_WEIGHT

# Pixels per (rad/s · second) of movement – tune with +/- keys
DEFAULT_SCALE = 280

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
    print(f"Connecting to {port} @ {BAUD} baud …", flush=True)

    try:
        # timeout=0: readline() returns immediately so we can poll every frame
        # (do not gate on in_waiting — on macOS /dev/cu.* it often stays 0 until a read).
        ser = Serial(port, BAUD, timeout=0)
    except SerialException as e:
        sys.exit(f"Cannot open port: {e}")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("✈ AIRTALK – Air Writing")
    font_big  = pygame.font.SysFont("monospace", 18, bold=True)
    font_hint = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()

    # Persistent drawing surface (survives every frame redraw)
    canvas = pygame.Surface((WIDTH, HEIGHT))
    canvas.fill(BG_COLOR)

    cx, cy = WIDTH // 2, HEIGHT // 2   # cursor position
    px, py = cx, cy                    # previous cursor position
    pen_down = False
    scale = DEFAULT_SCALE
    status_msg = "Waiting for READY or first IMU line …"

    # Do not reset_input_buffer(): if the board already sent READY, flushing would
    # leave us stuck forever with no "READY" substring in new data.
    ready = False
    _serial_print_count = 0

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
                    status_msg = "Cursor reset to centre"
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    scale = min(scale + 20, 600)
                    status_msg = f"Sensitivity: {scale}"
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    scale = max(scale - 20, 40)
                    status_msg = f"Sensitivity: {scale}"

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
                if "READY" in raw:
                    ready = True
                    status_msg = "PEN UP  |  Press button to draw"
                    if PRINT_SERIAL_DATA:
                        print("[serial] READY — IMU CSV lines should follow", flush=True)
                    continue
                # Missed READY (e.g. opened serial after boot) — accept first CSV row.
                parts_probe = [p.strip() for p in raw.split(",")]
                if len(parts_probe) == 5:
                    try:
                        float(parts_probe[0])
                        float(parts_probe[1])
                        float(parts_probe[2])
                        float(parts_probe[3])
                        _ = int(parts_probe[4])
                    except ValueError:
                        continue
                    ready = True
                    status_msg = "PEN UP  |  Press button to draw"
                    if PRINT_SERIAL_DATA:
                        print("[serial] implicit READY (CSV line) — drawing enabled", flush=True)
                else:
                    continue

            parts = [p.strip() for p in raw.split(",")]
            if len(parts) != 5:
                if PRINT_SERIAL_DATA and raw:
                    print(f"[serial skip] bad line ({len(parts)} fields): {raw!r}", flush=True)
                continue
            try:
                rot_x   = float(parts[0]) / 100.0   # gyro Z  rad/s
                rot_y   = float(parts[1]) / 100.0   # gyro X  rad/s
                lin_x   = float(parts[2]) / 100.0   # accel X m/s²
                lin_y   = float(parts[3]) / 100.0   # accel Y m/s²
                pen_down = int(parts[4]) == 1
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
                        f"lin=({lin_x:+.4f},{lin_y:+.4f}) pen={int(pen_down)}",
                        flush=True,
                    )

            # Dead zone on gyro to suppress idle drift
            if abs(rot_x) < DEAD_ZONE:
                rot_x = 0.0
            if abs(rot_y) < DEAD_ZONE:
                rot_y = 0.0

            # Sensor fusion: gyro gives smooth curves, accel helps straight lines
            move_x = (rot_x * GYRO_WEIGHT + lin_x * ACCEL_WEIGHT * 0.05) * scale * DT
            move_y = (rot_y * GYRO_WEIGHT + lin_y * ACCEL_WEIGHT * 0.05) * scale * DT

            px, py = cx, cy
            cx = max(0, min(WIDTH  - 1, cx + move_x))
            cy = max(0, min(HEIGHT - 1, cy - move_y))   # screen Y is flipped

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
            f"C=clear  R=reset  +/-=sensitivity({scale})  ESC=quit",
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
