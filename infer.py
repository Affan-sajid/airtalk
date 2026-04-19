"""
AIRTALK – Air Writing + live inference
--------------------------------------
Same serial + pygame loop as draw.py. When you release the pen button, the
current canvas is preprocessed (bbox crop → 64×64) and classified with
model_best.pt. Predictions print to the terminal and accumulate in a session list.

After GROQ_IDLE_SECONDS with the pen up and no new word, accumulated words are
sent to Groq (Llama) to form one natural sentence; that sentence is printed and
spoken via Groq TTS (Orpheus). Set `GROQ_API_KEY` in a `.env` file next to this
script, or export `GROQ_API_KEY` in the environment.

Run:  python3 infer.py  [optional_serial_port]

Controls match draw.py (C / R / S / +/- / brackets / ESC).
"""

from __future__ import annotations

import glob
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pygame
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image


def _load_serial_api():
    """Bind Serial + SerialException from pyserial."""
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


def find_port() -> str:
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
            "or set SERIAL_PORT in infer.py, or run:  python3 infer.py /dev/cu.usbmodemXXX"
        )
    return candidates[0]


# ── Serial / UI (same as draw.py) ────────────────────────────────────────────

SERIAL_PORT = "/dev/cu.usbmodem2101"
BAUD = 115200
WIDTH, HEIGHT = 900, 650
BG_COLOR = (15, 15, 20)
INK_COLOR = (255, 255, 255)
CURSOR_DOWN = (0, 220, 100)
CURSOR_UP = (200, 50, 50)
LINE_WIDTH = 3
DT = 0.01
DEAD_ZONE = 0.03
GYRO_WEIGHT = 1.0
VEL_DAMPING = 0.90
ACCEL_GAIN = 220.0
ACCEL_DEAD_ZONE = 0.15
GRAVITY = 9.81
GRAV_LPF_ALPHA = 0.985
ROT_GATE = 0.25
ROT_GATE_DAMPING = 0.70

CAPTURES_DIR = Path(__file__).resolve().parent / "captures"
DEFAULT_SCALE = 280
X_SENSITIVITY = 1.5
Y_SENSITIVITY = 1.38
PRINT_SERIAL_DATA = False
SERIAL_PRINT_EVERY = 10

# ── Groq LLM + TTS (after pause with pen up) ────────────────────────────────
# Seconds after the last CNN word (pen release) before we flush to Groq + TTS.
GROQ_IDLE_SECONDS = 2.0
GROQ_LLM_MODEL = "openai/gpt-oss-20b"
GROQ_TTS_MODEL = "canopylabs/orpheus-v1-english"
GROQ_TTS_VOICE = "troy"
# Optional override (prefer `.env` or the environment — do not commit real keys).
GROQ_API_KEY_ENV = ("GROQ_API_KEY",)

_env_file_loaded = False

def _load_project_env_file() -> None:
    """Load KEY=VALUE from `.env` in this directory; does not override existing os.environ."""
    global _env_file_loaded
    if _env_file_loaded:
        return
    _env_file_loaded = True
    path = Path(__file__).resolve().parent / ".env"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ[key] = val

# ── Model / preprocessing (matches train.ipynb) ─────────────────────────────

MODEL_PATH = Path(__file__).resolve().parent / "model_best.pt"
IMG_SIZE = 64
PADDING = 12
THRESHOLD = 30

CLASSES = sorted([p.name for p in CAPTURES_DIR.iterdir() if p.is_dir()])
NUM_CLASSES = len(CLASSES)

_infer_model = None
_infer_device: str | None = None

_infer_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
    T.Lambda(lambda x: x.repeat(3, 1, 1)),
])


def crop_to_strokes(
    img: Image.Image, threshold: int = THRESHOLD, padding: int = PADDING
) -> Image.Image:
    gray = img.convert("L")
    arr = np.array(gray)
    ys, xs = np.where(arr > threshold)
    if len(xs) == 0:
        return gray
    x0 = max(int(xs.min()) - padding, 0)
    y0 = max(int(ys.min()) - padding, 0)
    x1 = min(int(xs.max()) + padding, arr.shape[1] - 1)
    y1 = min(int(ys.max()) + padding, arr.shape[0] - 1)
    return gray.crop((x0, y0, x1, y1))


def preprocess(img: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    cropped = crop_to_strokes(img)
    return cropped.resize((size, size), Image.LANCZOS)


def build_model(num_classes: int) -> nn.Module:
    model = torchvision.models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def _load_infer_model() -> None:
    global _infer_model, _infer_device
    if _infer_model is not None:
        return
    if NUM_CLASSES == 0:
        sys.exit(
            f"No class folders under {CAPTURES_DIR}. "
            "Need subdirs (one per word) to know NUM_CLASSES for the checkpoint."
        )
    if not MODEL_PATH.is_file():
        sys.exit(f"Missing weights: {MODEL_PATH}")
    _infer_device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    _infer_model = build_model(NUM_CLASSES)
    _infer_model.load_state_dict(torch.load(MODEL_PATH, map_location=_infer_device))
    _infer_model.to(_infer_device)
    _infer_model.eval()
    print(
        f"[infer] Loaded {MODEL_PATH.name} on {_infer_device}  "
        f"({NUM_CLASSES} classes: {CLASSES})",
        flush=True,
    )


def predict_pil(img: Image.Image) -> tuple[str, float]:
    _load_infer_model()
    assert _infer_model is not None and _infer_device is not None
    processed = preprocess(img)
    tensor = _infer_transform(processed).unsqueeze(0).to(_infer_device)
    with torch.no_grad():
        logits = _infer_model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    idx = probs.argmax().item()
    confidence = probs[idx].item()
    return CLASSES[idx], confidence


def predict_canvas(canvas_surface: pygame.Surface) -> tuple[str, float]:
    # pygame 2+: tobytes; older: tostring
    raw = pygame.image.tobytes(canvas_surface, "RGB")
    w, h = canvas_surface.get_size()
    pil = Image.frombytes("RGB", (w, h), raw)
    return predict_pil(pil)


def _groq_api_key() -> str | None:
    _load_project_env_file()
    file_key = file_key = os.environ.get("GROQ_API_KEY_FILE", "").strip()
    if file_key:
        return file_key
    for name in GROQ_API_KEY_ENV:
        v = os.environ.get(name, "").strip()
        if v:
            return v
    return None


def naturalize_with_groq(words: list[str]) -> str | None:
    """Turn ordered CNN tokens into one natural sentence. Returns None on failure."""
    key = _groq_api_key()
    if not key:
        print(
            "[groq] No API key — add GROQ_API_KEY to `.env`, export "
            "GROQ_API_KEY, or set GROQ_API_KEY in infer.py. Skipping.",
            flush=True,
        )
        return None
    if not words:
        return None
    try:
        from groq import Groq
    except ImportError:
        print("[groq] Install: pip install groq", flush=True)
        return None
    client = Groq(api_key=key)
    joined = " ".join(words)
    prompt = f"""you are reconstructing a single intended sentence from a sequence of words detected from air-written input.

input word sequence:
{joined}

objective:
generate exactly one coherent, natural-sounding english sentence that best represents the writer’s intended meaning.

hard constraints:

output only one sentence. no explanations, no quotes, no meta text.
sentence must be in first person where applicable using “i / me / my”.
all input words must be used exactly once, unless grammatical normalization is required (e.g. tense, capitalization).
you may insert minimal functional words only (articles, auxiliary verbs, prepositions) strictly to ensure grammatical correctness. do not add new semantic content.

semantic normalization rules:

if input expresses emotions or physical states (e.g. tired, happy, hungry, sick): convert to
→ “i am feeling [word]” or “i feel [word]”
if input resembles an introduction (e.g. name present): convert to
→ “hi, my name is [name].” or equivalent natural variant
if input is structurally incomplete or ambiguous, infer the most probable natural intent while preserving all words

formatting rules:

capitalize first letter of the sentence
end with appropriate punctuation (. ! ?) based on tone

few-shot calibration:
hi name affan → hi, my name is affan.
i tired → i am feeling tired.
my name affan → my name is affan.
hi affan → hi, i’m affan!

final instruction:
produce the sentence now."""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that forms natural sentences from air-written words."},
                {"role": "user", "content": prompt},
            ],
            model=GROQ_LLM_MODEL,
            temperature=0.5,
            max_completion_tokens=128,
        )
        text = (chat_completion.choices[0].message.content or "").strip()
        if len(text) >= 2 and text[0] in ('"', "'") and text[-1] in ('"', "'"):
            text = text[1:-1].strip()
        return text or None
    except Exception as exc:
        print(f"[groq] Error: {exc}", flush=True)
        return None


def speak_sentence(text: str) -> None:
    text = text.strip()
    if not text:
        return
    key = _groq_api_key()
    if key:
        try:
            from groq import Groq
            client = Groq(api_key=key)
            response = client.audio.speech.create(
                model=GROQ_TTS_MODEL,
                voice=GROQ_TTS_VOICE,
                input=text,
                response_format="wav",
            )
            tmp_path = Path(__file__).resolve().parent / ".groq_tts_tmp.wav"
            response.write_to_file(str(tmp_path))
            if sys.platform == "darwin":
                subprocess.run(["afplay", str(tmp_path)], check=False)
            else:
                subprocess.run(["aplay", str(tmp_path)], check=False)
            try:
                tmp_path.unlink()
            except OSError:
                pass
            return
        except Exception as exc:
            print(f"[groq-tts] Error: {exc}", flush=True)
    if sys.platform == "darwin":
        subprocess.run(["say", text], check=False)
        return
    try:
        import pyttsx3

        eng = pyttsx3.init()
        eng.say(text)
        eng.runAndWait()
    except Exception as exc:
        print(f"[tts] No speech (macOS has `say`; else install pyttsx3): {exc}", flush=True)


def flush_phrase_to_groq_and_tts(words: list[str]) -> str:
    """
    Naturalize words into a sentence, speak it, and return the sentence string.
    Falls back to raw joined words when Groq is unavailable.
    """
    if not words:
        return ""
    final = naturalize_with_groq(words)
    if final:
        print(f"\n>>> FINAL: {final}\n", flush=True)
        speak_sentence(final)
        return final
    else:
        joined = " ".join(words)
        print(f"\n>>> FINAL (raw): {joined}\n", flush=True)
        speak_sentence(joined)
        return joined


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    if len(sys.argv) > 1:
        port = sys.argv[1]
    elif SERIAL_PORT:
        port = SERIAL_PORT
    else:
        port = find_port()
    print(f"Connecting to {port} @ {BAUD} baud …", flush=True)

    try:
        ser = Serial(port, BAUD, timeout=0)
    except SerialException as e:
        sys.exit(f"Cannot open port: {e}")

    _load_infer_model()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AIRTALK – Inference")

    class _DummyFont:
        @staticmethod
        def render(text, antialias, color):
            surf = pygame.Surface((1, 1), pygame.SRCALPHA)
            return surf

    try:
        font_big = pygame.font.SysFont("monospace", 18, bold=True)
        font_hint = pygame.font.SysFont("monospace", 13)
        font_sentence = pygame.font.SysFont("monospace", 32, bold=True)
        font_words = pygame.font.SysFont("monospace", 28, bold=True)
    except (NotImplementedError, ImportError):
        print("[warn] pygame.font unavailable; HUD text disabled", flush=True)
        font_big = _DummyFont()
        font_hint = _DummyFont()
        font_sentence = _DummyFont()
        font_words = _DummyFont()

    clock = pygame.time.Clock()

    canvas = pygame.Surface((WIDTH, HEIGHT))
    canvas.fill(BG_COLOR)

    cx, cy = WIDTH // 2, HEIGHT // 2
    px, py = cx, cy
    vx, vy = 0.0, 0.0
    grav_lp_x, grav_lp_y = 0.0, 0.0
    pen_down = False
    prev_pen_down = False
    scale = DEFAULT_SCALE
    accel_gain = ACCEL_GAIN
    status_msg = "Waiting for READY or first IMU line …"
    predictions: list[str] = []
    pending_groq_words: list[str] = []
    last_cnn_word_mono: float | None = None
    last_sentence: str = ""          # last Groq-generated sentence for on-screen display
    sentence_display_until: float = 0.0   # monotonic time until we keep showing the sentence

    ready = False
    _serial_print_count = 0
    accel_bias_z = 9.81

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_c:
                    canvas.fill(BG_COLOR)
                    predictions.clear()
                    pending_groq_words.clear()
                    last_sentence = ""
                    sentence_display_until = 0.0
                    status_msg = "Canvas cleared"
                elif event.key == pygame.K_r:
                    cx, cy = WIDTH // 2, HEIGHT // 2
                    px, py = cx, cy
                    vx, vy = 0.0, 0.0
                    grav_lp_x, grav_lp_y = 0.0, 0.0
                    ready = False
                    ser.write(b"R")
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
                    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S%fZ")[:-3]
                    save_path = CAPTURES_DIR / f"capture_{stamp}.png"
                    pygame.image.save(canvas, str(save_path))
                    canvas.fill(BG_COLOR)
                    status_msg = f"Saved + cleared: captures/capture_{stamp}.png"
                    print(f"[capture] Saved to {save_path}", flush=True)

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
                s = 100.0
                rot_x = -float(parts[0]) / s
                rot_y = -float(parts[1]) / s
                lin_x = float(parts[2]) / 100.0
                lin_y = float(parts[3]) / 100.0
                pen_down = int(parts[4]) == 1
                inferred_this_line = False
                if pen_down and not prev_pen_down:
                    canvas.fill(BG_COLOR)
                    cx, cy = WIDTH // 2, HEIGHT // 2
                    px, py = cx, cy
                    vx, vy = 0.0, 0.0
                    grav_lp_x, grav_lp_y = 0.0, 0.0
                    status_msg = "PEN DOWN – drawing"
                if not pen_down and prev_pen_down and ready:
                    label, conf = predict_canvas(canvas)
                    predictions.append(label)
                    pending_groq_words.append(label)
                    last_cnn_word_mono = time.monotonic()
                    print(f"\n>>> PREDICTION: '{label}'  ({conf:.1%} confidence)", flush=True)
                    print(f"    Session so far: {predictions}\n", flush=True)
                    status_msg = f"Last: '{label}' ({conf:.0%})  |  n={len(predictions)}"
                    inferred_this_line = True
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

            if abs(rot_x) < DEAD_ZONE:
                rot_x = 0.0
            if abs(rot_y) < DEAD_ZONE:
                rot_y = 0.0

            rot_mag = math.hypot(rot_x, rot_y)
            if rot_mag > 1e-6:
                peak = max(abs(rot_x), abs(rot_y))
                rot_x = rot_x * peak / rot_mag
                rot_y = rot_y * peak / rot_mag

            grav_ref = max(abs(accel_bias_z), 0.5)
            accel_norm = GRAVITY / grav_ref
            raw_ax = lin_x * accel_norm
            raw_ay = lin_y * accel_norm

            rotating = max(abs(rot_x), abs(rot_y)) > ROT_GATE
            if rotating:
                grav_lp_x = raw_ax
                grav_lp_y = raw_ay
            else:
                grav_lp_x = GRAV_LPF_ALPHA * grav_lp_x + (1 - GRAV_LPF_ALPHA) * raw_ax
                grav_lp_y = GRAV_LPF_ALPHA * grav_lp_y + (1 - GRAV_LPF_ALPHA) * raw_ay

            ax = raw_ax - grav_lp_x
            ay = raw_ay - grav_lp_y

            if abs(ax) < ACCEL_DEAD_ZONE:
                ax = 0.0
            if abs(ay) < ACCEL_DEAD_ZONE:
                ay = 0.0

            if rotating:
                vx *= ROT_GATE_DAMPING
                vy *= ROT_GATE_DAMPING
            else:
                vx = vx * VEL_DAMPING + ax * DT * accel_gain
                vy = vy * VEL_DAMPING + ay * DT * accel_gain

            if ax == 0.0 and abs(vx) < 0.5:
                vx = 0.0
            if ay == 0.0 and abs(vy) < 0.5:
                vy = 0.0

            move_x = (rot_x * GYRO_WEIGHT * scale * DT + vx * DT) * X_SENSITIVITY
            move_y = (rot_y * GYRO_WEIGHT * scale * DT + vy * DT) * Y_SENSITIVITY

            px, py = cx, cy
            cx = max(0, min(WIDTH - 1, cx + move_x))
            cy = max(0, min(HEIGHT - 1, cy + move_y))

            if pen_down and ready:
                pygame.draw.line(
                    canvas, INK_COLOR, (int(px), int(py)), (int(cx), int(cy)), LINE_WIDTH
                )
                status_msg = "PEN DOWN – drawing"
            else:
                if ready and not inferred_this_line:
                    status_msg = "PEN UP  |  Press button to draw"

        idle_sec = max(0.1, float(GROQ_IDLE_SECONDS))
        if pending_groq_words and not pen_down and last_cnn_word_mono is not None:
            elapsed = time.monotonic() - last_cnn_word_mono
            if elapsed >= idle_sec:
                sentence = flush_phrase_to_groq_and_tts(pending_groq_words[:])
                pending_groq_words.clear()
                last_cnn_word_mono = None
                if sentence:
                    last_sentence = sentence
                    sentence_display_until = time.monotonic() + 8.0  # show for 8 s
                    status_msg = f'"{sentence}"'
                else:
                    status_msg = "Phrase finalized — keep writing"
            else:
                left = idle_sec - elapsed
                words_preview = " ".join(pending_groq_words)
                status_msg = f"[{words_preview}] → sentence in {left:.1f}s…"

        screen.blit(canvas, (0, 0))

        now = time.monotonic()

        # ── Pending words strip — always visible at bottom ──────────────────
        pygame.draw.rect(screen, (20, 20, 30), (0, HEIGHT - 54, WIDTH, 54))
        if pending_groq_words:
            words_text = "  ".join(pending_groq_words)
            wsurf = font_words.render(words_text, True, (100, 200, 255))
            screen.blit(wsurf, (WIDTH // 2 - wsurf.get_width() // 2, HEIGHT - 48))
        elif predictions:
            wsurf = font_words.render(" ".join(predictions[-6:]), True, (80, 80, 100))
            screen.blit(wsurf, (WIDTH // 2 - wsurf.get_width() // 2, HEIGHT - 48))

        # ── Sentence overlay — show for 10 s after Groq returns ────────────
        if last_sentence and now < sentence_display_until:
            overlay = pygame.Surface((WIDTH, 120), pygame.SRCALPHA)
            overlay.fill((10, 10, 18, 230))
            screen.blit(overlay, (0, HEIGHT // 2 - 60))
            max_chars = max(1, WIDTH // 18)
            words_list = last_sentence.split()
            lines, line = [], ""
            for w in words_list:
                if len(line) + len(w) + 1 <= max_chars:
                    line = (line + " " + w).lstrip()
                else:
                    if line:
                        lines.append(line)
                    line = w
            if line:
                lines.append(line)
            y_start = HEIGHT // 2 - (len(lines) * 36) // 2
            for i, ln in enumerate(lines):
                surf = font_sentence.render(ln, True, (80, 240, 160))
                screen.blit(surf, (WIDTH // 2 - surf.get_width() // 2, y_start + i * 38))

        cursor_color = CURSOR_DOWN if pen_down else CURSOR_UP
        pygame.draw.circle(screen, cursor_color, (int(cx), int(cy)), 8)
        pygame.draw.circle(screen, (255, 255, 255), (int(cx), int(cy)), 8, 1)

        # ── Top HUD (status on left, hints on right, two rows) ─────────────
        pygame.draw.rect(screen, (30, 30, 40), (0, 0, WIDTH, 54))
        hud = font_big.render(status_msg[:80], True, (220, 220, 220))
        screen.blit(hud, (10, 6))
        hints = font_hint.render(
            f"release=word  pause{GROQ_IDLE_SECONDS:g}s=sentence  C=clear  R=recal  S=save  ESC=quit",
            True,
            (120, 120, 140),
        )
        screen.blit(hints, (10, 30))
        scale_info = font_hint.render(
            f"+/-=rot({scale})  [/]=trans({accel_gain:.0f})",
            True, (100, 100, 120),
        )
        screen.blit(scale_info, (WIDTH - scale_info.get_width() - 10, 30))

        pygame.display.flip()
        clock.tick(120)

    ser.close()
    pygame.quit()
    if pending_groq_words:
        print("[infer] Flushing unfinished phrase before exit …", flush=True)
        flush_phrase_to_groq_and_tts(pending_groq_words[:])  # speak on exit too
    if predictions:
        print(f"\n[infer] Session CNN words ({len(predictions)}): {predictions}", flush=True)
    print("Bye!", flush=True)


if __name__ == "__main__":
    main()
