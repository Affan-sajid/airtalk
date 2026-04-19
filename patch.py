import re

with open('gemini-infer.py', 'r') as f:
    content = f.read()

# Replace the Groq LLM + TTS section
old_section = """# ── Groq LLM + TTS (after pause with pen up) ────────────────────────────────
# Seconds after the last CNN word (pen release) before we flush to Groq + TTS.
LLM_IDLE_SECONDS = 2.0
GROQ_LLM_MODEL = "openai/gpt-oss-20b"
GROQ_TTS_MODEL = "canopylabs/orpheus-v1-english"
GROQ_TTS_VOICE = "troy"
# Optional override (prefer `.env` or the environment — do not commit real keys).
GROQ_API_KEY_ENV = ("GROQ_API_KEY",)"""

new_section = """# ── Gemini LLM + TTS (after pause with pen up) ────────────────────────────────
# Seconds after the last CNN word (pen release) before we flush to Gemini + TTS.
LLM_IDLE_SECONDS = 2.0
GEMINI_LLM_MODEL = "gemini-2.5-flash"
# Optional override (prefer `.env` or the environment — do not commit real keys).
GEMINI_API_KEY_ENV = ("GEMINI_API_KEY",)"""

content = content.replace(old_section, new_section)

# Replace _groq_api_key
old_key_func = """def _groq_api_key() -> str | None:
    _load_project_env_file()
    file_key = os.environ.get("GROQ_API_KEY_FILE", "").strip()
    if file_key:
        return file_key
    for name in GROQ_API_KEY_ENV:
        v = os.environ.get(name, "").strip()
        if v:
            return v
    return None"""

new_key_func = """def _gemini_api_key() -> str | None:
    _load_project_env_file()
    file_key = os.environ.get("GEMINI_API_KEY_FILE", "").strip()
    if file_key:
        return file_key
    for name in GEMINI_API_KEY_ENV:
        v = os.environ.get(name, "").strip()
        if v:
            return v
    return None"""

content = content.replace(old_key_func, new_key_func)

# Replace naturalize_with_gemini
old_nat_func = """def naturalize_with_gemini(words: list[str]) -> str | None:
    \"\"\"Turn ordered CNN tokens into one natural sentence using Structured Outputs.\"\"\"
    key = _groq_api_key()
    if not key or not words:
        return None
    try:
        from groq import Groq
    except ImportError:
        print("[groq] Install: pip install groq", flush=True)
        return None

    client = Groq(api_key=key)
    joined = " ".join(words)
    system_prompt = f\"\"\"You are a helpful assistant that reconstructs a single intended sentence from air-written words.
Input tokens: {joined}

RULES:
- Output ONLY the final sentence inside the JSON schema.
- Use first person ("I / me / my") where appropriate.
- Fix grammar (tense, caps) while keeping the original intent.
- Example: "hi name affan" -> "Hi, my name is Affan."
\"\"\"

    print(f"[groq] LLM Request: model={GROQ_LLM_MODEL}, words='{joined}'", flush=True)
    t0 = time.perf_counter()
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate the sentence now."},
            ],
            model=GROQ_LLM_MODEL,
            temperature=0.5,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "natural_sentence",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentence": {"type": "string"}
                        },
                        "required": ["sentence"],
                        "additionalProperties": False
                    }
                }
            },
        )
        import json
        content = chat_completion.choices[0].message.content
        data = json.loads(content or "{}")
        text = data.get("sentence", "").strip().strip('"').strip("'")
        
        latency = time.perf_counter() - t0
        u = getattr(chat_completion, "usage", None)
        usage_str = f" [P:{u.prompt_tokens} C:{u.completion_tokens}]" if u else ""
        
        if not text:
            print(f"[groq] Warning: LLM returned empty content after {latency:.3f}s. Raw: {repr(content)}", flush=True)
            return None

        print(f"[groq] LLM Success in {latency:.3f}s{usage_str}: \\"{text}\\"", flush=True)
        return text
    except Exception as exc:
        print(f"[groq] LLM Error after {time.perf_counter()-t0:.3f}s: {exc}", flush=True)
        return None"""

new_nat_func = """def naturalize_with_gemini(words: list[str]) -> str | None:
    \"\"\"Turn ordered CNN tokens into one natural sentence using Gemini.\"\"\"
    _load_project_env_file()
    if not words:
        return None
    try:
        from google import genai
        from google.genai.types import HttpOptions, GenerateContentConfig
    except ImportError:
        print("[gemini] Install: pip install google-genai", flush=True)
        return None

    try:
        client = genai.Client(http_options=HttpOptions(api_version="v1"))
    except Exception as e:
        print(f"[gemini] Failed to initialize client: {e}", flush=True)
        return None

    joined = " ".join(words)
    system_prompt = \"\"\"You are a helpful assistant that reconstructs a single intended sentence from air-written words.

RULES:
- Output ONLY the final sentence.
- Use first person ("I / me / my") where appropriate.
- Fix grammar (tense, caps) while keeping the original intent.
- Example: "hi name affan" -> "Hi, my name is Affan."
\"\"\"

    print(f"[gemini] LLM Request: model={GEMINI_LLM_MODEL}, words='{joined}'", flush=True)
    t0 = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=GEMINI_LLM_MODEL,
            contents=joined,
            config=GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.5,
            )
        )
        text = response.text.strip().strip('"').strip("'")
        
        latency = time.perf_counter() - t0
        
        if not text:
            print(f"[gemini] Warning: LLM returned empty content after {latency:.3f}s.", flush=True)
            return None

        print(f"[gemini] LLM Success in {latency:.3f}s: \\"{text}\\"", flush=True)
        return text
    except Exception as exc:
        print(f"[gemini] LLM Error after {time.perf_counter()-t0:.3f}s: {exc}", flush=True)
        return None"""

content = content.replace(old_nat_func, new_nat_func)

# Replace speak_sentence
old_speak_func = """def speak_sentence(text: str) -> None:
    text = text.strip()
    if not text:
        return
    key = _groq_api_key()
    if key:
        print(f"[groq-tts] Request: model={GROQ_TTS_MODEL}, voice={GROQ_TTS_VOICE}, text='{text}'", flush=True)
        t0 = time.perf_counter()
        try:
            from groq import Groq
            client = Groq(api_key=key)
            response = client.audio.speech.create(
                model=GROQ_TTS_MODEL, voice=GROQ_TTS_VOICE, input=text, response_format="wav"
            )
            tmp_path = Path(__file__).resolve().parent / ".groq_tts_tmp.wav"
            response.write_to_file(str(tmp_path))
            
            latency = time.perf_counter() - t0
            print(f"[groq-tts] Success in {latency:.3f}s. Playing via afplay/aplay...", flush=True)

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
            latency = time.perf_counter() - t0
            print(f"[groq-tts] Error after {latency:.3f}s: {exc}", flush=True)
    if sys.platform == "darwin":
        subprocess.run(["say", text], check=False)
        return
    try:
        import pyttsx3

        eng = pyttsx3.init()
        eng.say(text)
        eng.runAndWait()
    except Exception as exc:
        print(f"[tts] No speech (macOS has `say`; else install pyttsx3): {exc}", flush=True)"""

new_speak_func = """def speak_sentence(text: str) -> None:
    text = text.strip()
    if not text:
        return
    if sys.platform == "darwin":
        subprocess.run(["say", text], check=False)
        return
    try:
        import pyttsx3

        eng = pyttsx3.init()
        eng.say(text)
        eng.runAndWait()
    except Exception as exc:
        print(f"[tts] No speech (macOS has `say`; else install pyttsx3): {exc}", flush=True)"""

content = content.replace(old_speak_func, new_speak_func)

with open('gemini-infer.py', 'w') as f:
    f.write(content)
