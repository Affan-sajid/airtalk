import os
from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig

def _load_project_env_file() -> None:
    from pathlib import Path
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

_load_project_env_file()

try:
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="hi",
        config=GenerateContentConfig(
            system_instruction="You are a helpful assistant.",
            temperature=0.5,
        )
    )
    print("default success:", response.text)
except Exception as e:
    print("default error:", e)
