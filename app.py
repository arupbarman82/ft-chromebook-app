import os
import re
import json
import uuid
import shutil
import threading
import tempfile
import subprocess
from pathlib import Path
from functools import wraps

import httpx
from flask import (
    Flask, request, jsonify, render_template_string,
    redirect, url_for, session
)
from dotenv import load_dotenv
from werkzeug.security import check_password_hash, generate_password_hash

from openai import OpenAI
from faster_whisper import WhisperModel


# ====================
# Basic Config
# ====================
APP_TITLE = "Fundoo Tutor Metadata Writer"
HOST = os.getenv("FT_HOST", "127.0.0.1").strip()          # keep 127.0.0.1 for local-only
PORT = int(os.getenv("FT_PORT", "8787").strip())

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
ALLOWED_EXT = {".mp4", ".m4a", ".wav", ".webm"}

BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = BASE_DIR / "tmp_jobs"
TMP_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = BASE_DIR / "ft_config.json"   # stores API key + model settings (do NOT commit)
load_dotenv()


# ====================
# Login (ID + Password)
# ====================
# Set these in .env (recommended):
# FT_APP_USER=admin
# FT_APP_PASS=your_strong_password_here
# FLASK_SECRET_KEY=long_random_string
APP_USER = (os.getenv("FT_APP_USER") or "").strip()
APP_PASS = (os.getenv("FT_APP_PASS") or "").strip()
SECRET_KEY = (os.getenv("FLASK_SECRET_KEY") or "").strip()

AUTH_ENABLED = bool(APP_USER and APP_PASS)


# ====================
# Master Prompt (Backend only)
# ====================
MASTER_PROMPT = r"""
You are a metadata writer for Fundoo Tutor.

You analyse uploaded raw educational videos and produce YouTube Studio-ready metadata for either the Cambridge channel or the IB channel. Identify which applies from the video itself and never mix the two.

Tone must be calm, credible, authority-first, and fully human. Use British English. Do not use em dashes.
Do not mention YouTube, AI, prompts, automation, production tools, or third-party tools in public-facing outputs.
Ask questions one by one only.

You will receive:
- LINKS_MODE (one of: provided, checked_no_links, not_available, not_provided)
- VALIDATED_LINKS (a JSON list with fields: url, ok, title, reason)
- TRANSCRIPT (timestamped lines, MM:SS text)

A) Link-First Workflow (Hard Stop)
If LINKS_MODE is "not_provided", output exactly these 3 lines and STOP (no extra lines):
I can see the uploaded video file.
Have you checked the sheet for uploaded video links?
Please check the reporting sheet. You can find the uploaded video links from the reporting sheet. Use YouTube channel filters. Then copy and paste all the YouTube links from the sheet.

A2) Proceed Decision
If LINKS_MODE is "provided" → proceed and allow Watch Next.
If LINKS_MODE is "checked_no_links" → proceed without Watch Next.
If LINKS_MODE is "not_available" → proceed without Watch Next.
No follow-up questions about links.

B) Audio-First Understanding (Hard)
Treat TRANSCRIPT as the full spoken audio understanding. Do not guess unclear parts.

C) Core Strategy (Locked)
Authority first, sales later.
Teach the why, not just the what.
Keep value inside the video to protect watch time.
No hype, urgency, or pressure.
Evergreen unless explicitly time-bound.
Platform-neutral and country-neutral unless the video specifies.
Primary audience: Students, unless stated otherwise.

D) Titles (Hard)
Always output 3 options, labelled exactly:
Option 1 (Highest SEO Reach)
Option 2 (Parent High-Intent)
Option 3 (Authority Explainer)

Rules:
Keyword-intent validated
Title Case
60–75 characters
Colon only if genuinely needed
Year only if it improves search intent
Authoritative, not promotional

E) Description (Hard)
Length is fully video-based. No padding.
Exact order:
1. Hook (1–2 calm sentences)
2. Summary
3. What You Will Learn (short bullets)
4. Contact
5. Subscribe line
6. Disclaimer
7. Optional hashtags

After the Summary, you may insert up to 2:
Who This Video Is For
Common Confusions This Video Clears
How This Fits Into The Bigger Picture

Contact block (exact):
Call or WhatsApp: +91 78892 17144
Website:
Email: fundootutor@gmail.com

Include “book a demo” only if logically earned.

Subscribe line (exact casing, one sentence):
If this helped, subscribe for more clear, structured IB guidance.
or
If this helped, subscribe for more clear, structured Cambridge guidance.

Disclaimer:
IB: International Baccalaureate (IB) is a registered trademark of the International Baccalaureate Organisation, which is not affiliated with and does not endorse this content.
Cambridge: Cambridge is a registered trademark of Cambridge University Press and Assessment, which is not affiliated with and does not endorse this content.

F) Watch Next (Separate Section)
Only if LINKS_MODE is "provided".
Use only VALIDATED_LINKS where ok=true.
Select:
1) deepens current topic
2) next logical learning step
If fewer than two valid links remain, output the best possible remaining link(s).
If zero valid links remain, omit Watch Next entirely (including the heading).
Never invent links. Never use playlist links.

G) Education Type + Problems (Hard)
Education Type must be auto-selected from:
Concept overview
How-to
Problem walkthrough
Other only if genuinely applicable

Education Problems:
Exact timestamps MM:SS
One problem per line
Up to 8 only when justified
Learner-intent phrasing (what / how / why)
No answers, conclusions, or sales
Format: MM:SS text (no hyphens)
Timestamps strictly increasing, no duplicates

H) Tags (Hard)
One comma-separated line
Max 500 characters
Evergreen, high-intent only
No duplicates or near-duplicates
Do not mix IB and Cambridge
Include Fundoo Tutor only when natural

I) Final Output Order (Locked)
Output exactly in this order:
Title
Description
Watch Next (only if applicable)
Type (one line) + Education Problems immediately below
Tags

Do not output Chapters.

J) Final QA (Hard)
Before finalising, recheck and fix:
Timestamps strictly increasing, no duplicates
All titles 60–75 characters
Watch Next only if LINKS_MODE is "provided"
No em dashes/en dashes (—/–)
Tags <= 500 characters
""".strip()


# ====================
# Config storage (API key + models)
# ====================
def _safe_chmod_600(p: Path):
    try:
        os.chmod(p, 0o600)
    except Exception:
        pass

def load_runtime_config() -> dict:
    # Defaults from .env
    cfg = {
        "OPENAI_API_KEY": (os.getenv("OPENAI_API_KEY") or "").strip(),
        "OPENAI_MODEL": (os.getenv("OPENAI_MODEL") or "gpt-5.2-thinking").strip(),
        "OPENAI_FALLBACK_MODELS": (os.getenv("OPENAI_FALLBACK_MODELS") or "gpt-5-mini,gpt-4o").strip(),
        "REASONING_EFFORT": (os.getenv("REASONING_EFFORT") or "high").strip().lower(),
    }
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                cfg.update({k: (str(v).strip() if v is not None else "") for k, v in data.items()})
        except Exception:
            pass

    if cfg["REASONING_EFFORT"] not in {"low", "medium", "high"}:
        cfg["REASONING_EFFORT"] = "high"

    return cfg

def save_runtime_config(new_cfg: dict):
    CONFIG_PATH.write_text(json.dumps(new_cfg, indent=2), encoding="utf-8")
    _safe_chmod_600(CONFIG_PATH)


# ====================
# Dependency checks
# ====================
def which(cmd: str) -> str:
    p = shutil.which(cmd)
    return p or ""

def health_check() -> dict:
    ffmpeg_ok = bool(which("ffmpeg"))
    ffprobe_ok = bool(which("ffprobe"))
    return {
        "ffmpeg": ffmpeg_ok,
        "ffprobe": ffprobe_ok,
        "ffmpeg_path": which("ffmpeg"),
        "ffprobe_path": which("ffprobe"),
        "config_file": str(CONFIG_PATH),
        "config_exists": CONFIG_PATH.exists(),
        "auth_enabled": AUTH_ENABLED,
        "host": HOST,
        "port": PORT,
    }


# ====================
# Auth helpers
# ====================
def require_login(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not AUTH_ENABLED:
            return fn(*args, **kwargs)
        if session.get("logged_in") is True:
            return fn(*args, **kwargs)
        # For API calls, return 401
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not logged in"}), 401
        return redirect(url_for("login"))
    return wrapper


# ====================
# FFmpeg helpers
# ====================
def run_cmd(cmd: list[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "Command failed")
    return p.stdout.strip()

def ffprobe_duration(path: str) -> float:
    try:
        out = run_cmd([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ])
        return float(out)
    except Exception:
        return 0.0

def extract_audio_wav_16k_mono(src_path: str, out_wav: str):
    run_cmd([
        "ffmpeg", "-y", "-i", src_path,
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        out_wav
    ])


# ====================
# Link parsing + validation
# ====================
def parse_links(text: str) -> list[str]:
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    return [l for l in lines if l.startswith("http://") or l.startswith("https://")]

def validate_links(urls: list[str]) -> list[dict]:
    results = []
    if not urls:
        return results
    with httpx.Client(follow_redirects=True, timeout=12.0) as h:
        for u in urls:
            ok = False
            title = ""
            reason = ""
            try:
                r = h.get(u, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code != 200:
                    reason = f"HTTP {r.status_code}"
                else:
                    html = r.text
                    low = html.lower()
                    if "video unavailable" in low or "private video" in low:
                        reason = "Unavailable/private"
                    else:
                        m = re.search(r'property="og:title"\s+content="([^"]+)"', html)
                        if m:
                            title = m.group(1).strip()
                        ok = True
            except Exception as e:
                reason = str(e)
            results.append({"url": u, "ok": ok, "title": title, "reason": reason})
    return results


# ====================
# QA check
# ====================
def _mmss_to_sec(x: str) -> int:
    m, s = x.split(":")
    return int(m) * 60 + int(s)

def qa_check(output_text: str, link_mode: str, validated_links: list) -> list[str]:
    issues = []
    txt = output_text or ""
    lines = [l.rstrip("\n") for l in txt.splitlines()]
    non_empty = [l.strip() for l in lines if l.strip()]

    if "—" in txt or "–" in txt:
        issues.append("Contains an em dash/en dash (—/–).")

    required_labels = [
        "Option 1 (Highest SEO Reach)",
        "Option 2 (Parent High-Intent)",
        "Option 3 (Authority Explainer)",
    ]
    for lab in required_labels:
        if lab not in txt:
            issues.append(f"Missing title label: {lab}")

    # title length: next non-empty line after each label
    for lab in required_labels:
        idx = None
        for i, line in enumerate(lines):
            if line.strip() == lab:
                idx = i
                break
        if idx is None:
            continue
        j = idx + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            issues.append(f"No title text found after {lab}")
            continue
        title = lines[j].strip()
        L = len(title)
        if L < 60 or L > 75:
            issues.append(f"Title length {L} after {lab} (must be 60–75).")

    has_watch_next = any(l.strip().lower().startswith("watch next") for l in lines)
    valid_count = sum(1 for x in (validated_links or []) if x.get("ok"))

    if link_mode != "provided" and has_watch_next:
        issues.append("Watch Next present but links were not provided.")
    if link_mode == "provided" and valid_count == 0 and has_watch_next:
        issues.append("Watch Next present but no valid links remained.")

    if non_empty:
        last = non_empty[-1]
        if "," not in last:
            issues.append("Tags line not detected (expected one comma-separated line at the end).")
        elif len(last) > 500:
            issues.append(f"Tags line is {len(last)} chars (must be <= 500).")

    # Education Problems timestamps strictly increasing after "Type"
    type_idx = None
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith("type"):
            type_idx = idx
            break

    ts = []
    if type_idx is not None:
        for line in lines[type_idx + 1:]:
            m = re.match(r"^(\d{2}:\d{2})\s+.+", line.strip())
            if m:
                ts.append(m.group(1))

    if ts:
        secs = [_mmss_to_sec(x) for x in ts]
        if any(secs[i] >= secs[i + 1] for i in range(len(secs) - 1)):
            issues.append("Education Problems timestamps are not strictly increasing or contain duplicates.")
        if len(ts) > 8:
            issues.append(f"Education Problems has {len(ts)} lines (must be up to 8 when justified).")

    return issues


# ====================
# Whisper (lazy load)
# ====================
_whisper = None

def get_whisper():
    global _whisper
    if _whisper is None:
        # VAD kept OFF to avoid silero_vad missing asset issues
        _whisper = WhisperModel("small", device="cpu", compute_type="int8")
    return _whisper


# ====================
# OpenAI call (Responses first, Chat Completions fallback)
# ====================
def get_client(api_key: str) -> OpenAI:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Go to Settings and save your API key.")
    return OpenAI(api_key=api_key)

def call_openai_metadata(transcript: str, link_mode: str, validated_links: list, cfg: dict) -> str:
    msg = (
        f"LINKS_MODE: {link_mode}\n\n"
        f"VALIDATED_LINKS:\n{json.dumps(validated_links, indent=2)}\n\n"
        f"TRANSCRIPT:\n{transcript}\n"
    )

    primary = cfg.get("OPENAI_MODEL", "gpt-5.2-thinking").strip()
    fallbacks = [m.strip() for m in (cfg.get("OPENAI_FALLBACK_MODELS") or "").split(",") if m.strip()]
    models_to_try = [primary] + [m for m in fallbacks if m != primary]

    api_key = cfg.get("OPENAI_API_KEY", "").strip()
    effort = (cfg.get("REASONING_EFFORT") or "high").strip().lower()
    if effort not in {"low", "medium", "high"}:
        effort = "high"

    client = get_client(api_key)
    last_err = ""

    # 1) Try Responses API (preferred)
    for model_name in models_to_try:
        try:
            resp = client.responses.create(
                model=model_name,
                input=[
                    {"role": "system", "content": MASTER_PROMPT},
                    {"role": "user", "content": msg},
                ],
                reasoning={"effort": effort},
                store=False,
            )
            text = getattr(resp, "output_text", None)
            return (text or "").strip()
        except Exception as e:
            last_err = str(e)
            # if missing responses scope, try chat.completions next
            if "api.responses.write" in last_err:
                break

    # 2) Fallback: Chat Completions API
    for model_name in models_to_try:
        try:
            cc = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": MASTER_PROMPT},
                    {"role": "user", "content": msg},
                ],
            )
            out = (cc.choices[0].message.content or "").strip()
            if out:
                return out
            last_err = "Empty response from chat.completions."
        except Exception as e:
            last_err = str(e)

    raise RuntimeError(
        "OpenAI request failed.\n\n"
        f"Last error: {last_err}\n\n"
        "If you see missing scopes (example: api.responses.write), create a Project API key with All permissions "
        "or enable Write access for the Responses API in your key permissions."
    )


# ====================
# Flask App + Jobs
# ====================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

if not SECRET_KEY:
    # For login to work reliably, set FLASK_SECRET_KEY in .env
    SECRET_KEY = os.urandom(32).hex()
app.secret_key = SECRET_KEY

# Store password hash in memory (avoid keeping plain text in session)
PASS_HASH = generate_password_hash(APP_PASS) if AUTH_ENABLED else ""


jobs = {}
jobs_lock = threading.Lock()

def update_job(job_id, **kwargs):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)


# ====================
# UI Templates
# ====================
LOGIN_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{{ app_title }} - Login</title>
  <style>
    body{font-family:system-ui;max-width:520px;margin:40px auto;padding:0 14px}
    .card{border:1px solid #ddd;border-radius:16px;padding:16px}
    label{display:block;margin-top:10px}
    input{width:100%;padding:10px;border:1px solid #ccc;border-radius:12px}
    button{margin-top:14px;padding:10px 14px;border-radius:12px;border:1px solid #ccc;cursor:pointer}
    .err{color:#b00020;margin-top:10px}
    .muted{color:#666}
  </style>
</head>
<body>
  <h2 style="margin:0 0 10px 0">{{ app_title }}</h2>
  <p class="muted" style="margin-top:0">Please log in to continue.</p>
  <div class="card">
    <form method="post">
      <label>Username</label>
      <input name="user" autocomplete="username" required />
      <label>Password</label>
      <input name="pass" type="password" autocomplete="current-password" required />
      <button type="submit">Log in</button>
      {% if error %}<div class="err">{{ error }}</div>{% endif %}
    </form>
  </div>
</body>
</html>
"""

SETTINGS_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{{ app_title }} - Settings</title>
  <style>
    body{font-family:system-ui;max-width:900px;margin:24px auto;padding:0 12px}
    .top{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
    .card{border:1px solid #ddd;border-radius:16px;padding:16px;margin:14px 0}
    .muted{color:#666}
    label{display:block;margin-top:10px}
    input, textarea, select{width:100%;padding:10px;border:1px solid #ccc;border-radius:12px}
    textarea{height:80px}
    button{padding:10px 14px;border-radius:12px;border:1px solid #ccc;cursor:pointer}
    .row{display:flex;gap:12px;flex-wrap:wrap}
    .col{flex:1 1 340px}
    .ok{color:#0b7a0b}
    .err{color:#b00020}
    a{color:inherit}
  </style>
</head>
<body>
  <div class="top">
    <h2 style="margin:0">{{ app_title }} - Settings</h2>
    <a href="/" class="muted">Back</a>
    {% if auth_enabled %}<a href="/logout" class="muted">Logout</a>{% endif %}
  </div>

  <div class="card">
    <h3 style="margin-top:0">OpenAI Settings</h3>
    <p class="muted">Saved in <code>{{ config_path }}</code>. Do not commit this file to GitHub.</p>
    <form method="post">
      <div class="row">
        <div class="col">
          <label>OPENAI_API_KEY</label>
          <input name="OPENAI_API_KEY" type="password" value="{{ cfg.OPENAI_API_KEY }}" placeholder="Paste your API key" required />
        </div>
        <div class="col">
          <label>REASONING_EFFORT</label>
          <select name="REASONING_EFFORT">
            <option value="low" {% if cfg.REASONING_EFFORT=='low' %}selected{% endif %}>low</option>
            <option value="medium" {% if cfg.REASONING_EFFORT=='medium' %}selected{% endif %}>medium</option>
            <option value="high" {% if cfg.REASONING_EFFORT=='high' %}selected{% endif %}>high</option>
          </select>
        </div>
      </div>

      <div class="row">
        <div class="col">
          <label>OPENAI_MODEL</label>
          <input name="OPENAI_MODEL" value="{{ cfg.OPENAI_MODEL }}" />
        </div>
        <div class="col">
          <label>OPENAI_FALLBACK_MODELS (comma-separated)</label>
          <input name="OPENAI_FALLBACK_MODELS" value="{{ cfg.OPENAI_FALLBACK_MODELS }}" />
        </div>
      </div>

      <button type="submit">Save Settings</button>
      {% if msg %}<p class="{{ msg_class }}">{{ msg }}</p>{% endif %}
    </form>
  </div>

  <div class="card">
    <h3 style="margin-top:0">Health Check</h3>
    <pre>{{ health_pre }}</pre>
    <p class="muted">
      If ffmpeg is missing, install it in terminal:
      <code>sudo apt update && sudo apt install -y ffmpeg</code>
    </p>
  </div>
</body>
</html>
"""

APP_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{{ app_title }}</title>
  <style>
    body{font-family:system-ui;max-width:980px;margin:24px auto;padding:0 12px}
    .top{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
    .badge{font-size:12px;background:#f2f2f2;border:1px solid #ddd;border-radius:999px;padding:4px 10px}
    .links{margin-left:auto;display:flex;gap:12px;flex-wrap:wrap}
    .links a{color:inherit;text-decoration:none}
    .card{border:1px solid #ddd;border-radius:16px;padding:16px;margin:14px 0}
    .muted{color:#666}
    textarea{width:100%;height:110px;padding:10px;border:1px solid #ccc;border-radius:12px}
    pre{white-space:pre-wrap;background:#111;color:#eee;padding:12px;border-radius:12px;overflow:auto}
    button{padding:10px 14px;border-radius:12px;border:1px solid #ccc;cursor:pointer}
    .row{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start}
    .col{flex:1 1 360px}
    select{padding:10px;border-radius:12px;border:1px solid #ccc}
    input[type="file"]{width:100%}
    .overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);align-items:center;justify-content:center;z-index:9999}
    .modal{width:min(680px,92vw);background:#fff;border-radius:16px;padding:16px;box-shadow:0 20px 60px rgba(0,0,0,.35)}
    .barwrap{width:100%;background:#eee;border-radius:999px;height:14px;overflow:hidden}
    .bar{height:14px;width:0%;background:#111}
    .btnrow{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px}
    .warn{background:#fff7e6;border:1px solid #ffd28a;border-radius:16px;padding:12px;margin:14px 0}
    code{background:#f2f2f2;padding:2px 6px;border-radius:8px}
  </style>
</head>
<body>
  <div class="top">
    <h2 style="margin:0">{{ app_title }}</h2>
    <span class="badge">Local: http://{{ host }}:{{ port }}</span>
    <div class="links">
      <a href="/settings" class="muted">Settings</a>
      {% if auth_enabled %}<a href="/logout" class="muted">Logout</a>{% endif %}
    </div>
  </div>

  {% if warning %}
  <div class="warn">
    <b>Setup warning</b><br>
    <span class="muted">{{ warning }}</span>
  </div>
  {% endif %}

  <div class="card">
    <div class="row">
      <div class="col">
        <label class="muted">Upload media</label><br>
        <input id="file" type="file"
          accept=".mp4,.m4a,.wav,.webm,video/mp4,video/webm,audio/mp4,audio/x-m4a,audio/wav,audio/webm" />
      </div>

      <div class="col">
        <label class="muted">Links status</label><br>
        <select id="linkMode">
          <option value="not_provided">Not provided (Hard Stop)</option>
          <option value="checked_no_links" selected>Yes/checked but no links</option>
          <option value="not_available">Links not available</option>
          <option value="provided">Links provided</option>
        </select>

        <div style="margin-top:10px">
          <label class="muted">Links (one per line, only if provided)</label>
          <textarea id="links" placeholder="https://www.youtube.com/watch?v=..."></textarea>
        </div>

        <div class="btnrow">
          <button id="go">Process</button>
          <button id="clear" type="button">Clear</button>
        </div>
      </div>
    </div>
  </div>

  <div class="card"><h3 style="margin-top:0">Transcript</h3><pre id="t"></pre></div>
  <div class="card"><h3 style="margin-top:0">Metadata Output</h3><pre id="o"></pre></div>
  <div class="card"><h3 style="margin-top:0">Process Check</h3><pre id="qa"></pre></div>

  <div class="overlay" id="overlay">
    <div class="modal">
      <h3 style="margin:0 0 8px 0">Processing…</h3>
      <div class="barwrap"><div class="bar" id="bar"></div></div>
      <p class="muted" style="margin:10px 0 0 0" id="stage">Starting…</p>
      <p class="muted" style="margin:6px 0 0 0"><span id="pct">0</span>%</p>
    </div>
  </div>

<script>
const overlay = document.getElementById("overlay");
const bar = document.getElementById("bar");
const pct = document.getElementById("pct");
const stage = document.getElementById("stage");
const out = document.getElementById("o");
const transcript = document.getElementById("t");
const qa = document.getElementById("qa");
const fileInput = document.getElementById("file");

function showProgress(p, s){
  overlay.style.display = "flex";
  pct.textContent = String(p);
  bar.style.width = String(p) + "%";
  stage.textContent = s || "";
}
function hideProgress(){ overlay.style.display = "none"; }

async function poll(jobId){
  while(true){
    const r = await fetch("/api/status/" + jobId);
    if (r.status === 401) { window.location.href = "/login"; return; }
    const j = await r.json();

    if (j.error) {
      hideProgress();
      out.textContent = "ERROR:\\n" + j.error;
      return;
    }

    showProgress(j.progress || 0, j.stage || "Working…");

    if(j.done){
      hideProgress();
      transcript.textContent = j.transcript || "";
      out.textContent = j.metadata || "";
      if (j.qa_pass === true) {
        qa.textContent = "PASS: Output matches the key rules.";
      } else if (j.qa_pass === false) {
        qa.textContent = "FAIL:\\n" + (j.qa_issues || []).map(x => "- " + x).join("\\n");
      } else {
        qa.textContent = "";
      }
      return;
    }
    await new Promise(res => setTimeout(res, 600));
  }
}

document.getElementById("clear").onclick = () => {
  out.textContent = "";
  transcript.textContent = "";
  qa.textContent = "";
  document.getElementById("links").value = "";
  document.getElementById("linkMode").value = "checked_no_links";
  fileInput.value = "";
};

document.getElementById("go").onclick = async () => {
  out.textContent = "";
  transcript.textContent = "";
  qa.textContent = "";

  const f = fileInput.files[0];
  if(!f) return alert("Choose a file first.");

  const mode = document.getElementById("linkMode").value;
  const links = document.getElementById("links").value;

  if (mode === "provided" && (!links || links.trim().length === 0)) {
    return alert("You selected 'Links provided' but did not paste any links.");
  }

  const fd = new FormData();
  fd.append("file", f);
  fd.append("linkMode", mode);
  fd.append("links", links);

  showProgress(1, "Uploading…");
  const r = await fetch("/api/start", {method:"POST", body: fd});
  if (r.status === 401) { window.location.href = "/login"; return; }
  const j = await r.json();
  if(!r.ok){
    hideProgress();
    return alert(j.error || "Failed to start.");
  }
  await poll(j.job_id);
};
</script>
</body>
</html>
"""


# ====================
# Routes: Login / Logout / Settings
# ====================
@app.get("/login")
def login():
    if not AUTH_ENABLED:
        return redirect(url_for("index"))
    return render_template_string(LOGIN_HTML, app_title=APP_TITLE, error=None)

@app.post("/login")
def login_post():
    if not AUTH_ENABLED:
        return redirect(url_for("index"))
    u = (request.form.get("user") or "").strip()
    p = (request.form.get("pass") or "").strip()
    if u == APP_USER and check_password_hash(PASS_HASH, p):
        session["logged_in"] = True
        return redirect(url_for("index"))
    return render_template_string(LOGIN_HTML, app_title=APP_TITLE, error="Invalid username or password.")

@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.get("/settings")
@require_login
def settings():
    cfg = load_runtime_config()
    h = health_check()
    return render_template_string(
        SETTINGS_HTML,
        app_title=APP_TITLE,
        cfg=cfg,
        msg=None,
        msg_class="",
        health_pre=json.dumps(h, indent=2),
        config_path=str(CONFIG_PATH),
        auth_enabled=AUTH_ENABLED
    )

@app.post("/settings")
@require_login
def settings_post():
    cfg = load_runtime_config()
    new_cfg = {
        "OPENAI_API_KEY": (request.form.get("OPENAI_API_KEY") or "").strip(),
        "OPENAI_MODEL": (request.form.get("OPENAI_MODEL") or "").strip(),
        "OPENAI_FALLBACK_MODELS": (request.form.get("OPENAI_FALLBACK_MODELS") or "").strip(),
        "REASONING_EFFORT": (request.form.get("REASONING_EFFORT") or "high").strip().lower(),
    }
    if not new_cfg["OPENAI_API_KEY"]:
        msg = "API key is required."
        msg_class = "err"
    else:
        save_runtime_config(new_cfg)
        msg = "Saved."
        msg_class = "ok"
        cfg = new_cfg

    h = health_check()
    return render_template_string(
        SETTINGS_HTML,
        app_title=APP_TITLE,
        cfg=cfg,
        msg=msg,
        msg_class=msg_class,
        health_pre=json.dumps(h, indent=2),
        config_path=str(CONFIG_PATH),
        auth_enabled=AUTH_ENABLED
    )


# ====================
# Routes: Main + API
# ====================
@app.get("/")
@require_login
def index():
    h = health_check()
    cfg = load_runtime_config()
    warning = ""
    if not h["ffmpeg"] or not h["ffprobe"]:
        warning = "ffmpeg/ffprobe not found. Install: sudo apt update && sudo apt install -y ffmpeg"
    elif not cfg.get("OPENAI_API_KEY"):
        warning = "OPENAI_API_KEY not set. Go to Settings and save your API key."
    return render_template_string(APP_HTML, app_title=APP_TITLE, host=HOST, port=PORT, auth_enabled=AUTH_ENABLED, warning=warning)

@app.get("/api/health")
@require_login
def api_health():
    return jsonify(health_check())

@app.post("/api/start")
@require_login
def api_start():
    h = health_check()
    if not h["ffmpeg"] or not h["ffprobe"]:
        return jsonify({"error": "ffmpeg/ffprobe not found. Install: sudo apt update && sudo apt install -y ffmpeg"}), 400

    cfg = load_runtime_config()
    if not cfg.get("OPENAI_API_KEY"):
        return jsonify({"error": "OPENAI_API_KEY missing. Go to Settings and save your API key."}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported file type: {ext}. Use mp4, m4a, wav, webm."}), 400

    link_mode = (request.form.get("linkMode") or "checked_no_links").strip()
    links_text = request.form.get("links") or ""

    job_id = uuid.uuid4().hex[:10]
    uploaded_path = TMP_DIR / f"{job_id}{ext}"
    f.save(uploaded_path)

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "progress": 1,
            "stage": "Queued…",
            "done": False,
            "error": None,
            "transcript": "",
            "metadata": "",
            "qa_issues": [],
            "qa_pass": None,
        }

    t = threading.Thread(
        target=worker,
        args=(job_id, str(uploaded_path), link_mode, links_text),
        daemon=True
    )
    t.start()

    return jsonify({"job_id": job_id})

@app.get("/api/status/<job_id>")
@require_login
def api_status(job_id):
    with jobs_lock:
        j = jobs.get(job_id)
        if not j:
            return jsonify({"error": "Unknown job id"}), 404
        return jsonify(j)


def worker(job_id, uploaded_path: str, link_mode: str, links_text: str):
    try:
        # Hard stop before any work
        if link_mode == "not_provided":
            update_job(
                job_id,
                progress=100,
                stage="Stopped (Hard Stop).",
                done=True,
                transcript="",
                metadata=(
                    "I can see the uploaded video file.\n"
                    "Have you checked the sheet for uploaded video links?\n"
                    "Please check the reporting sheet. You can find the uploaded video links from the reporting sheet. Use YouTube channel filters. Then copy and paste all the YouTube links from the sheet."
                ),
                qa_pass=True,
                qa_issues=[],
            )
            return

        update_job(job_id, progress=5, stage="Extracting audio…")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / ("input" + Path(uploaded_path).suffix.lower())
            shutil.copy2(uploaded_path, src)

            wav = td / "audio.wav"
            extract_audio_wav_16k_mono(str(src), str(wav))
            dur = ffprobe_duration(str(wav)) or 0.0

            update_job(job_id, progress=15, stage="Loading speech model… (first run may take time)")
            model = get_whisper()

            update_job(job_id, progress=20, stage="Transcribing audio…")

            # Keep VAD OFF to avoid silero_vad missing assets
            segments, _info = model.transcribe(
                str(wav),
                language="en",
                vad_filter=False,
                beam_size=5
            )

            lines = []
            last_pct = 20
            for s in segments:
                lines.append(f"{int(s.start//60):02d}:{int(s.start%60):02d} {s.text.strip()}")
                if dur > 0:
                    covered = min(1.0, float(s.end) / dur)
                    p = 20 + int(55 * covered)  # up to ~75%
                    if p > last_pct:
                        last_pct = p
                        update_job(job_id, progress=p, stage="Transcribing audio…")

            transcript = "\n".join(lines).strip()
            if not transcript:
                raise RuntimeError("Transcript is empty. Check the audio track in your file.")

            update_job(job_id, progress=80, stage="Validating links…")
            urls = parse_links(links_text)
            validated = validate_links(urls) if (link_mode == "provided" and urls) else []

            update_job(job_id, progress=88, stage="Generating metadata…")
            cfg = load_runtime_config()
            metadata = call_openai_metadata(transcript, link_mode, validated, cfg)

            update_job(job_id, progress=96, stage="Final process check…")
            issues = qa_check(metadata, link_mode, validated)
            update_job(job_id, qa_issues=issues, qa_pass=(len(issues) == 0))

            update_job(job_id, progress=100, stage="Done.", done=True, transcript=transcript, metadata=metadata)

    except Exception as e:
        update_job(job_id, done=True, error=str(e), stage="Error", progress=100)
    finally:
        try:
            Path(uploaded_path).unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    # IMPORTANT: debug=False avoids the "double run" (two python PIDs) on Chromebook
    app.run(host=HOST, port=PORT, debug=False)
