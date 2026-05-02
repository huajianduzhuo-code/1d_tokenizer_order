"""VLM judge module for PromptBind-Hard (PBH).

Walks generated images for a run directory, asks each per-prompt question
to a VLM (GPT-4o or Gemini), normalizes the answer, and compares against
the dataset's `accepted` list. Results cached on disk so re-runs are free.

API keys: read from environment variables.
    OPENAI_API_KEY   for GPT-4o
    GOOGLE_API_KEY   for Gemini 2.0 Flash

To use: set them in your shell or in /home/hliu256/1d-tokenizer/.env (then
source it). Placeholders below mark the relevant calls — actual keys are
NEVER hard-coded.
"""

# ============================================================
# API KEY PLACEHOLDERS
# ============================================================
# Set these in your environment (do NOT commit real keys):
#   export OPENAI_API_KEY="sk-..."          # for GPT-4o
#   export GOOGLE_API_KEY="..."             # for Gemini
# ============================================================

import asyncio
import base64
import hashlib
import json
import os
import random
import re
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

# tqdm: progress bar; if missing, fall back to no-op
try:
    from tqdm.asyncio import tqdm_asyncio
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


SUPPORTED_PROVIDERS = ("openai", "gemini")
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

OPENAI_CONCURRENCY = 10
GEMINI_CONCURRENCY = 30

MAX_RETRIES = 5
BASE_BACKOFF = 2.0  # seconds
PER_CALL_TIMEOUT = 30.0  # seconds; SDK calls without explicit timeout can hang


# -----------------------------------------------------------
# Normalization
# -----------------------------------------------------------

_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}
_DIGIT_WORDS = {v: k for k, v in _NUM_WORDS.items()}

_YES_FORMS = {"yes", "yeah", "yep", "yup", "y", "true", "correct"}
_NO_FORMS = {"no", "nope", "n", "false", "incorrect"}


def _strip_punct(s: str) -> str:
    return re.sub(r"[^\w\s\-]", "", s)


def normalize_answer(answer: str, qtype: str) -> str:
    """Normalize a free-form VLM answer string into a canonical short form.

    Returns a normalized lowercase string. The returned string is intended
    to be compared against normalized members of the dataset's `accepted`
    list, not against the raw `answer` field.
    """
    if not isinstance(answer, str):
        answer = str(answer)
    s = _strip_punct(answer.strip().lower())
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""

    if qtype == "yesno":
        first_word = s.split()[0]
        if first_word in _YES_FORMS:
            return "yes"
        if first_word in _NO_FORMS:
            return "no"
        if "yes" in s.split():
            return "yes"
        if "no" in s.split():
            return "no"
        return s

    if qtype == "count":
        # Try to find a numeric token
        m = re.search(r"\b\d+\b", s)
        if m:
            return m.group()
        # Try number word
        for word in s.split():
            if word in _NUM_WORDS:
                return _NUM_WORDS[word]
        return s

    # mc / open: lowercase, plural-strip last word
    parts = s.split()
    if parts and parts[-1].endswith("s") and len(parts[-1]) > 3:
        parts[-1] = parts[-1].rstrip("s")
    return " ".join(parts)


def normalized_accepted_set(accepted: list[str], qtype: str) -> set[str]:
    out = set()
    for a in accepted:
        n = normalize_answer(a, qtype)
        out.add(n)
        if qtype == "count":
            # also accept the alternate digit/word form
            if n in _DIGIT_WORDS:
                out.add(_DIGIT_WORDS[n])
            if n in _NUM_WORDS:
                out.add(_NUM_WORDS[n])
    return out


# -----------------------------------------------------------
# Cache
# -----------------------------------------------------------

def _question_hash(question: dict) -> str:
    payload = json.dumps({
        "q": question["q"],
        "type": question["type"],
        "accepted": sorted(question["accepted"]),
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _cache_path(run_dir: Path, provider: str, prompt_id: str,
                sample_idx: int, qhash: str) -> Path:
    return run_dir / "judge_cache" / provider / prompt_id / \
        f"sample{sample_idx:03d}__{qhash}.json"


# -----------------------------------------------------------
# Image encoding
# -----------------------------------------------------------

def _encode_image_b64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# -----------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------

_SYSTEM = (
    "You are a strict visual question answerer. "
    "You look at the provided image and answer the question. "
    "Output ONLY a single JSON object — no prose, no markdown."
)


def _user_text(question: dict) -> str:
    qtype = question["type"]
    q = question["q"]
    if qtype == "count":
        instruct = (
            "Reply with a JSON object: {\"answer\": \"<arabic_numeral>\"}. "
            "Use only an Arabic numeral (e.g. \"3\"). "
            "If you cannot determine the count, return {\"answer\": \"unknown\"}."
        )
    elif qtype == "yesno":
        instruct = (
            "Reply with a JSON object: {\"answer\": \"yes\"} or "
            "{\"answer\": \"no\"}. "
            "If you cannot determine the answer, return {\"answer\": \"unknown\"}."
        )
    elif qtype == "mc":
        instruct = (
            "Reply with a JSON object: {\"answer\": \"<short_phrase>\"}. "
            "Use a single short phrase (one or two words). "
            "If you cannot determine the answer, return {\"answer\": \"unknown\"}."
        )
    else:
        instruct = (
            "Reply with a JSON object: {\"answer\": \"<short_text>\"}."
        )
    return f"Question: {q}\n{instruct}"


# -----------------------------------------------------------
# Provider call wrappers
# -----------------------------------------------------------

@dataclass
class JudgeRaw:
    """Raw VLM response, before normalization."""
    answer: str
    raw_text: str
    error: Optional[str] = None
    latency_ms: int = 0


async def _call_openai(client, image_b64: str, question: dict,
                       model: str) -> JudgeRaw:
    user_content = [
        {"type": "text", "text": _user_text(question)},
        {"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{image_b64}",
            "detail": "low",
        }},
    ]
    t0 = time.time()
    last_err: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={"type": "json_object"},
                    max_completion_tokens=512,
                    reasoning_effort="minimal",
                ),
                timeout=PER_CALL_TIMEOUT,
            )
            text = resp.choices[0].message.content or ""
            elapsed_ms = int((time.time() - t0) * 1000)
            try:
                data = json.loads(text)
                ans = str(data.get("answer", "")).strip()
                if not ans:
                    ans = "unknown"
                return JudgeRaw(answer=ans, raw_text=text, latency_ms=elapsed_ms)
            except json.JSONDecodeError:
                # Try regex extraction as fallback
                m = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
                if m:
                    return JudgeRaw(answer=m.group(1).strip(), raw_text=text,
                                    latency_ms=elapsed_ms,
                                    error="json_recovered")
                return JudgeRaw(answer="unknown", raw_text=text,
                                latency_ms=elapsed_ms, error="json_parse_fail")
        except Exception as e:
            last_err = e
            sleep_for = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(sleep_for)
    elapsed_ms = int((time.time() - t0) * 1000)
    return JudgeRaw(answer="unknown", raw_text="",
                    error=f"openai_exception: {last_err!r}",
                    latency_ms=elapsed_ms)


async def _call_gemini(client, image_bytes: bytes, question: dict,
                       model: str) -> JudgeRaw:
    # New google-genai SDK: client.aio.models.generate_content
    from google.genai import types as gtypes
    contents = [
        gtypes.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        gtypes.Part.from_text(text=_user_text(question)),
    ]
    config = gtypes.GenerateContentConfig(
        system_instruction=_SYSTEM,
        response_mime_type="application/json",
        max_output_tokens=128,
        temperature=0.0,
        thinking_config=gtypes.ThinkingConfig(thinking_budget=0),
    )
    t0 = time.time()
    last_err: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model, contents=contents, config=config,
                ),
                timeout=PER_CALL_TIMEOUT,
            )
            text = (resp.text or "").strip()
            elapsed_ms = int((time.time() - t0) * 1000)
            try:
                data = json.loads(text)
                ans = str(data.get("answer", "")).strip()
                if not ans:
                    ans = "unknown"
                return JudgeRaw(answer=ans, raw_text=text, latency_ms=elapsed_ms)
            except json.JSONDecodeError:
                m = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
                if m:
                    return JudgeRaw(answer=m.group(1).strip(), raw_text=text,
                                    latency_ms=elapsed_ms,
                                    error="json_recovered")
                return JudgeRaw(answer="unknown", raw_text=text,
                                latency_ms=elapsed_ms, error="json_parse_fail")
        except Exception as e:
            last_err = e
            sleep_for = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(sleep_for)
    elapsed_ms = int((time.time() - t0) * 1000)
    return JudgeRaw(answer="unknown", raw_text="",
                    error=f"gemini_exception: {last_err!r}",
                    latency_ms=elapsed_ms)


# -----------------------------------------------------------
# Provider clients
# -----------------------------------------------------------

def _make_openai_client(model: str):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Export it before judging with provider=openai. "
            "See the placeholder block at the top of pbh_judge.py.")
    try:
        from openai import AsyncOpenAI
    except ImportError as e:
        raise RuntimeError(
            "Package `openai` is not installed. Run: pip install openai") from e
    return AsyncOpenAI(api_key=api_key)


def _make_gemini_client():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Export it before judging with provider=gemini. "
            "See the placeholder block at the top of pbh_judge.py.")
    try:
        from google import genai
    except ImportError as e:
        raise RuntimeError(
            "Package `google-genai` is not installed. "
            "Run: pip install google-genai") from e
    return genai.Client(api_key=api_key)


# -----------------------------------------------------------
# Task assembly
# -----------------------------------------------------------

@dataclass
class JudgeTask:
    prompt_id: str
    category: str
    sample_idx: int
    image_path: Path
    qid: int
    question: dict
    qhash: str


def _walk_run_dir(run_dir: Path, records: list[dict]) -> list[JudgeTask]:
    """Build the flat task list for a run directory.

    Skips prompts whose sample dir is missing (generation incomplete) and
    samples whose PNG file is missing.
    """
    by_id = {r["id"]: r for r in records}
    tasks: list[JudgeTask] = []

    # Prompt dirs are named {global_idx:05d}
    for prompt_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        if not prompt_dir.name.isdigit():
            continue
        meta_file = prompt_dir / "metadata.json"
        if not meta_file.exists():
            continue
        try:
            with open(meta_file) as f:
                meta = json.load(f)
        except Exception:
            continue
        rid = meta.get("id")
        rec = by_id.get(rid)
        if rec is None:
            continue
        sample_dir = prompt_dir / "samples"
        if not sample_dir.exists():
            continue
        sample_files = sorted(p for p in sample_dir.iterdir()
                              if p.suffix.lower() == ".png")
        for sample_path in sample_files:
            try:
                sample_idx = int(sample_path.stem)
            except ValueError:
                continue
            for qid, question in enumerate(rec["questions"]):
                tasks.append(JudgeTask(
                    prompt_id=rid,
                    category=rec["category"],
                    sample_idx=sample_idx,
                    image_path=sample_path,
                    qid=qid,
                    question=question,
                    qhash=_question_hash(question),
                ))
    return tasks


# -----------------------------------------------------------
# Main entry
# -----------------------------------------------------------

async def _judge_one(task: JudgeTask, run_dir: Path, provider: str,
                     model: str, sem: asyncio.Semaphore,
                     client, force: bool) -> dict:
    cache_file = _cache_path(run_dir, provider, task.prompt_id,
                             task.sample_idx, task.qhash)
    if (not force) and cache_file.exists():
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            # Cache hit only counts if the cached response was produced by
            # the same model — switching judge model invalidates the cache.
            if cached.get("model") == model:
                cached["from_cache"] = True
                return cached
        except Exception:
            pass

    async with sem:
        try:
            if provider == "openai":
                image_b64 = await asyncio.to_thread(_encode_image_b64,
                                                    task.image_path)
                raw = await _call_openai(client, image_b64, task.question, model)
            elif provider == "gemini":
                image_bytes = await asyncio.to_thread(
                    lambda p: open(p, "rb").read(), task.image_path)
                raw = await _call_gemini(client, image_bytes, task.question, model)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:
            raw = JudgeRaw(answer="unknown", raw_text="",
                           error=f"call_exception: {e!r}\n{traceback.format_exc()}",
                           latency_ms=0)

    qtype = task.question["type"]
    accepted_norm = normalized_accepted_set(task.question["accepted"], qtype)
    pred_norm = normalize_answer(raw.answer, qtype)
    correct = pred_norm in accepted_norm

    record = {
        "prompt_id": task.prompt_id,
        "category": task.category,
        "sample_idx": task.sample_idx,
        "qid": task.qid,
        "question": task.question["q"],
        "qtype": qtype,
        "expected": task.question["answer"],
        "accepted": task.question["accepted"],
        "vlm_raw": raw.answer,
        "vlm_text": raw.raw_text,
        "vlm_normalized": pred_norm,
        "correct": correct,
        "provider": provider,
        "model": model,
        "error": raw.error,
        "latency_ms": raw.latency_ms,
        "from_cache": False,
    }

    # Write cache atomically
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(record, f)
    os.replace(tmp, cache_file)
    return record


async def judge_run_async(run_dir: Path, records: list[dict], provider: str,
                          model: Optional[str] = None,
                          force: bool = False) -> list[dict]:
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"provider must be one of {SUPPORTED_PROVIDERS}")

    if provider == "openai":
        model = model or DEFAULT_OPENAI_MODEL
        client = _make_openai_client(model)
        sem = asyncio.Semaphore(OPENAI_CONCURRENCY)
    else:
        model = model or DEFAULT_GEMINI_MODEL
        client = _make_gemini_client()
        sem = asyncio.Semaphore(GEMINI_CONCURRENCY)

    tasks = _walk_run_dir(run_dir, records)
    print(f"[judge:{provider}] {len(tasks)} tasks ({run_dir})")

    try:
        coros = [_judge_one(t, run_dir, provider, model, sem, client, force)
                 for t in tasks]
        if _HAS_TQDM:
            results = await tqdm_asyncio.gather(*coros, desc=f"judge:{provider}")
        else:
            results = await asyncio.gather(*coros)
    finally:
        # Explicitly close the SDK client so its httpx connection pool
        # finishes shutting down before asyncio.run() closes the event loop.
        # Skipping this leaves dangling AsyncClient.aclose() coroutines that
        # raise "Event loop is closed" during GC. Best-effort — different
        # SDK versions expose different shutdown methods.
        for closer in ("aclose", "close"):
            fn = getattr(client, closer, None)
            if fn is None:
                continue
            try:
                result = fn()
                if asyncio.iscoroutine(result):
                    await result
                break
            except Exception:
                pass

    # Persist a single newline-delimited dump for debugging
    dump_path = run_dir / f"judge_responses_{provider}.jsonl"
    with open(dump_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return results


def judge_run(run_dir: Path, records: list[dict], provider: str,
              model: Optional[str] = None, force: bool = False) -> list[dict]:
    """Synchronous wrapper around judge_run_async (for use from eval_pbh.py)."""
    return asyncio.run(
        judge_run_async(run_dir, records, provider, model, force))
