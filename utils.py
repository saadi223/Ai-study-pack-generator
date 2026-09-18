"""API access, extraction, safe errors, and Markdown export. No workflow imports."""
import json
import os
import math
import re
import time
from email.utils import parsedate_to_datetime
from pathlib import Path

import pymupdf
from groq import Groq, APIConnectionError, APIStatusError, AuthenticationError, RateLimitError

DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_SOURCE_CHARS = 18000
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_PAGES = 40
MAX_RATE_RETRIES = 6
MAX_AUTO_WAIT_SECONDS = 600
REQUEST_PACING_SECONDS = 65


class StudyError(Exception):
    """A user-safe error: never include raw provider messages or credentials."""


class RateLimitPause(StudyError):
    """Carry the provider cooldown into session state without exposing its body."""
    def __init__(self, message, seconds):
        super().__init__(message)
        self.retry_at = time.time() + seconds


def rate_limit_info(exc):
    """Extract only duration and quota category, never echo raw error text."""
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    message = str(error.get("message", "")) if isinstance(error, dict) else ""
    lower = message.lower()
    daily = bool(re.search(r"\b(tpd|rpd)\b|per day|daily", lower))
    category = "daily quota" if daily else "rate limit"
    seconds = None
    raw = exc.response.headers.get("retry-after", "")
    try:
        seconds = float(raw)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(raw).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            pass
    if seconds is None:
        hint = re.search(r"try again in\s+([0-9.dhms\s]+)", lower)
        if hint:
            units = {"d": 86400, "h": 3600, "m": 60, "s": 1}
            matches = re.findall(r"(\d+(?:\.\d+)?)\s*(ms|d|h|m|s)", hint.group(1))
            if matches:
                seconds = sum(float(n) * (0.001 if unit == "ms" else units[unit]) for n, unit in matches)
    known = seconds is not None and math.isfinite(seconds)
    seconds = max(1, math.ceil(seconds) + 1) if known else 60
    limit = re.search(r"\blimit\s*[:=]?\s*([\d,]+)", lower)
    requested = re.search(r"\brequested\s*[:=]?\s*([\d,]+)", lower)
    oversized = bool(limit and requested and int(requested.group(1).replace(",", "")) > int(limit.group(1).replace(",", "")))
    if oversized:
        text = "This single request exceeds Groq's reported quota limit. Waiting alone will not fix it. Use shorter notes and Brief detail, or a model/account with a sufficient limit. Changing inputs starts a new pack."
    elif known:
        text = f"Groq's {category} was reached. Retry after at least {seconds} seconds; completed stages are saved."
    else:
        text = f"Groq's {category} was reached. No reset time was provided. Wait at least 60 seconds before retrying; if it persists, check your Groq account Limits."
    return seconds, text, (not daily and not oversized and seconds <= MAX_AUTO_WAIT_SECONDS)


def wait_with_progress(seconds, notify=None, message="Waiting for Groq"):
    """Bounded, visible countdown. No API requests are sent during this wait."""
    seconds = max(0, math.ceil(seconds))
    for remaining in range(seconds, 0, -1):
        if notify and (remaining == seconds or remaining % 5 == 0):
            notify(f"{message}: continuing automatically in {remaining} seconds. Keep this tab open.")
        time.sleep(1)


def setting(name, default=""):
    """Streamlit Secrets first, environment second; works outside Streamlit too."""
    try:
        import streamlit as st
        value = st.secrets.get(name)
        if value:
            return str(value).strip()
    except (ImportError, FileNotFoundError):
        pass
    except Exception:
        # Missing or malformed secrets must not print secret-file contents.
        pass
    return os.environ.get(name, default).strip()


def get_client():
    key = setting("GROQ_API_KEY")
    if not key:
        raise StudyError("Add GROQ_API_KEY to Streamlit Secrets or your environment first.")
    # SDK retries are disabled: the workflow owns the bounded retry budgets.
    return Groq(api_key=key, timeout=90.0, max_retries=0)


def friendly_error(exc):
    if isinstance(exc, StudyError):
        return str(exc)
    if isinstance(exc, AuthenticationError):
        return "Groq rejected the key. Check GROQ_API_KEY and whether the key is active."
    if isinstance(exc, RateLimitError):
        return rate_limit_info(exc)[1]
    if isinstance(exc, APIConnectionError):
        return "Could not reach Groq, or the request timed out. Check your connection and retry."
    if isinstance(exc, APIStatusError):
        code = exc.status_code
        if code in (400, 404, 422):
            return "Groq rejected this request. Check model availability, JSON-mode support, and input size."
        if code == 403:
            return "Your Groq project does not have access to this model. Check model permissions."
        return "Groq is temporarily unavailable. Please retry later."
    return "The response could not be processed. Retry this stage or choose a smaller detail level."


def retry_delay(exc, attempt):
    """Return a bounded delay, or None for a non-retryable API error."""
    if isinstance(exc, RateLimitError):
        seconds, _, automatic = rate_limit_info(exc)
        return seconds if automatic else None
    if isinstance(exc, APIConnectionError):
        return 2 ** (attempt + 1)
    if isinstance(exc, APIStatusError) and exc.status_code >= 500:
        return 2 ** (attempt + 1)
    return None


def request_json(client, model, system, payload, schema, validate, notify=None,
                 max_tokens=6000):
    """Separate bounded budgets for 429s, malformed output, and transport errors.

    Up to six rate-limit retries, one response repair, and one transport retry.
    At most nine requests and 600 seconds of recovery waits per stage.
    """
    correction = ""
    rate_retries = malformed_retries = transport_retries = 0
    waited = 0
    for attempt in range(MAX_RATE_RETRIES + 3):
        try:
            options = {"reasoning_effort": "low"} if model in ("openai/gpt-oss-120b", "openai/gpt-oss-20b") else {}
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system +
                     "\nReturn ONLY a JSON object matching this JSON schema:\n" +
                     json.dumps(schema, separators=(",", ":")) + correction},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
                ],
                **options,
            )
            if not response.choices or response.choices[0].finish_reason != "stop":
                raise ValueError("The response was incomplete or refused.")
            value = json.loads(response.choices[0].message.content or "")
            return validate(value)
        except (ValueError, TypeError, KeyError) as exc:
            if malformed_retries >= 1:
                raise StudyError("This stage returned an incomplete or invalid response twice. Retry the stage; if it repeats, reduce detail or notes length.") from None
            malformed_retries += 1
            # Validation messages describe structure, never echo a raw response.
            correction = "\nThe previous response failed validation. Check every required field, question ID, goal reference, source reference, and minute total. Keep the result concise."
            # An invalid generated response still used quota. Pace its repair too.
            if waited + REQUEST_PACING_SECONDS > MAX_AUTO_WAIT_SECONDS:
                raise RateLimitPause("Automatic recovery wait budget reached. Completed stages are saved; retry later.", REQUEST_PACING_SECONDS) from None
            wait_with_progress(REQUEST_PACING_SECONDS, notify, "Preparing one response repair")
            waited += REQUEST_PACING_SECONDS
        except RateLimitError as exc:
            seconds, message, automatic = rate_limit_info(exc)
            if (not automatic or rate_retries >= MAX_RATE_RETRIES
                    or waited + seconds > MAX_AUTO_WAIT_SECONDS):
                raise RateLimitPause(message + " Automatic recovery is paused; check your Groq Limits if this repeats.", seconds) from None
            rate_retries += 1
            wait_with_progress(seconds, notify, f"Groq quota cooling down (automatic retry {rate_retries}/{MAX_RATE_RETRIES})")
            waited += seconds
        except (APIConnectionError, APIStatusError) as exc:
            delay = retry_delay(exc, transport_retries)
            if transport_retries >= 1 or delay is None:
                raise StudyError(friendly_error(exc)) from None
            if waited + delay > MAX_AUTO_WAIT_SECONDS:
                raise StudyError("Automatic recovery wait budget reached. Retry this stage later.") from None
            transport_retries += 1
            wait_with_progress(delay, notify, "Reconnecting to Groq")
            waited += delay
    raise StudyError("The request could not be completed.")


def extract_file(name, data):
    """Return (source sections, warnings). Never silently truncate supplied notes."""
    if len(data) > MAX_FILE_BYTES:
        raise StudyError("This file is over 5 MB. Upload a smaller file.")
    suffix = Path(name).suffix.lower()
    sections, warnings = [], []
    if suffix == ".txt":
        try:
            text = data.decode("utf-8-sig").strip()
        except UnicodeDecodeError:
            raise StudyError("Save your TXT file as UTF-8, then upload it again.") from None
        if text:
            sections.append({"label": name, "text": text})
    elif suffix == ".pdf":
        try:
            with pymupdf.open(stream=data, filetype="pdf") as doc:
                if doc.needs_pass:
                    raise StudyError("This PDF is password-protected. Upload an unlocked copy.")
                if len(doc) > MAX_PAGES:
                    raise StudyError("Use a PDF with 40 pages or fewer.")
                empty = []
                for page_number, page in enumerate(doc, 1):
                    text = page.get_text().strip()
                    if text:
                        sections.append({"label": f"{name}, page {page_number}", "text": text})
                    else:
                        empty.append(str(page_number))
                if empty and sections:
                    warnings.append("No text extracted from PDF pages " + ", ".join(empty) + ". These pages were not used; they may be scanned or blank.")
        except StudyError:
            raise
        except Exception:
            raise StudyError("This PDF could not be read. Upload a valid, unlocked PDF.") from None
    else:
        raise StudyError("Upload a selectable-text PDF or a UTF-8 TXT file.")
    if not sections:
        raise StudyError("No extractable text found. This may be a scanned PDF or an empty file. Use OCR first or paste the notes as text.")
    if sum(len(s["text"]) for s in sections) > MAX_SOURCE_CHARS:
        raise StudyError("The file has over 18,000 text characters. Upload a shorter chapter or selected pages.")
    return sections, warnings


def format_plan(plan):
    lines = ["## Learning plan", plan["title"], "", "### Learning goals"]
    lines += [f"- **{g['id']}**: {g['text']}" for g in plan["goals"]]
    lines += ["", "### Study schedule"]
    lines += [f"- **{b['minutes']} minutes**: {b['activity']} ({', '.join(b['goal_ids'])})" for b in plan["schedule"]]
    lines += ["", "### Information gaps"] + [f"- {x}" for x in plan["gaps"]]
    return "\n".join(lines)


def format_content(content):
    lines = ["## Explanations"]
    for section in content["sections"]:
        refs = ", ".join(section["source_ids"]) or "General model knowledge; not independently verified"
        lines += [f"### {section['heading']}", section["explanation"], f"**Example:** {section['example']}", f"*Sources: {refs}*", ""]
    lines += ["## Summary"] + [f"- {x}" for x in content["summary"]]
    lines += ["", "## Key terms"] + [f"- **{t['term']}**: {t['meaning']}" for t in content["key_terms"]]
    return "\n".join(lines)


def format_questions(assessment, answers=False):
    if answers:
        return "\n\n".join(f"**{a['question_id']}: {a['answer']}**\n\n{a['explanation']}" for a in assessment["answer_key"])
    lines = []
    for q in assessment["mcqs"]:
        lines += [f"### {q['id']}. {q['question']}", f"*Goal: {q['goal_id']}*"]
        lines += [f"- {letter}. {option}" for letter, option in zip("ABCD", q["options"])]
    for q in assessment["short_answers"]:
        lines += [f"### {q['id']}. {q['question']}", f"*Goal: {q['goal_id']}*"]
    return "\n\n".join(lines)


def format_download(pack, inputs, sources, review):
    lines = [f"# {pack['plan']['title']}",
             f"Level: {inputs['level']} | Language: {inputs['language']} | Study time: {inputs['minutes']} minutes",
             "AI-generated learning aid. AI review is not independent fact verification.",
             format_plan(pack["plan"]), format_content(pack["content"]), "## Flashcards"]
    lines += [f"**Q:** {c['front']}\n\n**A:** {c['back']}" for c in pack["content"]["flashcards"]]
    lines += ["## Practice questions", format_questions(pack["assessment"]),
              "## Answer key", format_questions(pack["assessment"], answers=True),
              "## Changes after review"] + [f"- {s}" for s in pack["changes"]]
    lines += ["## Remaining limitations"] + [f"- {s}" for s in pack["limitations"]]
    lines += ["## AI review of the draft", review["summary"]]
    lines += [f"- **{i['category']} — {i['location']}**: {i['issue']} Suggested fix: {i['fix']}" for i in review["issues"]]
    lines += ["## Source map"] + [f"- [{s['id']}] {s['label']}" for s in sources]
    if not sources:
        lines.append("No notes supplied. Based on model knowledge; check an authoritative textbook.")
    lines += ["## Extraction warnings"] + [f"- {w}" for w in inputs.get("source_warnings", [])]
    return "\n\n".join(lines) + "\n"
