"""Streamlit entry point: UI and per-session workflow state only."""
import hashlib

import streamlit as st

from utils import (DEFAULT_MODEL, StudyError, extract_file, format_content,
                   format_download, format_plan, format_questions,
                   friendly_error, get_client, setting)
from workflow import STAGES, fingerprint, new_state, run_workflow, sources_for, validate_inputs

st.set_page_config(page_title="AI Study Pack Generator", page_icon="📚", layout="wide")


def clear_all():
    for key in ("topic", "notes", "level", "language", "goals", "minutes", "detail", "model",
                "workflow", "input_signature", "reveal_answers", "extracted", "extraction_key"):
        st.session_state.pop(key, None)
    st.session_state["upload_epoch"] = st.session_state.get("upload_epoch", 0) + 1


def example(value, message):
    if not value.strip():
        st.caption("Example: " + message)


st.title("Personalized AI Study Pack Generator")
st.write("Turn a topic or your notes into a plan, explanations, flashcards, and practice questions.")
st.caption("Five AI stages: Planning → Content Generation → Assessment → Review → Refinement")
st.info("Notes are sent to Groq when you generate a pack. AI review checks clarity and source alignment; it is not independent fact verification.")

with st.sidebar:
    st.header("Study preferences")
    level = st.selectbox("Education level", ["Primary school", "Middle school", "O Level / Grade 9-10", "A Level / Grade 11-12", "Undergraduate", "Adult beginner"], index=2, key="level")
    language = st.selectbox("Study pack language", ["English", "Urdu", "Roman Urdu / Hinglish"], key="language")
    minutes = st.slider("Available study time (minutes)", 5, 240, 30, 5, key="minutes")
    detail = st.selectbox("Explanation detail", ["Brief", "Standard", "Detailed"], index=1, key="detail")
    with st.expander("Model settings"):
        model = st.text_input("Groq model name", value=setting("GROQ_MODEL", DEFAULT_MODEL), key="model", help="Use a chat model available to your Groq project with JSON object mode support.")
    st.caption("Results stay in this browser session. Download your pack before closing or refreshing the session.")

topic = st.text_input("What would you like to study?", placeholder="e.g. Newton's laws of motion", max_chars=250, key="topic")
example(topic, "Photosynthesis, Python loops, or O Level space physics")
notes = st.text_area("Paste your notes (optional)", placeholder="Paste the lesson or chapter you want to study...", height=160, max_chars=18000, key="notes")
example(notes, "Force = mass × acceleration. Force is measured in newtons.")
epoch = st.session_state.get("upload_epoch", 0)
uploaded = st.file_uploader("Or upload notes (PDF or TXT)", type=["pdf", "txt"], key=f"upload_{epoch}", help="Selectable-text PDF or UTF-8 TXT. Up to 5 MB, 40 PDF pages, and 18,000 combined text characters.")
if uploaded is None:
    st.caption("Example: Upload a short textbook chapter with selectable text.")
goals = st.text_area("What should you be able to do afterwards? (optional)", placeholder="e.g. Explain each law and solve basic force calculations", max_chars=1500, height=80, key="goals")
example(goals, "Define key terms and answer exam-style questions. Leave blank for suggested goals.")

data = uploaded.getvalue() if uploaded else b""
file_key = (uploaded.name, hashlib.sha256(data).hexdigest()) if uploaded else None
sources, warnings, input_error = [], [], None
if uploaded:
    try:
        if st.session_state.get("extraction_key") != file_key:
            st.session_state["extracted"] = extract_file(uploaded.name, data)
            st.session_state["extraction_key"] = file_key
        sources, warnings = st.session_state["extracted"]
        st.caption(f"Read {sum(len(s['text']) for s in sources):,} text characters from your file.")
    except StudyError as exc:
        input_error = str(exc)
for warning in warnings:
    st.warning(warning)

inputs = dict(topic=topic.strip(), notes=notes.strip(), level=level, language=language,
              goals=goals.strip(), minutes=minutes, detail=detail, model=model.strip(),
              sources=sources, source_warnings=warnings)
# Include even unreadable uploads in the signature: an invalid replacement must
# invalidate the previous pack too, not accidentally keep it visible.
signature = fingerprint({**inputs, "upload_identity": file_key})
if st.session_state.get("input_signature") != signature:
    had_results = bool(st.session_state.get("workflow", {}).get("outputs"))
    st.session_state["workflow"] = new_state(inputs)
    st.session_state["input_signature"] = signature
    st.session_state["reveal_answers"] = False
    if had_results:
        st.info("Your inputs changed. The previous results were cleared. Generate a new pack.")
state = st.session_state["workflow"]
try:
    validate_inputs(inputs)
except StudyError as exc:
    input_error = input_error or str(exc)
if input_error:
    st.info(input_error)

buttons = st.columns(4)
has_outputs = bool(state["outputs"])
generate = buttons[0].button("Generate", type="primary", disabled=bool(input_error) or has_outputs)
regenerate = buttons[1].button("Regenerate", disabled=bool(input_error) or not has_outputs,
                                 help="Discard the current results and run all five stages again.")
retry = buttons[2].button("Retry Failed Stage", disabled=bool(input_error) or not state["failed_stage"])
buttons[3].button("Clear", on_click=clear_all)

if generate or regenerate or retry:
    if regenerate:
        st.session_state["workflow"] = new_state(inputs)
        state = st.session_state["workflow"]
    st.session_state["reveal_answers"] = False
    bar = st.progress(len(state["outputs"]) / 5)
    status = st.empty()

    def show_progress(done, stage, message):
        bar.progress(done / 5)
        status.info(f"{stage}: {message}")

    try:
        client = get_client()
        try:
            run_workflow(client, inputs, state, show_progress)
        finally:
            client.close()
    except Exception as exc:
        state["error"] = friendly_error(exc)
        state["failed_stage"] = next((s for s in STAGES if s not in state["outputs"]), "Planning")
    st.rerun()

if state["error"]:
    st.error(f"{state['failed_stage']} stopped: {state['error']}")
    st.caption("Completed stages are saved. Retry Failed Stage resumes from the first unfinished stage.")
if state["outputs"]:
    st.caption("Completed: " + " · ".join(s for s in STAGES if s in state["outputs"]))

pack = state["outputs"].get("Refinement")
if pack:
    st.success("Your study pack is ready.")
    tabs = st.tabs(["Learning plan", "Study notes", "Flashcards", "Practice questions", "Review & sources"])
    with tabs[0]:
        st.markdown(format_plan(pack["plan"]))
    with tabs[1]:
        st.markdown(format_content(pack["content"]))
    with tabs[2]:
        for i, card in enumerate(pack["content"]["flashcards"], 1):
            st.write(f"**Card {i}:** {card['front']}")
            with st.expander(f"Reveal card {i}"):
                st.write(card["back"])
    with tabs[3]:
        st.markdown(format_questions(pack["assessment"]))
        if st.checkbox("Reveal assessment answers", key="reveal_answers"):
            st.subheader("Answer key")
            st.markdown(format_questions(pack["assessment"], answers=True))
    with tabs[4]:
        st.subheader("Changes after review")
        for change in pack["changes"]:
            st.write("• " + change)
        st.subheader("Remaining limitations")
        for limitation in pack["limitations"]:
            st.write("• " + limitation)
        st.warning("AI review is not independent fact verification. Source references indicate claimed alignment, not proof that a claim is correct.")
        review = state["outputs"]["Review"]
        with st.expander("AI review of the earlier draft"):
            st.write(review["summary"])
            # Issue text can quote answers; hide it until answers are revealed.
            if st.session_state.get("reveal_answers"):
                for issue in review["issues"]:
                    st.write(f"**{issue['category']} — {issue['location']}**: {issue['issue']} Fix: {issue['fix']}")
            else:
                st.caption("Detailed issues may contain answers. Reveal assessment answers to see them.")
        st.subheader("Supplied sources")
        for source in sources_for(inputs):
            with st.expander(f"[{source['id']}] {source['label']}"):
                st.text(source["text"])
        if not sources_for(inputs):
            st.write("No notes supplied. This pack uses model knowledge; consult a reliable textbook.")
    st.caption("The complete Markdown download includes the answer key.")
    st.download_button("Download Study Pack (.md)",
                       data=format_download(pack, inputs, sources_for(inputs), state["outputs"]["Review"]),
                       file_name="study-pack.md", mime="text/markdown", on_click="ignore")
