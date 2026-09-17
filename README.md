# Personalized AI Study Pack Generator

Python + Groq + Streamlit. Beginner-friendly Hinglish setup guide.

## 1. Five-stage workflow aur context

Har stage ek separate Groq request hai. Yeh five-stage sequential AI workflow hai;
five independent experts ya independent fact-checkers nahi hain.

| Stage | Kya karta hai | Context milta hai |
|---|---|---|
| Planning | Topic/notes samajh kar goals aur exact time-budget schedule | Preferences + original sources |
| Content Generation | Explanations, summary, terms, examples, flashcards | Preferences + sources + plan |
| Assessment | MCQs, short questions, separate explained answer key | Preferences + sources + plan + content |
| Review | Clarity, consistency, level, alignment, unsupported claims, gaps | Preferences + original sources + all drafts |
| Refinement | Review apply karke complete final pack | Preferences + sources + all drafts + review |

Pydantic har output ka structure check karta hai. Python checks schedule total,
source IDs, goal coverage, MCQ options, aur question/answer matching bhi validate
karte hain. Yeh factual correctness ka proof nahi hai. Semantic source alignment
AI review karta hai, jis se errors phir bhi ho sakte hain.

Har successful stage session state mein save hota hai. Failure par completed stages
repeat nahi hote. Retry Failed Stage remaining workflow chalata hai. Inputs/model
change karne se **all** stages invalidate hote hain: sab first plan par depend karte
hain. Regenerate bhi sab dobara chalata hai. Clear inputs aur results reset karta hai.

## 2. Project files

Repository root mein exactly yeh six files rakhein:

```text
study-pack-generator/
    app.py
    workflow.py
    utils.py
    requirements.txt
    README.md
    .gitignore
```

| File | Responsibility |
|---|---|
| app.py | Interface, inputs, tabs, session state, buttons, progress |
| workflow.py | Five stage functions, prompts, schemas, validation, resume |
| utils.py | Official Groq client, secrets, limited retries, extraction, export |
| requirements.txt | Four required packages |
| README.md | Complete setup, testing, deployment, troubleshooting |
| .gitignore | Secrets aur local generated files Git se exclude karta hai |

Imports ka direction: `app → workflow → utils`, aur `app → utils`.
`utils.py` kabhi `app.py` ya `workflow.py` import nahi karta.

Complete code teen provided `.py` files mein hai; koi placeholder function nahi hai.
Default model `openai/gpt-oss-120b` hai; sidebar mein ya `GROQ_MODEL` secret se change
kar sakte hain. Selected model ko chat completions + JSON object mode support karna
chahiye. JSON mode ke saath local validation zaroori hai.

## 3. Google Colab setup aur secure API test

### A. Notebook aur files

1. https://colab.research.google.com kholein aur **New notebook** select karein.
2. Left side **Files** icon se `app.py`, `workflow.py`, `utils.py`, aur
   `requirements.txt` upload karein. Upload `/content` mein karein.
3. Har neeche wala code block alag Colab cell mein run karein. App source ko
   notebook cells mein paste karna zaroori nahi; uploaded modules import honge.

```python
%cd /content
%pip install -q -r requirements.txt
```

Agar updated packages ke liye restart message aaye, runtime restart karein, phir
secret-loading cell dobara run karein. Colab files runtime delete hone par lost ho
sakti hain, isliye original six files apne computer/GitHub par rakhein.

### B. API key securely add karein

1. https://console.groq.com/keys par apne account mein API key banayein.
2. Colab left sidebar mein **key icon / Secrets** kholein.
3. New secret ka exact name **GROQ_API_KEY** rakhein.
4. Value mein actual key paste karein aur **Notebook access** enable karein.
5. API key ko notebook code, output, screenshot, GitHub ya chat mein paste na karein.

```python
import os
from google.colab import userdata

try:
    key = userdata.get("GROQ_API_KEY")
    if not key:
        raise ValueError("Empty secret")
    os.environ["GROQ_API_KEY"] = key
    del key
    print("Secret loaded securely.")
except Exception:
    print("Check the secret name GROQ_API_KEY and enable Notebook access.")
```

API test cell: sirf status print hoga, key aur raw API errors nahi.

```python
from utils import get_client, friendly_error

client = None
try:
    client = get_client()
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "Reply with OK only."}],
        max_completion_tokens=256,
    )
    print("API connection successful." if response.choices else "No response received.")
except Exception as exc:
    print(friendly_error(exc))
finally:
    if client is not None:
        client.close()
```

Yeh connectivity test hai; complete JSON workflow test next section mein hai.
`GROQ_API_KEY` authentication key hai; model name alag setting hai.

## 4. Complete workflow test in Colab

Uploaded `app.py`, `workflow.py`, `utils.py` complete executable source files hain.
Files edit karne ke baad Colab runtime restart aur secret-loading cell re-run karein,
taake cached imports purana code na chalayein.

### A. Small end-to-end example

```python
from workflow import new_state, run_workflow, sources_for
from utils import get_client, format_download, friendly_error

inputs = {
    "topic": "Newton's second law",
    "notes": "Force equals mass multiplied by acceleration: F = ma. "
             "Force is measured in newtons, mass in kilograms, and acceleration "
             "in metres per second squared. A 2 kg object accelerating at "
             "3 metres per second squared requires a net force of 6 N.",
    "sources": [],
    "source_warnings": [],
    "level": "O Level / Grade 9-10",
    "language": "English",
    "goals": "Explain F = ma and solve a simple force calculation.",
    "minutes": 20,
    "detail": "Brief",
    "model": "openai/gpt-oss-120b",
}
state = new_state(inputs)
client = None
try:
    client = get_client()
    run_workflow(
        client, inputs, state,
        progress=lambda done, stage, message: print(f"{done}/5 | {stage}: {message}"),
    )
    if state["error"]:
        print("Stopped:", state["failed_stage"], state["error"])
    else:
        print("All five stages completed:", list(state["outputs"]))
except Exception as exc:
    print(friendly_error(exc))
finally:
    if client is not None:
        client.close()
```

### B. Failed stage resume

Rate limit ho to pehle wait karein; phir yeh cell chalayein. `state = new_state(...)`
dobara **na** run karein; usse completed progress reset ho jayegi.

```python
client = None
try:
    client = get_client()
    run_workflow(client, inputs, state,
                 progress=lambda done, stage, message: print(stage, message))
    print(state["error"] or "Study pack ready.")
except Exception as exc:
    print(friendly_error(exc))
finally:
    if client is not None:
        client.close()
```

### C. Markdown download

```python
from pathlib import Path
from google.colab import files

if "Refinement" in state["outputs"]:
    pack = state["outputs"]["Refinement"]
    markdown = format_download(pack, inputs, sources_for(inputs), state["outputs"]["Review"])
    Path("study-pack.md").write_text(markdown, encoding="utf-8")
    files.download("study-pack.md")
else:
    print("Complete the failed stage before downloading.")
```

### D. Test PDF/TXT extraction (optional)

```python
from google.colab import files
from utils import extract_file

uploaded = files.upload()
if uploaded:
    filename = next(iter(uploaded))
    try:
        sections, warnings = extract_file(filename, uploaded[filename])
        inputs["sources"] = sections
        inputs["source_warnings"] = warnings
        inputs["notes"] = ""  # use uploaded notes instead of the earlier sample
        print("Text extracted:", sum(len(s["text"]) for s in sections), "characters")
        for warning in warnings:
            print(warning)
        print("Run the resume cell: changed inputs automatically reset the workflow.")
    except Exception as exc:
        print(friendly_error(exc))
```

Colab mein hum Python workflow test kar rahe hain. Streamlit UI ko ordinary Colab
cell output mein interactive website ki tarah render nahi kiya ja sakta. Final UI
Streamlit Community Cloud par test karein, ya local computer par:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Local run ke liye project folder ke andar `.streamlit/secrets.toml` banayein aur
Section 7 ka TOML add karein. Yeh local secret file six committed files ka part
nahi hai; `.gitignore` ise exclude karta hai.

### E. UI acceptance checks

| Test | Expected result |
|---|---|
| No topic, notes, or file | Helpful message; Generate disabled |
| Topic only | Complete pack with model-knowledge limitation |
| Notes only | Topic inferred; content tied to source IDs |
| Partial notes / unrelated requested goal | Unsupported scope appears as a gap |
| Type in an empty field, then click outside it | Example below the field disappears on rerun |
| Clear that field and click outside | Example returns |
| Scanned/blank PDF | No-extractable-text explanation; generate disabled |
| PDF with some empty pages | Warning naming omitted pages; extracted pages usable |
| Invalid key or inaccessible model | Friendly error without the key or raw provider text |
| Failure halfway through | Completed stages retained; Retry resumes remaining stages |
| Change topic, language, file, time, or model | Old results cleared |
| Regenerate | All five stages run again |
| Practice questions | Answers absent until Reveal assessment answers checked |
| Download | UTF-8 Markdown with questions, answers, source map, review, limitations |
| Clear | Inputs, upload, and results reset |

Text input examples update on Streamlit's normal rerun (usually Enter or blur),
not on every keystroke. The app deliberately does not use a form, which would
delay updates until form submission.

## 5. GitHub par upload

1. https://github.com par sign in karein.
2. **New repository** select karein; name `ai-study-pack-generator` rakhein.
3. Repository create karein. Public/private apni preference ke mutabiq choose karein.
4. **Add file → Upload files** mein six provided files upload karein.
5. Files repository ke root mein hon. `app.py` kisi unnecessary nested folder mein
   na ho, warna deployment main-file path change karna padega.
6. Agar computer `.gitignore` hidden rakhta hai, GitHub **Add file → Create new
   file** se exact `.gitignore` name banayein aur provided content paste karein.
7. Commit message `Add study pack generator` likh kar **Commit changes** karein.
8. Verify karein: teen `.py` files aur three supporting files nazar aani chahiye.

API key, `.streamlit/secrets.toml`, personal notes, ya key-containing notebook
upload na karein. `.gitignore` already-committed secrets remove nahi karta. Agar
key accidentally commit ho gayi ho, Groq console se revoke/rotate karein.

## 6. Streamlit Community Cloud deployment

1. https://share.streamlit.io par sign in karein aur GitHub connect karein.
2. **Create app / Deploy an app** choose karein.
3. Repository `ai-study-pack-generator` select karein.
4. Correct branch select karein, usually `main`.
5. **Main file path = `app.py`** set karein.
6. **Advanced settings** mein Python **3.11** choose karein, if offered.
7. Secrets box mein Section 7 ka configuration add karein.
8. **Deploy** karein. Cloud `requirements.txt` se packages install karega.
9. App open hone ke baad pehle Brief + 20 minutes + short topic se test karein.
10. Generated app URL share kar sakte hain. Code updates GitHub mein commit karne
    par Community Cloud repository changes pick up karta hai; deploy logs check karein.

Cloud app par requests aapke server-side Groq key/quota use karti hain. Session
state users ke liye separate hai; key UI mein expose nahi hoti. Is demo mein
authentication ya per-user quota controls implemented nahi hain.

## 7. Streamlit Secrets aur troubleshooting

Deployment ke **Advanced settings → Secrets**, ya deployed app ke **Settings →
Secrets**, mein yeh TOML paste karein. Placeholder ki jagah actual key **sirf
Secrets editor mein** add karein:

```toml
GROQ_API_KEY = "paste-your-actual-groq-key-here"
GROQ_MODEL = "openai/gpt-oss-120b"
```

Double quotes use karein. `GROQ_MODEL` optional hai. Save karein; zaroorat par app
reboot karein. Key ka exact name har environment mein `GROQ_API_KEY` hi hai.

| Issue | Fix |
|---|---|
| Colab secret load failed | Exact name check karein; Notebook access ON karein |
| Missing GROQ_API_KEY | Cloud Secrets add/save karein; Colab mein secret-loading cell run karein |
| Authentication error | Key active hai? Correct Groq project ki key use karein |
| Model request rejected | Model ID/permissions/JSON support verify karein; shorter inputs try karein |
| Rate limit | Wait; quota check; Brief detail aur smaller notes use; Retry Failed Stage |
| Connection timeout | Network/provider availability check; retry later |
| Invalid response twice | Retry stage; persistent ho to Brief detail ya supported model use |
| ModuleNotFoundError | All six files same root mein; requirements install; local module names exact |
| Cannot import app/module | Entry point app.py; circular imports introduce na karein |
| Scanned PDF | OCR first; selectable-text PDF/TXT upload ya text paste karein |
| UTF-8 error | Text editor mein Save As UTF-8 karein |
| Notes too long | 18,000 combined characters tak reduce; chapter-wise packs banayein |
| Cloud says no main file | GitHub path/capitalization check; main path app.py |
| Old output after code edits in Colab | Runtime restart; files still exist check; reinstall/load secret again |
| Results gone after refresh | Session memory persistent storage nahi; Markdown download karein |

## Design limits and security

- Exactly three application Python files. Pydantic provides nested output
  validation; PyMuPDF extracts selectable PDF text. No vector database or LangChain.
- Each stage uses at most **two API calls total**, including malformed responses,
  transient failures, and short rate-limit retries. SDK retries are disabled.
  Normal run: five calls. Worst case per run: ten. Manual retries are user-triggered.
- No silent note truncation. Max 5 MB/file, 40 PDF pages, 18,000 combined characters.
  Image-only parts, charts, and complex PDF reading order may not extract faithfully,
  even if a page also contains some text. OCR is not included.
- Prompt injection protection treats uploaded text as data in a separate JSON
  payload, never as system instructions. This reduces risk but cannot guarantee
  a model will never follow malicious text. No model tools or code execution exist.
- Source IDs are structurally checked; supporting meaning and factual truth are
  not deterministically verified. Review is the same model checking its earlier work.
- Session results persist across ordinary UI reruns, not server restarts/session
  disconnects. No database, disk-based user history, or shared result cache.
- Answer key is hidden in the UI until reveal. Full Markdown intentionally includes
  it. Study explanations and flashcards teach the same concepts as the questions.
- Notes, goals, and generated stages are sent to Groq. No API key is inserted into
  prompts, output, or custom error messages.

## Official references

- [Groq Python SDK](https://github.com/groq/groq-python)
- [Groq JSON / structured outputs](https://console.groq.com/docs/structured-outputs)
- [Groq supported models](https://console.groq.com/docs/models)
- [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
- [Streamlit Secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)

Dependency ranges permit compatible updates; this is a starter project, not a
fully locked reproducible environment. See the accompanying delivery message for
which tests were actually run. A live Groq test requires your own active key.
