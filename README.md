# Agentic — AI course labs

## Setup

From the repo root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run `source .venv/bin/activate` again in each new terminal session. You can tell
it worked because your prompt gains a `(.venv)` prefix — that means `python` and
`pip` now point inside the venv instead of at your system Python.

For lab 2, also start Ollama and pull the two models the scripts ask for:

```bash
ollama serve                 # leave running in its own terminal
ollama pull mistral          # used by musicChatbot.py
ollama pull phi3             # used by gradioMusicChatbot.py
```

## Adding a package

Install it, then record it so the next person gets it too:

```bash
pip install <package>
pip freeze > requirements.txt
```
