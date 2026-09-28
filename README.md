# Agentic — AI course labs

## Setup

From the repo root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run `source .venv/bin/activate` again in each new terminal session. You can tell it worked because your prompt gains a `(.venv)` prefix — that means `python` and `pip` now point inside the venv instead of at your system Python.

For lab 2, also start Ollama and pull the two models the scripts ask for:

```bash
ollama serve                 
ollama pull mistral          # used by musicChatbot.py
ollama pull phi3             # used by gradioMusicChatbot.py
ollama list                  # check that the ollama models are pulled correctly
```

## Getting a Hugging Face token

1. Log in at [huggingface.co](https://huggingface.co).
2. Click your profile picture -> **Settings** -> **Access Tokens**.
3. Click **Create new token**, pick the **Read** type, and give it a name (e.g. `agentic-class`).
4. Copy the token right away. It starts with `hf_` and is only shown once. If you lose it, delete it and create a new one.
5. In the repo root, add your token to the `.env`, based on teh `.env.example`.

The scripts call `load_dotenv()`, so they pick it up automatically. `.env` is in `.gitignore`, so it won't be committed. Never paste the token into code, chat, or anything you submit. If it leaks, delete it on the Access Tokens page and make a new one.

## Adding a package

Install it, then record it so the next person gets it too:

```bash
pip install <package>
pip freeze > requirements.txt
```
