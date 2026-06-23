# Deploying DocuMind

## TL;DR
- **Vercel: no.** It's for frontends/light serverless functions. DocuMind ships
  PyTorch + a ~1 GB model, exceeds Vercel's function size/memory/time limits, and
  needs persistent storage. Wrong tool.
- **GitHub: yes, do this first** (5 min, zero risk — your portfolio asset).
- **Live demo: Hugging Face Spaces (free)**, using the included `Dockerfile` +
  `app_gradio.py`. Render/Railway/Fly also work, but their free tiers (512 MB RAM)
  will OOM on torch — use a paid tier there.

---

## Step 1 — Push to GitHub

```bash
cd DocuMind                      # the project folder

git init
git add .
git commit -m "DocuMind: local RAG document assistant (FAISS + flan-t5)"
```

Create an EMPTY repo on github.com named `DocuMind` (no README/license — you
already have them). Then:

```bash
git branch -M main
git remote add origin https://github.com/Sweathanath132/DocuMind.git
git push -u origin main
```

> `storage/`, `__pycache__/`, and venvs are already in `.gitignore`, so you won't
> commit the index or model cache.

If `git push` asks for a password, GitHub no longer accepts your account password —
use a **Personal Access Token**: GitHub → Settings → Developer settings → Personal
access tokens → Tokens (classic) → generate one with `repo` scope, and paste it as
the password.

---

## Step 2 — Live demo on Hugging Face Spaces (free)

1. Create a free account at https://huggingface.co
2. Click **New Space** → name it `documind` → **SDK: Docker** → **CPU basic (free)**.
3. Push your code into the Space (it's a git repo):

```bash
# from the DocuMind folder
git remote add space https://huggingface.co/spaces/<your-username>/documind
git push space main
```

4. Make sure the Space's `README.md` starts with the Docker frontmatter so HF runs
   it on port 7860. Easiest: open the Space → Files → README.md → edit → paste the
   contents of **`hf_space_README.md`** (included in this folder) → commit.

5. HF builds the image and gives you a public URL like
   `https://<username>-documind.hf.space`. First query is slow (downloads the model
   once, then cached). Upload a doc, ask a question — done.

> Put that live URL on your resume/LinkedIn and in the repo description.
> "Deployed end-to-end" is exactly the line recruiters reward.

---

## Step 3 — Test it locally before you deploy (recommended)

```bash
pip install -r requirements.txt
python app_gradio.py            # open http://127.0.0.1:7860
```
Upload `sample_docs/sagility_facts.txt`, build the index, and ask
"How much revenue did Sagility report in FY25?" — you should get an answer with a
cited source passage. Once this works locally, the Space will work too.

---

## What about Render / Railway / Fly.io?
All three deploy the included `Dockerfile` directly. The model needs ~1.5–2 GB RAM
to load, so pick a paid instance (≥2 GB). Free tiers will crash with out-of-memory.
For a free demo, Hugging Face Spaces is the better choice.
