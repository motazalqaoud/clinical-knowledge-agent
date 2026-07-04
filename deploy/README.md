# Deploying to a Hugging Face Space

This folder holds a ready-to-use Space README (`space_README.md`) with the
YAML front matter Spaces require. It's kept separate from the repo's main
`README.md` so the two don't collide — this repo needs a GitHub-style
README, a Space needs one with `sdk: gradio` front matter at the very top.

## Steps (run these yourself — this requires your own Hugging Face login
## and network access, which this build environment does not have)

1. On huggingface.co, click your profile icon → **New Space**.
   - SDK: **Gradio**
   - Hardware: CPU basic (free tier) is enough to start
   - Visibility: your choice
2. Hugging Face gives you a git remote URL, e.g.
   `https://huggingface.co/spaces/<your-username>/<space-name>`.
3. From your local clone of this repo:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main:main
   ```
   You'll be prompted for your Hugging Face username and an access token
   (create one at https://huggingface.co/settings/tokens if you don't have
   one) — use the token as the password.
4. Replace the Space's `README.md` with this folder's `space_README.md`
   (it needs to be the Space's actual `README.md`, not live in a `deploy/`
   subfolder):
   ```bash
   cp deploy/space_README.md README.md
   git add README.md
   git commit -m "Add Space front matter for Hugging Face deployment"
   git push space main:main
   ```
   Do this on a throwaway local copy or a dedicated deploy branch — don't
   overwrite the GitHub-facing `README.md` on your main branch.
5. The Space builds automatically, installs `requirements.txt`, and
   downloads `all-MiniLM-L6-v2` and `flan-t5-large` from the Hub itself
   the first time you click "Ingest" / ask a question — no local download
   needed.
