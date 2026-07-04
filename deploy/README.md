# Deploying to a Hugging Face Space

This folder holds a ready-to-use Space README (`space_README.md`) with the
YAML front matter Spaces require. It's kept separate from the repo's main
`README.md` so the two don't collide — this repo needs a GitHub-style
README, a Space needs one with `sdk: gradio` front matter at the very top.

## Option A: automatic sync (recommended once set up)

`.github/workflows/sync-to-huggingface.yml` pushes this repo to your
Space automatically on every push to `main` — no more manual
copy-paste. It needs one one-time manual step from you (this requires
your Hugging Face account, which I can't access):

1. Create a Hugging Face access token with **write** permission at
   https://huggingface.co/settings/tokens.
2. On GitHub, go to this repo's **Settings → Secrets and variables →
   Actions → New repository secret**.
3. Name it `HF_TOKEN`, paste the token as the value, save.

That's it — every push to `main` now syncs to
`https://huggingface.co/spaces/motazalqaoud/clinical-knowledge-agent`
automatically, swapping in this folder's `space_README.md` as the
Space's front matter. If the secret isn't set, the workflow just skips
the sync step (it won't fail your other CI checks).

You can also trigger it manually anytime from the **Actions** tab →
**Sync to Hugging Face Space** → **Run workflow**, without waiting for
a push.

## Option B: manual (steps below — no GitHub secret needed)

Run these yourself — this requires your own Hugging Face login and
network access, which this build environment does not have.

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
   downloads `all-MiniLM-L6-v2` and `Qwen2.5-1.5B-Instruct` from the Hub itself
   the first time you click "Ingest" / ask a question — no local download
   needed.
