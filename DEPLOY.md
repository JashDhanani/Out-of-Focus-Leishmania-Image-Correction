# Deploying the demo (Hugging Face Spaces, free)

Everything the demo needs is self-contained in `webapp/`:

```
webapp/
├── app.py                              # Gradio UI + inference
├── model.py                            # standalone generator (torch only)
├── requirements.txt
├── README.md                           # Space metadata (sdk: gradio, ...)
├── checkpoints/leishmania_generator.pt # ~23MB, Gs weights only (fp16)
└── examples/*.jpg                      # sample out-of-focus images
```

## 1. Create a free Hugging Face account
https://huggingface.co/join -- no payment info required.

## 2. Create a new Space
Go to https://huggingface.co/new-space and fill in:
- **Space name**: anything, e.g. `out-of-focus-correction`
- **License**: MIT (or your choice)
- **SDK**: **Gradio**
- **Hardware**: **CPU basic** (free tier -- this is enough; the model is small)

Click **Create Space**.

## 3. Upload the files
The Space needs the *contents* of `webapp/` at its **root** (not nested in a
`webapp/` folder) -- `app_file: app.py` in the README's metadata is resolved
relative to the Space root.

Easiest way (no git/git-lfs needed): open your new Space's **Files** tab in
the browser and drag in everything from inside `webapp/`:
`app.py`, `model.py`, `requirements.txt`, `README.md`,
`checkpoints/leishmania_generator.pt`, and the `examples/` folder.

(If you prefer git: `git clone` the Space's repo URL shown on its page, copy
the contents of `webapp/` into it, `git add -A && git commit && git push`.
Hugging Face auto-detects the `.pt` file needs Git LFS and will prompt you;
run `git lfs install` first if you go this route.)

## 4. Wait for the build
The Space rebuilds automatically after upload (~1-2 minutes: installs
`requirements.txt`, then launches `app.py`). Watch the **Logs** tab if
anything fails -- usually a missing/misnamed dependency.

## 5. Test it
Once it says "Running", open the app tab: upload an image or click a sample,
hit **Correct image**, and the result appears next to it with a built-in
download icon (top-right corner of the output image).

## Notes
- **Gradio's version is pinned in `README.md`** (`sdk_version: 5.27.0`), not
  `requirements.txt` -- Spaces provisions the SDK itself from that field.
  `requirements.txt` only lists the extra packages `app.py`/`model.py` need
  (torch, pillow, etc). If you ever change `sdk_version`, check it isn't
  in the 4.26-4.44.x range: those releases have a
  [known bug](https://github.com/gradio-app/gradio/issues/11084) that
  crashes every request with `TypeError: argument of type 'bool' is not
  iterable` -- which is what happened during this project's first deploy
  attempt.
- **Free tier caveat**: CPU Spaces sleep after a period of inactivity and
  cold-start (~30s) on the next visit. This is fine for a learning project;
  there's no cost either way.
- **Keeping GitHub as your source of truth**: nothing stops you from also
  pushing this whole repo to GitHub for version history/portfolio purposes --
  Spaces and GitHub are just two different git remotes for the same code.
  (Hugging Face also supports linking a Space directly to a GitHub repo via
  Settings -> "Sync with GitHub" if you want push-to-deploy from there instead
  of uploading manually.)
- If you retrain and want to update the live demo, re-run
  `scripts/export_generator.py` against the new checkpoint and re-upload
  `checkpoints/leishmania_generator.pt` -- the Space rebuilds automatically.
