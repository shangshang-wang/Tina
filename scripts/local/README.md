# Local Machine Scripts

Scripts under `scripts/local/` are launch helpers for the current machine setup.
They may encode assumptions about GPU count, GPU IDs, local checkpoint paths,
offline cache behavior, or queue/log directories.

Treat these as examples or machine-specific runbooks, not portable Tina entry
points. Portable training, evaluation, and analysis logic should stay under
`scripts/train/`, `scripts/eval/`, `scripts/data/`, `scripts/analysis/`, and
`tina/`.

These scripts do not include generated data, model weights, rollout outputs, or
logs. Those artifacts are ignored by `.gitignore`.

See `ENVIRONMENT.md` for the package/build notes for this specific machine.
