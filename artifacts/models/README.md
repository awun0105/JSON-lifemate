# models/

Trained models (SPEC 6.2.6). Weights never go into git; they live on Kaggle Models or a private HF repo.

- `registry.csv`: one row per trained model: `name, base, data_version, checkpoint, val_seen, val_unseen, location`.
  Appended by `python -m lifemate.train.llm_pipeline` (stage `register`).
- `<name>/train_config.json`: full train config, `code_version`, data sha256, seed, GPU and package versions,
  chosen checkpoint and its eval_unseen loss.
- `<name>/metrics.jsonl`: train and eval loss per logging step.
- `<name>/eval.json`: agent scores of the candidate checkpoints on val_seen / val_unseen, with run folders.

Name format: `ft-<base>-sft<v>-r<nn>`, e.g. `ft-base4b-sft1-r01`.
