# Runbook: the validation campaign on a GPU machine

This runbook is for whoever runs the campaign on the GPU machine, a person or an agent
working in a loop. The work is fixed in advance as the steps of `ml/training/campaign.py`:

1. Train hyperparameter set `trial_55` on the five random splits (`random_42`, `random_101`,
   `random_143`, `random_404`, `random_666`).
2. Evaluate it, and select between `trial_55` and Leon's `trial_0` by mean validation loss.
3. With the selected set: train the out-of-sample split (`oos_styryl`) and a model on all
   data (`full`), evaluate the out-of-sample run, and record the selected set's five
   random-split models as the ensemble.

If `trial_0` is selected, its existing models stay as they are and only the out-of-sample
and full models are trained with it. Nothing is ever retrained or overwritten: the driver
skips finished steps and refuses to replace a model file.

## Rules

- **Scope.** Work only inside `ml/`. Do not open, read, list or search anything else in the
  repository. `pyproject.toml` and `uv.lock` are used by `uv`, not read.
- **Only the driver.** Training, evaluation, selection, commits and pushes happen only through
  `uv run --extra ml python -m ml.training.campaign ...`. Do not run `ml.training.train`,
  `evaluate`, `select_hyperparameters` or `test_predictions` directly, and do not write your
  own scripts.
- **Change nothing that defines the results.** Never change the hyperparameter values, the
  training settings (epochs, batch size, learning rate and its schedule, early stopping,
  seeds), the dataset, the split files, the selection rule, or any existing result,
  checkpoint or log. Never delete anything. The driver writes all results itself.
- **Minor fixes are allowed when a step cannot run.** If a step fails because of this
  machine (paths, library or driver versions, device handling) or because of a bug in the
  plumbing of `ml/training/` or `ml/model/`, you may make the smallest change inside `ml/`
  that lets it run, without touching anything in the rule above. For every such change:
  1. append an entry to `ml/results/logs/modifications.log`: date, the file and lines
     changed, the error that made it necessary, and what the change does;
  2. commit the change and that entry together, on their own, with a message saying what
     was fixed and why, and push;
  3. name it in your next report.
  The driver refuses to run while a changed file outside `ml/results/` is uncommitted, so a
  fix has to be committed before the step is retried. If a failure cannot be fixed within
  these limits, or a step fails twice after a fix, stop and report.
- **Git.** Use git through `campaign sync` and `campaign run ... --commit --push`, and by
  hand only to commit and push a fix as described above. Read-only `git status`, `git log`
  and `git diff` are allowed. Never amend, rebase, reset, revert, stash or force-push.
- **No attribution.** No commit, file or message may mention an AI, an assistant or a tool
  that did the work. The driver writes the commit messages; do not add trailers.
- **Environment.** `uv sync --extra ml` is the only install. Nothing else gets installed or
  upgraded.
- **Otherwise, on any error, stop.** If a run fails and no minor fix applies, a step stays
  unfinished after its process has ended, the driver refuses something you do not
  understand, or anything is unexpected, stop the loop and report what happened.

## Setup (once, by the person)

```
git pull
git sparse-checkout set ml        # leaves only ml/ and the root files on disk
uv sync --extra ml
uv run --extra ml python -m ml.training.campaign check
```

`check` prints the torch version, the GPUs torch sees and the last commit, and confirms that
no file outside `ml/results/` has been changed. Pushing needs git credentials for the
repository on this machine. `git sparse-checkout disable` brings the other folders back.

## The loop

Each iteration:

1. `campaign sync` brings in commits pushed from elsewhere. If it fails, stop and report.
2. `campaign status` lists every step as `done`, `running`, `ready` or `blocked`.
3. If every step is `done`, write the final report (below) and end the loop.
4. For a `running` step, look at the end of `ml/results/logs/<step>.log`. Epoch lines mean
   it is progressing. If its process has ended but the step is not `done`, read the log:
   apply a minor fix if the rules allow one, otherwise stop and report.
5. For `ready` steps, `campaign gpus` lists the free GPUs (no compute process, little memory
   in use). Start one ready step per free GPU, never two steps on one GPU, in the background:

   ```
   nohup uv run --extra ml python -m ml.training.campaign run <step> --device cuda:<i> --commit --push > /dev/null 2>&1 &
   ```

   The step writes its own log, commits its results and pushes when it finishes. Steps that
   need no GPU for long (`evaluate_*`, `select`, `ensemble`) run the same way, on any GPU
   or with `--device cpu`. If `CUDA_VISIBLE_DEVICES` is set, `cuda:<i>` counts only the
   GPUs it lists; check with `campaign check` that the index means what you intend. If the
   machine assigns GPUs through a job scheduler, submit each step as a job that runs exactly
   the command above with one GPU instead.
6. If nothing is ready to start because all GPUs are busy, or steps are running, wait and
   check again later. A training run takes hours; checking every 20 to 30 minutes is enough.

The order is enforced by the driver: the five `train_trial_55_*` steps can run in parallel;
`evaluate_trial_55` follows them, then `select`; after `select`, `train_oos_styryl`,
`train_full` and `ensemble` can run in parallel, and `evaluate_selected` follows
`train_oos_styryl`.

## Reports

After `select` is done, report which set was selected, from
`ml/results/hyperparameter_selection.csv`.

At the end, report:
- the selected set and the mean validation loss of both sets
  (`ml/results/hyperparameter_selection.csv`),
- the test metrics of the selected set (`ml/results/<set>/metrics_random_splits.csv` and the
  `oos_styryl` rows of `ml/results/<set>/metrics.csv`),
- the number of epochs of the full model (its log),
- every entry of `ml/results/logs/modifications.log`, if there is one,
- that every step is `done` and `git status` shows nothing uncommitted.

Quote numbers from those files as they are; do not compute or round anything yourself.
