---
title: Submitting batch jobs well
tags: [hpc, slurm, cluster, workflow, checklist]
created: 2026-09-07
---

# Submitting batch jobs well

Notes written after losing a 24 h run and three submissions to avoidable
mistakes. Every rule here has a scar behind it — see [[Cluster postmortem]].

The theme: **a batch job fails silently by default.** You are not at the
keyboard, `squeue` only shows live jobs, and a job that dies in one second looks
exactly like a job that was never submitted. Everything below is about making
failures loud and early.

---

## 1. Never trust that it queued

`squeue` shows *running and pending* jobs. A job that failed at startup is gone
before you can type. The audit log is `sacct`.

```bash
sbatch --array=0-5 job.sbatch          # prints: Submitted batch job 541974
squeue -u $USER                        # live only
sacct -j 541974 --format=JobID,State,ExitCode,Elapsed,Start,End
sacct -X --starttime today --format=JobID,JobName,State,ExitCode,Elapsed
```

`-X` collapses the `.batch`/`.extern` sub-steps. Exit codes worth memorising:

| ExitCode | Meaning | Usual cause |
|---|---|---|
| `0:0` | success | — |
| `127:0` | command not found | interpreter not on `PATH`, module not loaded |
| `126:0` | cannot execute binary | wrong architecture — a venv copied between machines |
| `1:0` | script exited 1 | your code, or `set -e` firing early |
| `0:53` / `0:2` | could not open output file | the `--output` directory does not exist |
| `CANCELLED` | killed externally | QOS limit, `--time` over the partition cap, OOM |

> [!warning] `Elapsed 00:00:00` with `FAILED`
> The job never really started. Look at §2 and §3 before anything else — your
> code was not even reached.

---

## 2. Directories SLURM needs must exist *before* submission

`#SBATCH --output=logs/%A_%a.out` is opened by SLURM **before your script runs**.
A `mkdir -p logs` inside the script is too late — the job dies with no output and
no trace, which reads exactly like the submission being ignored.

```bash
#SBATCH --output=logs/%A_%a.out    # <- opened before line 1 of the body
...
mkdir -p logs                      # <- far too late, never reached
```

Two fixes, use both:

```bash
# 1. keep the directory in version control (git ignores empty dirs)
mkdir -p logs && touch logs/.gitkeep && git add -f logs/.gitkeep
```

```bash
# 2. submit through a wrapper that guarantees the preconditions
cat > submit.sh <<'SH'
#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs outputs
sbatch --array="${1:-0-5}" job.sbatch
SH
chmod +x submit.sh
```

---

## 3. Environments are platform-specific — never copy them

A virtualenv contains a real interpreter binary. Copying one from a laptop to a
cluster replaces a working environment with a foreign binary.

- macOS venv on a Linux node → `cannot execute binary file` (**126**)
- venv built against a module-provided Python, module not loaded → `python: command not found` (**127**)

Build it once on the cluster, and exclude it from every sync:

```bash
rsync -avz \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude 'outputs/' \
    --exclude '*.pt' \
    ./ user@cluster:project/
```

Add `.venv/` to `.gitignore`. Excluding `outputs/` matters just as much — pushing
a stale local copy can overwrite results computed on the cluster.

Then call the interpreter **by absolute path** rather than trusting `PATH`, and
check it before anything expensive:

```bash
source .venv/bin/activate
PYTHON="${PYTHON:-$PWD/.venv/bin/python}"
if ! "${PYTHON}" -V >/dev/null 2>&1; then
    echo "FATAL: ${PYTHON} does not run on $(hostname)." >&2
    echo "       -> $(readlink -f "${PYTHON}" 2>/dev/null || echo missing)" >&2
    exit 1
fi
echo "python: ${PYTHON} ($("${PYTHON}" -V 2>&1))"
```

---

## 4. Test on a *compute* node, not the login node

The login node often has modules, filesystems and CPU features the compute nodes
lack. A five-minute interactive job costs nothing and catches almost everything:

```bash
srun -p normal --time=00:10:00 --pty bash        # interactive shell
# or a one-shot smoke test:
srun -p normal --time=00:10:00 bash -c \
  './.venv/bin/python train.py --config configs/big.yaml --dry-run'
```

Give the real entry point a `--dry-run` that builds everything and stops before
the first expensive step. It exercises the whole import path, the config parsing
and the data setup in seconds.

---

## 5. Budget the wall clock from a *measured* rate

Estimating from a laptop is how you request 24 h for a 56 h job. Measure once,
then extrapolate:

```bash
grep "i=" logs/prev_run.out | tail -20     # timestamps + iteration numbers
```

```
04:36:59 ... i=38000
05:21:25 ... i=40000        -> 2666 s / 2000 it = 1.33 s/it
```

Then `total_iterations × rate`, and add 30–50% margin. Ask for more `--time`
than you need — an over-request costs a slightly worse queue position; an
under-request costs the whole run.

> [!tip] Overcommitted threads are a common slowdown
> One BLAS thread per allocated core, or the run is slower than your laptop:
> ```bash
> export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
> export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
> ```

---

## 6. Make every long job resumable

Assume the job will be interrupted: wall clock, preemption, node failure. A run
that cannot resume is a run you will pay for twice.

Checkpoint **everything the optimizer needs**, not just the weights — optimizer
moments, LR-schedule position, RNG state, and the epoch/step counter:

```python
torch.save({
    "state_dict": model.state_dict(),
    "optimizer":  optimizer.state_dict(),
    "scheduler":  scheduler.state_dict(),
    "step":       step,
    "rng":        torch.get_rng_state(),
}, tmp)
tmp.replace(path)          # write-then-rename: a kill mid-write leaves the old
                           # checkpoint intact rather than a truncated one
```

Then `--requeue` becomes safe and preemptible partitions become usable:

```bash
#SBATCH --requeue
```

> [!warning] Two traps around resuming
> **Resuming the wrong thing.** If the code auto-resumes from a checkpoint in the
> output directory, re-running a *fixed* configuration into the *same* directory
> silently continues the broken run. Version the output path
> (`outputs/run_v2/`) whenever you change something that matters.
>
> **Only saving on success.** If the final artifacts are written after the last
> epoch, an interrupted run looks like it produced nothing. Write the usable
> artifact at every checkpoint, and have the loader fall back to the resume file.

---

## 7. Make failures loud

The 24 h loss was not a crash — the model died at hour 8 and the job kept
burning cycles on a dead network. Log the health metric that would have shown it:

```python
if grad_norm < 1e-5:
    logger.warning("step %d: gradient norm %.2e — likely saturated; check X",
                   step, grad_norm)
```

Print the environment banner at the top of every job, so a log tells you what
actually ran:

```bash
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}  task ${SLURM_ARRAY_TASK_ID:-0}"
echo "partition : ${SLURM_JOB_PARTITION:-?}"
echo "config    : ${CONFIG}"
echo "run dir   : ${RUN}"
echo "commit    : $(git rev-parse --short HEAD 2>/dev/null || echo 'not a repo')"
echo "resuming  : $([ -f "${RUN}/state.pt" ] && echo YES || echo 'no')"
```

The commit hash is the single most valuable line: six months later it is the
only way to know which version of the code produced a result.

> [!note] `set -euo pipefail` cuts both ways
> It stops a broken job early — good — but combined with SLURM's redirection it
> makes "died at startup" and "never started" look identical. Keep it, and pair
> it with the banner above so the log always shows *how far* it got.

---

## 8. Use job arrays, one directory per task

One submission, N independent runs, individually requeueable:

```bash
sbatch --array=0-5 job.sbatch          # 6 tasks
sbatch --array=0-5%2 job.sbatch        # at most 2 running at once
sbatch --array=0,3,5 job.sbatch        # only the ones that failed
```

Map the index to a configuration inside the script, and derive a **distinct**
output directory per task:

```bash
IDX="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$(( IDX / 2 ))
if (( IDX % 2 == 0 )); then CONFIG=configs/a.yaml; TAG=a
else                        CONFIG=configs/b.yaml; TAG=b; fi
RUN="outputs/${TAG}_s${SEED}"
```

`%A` is the array job id and `%a` the task index, so `--output=logs/%A_%a.out`
gives one log per task.

---

## 9. Shrink the problem before you scale it

The most valuable thing built during the failure was a **4-minute reproduction**
of a 24-hour bug: same architecture, 1/16 the data, 1/50 the iterations. It
reproduced the failure exactly and turned a one-run-per-day debug loop into a
one-run-per-coffee loop.

Before any big submission, run the identical pipeline at toy scale:

```bash
python train.py --config configs/big.yaml \
    --set data.n=128 training.steps=2500 \
    --output-dir outputs/smoke
```

If it cannot survive five minutes locally, it will not survive 24 h on a node.

---

## Pre-submission checklist

- [ ] `logs/` and `outputs/` exist
- [ ] venv built **on the cluster**, never copied
- [ ] `--dry-run` passes **on a compute node** via `srun`
- [ ] `--time` from a measured rate, plus margin
- [ ] output directory is fresh, or resuming deliberately
- [ ] checkpointing on, `--requeue` set
- [ ] banner prints config, run dir, commit hash
- [ ] smoke run at toy scale passed
- [ ] `sacct` checked 60 s after submission — not `squeue`

---

## Related

- [[Cluster postmortem]] — the run that motivated all of this
- [[PIBE implementation]]
