"""
kaggle_pipeline.py
==================
Run ONE cell of the matched-control experiment on a free Kaggle GPU notebook:
train the victim, then run the FGSM-evasion / threshold-MIA evaluation, the
paper's shadow-model MIA (+BCVA) and AFMIA (+BCVA) on it.

Built around Kaggle's limits
----------------------------
* 12-hour sessions — everything is resumable.  The pipeline stops cleanly
  before --time_budget_hours, Kaggle saves /kaggle/working as the notebook's
  output, and the next session (with that output attached as an input)
  restores it and continues where it stopped.
* 2 × T4 GPUs — training runs on one GPU; afterwards the three jobs share
  both GPUs (longest first).  Works with one GPU too, just slower.
* 4 vCPUs — DataLoader workers are capped accordingly.
* Dataset — the public Kaggle copy of NIH ChestX-ray14 is located under
  /kaggle/input automatically (nothing to upload).  The membership split and
  labels come from manifest_portable.csv.gz, i.e. exactly the split used by
  the paper and by the local runs.

Cells: std_reg (A), fgsm_reg (B, also yields B2 = robust early stopping),
std_noreg (C), fgsm_noreg (D).

Usage (in the notebook)
-----------------------
  python Matched_Experiments/kaggle/kaggle_pipeline.py --cell fgsm_noreg --seeds 42
  python Matched_Experiments/kaggle/kaggle_pipeline.py --cell std_reg --smoke   # ~25-min test
"""

import argparse
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE        = os.path.dirname(os.path.abspath(__file__))
MATCHED_DIR = os.path.dirname(HERE)
PORTABLE_MANIFEST = os.path.join(HERE, "manifest_portable.csv.gz")

TRAIN_CELLS = ["std_reg", "fgsm_reg", "std_noreg", "fgsm_noreg"]
EXIT_PAUSED = 75                  # train_matched_victim.py: deadline reached, state saved
GRACE_S     = 10 * 60             # training must stop this long before the hard deadline
MIN_START_S = 20 * 60             # never launch a job with less time than this left
EVAL_MIN_S  = 75 * 60             # evaluate_victim.py is not resumable: needs this much time
HEARTBEAT_S = 10 * 60

LOG_FILE = None


def n_cpus():
    """CPUs this process may use (respects container / taskset limits)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


def log(msg):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


def code_version():
    try:
        return subprocess.run(["git", "-C", MATCHED_DIR, "log", "-1", "--format=%h %ci"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def fatal(msg):
    log("FATAL: " + msg)
    sys.exit(1)


# ─── Environment ──────────────────────────────────────────────────────────────

GPU_CHECK = r"""
import sys, torch
n = torch.cuda.device_count()
print("COUNT", n)
for i in range(n):
    name = torch.cuda.get_device_name(i)
    try:
        d = torch.device(f"cuda:{i}")
        x = torch.randn(4, 3, 32, 32, device=d, requires_grad=True)
        conv = torch.nn.Conv2d(3, 8, 3).to(d)
        with torch.autocast("cuda", dtype=torch.float16):
            y = conv(x).float().sum()
        y.backward()
        torch.cuda.synchronize(d)
        print("OK", i, name)
    except Exception as e:
        print("FAIL", i, name, repr(e)[:200])
"""


def check_gpus():
    out = subprocess.run([sys.executable, "-c", GPU_CHECK], capture_output=True, text=True)
    lines = out.stdout.splitlines()
    ok = [l.split(maxsplit=2) for l in lines if l.startswith("OK")]
    bad = [l for l in lines if l.startswith("FAIL")]
    for l in bad:
        log(f"GPU check failed: {l}")
    if bad and not ok:
        fatal("no usable GPU. If the accelerator is a P100, recent PyTorch builds no longer "
              "support it — switch the notebook to 'GPU T4 x2' (Settings → Accelerator).")
    if not ok:
        fatal("no GPU found — enable an accelerator (Settings → Accelerator → GPU T4 x2) "
              f"and rerun.\n{out.stderr[-500:]}")
    for _, i, name in ok:
        log(f"GPU {i}: {name}")
    return [int(i) for _, i, _ in ok]


def find_nih(input_root):
    """Locate Data_Entry_2017.csv and every PNG of the Kaggle NIH dataset."""
    pats = lambda tail: [os.path.join(input_root, *(["*"] * d), tail) for d in (1, 2, 3)]
    csvs = sorted({p for pat in pats("Data_Entry_2017.csv") for p in glob.glob(pat)})
    img_dirs = sorted({p for pat in pats(os.path.join("images_*", "images")) for p in glob.glob(pat)})
    if not img_dirs:   # flat layout fallback
        img_dirs = sorted({p for pat in pats("images") for p in glob.glob(pat) if os.path.isdir(p)})
    index = {}
    for d in img_dirs:
        with os.scandir(d) as it:
            for e in it:
                if e.name.lower().endswith(".png") and not e.name.startswith("._"):
                    index.setdefault(e.name, e.path)
    return (csvs[0] if csvs else None), index, img_dirs


def build_manifest(portable, index, out_csv, data_entry):
    import pandas as pd
    df = pd.read_csv(portable)
    missing = ~df["image"].isin(index.keys())
    if missing.any():
        fatal(f"{int(missing.sum())} of {len(df)} images of the split are not in the attached "
              f"dataset (e.g. {df.loc[missing, 'image'].iloc[:3].tolist()}). Attach the "
              "'NIH Chest X-rays' dataset (nih-chest-xrays/data) to the notebook.")
    if data_entry:
        de = pd.read_csv(data_entry, usecols=["Image Index", "Finding Labels"])
        m = df.merge(de, left_on="image", right_on="Image Index", how="left")
        diff = int((m["label"] != m["Finding Labels"]).sum())
        if diff:
            log(f"WARNING: {diff} images have different labels in the Kaggle CSV; "
                "the paper's labels from manifest_portable.csv.gz are used.")
    df["path"] = df["image"].map(index)
    df[["path", "label", "label_idx", "split"]].to_csv(out_csv, index=False)
    log(f"Manifest: {len(df)} images "
        f"({int((df.split == 'member').sum())} member / {int((df.split == 'nonmember').sum())} "
        f"non-member) → {out_csv}")


def copy_tree(src, dst):
    """Copy a (read-only) input tree into a writable location."""
    for root, dirs, files in os.walk(src):
        target = os.path.join(dst, os.path.relpath(root, src))
        os.makedirs(target, exist_ok=True)
        for f in files:
            shutil.copyfile(os.path.join(root, f), os.path.join(target, f))


def restore_previous(input_root, results_dir, results_name, cell):
    """Bring back the output of the previous session (attached as an input)."""
    cands = []
    for d in (1, 2, 3):
        cands += glob.glob(os.path.join(input_root, *(["*"] * d), results_name, "kaggle_state.json"))
    states = []
    for c in cands:
        try:
            with open(c) as f:
                states.append((json.load(f), os.path.dirname(c)))
        except Exception:
            pass
    if not states:
        return ["No previous output attached — starting fresh."]
    states.sort(key=lambda s: s[0].get("updated", ""))
    st, src = states[-1]
    msgs = []
    if st.get("cell") != cell:
        msgs.append(f"WARNING: the attached output is for cell '{st.get('cell')}', not '{cell}'. "
                    "Its files are restored too (run names differ, nothing is overwritten).")
    local = os.path.join(results_dir, "kaggle_state.json")
    if os.path.exists(local):
        with open(local) as f:
            if json.load(f).get("updated", "") >= st.get("updated", ""):
                return msgs + ["Local results are newer than the attached output — not restoring."]
    t0 = time.time()
    copy_tree(src, results_dir)
    return msgs + [f"Restored previous output from {src} ({time.time() - t0:.0f}s; "
                   f"last session ended {st.get('updated')})."]


def choose_cache_dir(work_dir):
    for base in ("/kaggle/temp", "/kaggle/tmp", "/tmp"):
        try:
            os.makedirs(os.path.join(base, "matched_cache"), exist_ok=True)
            if shutil.disk_usage(base).free > 8e9:
                return os.path.join(base, "matched_cache")
        except OSError:
            continue
    return os.path.join(work_dir, "cache")


# ─── Tasks ────────────────────────────────────────────────────────────────────

def plan_tasks(cell, seeds, robust, n_slots):
    job_order = ("afmia", "shadow_mia", "eval") if n_slots > 1 else ("eval", "shadow_mia", "afmia")
    tasks = []
    for seed in seeds:
        run = f"{cell}_seed{seed}"
        tasks.append(dict(name=f"train_{run}", kind="train", run=run, seed=seed, deps=[]))
        victims = [run] + ([f"fgsm_reg_robustES_seed{seed}"] if cell == "fgsm_reg" and robust else [])
        for v in victims:
            for kind in job_order:
                tasks.append(dict(name=f"{kind}_{v}", kind=kind, run=v, seed=seed,
                                  deps=[f"train_{run}"]))
    return tasks


def is_done(t, R):
    run = t["run"]
    if t["kind"] == "train":
        vdir = os.path.join(R, "victims", run)
        ok = (os.path.exists(os.path.join(vdir, "model_meta.json"))
              and not os.path.exists(os.path.join(vdir, "train_state.pt")))
        if run.startswith("fgsm_reg_seed"):
            b2 = run.replace("fgsm_reg_", "fgsm_reg_robustES_")
            ok = ok and os.path.exists(os.path.join(R, "victims", b2, "model_meta.json"))
        return ok
    if t["kind"] == "eval":
        return os.path.exists(os.path.join(R, "eval", run, "threshold_mia.json"))
    return os.path.exists(os.path.join(R, "attacks", t["kind"], run, "results.json"))


def task_cmd(t, R, cache_dir, train_deadline, smoke):
    py = sys.executable
    seed = str(t["seed"])
    smoke_flag = ["--smoke"] if smoke else []
    if t["kind"] == "train":
        cell = t["run"].rsplit("_seed", 1)[0]
        cmd = [py, os.path.join(MATCHED_DIR, "train_matched_victim.py"),
               "--training", "fgsm" if cell.startswith("fgsm") else "standard",
               "--regime", "noreg" if cell.endswith("noreg") else "reg",
               "--seed", seed, "--out_dir", os.path.join(R, "victims", t["run"]),
               "--cache_dir", cache_dir, "--num_workers", str(n_cpus()),
               "--deadline", f"{train_deadline:.0f}"]
        if cell == "fgsm_reg":
            cmd += ["--robust_es_out_dir", os.path.join(R, "victims", f"fgsm_reg_robustES_seed{seed}")]
        return cmd + smoke_flag
    common = ["--victim_dir", os.path.join(R, "victims", t["run"]), "--seed", seed,
              "--cache_dir", cache_dir]
    if t["kind"] == "eval":
        return [py, os.path.join(MATCHED_DIR, "evaluate_victim.py"), *common,
                "--out_dir", os.path.join(R, "eval", t["run"])] + smoke_flag
    return [py, os.path.join(MATCHED_DIR, "run_shadow_attacks.py"), "--attack", t["kind"], *common,
            "--out_dir", os.path.join(R, "attacks", t["kind"], t["run"]),
            "--splits_dir", os.path.join(R, "splits")] + smoke_flag


def last_line(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4000))
            lines = [l for l in f.read().decode(errors="replace").splitlines() if l.strip()]
        return lines[-1][:160] if lines else ""
    except OSError:
        return ""


def tail(path, n=25):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def run_scheduler(tasks, slots, R, cache_dir, manifest, hard_deadline, smoke):
    env_base = dict(os.environ, MATCHED_MANIFEST=manifest, PYTHONUNBUFFERED="1")
    cpus = n_cpus()
    status = {t["name"]: ("done" if is_done(t, R) else "pending") for t in tasks}
    running = {}              # slot index → (task, Popen, start time)
    last_beat = 0.0
    eval_min = 5 * 60 if smoke else EVAL_MIN_S

    def deps_ok(t):
        return all(status.get(d) == "done" for d in t["deps"])

    while True:
        now = time.time()

        # Reap finished tasks
        for slot, (t, proc, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[slot]
            mins = (now - t0) / 60
            logf = os.path.join(R, "logs", f"{t['name']}.log")
            if rc == 0 and is_done(t, R):
                status[t["name"]] = "done"
                open(os.path.join(R, ".markers", f"{t['name']}.done"), "w").close()
                log(f"DONE    {t['name']}  ({mins:.0f} min)")
            elif rc == EXIT_PAUSED:
                status[t["name"]] = "paused"
                log(f"PAUSED  {t['name']}  ({mins:.0f} min) — continues next session")
            else:
                status[t["name"]] = "failed"
                log(f"FAILED  {t['name']}  (exit {rc}, {mins:.0f} min). Last log lines:\n"
                    + tail(logf))

        # Hard deadline: stop everything (all work is checkpointed)
        if now >= hard_deadline:
            for slot, (t, proc, _) in running.items():
                log(f"Time budget reached — stopping {t['name']} (progress is saved)")
                proc.terminate()
            for slot, (t, proc, _) in running.items():
                try:
                    proc.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    proc.kill()
                status[t["name"]] = "interrupted"
            running.clear()
            break

        # Launch ready tasks on free slots
        running_names = {t["name"] for t, _, _ in running.values()}
        for slot, gpu in enumerate(slots):
            if slot in running:
                continue
            remaining = hard_deadline - now
            for t in tasks:
                if status[t["name"]] != "pending" or t["name"] in running_names or not deps_ok(t):
                    continue
                if remaining < MIN_START_S or (t["kind"] == "eval" and remaining < eval_min):
                    continue
                cmd = task_cmd(t, R, cache_dir, hard_deadline - GRACE_S, smoke)
                env = dict(env_base, CUDA_VISIBLE_DEVICES=str(gpu),
                           MATCHED_NUM_WORKERS=str(cpus if t["kind"] == "train" else max(1, cpus // 2)),
                           OMP_NUM_THREADS=str(max(1, cpus // 2)))
                logf = os.path.join(R, "logs", f"{t['name']}.log")
                with open(logf, "a") as f:
                    f.write(f"\n===== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} "
                            f"session start (GPU {gpu}) =====\n")
                proc = subprocess.Popen(cmd, stdout=open(logf, "a"), stderr=subprocess.STDOUT,
                                        env=env, cwd=MATCHED_DIR)
                running[slot] = (t, proc, now)
                status[t["name"]] = "running"
                running_names.add(t["name"])
                log(f"START   {t['name']}  on GPU {gpu}  "
                    f"({remaining / 3600:.1f} h left in this session)")
                break

        if not running:
            break

        if now - last_beat > HEARTBEAT_S:
            last_beat = now
            left = (hard_deadline - now) / 3600
            for t, _, t0 in running.values():
                log(f"  … {t['name']} running {(now - t0) / 60:.0f} min | "
                    f"{last_line(os.path.join(R, 'logs', t['name'] + '.log'))}")
            log(f"  ({left:.1f} h left in the time budget)")
        time.sleep(15)

    for t in tasks:          # tasks never started this session stay pending
        if status[t["name"]] == "running":
            status[t["name"]] = "interrupted"
    return status


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global LOG_FILE, GRACE_S, MIN_START_S
    p = argparse.ArgumentParser(description="Matched experiment — one cell on Kaggle")
    p.add_argument("--cell", required=True, choices=TRAIN_CELLS)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--no_robust_es", action="store_true",
                   help="fgsm_reg only: do not attack the robust-early-stopping model (B2)")
    p.add_argument("--time_budget_hours", type=float, default=None,
                   help="stop cleanly after this long (default 11.0; 1.5 with --smoke)")
    p.add_argument("--work_dir", default="/kaggle/working")
    p.add_argument("--input_root", default="/kaggle/input")
    p.add_argument("--cache_dir", default=None, help="default: a non-output temp dir")
    p.add_argument("--gpu_slots", default=None,
                   help="comma-separated GPU ids, one job per entry (default: all GPUs)")
    p.add_argument("--portable_manifest", default=PORTABLE_MANIFEST)
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end test run")
    # margins (minutes) — only changed for testing the pause/resume logic
    p.add_argument("--grace_minutes", type=float, default=GRACE_S / 60, help=argparse.SUPPRESS)
    p.add_argument("--min_start_minutes", type=float, default=MIN_START_S / 60, help=argparse.SUPPRESS)
    args = p.parse_args()
    GRACE_S, MIN_START_S = args.grace_minutes * 60, args.min_start_minutes * 60

    t_start = time.time()
    budget_h = args.time_budget_hours or (1.5 if args.smoke else 11.0)
    hard_deadline = t_start + budget_h * 3600
    results_name = "results_smoke" if args.smoke else "results"
    R = os.path.join(args.work_dir, results_name)
    for d in ("logs", ".markers", "victims", "eval", "attacks", "splits"):
        os.makedirs(os.path.join(R, d), exist_ok=True)

    # 1. Previous session's output (before this session's log file is opened)
    restore_msgs = restore_previous(args.input_root, R, results_name, args.cell)

    LOG_FILE = os.path.join(R, "logs", "kaggle_pipeline.log")
    log("=" * 78)
    log(f"Matched experiment on Kaggle — cell {args.cell}, seeds {args.seeds}"
        f"{' [SMOKE TEST]' if args.smoke else ''}; time budget {budget_h:.1f} h")
    log("=" * 78)
    for m in restore_msgs:
        log(m)
    commit = code_version()
    log(f"Code version: {commit}")
    prev_state = os.path.join(R, "kaggle_state.json")
    if os.path.exists(prev_state):
        with open(prev_state) as f:
            prev_commit = json.load(f).get("code_version")
        if prev_commit and prev_commit != commit:
            log(f"WARNING: previous session ran code {prev_commit}; this one runs {commit}. "
                "Keep all sessions and accounts on the same commit for comparable results.")

    # 2. GPUs
    gpus = check_gpus()
    slots = [int(g) for g in args.gpu_slots.split(",")] if args.gpu_slots else gpus
    log(f"CPU cores: {n_cpus()}  |  job slots on GPUs: {slots}")

    # 3. Dataset → manifest with this machine's paths
    data_entry, index, img_dirs = find_nih(args.input_root)
    log(f"NIH dataset: {len(index)} PNGs in {len(img_dirs)} folder(s)"
        f"{'; labels CSV ' + data_entry if data_entry else ''}")
    cache_dir = args.cache_dir or choose_cache_dir(args.work_dir)
    os.makedirs(cache_dir, exist_ok=True)
    manifest = os.path.join(cache_dir, "manifest_kaggle.csv")   # temp: not saved as output
    build_manifest(args.portable_manifest, index, manifest, data_entry)
    os.environ["MATCHED_MANIFEST"] = manifest

    # 4. Image cache (rebuilt each session in a temp dir; ~10 min)
    log(f"Image cache: {cache_dir}")
    t0 = time.time()
    with open(os.path.join(R, "logs", "build_cache.log"), "a") as f:
        rc = subprocess.call([sys.executable, os.path.join(MATCHED_DIR, "build_image_cache.py"),
                              "--cache_dir", cache_dir, "--zip_dir", "/nonexistent",
                              "--workers", str(n_cpus()), "--verify", "300"],
                             stdout=f, stderr=subprocess.STDOUT,
                             env=dict(os.environ, MATCHED_MANIFEST=manifest))
    if rc != 0:
        fatal("image cache build failed:\n" + tail(os.path.join(R, "logs", "build_cache.log")))
    log(f"Image cache ready ({(time.time() - t0) / 60:.1f} min)")

    # 5. Train + attack
    tasks = plan_tasks(args.cell, args.seeds, not args.no_robust_es, len(slots))
    done_names = [t["name"] for t in tasks if is_done(t, R)]
    log(f"{len(tasks)} tasks planned, {len(done_names)} already done"
        + (f": {', '.join(done_names)}" if done_names else ""))
    status = run_scheduler(tasks, slots, R, cache_dir, manifest, hard_deadline, args.smoke)

    # 6. Summary of everything in this output
    subprocess.call([sys.executable, os.path.join(MATCHED_DIR, "aggregate_results.py"),
                     "--results_dir", R], stdout=open(os.path.join(R, "logs", "aggregate.log"), "a"),
                    stderr=subprocess.STDOUT, env=dict(os.environ, MATCHED_MANIFEST=manifest))

    complete = all(v == "done" for v in status.values())
    failed = [k for k, v in status.items() if v == "failed"]
    state_path = os.path.join(R, "kaggle_state.json")
    prev = {}
    if os.path.exists(state_path):
        with open(state_path) as f:
            prev = json.load(f)
    sessions = prev.get("sessions", []) + [{
        "start": datetime.datetime.fromtimestamp(t_start).isoformat(timespec="seconds"),
        "end": datetime.datetime.now().isoformat(timespec="seconds"),
        "gpus": slots, "code_version": commit, "status": status}]
    state = {"cell": args.cell, "seeds": args.seeds, "robust_es": not args.no_robust_es,
             "code_version": commit,
             "smoke": args.smoke, "complete": complete, "failed": failed,
             "updated": datetime.datetime.now().isoformat(timespec="seconds"),
             "tasks": status, "sessions": sessions}
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    if complete:
        verdict = ("COMPLETE — every task for this cell is done. Download this notebook's "
                   "output (results/) and merge it with the other accounts' outputs.")
    elif failed:
        verdict = (f"FAILED: {', '.join(failed)} — see results/logs/<task>.log. Fix the cause, "
                   "then run a new version with this output attached; finished work is kept.")
    else:
        verdict = ("NOT FINISHED (time budget reached) — click 'Save Version' again with THIS "
                   "version's output attached as input; it continues automatically.")
    with open(os.path.join(R, "STATUS.md"), "w") as f:
        f.write(f"# Kaggle run status — cell `{args.cell}`, seeds {args.seeds}\n\n")
        f.write(f"Updated {state['updated']} after {len(sessions)} session(s).\n\n")
        f.write(f"**{verdict}**\n\n| Task | Status |\n|---|---|\n")
        for k, v in status.items():
            f.write(f"| {k} | {v} |\n")
        if os.path.exists(os.path.join(R, "summary", "REPORT.md")):
            f.write("\nPartial results: `summary/REPORT.md`\n")
    log("=" * 78)
    for k, v in status.items():
        log(f"  {v:<12s} {k}")
    log(verdict)
    log("=" * 78)


if __name__ == "__main__":
    main()
