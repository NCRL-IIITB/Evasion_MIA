"""
kaggle_pipeline.py  (Matched_Experiments_v3 — the experiment lab)
=================================================================
Run any set of victims through the full attack battery on a free Kaggle GPU
notebook.  Victims come from two sources:

  --victims std fgsm_rs:4:0.5 fgsm_rs:8:0.5 …   trained here (no augmentation,
        35 epochs, last epoch); 'std' = standard training, otherwise
        method:eps(/255):alpha with method ∈ fgsm_rs | nfgsm | fgsm | pgd3
  --import_runs std_noreg_seed42 fgsm_noreg_seed42 …   copied from attached
        outputs of earlier runs (v1 Matched_Experiments → name prefix 'v1-',
        v2 → 'v2-'), e.g. the C and D victims of the 'results 3' / 'results 5'
        folders, or the regularised A / B / B2.  No training: attack-only.

Tasks per victim (dependencies in brackets):
  train_<run>                   (—)                     only for trained victims
  eval_<run>                    (train)                 v2 evaluate_victim.py
  shadows_at_<run>, shadows_std_<run>   (train)         v2 train_shadows.py
  attack_afmia_<run>            (shadows_at)            v2 run_attacks.py (paper + v2 AFMIA)
  attack_shadow_mia_std_<run>   (shadows_std)           v2 run_attacks.py (paper Shadow-MIA)
  attack_shadow_mia_at_<run>    (shadows_at)            v2 run_attacks.py
  attack_lab_<run>              (shadows_std, shadows_at)  v3 lab_attacks.py

Same Kaggle mechanics as v1/v2: 12-hour sessions (stops after 11 h; attach the
output and rerun to continue), 2 × T4, 4 vCPUs, NIH dataset under /kaggle/input,
the paper's split (Matched_Experiments/kaggle/manifest_portable.csv.gz).
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

HERE   = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.dirname(HERE)
V2_DIR = os.path.join(os.path.dirname(V3_DIR), "Matched_Experiments_v2")
sys.path.insert(0, V2_DIR)

EXIT_PAUSED = 75
GRACE_S     = 10 * 60
MIN_START_S = 20 * 60
HEARTBEAT_S = 10 * 60
RESULTS     = "results_v3"
LOG_FILE    = None


def n_cpus():
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


def fatal(msg):
    log("FATAL: " + msg)
    sys.exit(1)


def code_version():
    try:
        return subprocess.run(["git", "-C", V3_DIR, "log", "-1", "--format=%h %ci"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


# ─── Environment ──────────────────────────────────────────────────────────────

GPU_CHECK = r"""
import torch
for i in range(torch.cuda.device_count()):
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


def check_gpus(allow_cpu):
    out = subprocess.run([sys.executable, "-c", GPU_CHECK], capture_output=True, text=True)
    ok = [l.split(maxsplit=2) for l in out.stdout.splitlines() if l.startswith("OK")]
    for l in out.stdout.splitlines():
        if l.startswith("FAIL"):
            log(f"GPU check failed: {l}")
    if not ok:
        if allow_cpu:
            log("No GPU — running on CPU (--allow_cpu; pipeline test only)")
            return ["cpu"]
        fatal("no usable GPU — Settings → Accelerator → GPU T4 x2 (not P100), then rerun.\n"
              + out.stderr[-500:])
    for _, i, name in ok:
        log(f"GPU {i}: {name}")
    return [int(i) for _, i, _ in ok]


def find_nih(input_root):
    pats = lambda tail: [os.path.join(input_root, *(["*"] * d), tail) for d in (1, 2, 3)]  # noqa
    csvs = sorted({p for pat in pats("Data_Entry_2017.csv") for p in glob.glob(pat)})
    img_dirs = sorted({p for pat in pats(os.path.join("images_*", "images")) for p in glob.glob(pat)})
    if not img_dirs:
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
            log(f"WARNING: {diff} labels differ in the Kaggle CSV; the paper's labels are used.")
    df["path"] = df["image"].map(index)
    df[["path", "label", "label_idx", "split"]].to_csv(out_csv, index=False)
    log(f"Manifest: {len(df)} images ({int((df.split == 'member').sum())} member / "
        f"{int((df.split == 'nonmember').sum())} non-member) → {out_csv}")


def copy_tree(src, dst):
    for root, _, files in os.walk(src):
        target = os.path.join(dst, os.path.relpath(root, src))
        os.makedirs(target, exist_ok=True)
        for f in files:
            shutil.copyfile(os.path.join(root, f), os.path.join(target, f))


def restore_previous(input_root, R, results_name):
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
        return ["No previous v3 output attached — starting fresh."]
    states.sort(key=lambda s: s[0].get("updated", ""))
    st, src = states[-1]
    msgs = []
    local = os.path.join(R, "kaggle_state.json")
    if os.path.exists(local):
        with open(local) as f:
            if json.load(f).get("updated", "") >= st.get("updated", ""):
                return msgs + ["Local results are newer than the attached output — not restoring."]
    t0 = time.time()
    copy_tree(src, R)
    return msgs + [f"Restored previous output from {src} ({time.time() - t0:.0f}s; last session "
                   f"ended {st.get('updated')})."]


def import_runs(input_root, R, names, smoke):
    """Copy finished victims (model.pth + meta + history + checkpoints) from attached
    outputs of earlier experiments.  v1 victims get the prefix 'v1-', v2 victims 'v2-'
    (v1's and v2's 'fgsm_noreg_seed42' are different recipes).  Returns imported names."""
    imported = []
    for name in names:
        cands = []
        for d in (1, 2, 3, 4, 5):
            cands += glob.glob(os.path.join(input_root, *(["*"] * d), "victims", name,
                                            "model_meta.json"))
        found = False
        for meta_path in sorted(set(cands)):
            src = os.path.dirname(meta_path)
            if any(part.startswith(RESULTS) for part in src.split(os.sep)):
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            if (bool(meta.get("smoke")) != smoke or not os.path.exists(os.path.join(src, "model.pth"))
                    or os.path.exists(os.path.join(src, "train_state.pt"))):
                continue
            prefix = "v2-" if str(meta.get("experiment", "")).startswith("Matched_Experiments_v2") else "v1-"
            run = prefix + name
            dst = os.path.join(R, "victims", run)
            if not os.path.exists(os.path.join(dst, "model_meta.json")):
                copy_tree(src, dst)
                meta.update(imported_from=src, original_run=name,
                            regime=meta.get("regime", "noreg"))
                with open(os.path.join(dst, "model_meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)
                log(f"Imported {name} from {src} as {run}")
            if run not in imported:
                imported.append(run)
            found = True
        if not found:
            log(f"WARNING: no finished victim '{name}' found under {input_root} — attach the old "
                "output (a Kaggle notebook output or an uploaded dataset of the results folder).")
    return imported


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

KIND_ORDER = ["shadows_at", "shadows_std", "eval", "attack_afmia", "attack_shadow_mia_std",
              "attack_lab", "attack_shadow_mia_at"]
DEPS = {"eval": ["train"], "shadows_at": ["train"], "shadows_std": ["train"],
        "attack_afmia": ["shadows_at"], "attack_shadow_mia_std": ["shadows_std"],
        "attack_shadow_mia_at": ["shadows_at"], "attack_lab": ["shadows_std", "shadows_at"]}
ATTACK_SUBDIR = {"attack_afmia": "afmia", "attack_shadow_mia_std": "shadow_mia_std",
                 "attack_shadow_mia_at": "shadow_mia_at", "attack_lab": "lab"}


def plan_tasks(trained, imported, seeds):
    """trained: list of (cell_id, recipe) trained here for every seed; imported: run
    names already present in R/victims (their seed is in the name).  All trainings
    come first so that several victims train in parallel on the two GPUs."""
    runs = [(f"{cid}_seed{s}", s, rec) for s in seeds for cid, rec in trained]
    runs += [(r, int(r.rsplit("_seed", 1)[1]), {"imported": True}) for r in imported]
    tasks = [dict(name=f"train_{run}", kind="train", run=run, seed=seed, rec=rec, deps=[])
             for run, seed, rec in runs]
    for kind in KIND_ORDER:
        for run, seed, rec in runs:
            tasks.append(dict(name=f"{kind}_{run}", kind=kind, run=run, seed=seed, rec=rec,
                              deps=[f"{d}_{run}" for d in DEPS[kind]]))
    return tasks


def is_done(t, R):
    run, k = t["run"], t["kind"]
    if k == "train":
        v = os.path.join(R, "victims", run)
        return (os.path.exists(os.path.join(v, "model_meta.json"))
                and not os.path.exists(os.path.join(v, "train_state.pt")))
    if k == "eval":
        return os.path.exists(os.path.join(R, "eval", run, "threshold_mia.json"))
    if k.startswith("shadows_"):
        return os.path.exists(os.path.join(R, "attacks", run, k, "DONE.json"))
    return os.path.exists(os.path.join(R, "attacks", run, ATTACK_SUBDIR[k], "results.json"))


def task_cmd(t, R, cache_dir, deadline, smoke):
    py, seed = sys.executable, str(t["seed"])
    sm = ["--smoke"] if smoke else []
    vdir = os.path.join(R, "victims", t["run"])
    adir = os.path.join(R, "attacks", t["run"])
    k = t["kind"]
    if k == "train":
        rec = t["rec"]
        cmd = [py, os.path.join(V2_DIR, "train_victim.py"), "--training", rec["training"],
               "--seed", seed, "--out_dir", vdir, "--cache_dir", cache_dir,
               "--num_workers", str(n_cpus()), "--deadline", f"{deadline:.0f}"]
        if rec["training"] == "fgsm":
            cmd += ["--method", rec["method"], "--eps", str(rec["eps"]), "--alpha", str(rec["alpha"])]
        return cmd + sm
    if k == "eval":
        return [py, os.path.join(V2_DIR, "evaluate_victim.py"), "--victim_dir", vdir, "--seed", seed,
                "--out_dir", os.path.join(R, "eval", t["run"]), "--cache_dir", cache_dir] + sm
    if k.startswith("shadows_"):
        return [py, os.path.join(V2_DIR, "train_shadows.py"), "--kind", k.split("_")[1],
                "--victim_dir", vdir, "--seed", seed, "--attack_dir", adir,
                "--cache_dir", cache_dir, "--splits_dir", os.path.join(R, "splits"),
                "--deadline", f"{deadline:.0f}"] + sm
    if k == "attack_lab":
        return [py, os.path.join(V3_DIR, "lab_attacks.py"), "--victim_dir", vdir, "--seed", seed,
                "--attack_dir", adir, "--cache_dir", cache_dir] + sm
    attack = ["--attack", "afmia"] if k == "attack_afmia" else \
        ["--attack", "shadow_mia", "--shadow_kind", k.rsplit("_", 1)[1]]
    return [py, os.path.join(V2_DIR, "run_attacks.py"), *attack, "--victim_dir", vdir,
            "--seed", seed, "--attack_dir", adir, "--cache_dir", cache_dir] + sm


def tail(path, n=25):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def last_line(path):
    lines = [l for l in tail(path, 5).splitlines() if l.strip()]
    return lines[-1][:160] if lines else ""


def run_scheduler(tasks, slots, R, cache_dir, manifest, hard_deadline, smoke):
    env_base = dict(os.environ, MATCHED_MANIFEST=manifest, PYTHONUNBUFFERED="1")
    cpus = n_cpus()
    status = {t["name"]: ("done" if is_done(t, R) else "pending") for t in tasks}
    running, last_beat = {}, 0.0

    def deps_ok(t):
        return all(status.get(d) == "done" for d in t["deps"])

    while True:
        now = time.time()
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
                log(f"FAILED  {t['name']}  (exit {rc}, {mins:.0f} min). Last log lines:\n" + tail(logf))

        if now >= hard_deadline:
            for t, proc, _ in running.values():
                log(f"Time budget reached — stopping {t['name']} (progress is saved)")
                proc.terminate()
            for t, proc, _ in running.values():
                try:
                    proc.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    proc.kill()
                status[t["name"]] = "interrupted"
            running.clear()
            break

        names = {t["name"] for t, _, _ in running.values()}
        for slot, gpu in enumerate(slots):
            if slot in running:
                continue
            remaining = hard_deadline - now
            for t in tasks:
                if status[t["name"]] != "pending" or t["name"] in names or not deps_ok(t):
                    continue
                if remaining < MIN_START_S:
                    continue
                cmd = task_cmd(t, R, cache_dir, hard_deadline - GRACE_S, smoke)
                env = dict(env_base, MATCHED_NUM_WORKERS=str(cpus if t["kind"] == "train"
                                                              else max(1, cpus // 2)),
                           OMP_NUM_THREADS=str(max(1, cpus // 2)))
                if gpu == "cpu":
                    env.update(V2_ALLOW_CPU="1", CUDA_VISIBLE_DEVICES="")
                else:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                logf = os.path.join(R, "logs", f"{t['name']}.log")
                with open(logf, "a") as f:
                    f.write(f"\n===== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} session start "
                            f"(GPU {gpu}) =====\n")
                proc = subprocess.Popen(cmd, stdout=open(logf, "a"), stderr=subprocess.STDOUT,
                                        env=env, cwd=V2_DIR)
                running[slot] = (t, proc, now)
                status[t["name"]] = "running"
                names.add(t["name"])
                log(f"START   {t['name']}  on GPU {gpu}  ({remaining / 3600:.1f} h left)")
                break

        if not running:
            break
        if now - last_beat > HEARTBEAT_S:
            last_beat = now
            for t, _, t0 in running.values():
                log(f"  … {t['name']} running {(now - t0) / 60:.0f} min | "
                    f"{last_line(os.path.join(R, 'logs', t['name'] + '.log'))}")
            log(f"  ({(hard_deadline - now) / 3600:.1f} h left in the time budget)")
        time.sleep(15)

    for t in tasks:
        if status[t["name"]] == "running":
            status[t["name"]] = "interrupted"
    return status


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global LOG_FILE, GRACE_S, MIN_START_S
    import v2_common as v2

    p = argparse.ArgumentParser(description="Matched experiment v3 (lab) on Kaggle")
    p.add_argument("--label", default="lab", help="name of this account's run (for STATUS.md)")
    p.add_argument("--victims", nargs="*", default=[],
                   help="victims to train: 'std' or method:eps255:alpha (fgsm_rs|nfgsm|fgsm|pgd3)")
    p.add_argument("--import_runs", nargs="*", default=[],
                   help="finished victims to copy from attached outputs, e.g. std_noreg_seed42")
    p.add_argument("--seeds", nargs="+", type=int, default=[42], help="seeds of trained victims")
    p.add_argument("--time_budget_hours", type=float, default=None)
    p.add_argument("--work_dir", default="/kaggle/working")
    p.add_argument("--input_root", default="/kaggle/input")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--gpu_slots", default=None)
    p.add_argument("--portable_manifest", default=v2.PORTABLE_MANIFEST)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--allow_cpu", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--grace_minutes", type=float, default=GRACE_S / 60, help=argparse.SUPPRESS)
    p.add_argument("--min_start_minutes", type=float, default=MIN_START_S / 60, help=argparse.SUPPRESS)
    args = p.parse_args()
    GRACE_S, MIN_START_S = args.grace_minutes * 60, args.min_start_minutes * 60

    trained = []
    for spec in args.victims:
        if spec == "std":
            trained.append(("std_noreg", {"training": "standard"}))
            continue
        method, eps, alpha = v2.parse_fgsm_variant(spec)
        trained.append((v2.victim_cell_id("fgsm_noreg", method, eps, alpha),
                        {"training": "fgsm", "method": method, "eps": eps, "alpha": alpha}))
    if not trained and not args.import_runs:
        fatal("nothing to do — give --victims and/or --import_runs")

    t_start = time.time()
    budget_h = args.time_budget_hours or (1.5 if args.smoke else 11.0)
    hard_deadline = t_start + budget_h * 3600
    results_name = RESULTS + ("_smoke" if args.smoke else "")
    R = os.path.join(args.work_dir, results_name)
    for d in ("logs", ".markers", "victims", "eval", "attacks", "splits"):
        os.makedirs(os.path.join(R, d), exist_ok=True)
    restore_msgs = restore_previous(args.input_root, R, results_name)

    LOG_FILE = os.path.join(R, "logs", "kaggle_pipeline.log")
    log("=" * 78)
    log(f"Matched experiment v3 (lab) — '{args.label}': train {[c for c, _ in trained]} "
        f"(seeds {args.seeds}), import {args.import_runs}"
        f"{' [SMOKE TEST]' if args.smoke else ''}; budget {budget_h:.1f} h")
    log("=" * 78)
    for m in restore_msgs:
        log(m)
    commit = code_version()
    log(f"Code version: {commit}")
    imported = import_runs(args.input_root, R, args.import_runs, args.smoke) if args.import_runs else []

    gpus = check_gpus(args.allow_cpu)
    slots = [int(g) for g in args.gpu_slots.split(",")] if args.gpu_slots else gpus
    log(f"CPU cores: {n_cpus()}  |  job slots: {slots}")

    data_entry, index, img_dirs = find_nih(args.input_root)
    log(f"NIH dataset: {len(index)} PNGs in {len(img_dirs)} folder(s)")
    cache_dir = args.cache_dir or choose_cache_dir(args.work_dir)
    os.makedirs(cache_dir, exist_ok=True)
    manifest = os.path.join(cache_dir, "manifest_kaggle.csv")
    build_manifest(args.portable_manifest, index, manifest, data_entry)
    os.environ["MATCHED_MANIFEST"] = manifest

    log(f"Image cache: {cache_dir}")
    t0 = time.time()
    with open(os.path.join(R, "logs", "build_cache.log"), "a") as f:
        rc = subprocess.call([sys.executable, v2.BUILD_CACHE_SCRIPT, "--cache_dir", cache_dir,
                              "--zip_dir", "/nonexistent", "--workers", str(n_cpus()),
                              "--verify", "300"], stdout=f, stderr=subprocess.STDOUT,
                             env=dict(os.environ, MATCHED_MANIFEST=manifest))
    if rc != 0:
        fatal("image cache build failed:\n" + tail(os.path.join(R, "logs", "build_cache.log")))
    log(f"Image cache ready ({(time.time() - t0) / 60:.1f} min)")

    tasks = plan_tasks(trained, imported, args.seeds)
    done = [t["name"] for t in tasks if is_done(t, R)]
    log(f"{len(tasks)} tasks planned, {len(done)} already done" + (f": {', '.join(done)}" if done else ""))
    status = run_scheduler(tasks, slots, R, cache_dir, manifest, hard_deadline, args.smoke)

    subprocess.call([sys.executable, os.path.join(V3_DIR, "aggregate_lab.py"), "--results_dir", R]
                    + (["--bootstrap", "100"] if args.smoke else []),
                    stdout=open(os.path.join(R, "logs", "aggregate.log"), "a"),
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
    state = {"label": args.label, "trained": [v for v, _ in trained], "imported": imported,
             "seeds": args.seeds,
             "code_version": commit, "smoke": args.smoke, "complete": complete, "failed": failed,
             "updated": datetime.datetime.now().isoformat(timespec="seconds"),
             "tasks": status, "sessions": sessions}
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    if complete:
        verdict = ("COMPLETE — every task is done. Download this notebook's output "
                   f"({results_name}/) and merge it with the other accounts' outputs "
                   "(Matched_Experiments_v3/kaggle/merge_kaggle_results.py).")
    elif failed:
        verdict = (f"FAILED: {', '.join(failed)} — see {results_name}/logs/<task>.log. Fix the "
                   "cause, then run a new version with this output attached; finished work is kept.")
    else:
        verdict = ("NOT FINISHED (time budget reached) — Save Version again with THIS version's "
                   "output attached as input; it continues automatically.")
    with open(os.path.join(R, "STATUS.md"), "w") as f:
        f.write(f"# Kaggle run status (v3 lab) — `{args.label}`: trained {[c for c, _ in trained]} "
                f"seeds {args.seeds}, imported {imported}\n\n")
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
