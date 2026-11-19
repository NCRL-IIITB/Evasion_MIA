"""
run.py — the whole experiment of THIS folder, resumable across Kaggle sessions.

  python run.py                         full run (on Kaggle: /kaggle/working/results_<EXPERIMENT>)
  python run.py --smoke                 ~20-40 min pipeline test (tiny data, 2 epochs)
  python run.py --results DIR           choose the output folder
  python run.py --hours 11              session budget (default 11 h; Kaggle kills at 12 h)
  python run.py --partial               run attacks/report with whatever signals exist
  python run.py --report_only           only rebuild summary/REPORT.md
  python run.py --dataset_override synthetic --smoke    CPU test without downloads

Stages (all state lives in files, so a new session continues where the last stopped):
  train:<model>   train one zoo model (resumable per epoch)               GPU
  sig:<model>     query it with every probe + evasion diagnostics         GPU
  attack          every attack x setting x target on the cached signals   CPU
  report          summary/REPORT.md, CSVs, figures                        CPU
Both GPUs of a Kaggle "T4 x2" session are used (one worker per GPU).
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config  # noqa: E402
from fmia.utils import EXIT_PAUSED, fmt_s, load_json, log, save_json, set_log_file, time_left  # noqa: E402

ON_KAGGLE = os.path.isdir("/kaggle/working")
TIMES = dict(min_start_train=15, min_start_sig=25, min_start_attack=15)
TIMES.update(getattr(config, "TIMES", {}))


def default_results(smoke):
    base = "/kaggle/working" if ON_KAGGLE else HERE
    return os.path.join(base, f"results_{config.EXPERIMENT}" + ("_smoke" if smoke else ""))


def restore_previous(res):
    """Copy the attached output of the previous session (Add Input -> this notebook's output)."""
    if os.path.isdir(res) and os.listdir(res):
        return
    name = os.path.basename(res)
    cands = []
    for depth in range(1, 6):
        cands += glob.glob("/kaggle/input/" + "*/" * depth + name)
    cands = [c for c in cands if os.path.isdir(c) and os.path.isdir(os.path.join(c, "models"))]
    if not cands:
        return
    best = max(cands, key=lambda c: os.path.getmtime(os.path.join(c, "STATUS.md"))
               if os.path.exists(os.path.join(c, "STATUS.md")) else 0)
    print(f"restoring previous output from {best}", flush=True)
    shutil.copytree(best, res, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.tmp", "*.tmp.npz"))


def code_version():
    try:
        r = subprocess.run(["git", "-C", HERE, "log", "-1", "--format=%h %ci"], capture_output=True, text=True)
        return r.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


# ─── workers ──────────────────────────────────────────────────────────────────

def worker(args, plan, res):
    import torch
    from fmia.data import load_dataset
    from fmia.signals import evaluate_evasion, extract_signals
    from fmia.train import train_model

    if args.worker == "attack":
        from fmia.runner import run_all
        r = run_all(plan, res, min_left_s=60 * TIMES["min_start_attack"])
        sys.exit(EXIT_PAUSED if r == "paused" else 0)

    kind, mid = args.worker.split(":", 1)
    m = plan.models[mid]
    dconf = plan.datasets[m["dataset"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    ds = load_dataset(m["dataset"], dconf.get("n_pop"), dconf.get("n_test"))
    mdir = os.path.join(res, "models", mid)
    if kind == "train":
        spec = dict(m)
        spec["keep"] = plan.keep_mask(mid, ds.n_pop)
        log(f"train {mid} ({m['arch']}, {m['train']['method']}) on {ds.describe()}, "
            f"{int(spec['keep'].sum())} members")
        r = train_model(spec, ds, mdir, device)
        sys.exit(EXIT_PAUSED if r == "paused" else 0)
    if kind == "sig":
        probes = plan.probes[m["dataset"]]
        src = {f: [os.path.join(res, "models", s["model_id"]) for s in plan.sources_of(f)]
               for f in plan.probe_source_families(m["dataset"])}
        out = os.path.join(res, "signals", f"{mid}.npz")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        log(f"signals {mid}: {len(probes)} probes, sources {list(src)}")
        if not os.path.exists(out):
            extract_signals(mdir, ds, probes, src, out, device, dconf.get("extract_bs", 500))
        keep = plan.keep_mask(mid, ds.n_pop)
        ev = evaluate_evasion(mdir, ds, keep, dconf.get("evasion_eps", [2, 8]),
                              dconf.get("robust_eps", 8), device, bs=dconf.get("extract_bs", 500))
        save_json(os.path.join(res, "evasion", f"{mid}.json"), ev)
        sys.exit(0)
    raise SystemExit(f"unknown worker task {args.worker}")


# ─── driver ───────────────────────────────────────────────────────────────────

def is_done(res, task):
    kind, _, mid = task.partition(":")
    if kind == "train":
        return os.path.exists(os.path.join(res, "models", mid, "meta.json"))
    if kind == "sig":
        return (os.path.exists(os.path.join(res, "signals", f"{mid}.npz")) and
                os.path.exists(os.path.join(res, "evasion", f"{mid}.json")))
    return False


def probed_models(plan):
    """Models that need signals: every non-source model (sources only build transfer probes)."""
    return [m for m, d in plan.models.items() if not d["is_source"]]


def gpu_slots():
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    return [str(i) for i in range(n)] or [None]


def launch(args, res, task, slot):
    env = dict(os.environ)
    if slot is not None:
        env["CUDA_VISIBLE_DEVICES"] = slot
    cmd = [sys.executable, "-u", os.path.join(HERE, "run.py"), "--worker", task, "--results", res]
    if args.smoke:
        cmd.append("--smoke")
    if args.dataset_override:
        cmd += ["--dataset_override", args.dataset_override]
    logf = open(os.path.join(res, "logs", task.replace(":", "_") + ".log"), "a")
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env), logf


def tail(path, n=15):
    try:
        with open(path) as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def status(plan, res, failures, note=""):
    n_m = len(plan.models)
    probed = probed_models(plan)
    tr = sum(is_done(res, f"train:{m}") for m in plan.models)
    sg = sum(is_done(res, f"sig:{m}") for m in probed)
    pairs = [(s["name"], t) for s in plan.settings for t in plan.targets_of(s)]
    at = sum(os.path.exists(os.path.join(res, "attacks", f"{s}__{t}.json")) for s, t in pairs)
    complete = tr == n_m and sg == len(probed) and at == len(pairs) and \
        os.path.exists(os.path.join(res, "summary", "REPORT.md"))
    line = f"models trained {tr}/{n_m}, signals {sg}/{len(probed)}, attack pairs {at}/{len(pairs)}"
    head = ("**COMPLETE** — download the output; the report is `summary/REPORT.md`." if complete else
            "**NOT FINISHED** — attach this notebook's latest output as an input (Add Input → Your Work "
            "→ Notebooks) and Save Version again; it resumes where it stopped.")
    txt = [f"# {plan.experiment}{' (smoke)' if plan.smoke else ''} — status", "",
           f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')}; code {code_version()}.", "", head, "",
           f"* {line}", f"* session time left: {fmt_s(time_left())}"]
    if failures:
        txt += ["", "Failed tasks this session (see `logs/<task>.log`):", ""]
        txt += [f"* `{t}` × {c}" for t, c in failures.items()]
    if note:
        txt += ["", note]
    with open(os.path.join(res, "STATUS.md"), "w") as f:
        f.write("\n".join(txt) + "\n")
    save_json(os.path.join(res, "kaggle_state.json"),
              {"updated": time.time(), "trained": tr, "signals": sg, "attack_pairs": at,
               "n_models": n_m, "n_pairs": len(pairs), "complete": complete, "code": code_version()})
    return complete, line


def schedule(args, plan, res):
    order = plan.train_order()
    train_tasks = [f"train:{m}" for m in order]
    probed = set(probed_models(plan))
    sig_tasks = [f"sig:{m}" for m in order if m in probed]
    deps = {f"sig:{m}": [f"train:{d}" for d in plan.signal_deps(m)] for m in probed}
    slots = gpu_slots()
    log(f"GPU slots: {slots}")
    running, failures, paused = {}, Counter(), False
    last_beat = 0.0

    def ready_task():
        busy = {t for t, _, _ in running.values()}
        for t in sig_tasks:
            if t not in busy and failures[t] < 2 and not is_done(res, t) and \
                    all(is_done(res, d) for d in deps[t]):
                return t
        for t in train_tasks:
            if t not in busy and failures[t] < 2 and not is_done(res, t):
                return t
        return None

    while True:
        for slot, (task, p, logf) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            logf.close()
            del running[slot]
            if rc == 0:
                log(f"done   {task}")
            elif rc == EXIT_PAUSED:
                log(f"paused {task} (deadline)")
                paused = True
            else:
                failures[task] += 1
                log(f"FAILED {task} (exit {rc}):\n" + tail(os.path.join(res, "logs", task.replace(":", "_") + ".log")))
        if not paused:
            for slot in slots:
                if slot in running:
                    continue
                t = ready_task()
                if t is None:
                    break
                need = TIMES["min_start_sig"] if t.startswith("sig") else TIMES["min_start_train"]
                if time_left() < need * 60:
                    log(f"not starting {t}: {fmt_s(time_left())} left")
                    paused = True
                    break
                p, logf = launch(args, res, t, slot)
                running[slot] = (t, p, logf)
                log(f"start  {t} on GPU {slot}")
        if not running and (paused or ready_task() is None):
            break
        if time.time() - last_beat > 600:
            last_beat = time.time()
            _, line = status(plan, res, failures)
            log(f"heartbeat: {line}; running {[t for t, _, _ in running.values()]}; left {fmt_s(time_left())}")
        time.sleep(5)
    return failures, paused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--report_only", action="store_true")
    ap.add_argument("--dataset_override")
    ap.add_argument("--worker")
    args = ap.parse_args()

    from fmia.plan import Plan
    res = os.path.abspath(args.results or default_results(args.smoke))
    plan = Plan(config, smoke=args.smoke, dataset_override=args.dataset_override)
    if args.worker:
        set_log_file(None)
        return worker(args, plan, res)

    restore_previous(res)
    for d in ("logs", "models", "signals", "evasion", "attacks", "summary"):
        os.makedirs(os.path.join(res, d), exist_ok=True)
    set_log_file(os.path.join(res, "logs", "driver.log"))
    if "FMIA_DEADLINE" not in os.environ:
        os.environ["FMIA_DEADLINE"] = str(time.time() + args.hours * 3600)
    log(f"=== {plan.experiment}{' SMOKE' if args.smoke else ''}: {len(plan.models)} models, "
        f"{len(plan.settings)} settings, results {res}, code {code_version()}, budget {fmt_s(time_left())}")

    from fmia.report import write_report
    if args.report_only:
        _, line = status(plan, res, Counter())
        log(f"report: {write_report(plan, res, line)}")
        return

    from fmia.data import prefetch
    for key in plan.datasets:
        log(f"dataset {key}: fetching")
        prefetch(key)

    failures, paused = schedule(args, plan, res)
    all_sig = all(is_done(res, f"sig:{m}") for m in probed_models(plan))
    if (all_sig or args.partial) and time_left() > 60 * TIMES["min_start_attack"]:
        log("attacks: start")
        cmd = [sys.executable, "-u", os.path.join(HERE, "run.py"), "--worker", "attack", "--results", res]
        if args.smoke:
            cmd.append("--smoke")
        if args.dataset_override:
            cmd += ["--dataset_override", args.dataset_override]
        with open(os.path.join(res, "logs", "attack.log"), "a") as logf:
            rc = subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT)
        if rc not in (0, EXIT_PAUSED):
            failures["attack"] += 1
            log("FAILED attack:\n" + tail(os.path.join(res, "logs", "attack.log"), 30))
    elif not all_sig:
        log("attacks: waiting for all signals (use --partial to attack what exists)")
    _, line = status(plan, res, failures)
    try:
        log(f"report: {write_report(plan, res, line)}")
    except Exception as e:                      # the report must never lose the session's work
        failures["report"] += 1
        log(f"report FAILED: {e!r}")
    complete, line = status(plan, res, failures)
    log(f"=== {'COMPLETE' if complete else 'NOT FINISHED'}: {line}")


if __name__ == "__main__":
    main()
