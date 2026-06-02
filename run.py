#!/usr/bin/env python3
"""Run queue: snapshot a train .py, enqueue it, run it on a chosen set of GPUs.

  python run.py add <file.py> <name> [opts]   freeze the .py, enqueue it, open a tmux window
  python run.py worker                          GPU-aware scheduler (window 0 of the tmux session)
  python run.py ls                              show the queue
  python run.py rm <id>                         remove a queued run (snapshot + its window)
  python run.py cancel <id>                     cancel a queued run, or kill it if running
  python run.py cell <id>                       (internal) the per-run tmux window: wait then run

`add` options:
  --gpus 2,3     GPUs this run is ALLOWED to use (default: all GPUs). The scheduler
                 places the run on whichever `--nproc` of these are free.
  --nproc 1      number of GPUs this run needs (default: 1, i.e. a single-GPU run)
  --bs 32        override Hyperparameters.batch_size in the frozen snapshot
  --lr 0.002     override Hyperparameters.learning_rate in the frozen snapshot
  --dbs 16       override Hyperparameters.device_batch_size in the frozen snapshot

Each run gets its own tmux window (named after the run) in the worker's session.
The window shows a WAITING banner while queued, then live training output once the
scheduler grants its turn. The worker is a GPU-aware scheduler: it runs as many
queued runs in parallel as fit on free GPUs, each pinned via CUDA_VISIBLE_DEVICES
to the GPUs it was assigned. A single-GPU run (--nproc 1 --gpus 2,3) lands on GPU 2
or 3, whichever is free, so two such runs execute side by side.
The frozen snapshot in snapshots/ is the single source of truth for what ran.
"""
import os
import re
import sys
import glob
import json
import time
import shlex
import shutil
import signal
import tempfile
import datetime
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
QUEUE_DIR = os.path.join(ROOT, "queue")
SNAP_DIR = os.path.join(ROOT, "snapshots")
TMUX_SESSION = "worker"  # session that hosts the worker (win 0) and one window per run
NUM_GPUS = 4             # total GPUs on this box; default allowed set is all of them
POLL = 2.0               # seconds between queue polls

QUEUED, RUNNING, DONE, FAILED, CANCELLED = (
    "queued", "running", "done", "failed", "cancelled")


# ---------------------------------------------------------------- queue helpers
def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def entry_path(eid):
    return os.path.join(QUEUE_DIR, eid + ".json")


def load(eid):
    with open(entry_path(eid)) as f:
        return json.load(f)


def save(entry):
    """Atomically write queue/<id>.json."""
    os.makedirs(QUEUE_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=QUEUE_DIR, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(entry, f, indent=2)
        f.write("\n")
    os.replace(tmp, entry_path(entry["id"]))


def all_entries():
    """All queue entries sorted by sequence number."""
    out = []
    for p in glob.glob(os.path.join(QUEUE_DIR, "*.json")):
        try:
            with open(p) as f:
                out.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue  # skip half-written / transient files
    return sorted(out, key=lambda e: e.get("seq", 0))


# ------------------------------------------------------------------ tmux helpers
def tmux(*args):
    return subprocess.run(["tmux", *args], capture_output=True, text=True)


def session_exists():
    return tmux("has-session", "-t", TMUX_SESSION).returncode == 0


def window_exists(eid):
    r = tmux("list-windows", "-t", TMUX_SESSION, "-F", "#{window_name}")
    return eid in r.stdout.split()


def open_window(eid):
    """Open a tmux window that runs this run's cell (waits, then runs live)."""
    if not session_exists():
        print("warning: tmux session %r is not running, so no window was created.\n"
              "         start the worker first:  python3 run.py worker" % TMUX_SESSION)
        return False
    if window_exists(eid):
        return True
    cellcmd = "%s %s cell %s; exec bash" % (
        shlex.quote(sys.executable), shlex.quote(os.path.join(ROOT, "run.py")), eid)
    # -a: insert after the current window so tmux picks the next free index
    # (plain `-t SESSION` resolves to window 0 and fails with "index 0 in use")
    r = tmux("new-window", "-a", "-t", TMUX_SESSION, "-n", eid, "-c", ROOT, cellcmd)
    if r.returncode != 0:
        print("warning: could not create tmux window for %s: %s" % (eid, r.stderr.strip()))
        return False
    tmux("set-window-option", "-t", "%s:%s" % (TMUX_SESSION, eid),
         "automatic-rename", "off")  # keep our chosen name
    return True


def kill_window(eid):
    tmux("kill-window", "-t", "%s:%s" % (TMUX_SESSION, eid))


def kill_pgid(pid, proc=None):
    """Best-effort terminate a process group (the shell + torchrun + tee)."""
    if not pid:
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    for _ in range(10):
        if proc is not None and proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


# --------------------------------------------------------------- snapshot / add
def _patch_hyperparam(text, field, typ, value):
    """Replace a `<field> : <typ> = ...` line in the snapshot's Hyperparameters.

    Patches are baked into the frozen snapshot so it stays the exact source of
    truth for what ran. Raises if the field isn't found (so a typo can't silently
    run the unmodified default).
    """
    pat = re.compile(r"^(\s*%s\s*:\s*%s\s*=\s*).*$" % (field, typ), re.MULTILINE)
    new, n = pat.subn(r"\g<1>%s  # overridden by run.py" % value, text)
    if n != 1:
        sys.exit("could not patch %s (matched %d lines, expected 1)" % (field, n))
    return new


def cmd_add(argv):
    if len(argv) < 2:
        sys.exit("usage: python3 run.py add <file.py> <name> "
                 "[--gpus 2,3] [--nproc 1] [--bs 32] [--lr 0.002] [--dbs 16]")
    src, name, opts = argv[0], argv[1], argv[2:]

    # parse the optional flags
    gpus = nproc = bs = lr = dbs = None
    i = 0
    while i < len(opts):
        k = opts[i]
        if i + 1 >= len(opts):
            sys.exit("flag %s needs a value" % k)
        v = opts[i + 1]
        if k == "--gpus":   gpus = v
        elif k == "--nproc": nproc = int(v)
        elif k == "--bs":   bs = v
        elif k == "--lr":   lr = v
        elif k == "--dbs":  dbs = v
        else:               sys.exit("unknown flag %r" % k)
        i += 2

    allowed = ([int(g) for g in gpus.split(",") if g != ""] if gpus is not None
               else list(range(NUM_GPUS)))
    if not allowed:
        sys.exit("--gpus must list at least one GPU")
    if nproc is None:
        nproc = 1  # default to a single GPU (run many 1-GPU experiments in parallel);
                   # allowed-GPUs still defaults to all, so the scheduler load-balances them
    if nproc > len(allowed):
        sys.exit("--nproc %d exceeds the %d allowed GPUs %s" % (nproc, len(allowed), allowed))

    src = os.path.abspath(src)
    if not (os.path.isfile(src) and src.endswith(".py")):
        sys.exit("source must be an existing .py file: %s" % src)

    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-_.")
    if not name:
        sys.exit("run name must contain at least one letter, digit, '.', '-' or '_'")
    seq = max([e.get("seq", 0) for e in all_entries()], default=0) + 1
    eid = "%03d_%s" % (seq, name)

    os.makedirs(SNAP_DIR, exist_ok=True)
    snap = os.path.join(SNAP_DIR, eid + ".py")
    with open(src) as f:
        text = f.read()
    if bs is not None:  text = _patch_hyperparam(text, "batch_size", "int", bs)
    if lr is not None:  text = _patch_hyperparam(text, "learning_rate", "float", lr)
    if dbs is not None: text = _patch_hyperparam(text, "device_batch_size", "int", dbs)
    with open(snap, "w") as f:  # frozen (optionally patched) snapshot = source of truth
        f.write(text)

    save({
        "id": eid, "seq": seq, "name": name,
        "source": src,
        "snapshot": os.path.relpath(snap, ROOT),
        "console": os.path.relpath(os.path.join(QUEUE_DIR, eid + ".console.log"), ROOT),
        "allowed_gpus": allowed,
        "nproc": nproc,
        "assigned_gpus": None,
        "status": QUEUED,
        "submitted_at": now(), "started_at": None, "finished_at": None,
        "pid": None, "run_uuid": None, "logfile": None,
        "exit_code": None, "cancel_requested": False,
    })
    overrides = ", ".join(
        "%s=%s" % (k, v) for k, v in (("bs", bs), ("lr", lr), ("dbs", dbs)) if v is not None)
    print("queued %s  (gpus=%s nproc=%d%s)  snapshot: snapshots/%s.py" % (
        eid, allowed, nproc, "  " + overrides if overrides else "", eid))
    open_window(eid)


# ------------------------------------------------------------- per-run tmux cell
def queue_position(eid):
    q = [e["id"] for e in all_entries() if e["status"] == QUEUED]
    return (q.index(eid) + 1, len(q)) if eid in q else None


def cmd_cell(eid):
    # phase 1: wait for the worker to grant our turn
    while load(eid)["status"] == QUEUED and not load(eid).get("cancel_requested"):
        e = load(eid)
        pos = queue_position(eid)
        sys.stdout.write("\033[2J\033[H")  # clear screen
        print("=== %s ===" % eid)
        print("status   : WAITING in queue")
        if pos:
            print("position : %d of %d queued" % pos)
        print("submitted: %s" % e["submitted_at"])
        print("snapshot : %s" % e["snapshot"])
        print("\n(switches to live training output once the worker starts this run)")
        sys.stdout.flush()
        time.sleep(POLL)

    e = load(eid)
    if e["status"] == CANCELLED or e.get("cancel_requested"):
        if e["status"] != CANCELLED:
            e.update(status=CANCELLED, finished_at=now())
            save(e)
        print("\n%s was cancelled before it started." % eid)
        return
    if e["status"] != RUNNING:
        print("\n%s: unexpected status %r, not launching." % (eid, e["status"]))
        return

    # phase 2: run it live in this window (tee to console log for persistence)
    snapshot = os.path.join(ROOT, e["snapshot"])
    console = os.path.join(ROOT, e["console"])
    gpus = e.get("assigned_gpus") or list(range(e["nproc"]))
    cuda_visible = ",".join(str(g) for g in gpus)
    sys.stdout.write("\033[2J\033[H")
    print("=== %s : STARTING on GPU(s) %s (%d proc) ===\n" % (eid, cuda_visible, e["nproc"]))
    # `exit ${PIPESTATUS[0]}` propagates torchrun's exit code, not tee's (tee
    # exits 0 on EOF, which would mask a crashed/aborted run as a clean success).
    # --standalone picks its own free rendezvous port, so parallel runs don't collide.
    cmd = "torchrun --standalone --nproc_per_node=%d %s 2>&1 | tee %s; exit ${PIPESTATUS[0]}" % (
        e["nproc"], shlex.quote(snapshot), shlex.quote(console))
    # CUDA_VISIBLE_DEVICES pins this run to its assigned GPUs (each appears as cuda:0..);
    # own session/process-group so cancel can kill the whole torchrun tree;
    # bash (not the default /bin/sh) so ${PIPESTATUS} is available
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=cuda_visible)
    proc = subprocess.Popen(cmd, shell=True, executable="/bin/bash",
                            cwd=ROOT, start_new_session=True, env=env)
    e = load(eid); e["pid"] = proc.pid; save(e)

    # the training script names its log logs/<run_name>_<uuid>.txt where run_name is
    # the snapshot stem (== eid), so we find OUR log unambiguously even with parallel
    # runs by globbing the eid prefix (the old before/after diff misattributed under
    # concurrency).
    logpat = os.path.join(ROOT, "logs", eid + "_*.txt")
    logfile_found = False
    while proc.poll() is None:
        if not logfile_found:
            matches = glob.glob(logpat)
            if matches:
                lf = max(matches, key=os.path.getmtime)
                e = load(eid)
                e["logfile"] = os.path.relpath(lf, ROOT)
                e["run_uuid"] = os.path.splitext(os.path.basename(lf))[0]
                save(e)
                logfile_found = True
        if load(eid).get("cancel_requested"):
            print("\n[cell] cancel requested, terminating %s" % eid)
            kill_pgid(proc.pid, proc)
            break
        time.sleep(1.0)
    ret = proc.wait()

    e = load(eid)
    status = CANCELLED if e.get("cancel_requested") else (DONE if ret == 0 else FAILED)
    e.update(status=status, finished_at=now(), exit_code=ret, pid=None)
    save(e)
    print("\n=== %s : %s (exit %s) ===" % (eid, status.upper(), ret))


# --------------------------------------------------------------------- worker
def busy_gpus(entries):
    """GPUs currently held by running runs."""
    held = set()
    for e in entries:
        if e["status"] == RUNNING and e.get("assigned_gpus"):
            held.update(e["assigned_gpus"])
    return held


def pick_gpus(allowed, nproc, busy):
    """Return `nproc` free GPUs from `allowed`, or None if not enough are free."""
    free = [g for g in allowed if g not in busy]
    return free[:nproc] if len(free) >= nproc else None


def cmd_worker():
    print("[worker] GPU-aware scheduler started, polling %s" % QUEUE_DIR)
    while True:
        # 1. reap runs whose tmux window vanished (user closed it mid-run)
        for e in all_entries():
            if e["status"] == RUNNING and not window_exists(e["id"]):
                kill_pgid(e.get("pid"))
                e.update(status=FAILED, finished_at=now()); save(e)
                print("[worker] %s window vanished -> failed (freed %s)"
                      % (e["id"], e.get("assigned_gpus")))

        # 2. retire cancelled-while-queued runs so they don't block scheduling
        for e in all_entries():
            if e["status"] == QUEUED and e.get("cancel_requested"):
                e.update(status=CANCELLED, finished_at=now()); save(e)
                print("[worker] %s cancelled before start" % e["id"])

        # 3. place every queued run that fits on currently-free GPUs (parallel).
        #    We scan in submission order but backfill: a run that doesn't fit is
        #    skipped rather than blocking smaller runs behind it, maximizing GPU use.
        busy = busy_gpus(all_entries())
        for e in [x for x in all_entries() if x["status"] == QUEUED]:
            if e.get("cancel_requested"):
                continue
            gpus = pick_gpus(e["allowed_gpus"], e["nproc"], busy)
            if gpus is None:
                continue  # not enough of its allowed GPUs free right now
            open_window(e["id"])  # ensure a window exists (e.g. added while session down)
            e = load(e["id"])
            e.update(status=RUNNING, started_at=now(), assigned_gpus=gpus); save(e)
            busy.update(gpus)  # reserve within this pass so the next run won't double-book
            print("[worker] granted %s on GPU(s) %s" % (e["id"], gpus))

        time.sleep(POLL)


# -------------------------------------------------------------------- ls/rm/etc
def cmd_ls():
    entries = all_entries()
    if not entries:
        print("queue empty")
        return
    print("%-26s %-9s %-10s %s" % ("ID", "STATUS", "GPUS", "LOGFILE / SNAPSHOT"))
    for e in entries:
        loc = e.get("logfile") or e["snapshot"]
        extra = ""
        if e["status"] == FAILED and e.get("exit_code") is not None:
            extra = " (exit %s)" % e["exit_code"]
        # show where it ran (assigned) if known, else what it's allowed to use
        gpus = e.get("assigned_gpus") or e.get("allowed_gpus") or []
        gpustr = ",".join(str(g) for g in gpus) if gpus else "-"
        print("%-26s %-9s %-10s %s%s" % (e["id"], e["status"], gpustr, loc, extra))


def cmd_rm(eid):
    e = load(eid)
    if e["status"] == RUNNING:
        sys.exit("%s is running; use `cancel %s` first" % (eid, eid))
    kill_window(eid)
    snap = os.path.join(ROOT, e["snapshot"])
    if os.path.exists(snap):
        os.remove(snap)
    os.remove(entry_path(eid))
    print("removed %s" % eid)


def cmd_cancel(eid):
    e = load(eid)
    if e["status"] in (DONE, FAILED, CANCELLED):
        sys.exit("%s already %s" % (eid, e["status"]))
    e["cancel_requested"] = True
    save(e)
    print("cancel requested; %s will be killed shortly" % eid
          if e["status"] == RUNNING else
          "%s will be skipped" % eid)


# -------------------------------------------------------------------------- cli
def main(argv):
    if not argv:
        sys.exit(__doc__)
    cmd, rest = argv[0], argv[1:]
    if cmd == "add":
        cmd_add(rest)
    elif cmd == "worker":
        cmd_worker()
    elif cmd == "cell":
        cmd_cell(*rest)
    elif cmd in ("ls", "list"):
        cmd_ls()
    elif cmd == "rm":
        cmd_rm(*rest)
    elif cmd == "cancel":
        cmd_cancel(*rest)
    else:
        sys.exit("unknown command %r\n%s" % (cmd, __doc__))


if __name__ == "__main__":
    os.chdir(ROOT)
    main(sys.argv[1:])
