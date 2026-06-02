#!/usr/bin/env python3
"""Run queue: snapshot a train .py, enqueue it, run serially on 4 GPUs.

  python run.py add <file.py> <name>   freeze the .py, enqueue it, open a tmux window
  python run.py worker                  serial gatekeeper (window 0 of the tmux session)
  python run.py ls                      show the queue
  python run.py rm <id>                 remove a queued run (snapshot + its window)
  python run.py cancel <id>             cancel a queued run, or kill it if running
  python run.py cell <id>               (internal) the per-run tmux window: wait then run

Each run gets its own tmux window (named after the run) in the worker's session.
The window shows a WAITING banner while queued, then live training output once the
worker grants its turn. The worker is a pure serial gatekeeper: one run at a time.
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
NPROC = 4                # all runs use 4 GPUs
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


def existing_logfiles():
    return set(glob.glob(os.path.join(ROOT, "logs", "*.txt")))


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
def cmd_add(src, name=None):
    if name is None:
        sys.exit("a run name is required:  python3 run.py add <file.py> <name>")
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
    shutil.copyfile(src, snap)  # verbatim freeze = source of truth

    save({
        "id": eid, "seq": seq, "name": name,
        "source": src,
        "snapshot": os.path.relpath(snap, ROOT),
        "console": os.path.relpath(os.path.join(QUEUE_DIR, eid + ".console.log"), ROOT),
        "nproc": NPROC,
        "status": QUEUED,
        "submitted_at": now(), "started_at": None, "finished_at": None,
        "pid": None, "run_uuid": None, "logfile": None,
        "exit_code": None, "cancel_requested": False,
    })
    print("queued %s  (snapshot: snapshots/%s.py)" % (eid, eid))
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
    before = existing_logfiles()  # serial -> the new logs/*.txt is unambiguously ours
    sys.stdout.write("\033[2J\033[H")
    print("=== %s : STARTING on %d GPUs ===\n" % (eid, e["nproc"]))
    # `exit ${PIPESTATUS[0]}` propagates torchrun's exit code, not tee's (tee
    # exits 0 on EOF, which would mask a crashed/aborted run as a clean success).
    cmd = "torchrun --standalone --nproc_per_node=%d %s 2>&1 | tee %s; exit ${PIPESTATUS[0]}" % (
        e["nproc"], shlex.quote(snapshot), shlex.quote(console))
    # own session/process-group so cancel can kill the whole torchrun tree;
    # bash (not the default /bin/sh) so ${PIPESTATUS} is available
    proc = subprocess.Popen(cmd, shell=True, executable="/bin/bash",
                            cwd=ROOT, start_new_session=True)
    e = load(eid); e["pid"] = proc.pid; save(e)

    logfile_found = False
    while proc.poll() is None:
        if not logfile_found:
            new = existing_logfiles() - before
            if new:
                lf = max(new, key=os.path.getmtime)
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
def cmd_worker():
    print("[worker] gatekeeper started, polling %s" % QUEUE_DIR)
    while True:
        nxt = next((e for e in all_entries() if e["status"] == QUEUED), None)
        if nxt is None:
            time.sleep(POLL)
            continue
        eid = nxt["id"]
        if load(eid).get("cancel_requested"):
            e = load(eid); e.update(status=CANCELLED, finished_at=now()); save(e)
            print("[worker] %s cancelled before start" % eid)
            continue
        # make sure the run has a window (e.g. it was added while the session was down)
        open_window(eid)
        # grant the run its turn; its cell window sees RUNNING and launches torchrun
        e = load(eid); e.update(status=RUNNING, started_at=now()); save(e)
        print("[worker] granted %s, waiting for it to finish" % eid)
        # wait until the cell reports a terminal status (or its window vanished)
        while True:
            e = load(eid)
            if e["status"] in (DONE, FAILED, CANCELLED):
                break
            if not window_exists(eid):  # user closed the window mid-run
                kill_pgid(e.get("pid"))
                e.update(status=FAILED, finished_at=now())
                save(e)
                print("[worker] %s window vanished -> failed" % eid)
                break
            time.sleep(POLL)
        print("[worker] %s -> %s" % (eid, load(eid)["status"]))


# -------------------------------------------------------------------- ls/rm/etc
def cmd_ls():
    entries = all_entries()
    if not entries:
        print("queue empty")
        return
    print("%-22s %-9s %s" % ("ID", "STATUS", "LOGFILE / SNAPSHOT"))
    for e in entries:
        loc = e.get("logfile") or e["snapshot"]
        extra = ""
        if e["status"] == FAILED and e.get("exit_code") is not None:
            extra = " (exit %s)" % e["exit_code"]
        print("%-22s %-9s %s%s" % (e["id"], e["status"], loc, extra))


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
        cmd_add(*rest)
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
