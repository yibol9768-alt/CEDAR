import os
import re
import time
import glob
import tarfile
import subprocess

COLAB = os.path.expanduser("~/.local/bin/colab")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_LOCAL = os.path.join(ROOT, "results_v3", "model_preds_v3")
CKPT_LOCAL = os.path.join(ROOT, "results_v3", "checkpoints")
LOG_LOCAL = os.path.join(ROOT, "logs", "colab_workers")
os.makedirs(PRED_LOCAL, exist_ok=True)
os.makedirs(CKPT_LOCAL, exist_ok=True)
os.makedirs(LOG_LOCAL, exist_ok=True)

SESSIONS = ["cedar_v3_f0", "cedar_v3_f1", "cedar_v3_f2", "cedar_v3_f3"]
REQUIRED_CKPTS = [
    "mb_v3_cedar_split_op_seed42.pt",
    "mb_v3_strong_combined_no_nsd_seed42.pt",
    "db_v3_cedar_split_op_seed42.pt",
]
# Already trained on the Mac (JSON synced); the 1.6GB .pt is relayed separately, so never retrain it.
PENDING_RELAY_CKPTS = ["db_v3_cedar_split_op_seed42.pt"]
LOOP_REQUIRED_CKPTS = [c for c in REQUIRED_CKPTS if c not in PENDING_RELAY_CKPTS]

WORKER_SPECS = [
    ("cedar_v3_f0", ["CEDAR_WORKER_ID=0", "CEDAR_NUM_WORKERS=4"], "v3_4way_worker.py"),
    ("cedar_v3_f1", ["CEDAR_WORKER_ID=1", "CEDAR_NUM_WORKERS=4"], "v3_4way_worker.py"),
    ("cedar_v3_f2", ["CEDAR_WORKER_ID=2", "CEDAR_NUM_WORKERS=4"], "v3_4way_worker.py"),
    ("cedar_v3_f3", ["CEDAR_WORKER_ID=3", "CEDAR_NUM_WORKERS=4"], "v3_4way_worker.py"),
]

JSON_RE = re.compile(r"^[a-zA-Z0-9_]+\.json$")


def run_cmd(args, timeout=180):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        print(f"  [WARN] subprocess {args[:4]} exception: {e}", flush=True)
        return None


def prepare_and_upload_bundles():
    full_tar = "/tmp/cedar_v3_fresh_full_bundle.tar.gz"
    archived_ckpts = [
        ck for ck in REQUIRED_CKPTS
        if os.path.exists(os.path.join(CKPT_LOCAL, ck))
        and os.path.getsize(os.path.join(CKPT_LOCAL, ck)) > 10_000_000
    ]
    archived_ckpts = sorted(set(archived_ckpts) | set(PENDING_RELAY_CKPTS))
    with tarfile.open(full_tar, "w:gz") as tf:
        for rel in [
            "src/cedar_model.py",
            "src/cedar_v2_model.py",
            "scripts/v3_train_cedar_v2_suite.py",
            "scripts/v3_4way_worker.py",
            "scripts/v3_theorem2_convergence_and_geometry.py",
        ]:
            tf.add(os.path.join(ROOT, rel), arcname=rel)
        for df in sorted(glob.glob(os.path.join(ROOT, "data", "*.json"))):
            tf.add(df, arcname=f"cedar_data/{os.path.basename(df)}")
        for df2 in sorted(glob.glob(os.path.join(ROOT, "data_v2", "*.json"))):
            tf.add(df2, arcname=f"cedar_data_v2/{os.path.basename(df2)}")
        for jf in sorted(glob.glob(os.path.join(PRED_LOCAL, "*.json"))):
            tf.add(jf, arcname=f"cached_preds/{os.path.basename(jf)}")

    sz_mb = os.path.getsize(full_tar) / (1024 * 1024)
    print(f"  [BUNDLE] Created {full_tar} ({sz_mb:.1f} MB) | local_archived_ckpts={archived_ckpts}", flush=True)

    unpack_py = "/tmp/cedar_v3_unpack_fresh.py"
    with open(unpack_py, "w") as f:
        f.write(
            "import os, glob, shutil, tarfile, subprocess, sys\n"
            "for pkg in ['sentencepiece', 'tiktoken', 'protobuf']:\n"
            "    try:\n"
            "        __import__(pkg if pkg != 'protobuf' else 'google.protobuf')\n"
            "    except ImportError:\n"
            "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])\n"
            "os.makedirs('/tmp/ext_v3_fresh', exist_ok=True)\n"
            "tarfile.open('/content/cedar_fresh_bundle.tar.gz').extractall('/tmp/ext_v3_fresh')\n"
            "for s_d, d_d in [\n"
            "    ('src', '/content/src'),\n"
            "    ('scripts', '/content/scripts'),\n"
            "    ('cedar_data', '/content/cedar_data'),\n"
            "    ('cedar_data_v2', '/content/cedar_data_v2'),\n"
            "    ('cached_preds', '/content/cedar_results_v3/model_preds_v3'),\n"
            "]:\n"
            "    p = f'/tmp/ext_v3_fresh/{s_d}'\n"
            "    if os.path.exists(p):\n"
            "        shutil.copytree(p, d_d, dirs_exist_ok=True)\n"
            "for init_p in ['/content/src/__init__.py', '/content/scripts/__init__.py']:\n"
            "    if not os.path.exists(init_p):\n"
            "        open(init_p, 'w').close()\n"
            "os.makedirs('/content/cedar_results_v3/model_preds_v3', exist_ok=True)\n"
            "os.makedirs('/content/cedar_results_v3/checkpoints', exist_ok=True)\n"
            f"for ck in {archived_ckpts!r}:\n"
            "    ck_p = f'/content/cedar_results_v3/checkpoints/{ck}'\n"
            "    if not os.path.exists(ck_p):\n"
            "        open(ck_p, 'wb').write(b'LOCAL_ARCHIVED')\n"
            "shutil.rmtree('/tmp/ext_v3_fresh', ignore_errors=True)\n"
            "n_j = len(glob.glob('/content/cedar_results_v3/model_preds_v3/*.json'))\n"
            "n_d1 = len(glob.glob('/content/cedar_data/*.json'))\n"
            "n_d2 = len(glob.glob('/content/cedar_data_v2/*.json'))\n"
            "print(f'UNPACK_FRESH_OK cached_jsons={n_j} d1={n_d1} d2={n_d2}')\n"
        )

    for sess in SESSIONS:
        up = run_cmd([COLAB, "upload", "-s", sess, full_tar, "/content/cedar_fresh_bundle.tar.gz"], timeout=360)
        assert up and up.returncode == 0, f"Upload failed on {sess}: {up.stderr if up else 'None'}"
        ex = run_cmd([COLAB, "exec", "-s", sess, "-f", unpack_py, "--timeout", "180"], timeout=210)
        assert ex and ex.returncode == 0 and "UNPACK_FRESH_OK" in ex.stdout, (
            f"Unpack failed on {sess}: {ex.stdout if ex else ''} {ex.stderr if ex else ''}"
        )
        print(f"  [OK] {sess} -> {ex.stdout.strip()}", flush=True)


def spawn_worker(sess, envs, script_name):
    cmd = [COLAB, "exec", "-s", sess]
    for ev in envs:
        cmd.extend(["--env", ev])
    cmd.extend(["-f", os.path.join(ROOT, "scripts", script_name), "--timeout", "14400"])
    log_path = os.path.join(LOG_LOCAL, f"{sess}.log")
    log_f = open(log_path, "a")
    log_f.write(f"\n=== SPAWN {time.strftime('%H:%M:%S')} {sess} {script_name} {envs} ===\n")
    log_f.flush()
    p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)
    print(f"  [LAUNCHED] {sess} -> {script_name} ({envs}) pid={p.pid}", flush=True)
    return (p, log_f, envs, script_name, log_path)


def sync_once():
    for sess in SESSIONS:
        r_j = run_cmd([COLAB, "ls", "-s", sess, "/content/cedar_results_v3/model_preds_v3"], timeout=45)
        if r_j and r_j.returncode == 0:
            for line in r_j.stdout.splitlines():
                fn = line.strip()
                if JSON_RE.match(fn):
                    dst = os.path.join(PRED_LOCAL, fn)
                    if not (os.path.exists(dst) and os.path.getsize(dst) > 1000):
                        part = dst + ".part"
                        dl = run_cmd([COLAB, "download", "-s", sess, f"/content/cedar_results_v3/model_preds_v3/{fn}", part], timeout=120)
                        if dl and dl.returncode == 0 and os.path.exists(part) and os.path.getsize(part) > 1000:
                            os.replace(part, dst)
                        elif os.path.exists(part):
                            os.remove(part)

        r_c = run_cmd([COLAB, "ls", "-s", sess, "/content/cedar_results_v3/checkpoints"], timeout=45)
        if r_c and r_c.returncode == 0:
            for line in r_c.stdout.splitlines():
                ck = line.strip()
                if ck in REQUIRED_CKPTS:
                    dst = os.path.join(CKPT_LOCAL, ck)
                    if not (os.path.exists(dst) and os.path.getsize(dst) > 10_000_000):
                        part = dst + ".part"
                        dl = run_cmd([COLAB, "download", "-s", sess, f"/content/cedar_results_v3/checkpoints/{ck}", part], timeout=600)
                        if dl and dl.returncode == 0 and os.path.exists(part) and os.path.getsize(part) > 10_000_000:
                            os.replace(part, dst)
                        elif os.path.exists(part):
                            os.remove(part)


def valid_local_ckpts():
    return sorted([
        f for f in os.listdir(CKPT_LOCAL)
        if f.endswith(".pt") and not f.endswith(".part")
        and os.path.getsize(os.path.join(CKPT_LOCAL, f)) > 10_000_000
    ])


def main():
    existing_j = len(glob.glob(os.path.join(PRED_LOCAL, "*.json")))
    print(f"=== [1/4] Uploading full code + data + {existing_j} cached JSONs to 4 fresh GPUs ===", flush=True)
    prepare_and_upload_bundles()

    print("=== [2/4] Launching 4 Evenly Sharded Workers (16 remaining jobs/GPU) ===", flush=True)
    procs = {}
    for sess, envs, script_name in WORKER_SPECS:
        procs[sess] = spawn_worker(sess, envs, script_name)

    print("=== [3/4] Entering 75s Keepalive + Incremental Sync Loop ===", flush=True)
    restarts = {s: 0 for s in SESSIONS}
    while True:
        time.sleep(75)
        sync_once()
        jsons = sorted([f for f in os.listdir(PRED_LOCAL) if f.endswith(".json") and not f.endswith(".part")])
        ckpts = valid_local_ckpts()
        have_req_ckpts = all(c in ckpts for c in LOOP_REQUIRED_CKPTS)

        if len(jsons) >= 116 and have_req_ckpts:
            print("=== ALL 116 PREDICTION JSONS & ALL 3 CANONICAL CHECKPOINTS SYNCED LOCALLY! ===", flush=True)
            break

        alive_sessions = []
        for sess, (p, lf, envs, sname, lpath) in list(procs.items()):
            rc = p.poll()
            if rc is None:
                alive_sessions.append(sess)
            elif rc != 0 and restarts[sess] < 3:
                restarts[sess] += 1
                print(f"  [SELF-HEAL] {sess} exited with rc={rc}; restarting kernel & relaunching ({restarts[sess]}/3)...", flush=True)
                lf.close()
                run_cmd([COLAB, "restart-kernel", "-s", sess], timeout=90)
                procs[sess] = spawn_worker(sess, envs, sname)
                alive_sessions.append(sess)

        ts = time.strftime("%H:%M:%S")
        print(
            f"[LIVE SYNC {ts}] JSONs: {len(jsons)}/116 | Ckpts: {len(ckpts)} ({ckpts}) | Alive: {alive_sessions}",
            flush=True,
        )

        if not alive_sessions:
            sync_once()
            jsons = sorted([f for f in os.listdir(PRED_LOCAL) if f.endswith(".json") and not f.endswith(".part")])
            print(f"[ALL WORKERS EXITED] Final JSON count: {len(jsons)}/116", flush=True)
            break

    for s, (p, lf, _, _, _) in procs.items():
        if p.poll() is None:
            p.terminate()
        lf.close()
    run_cmd(["pkill", "-f", "colab exec -s cedar_v3_f"], timeout=15)
    time.sleep(2)

    jsons = sorted([f for f in os.listdir(PRED_LOCAL) if f.endswith(".json") and not f.endswith(".part")])
    ckpts = valid_local_ckpts()
    assert len(jsons) == 116, f"Expected 116 JSONs before Theorem 1 geometry, got {len(jsons)}"
    assert all(c in ckpts for c in LOOP_REQUIRED_CKPTS), f"Missing required checkpoints: {ckpts}"

    print("=== [4/4] Running Theorem 1 Disjoint-Pool Convergence & Geometry on cedar_v3_f0 ===", flush=True)
    run_cmd([COLAB, "restart-kernel", "-s", "cedar_v3_f0"], timeout=90)
    seed42_pt = os.path.join(CKPT_LOCAL, "mb_v3_cedar_split_op_seed42.pt")
    # Ensure full >10MB mb_v3_cedar_split_op_seed42.pt is on cedar_v3_f0
    up_ck = run_cmd(
        [COLAB, "upload", "-s", "cedar_v3_f0", seed42_pt, "/content/cedar_results_v3/checkpoints/mb_v3_cedar_split_op_seed42.pt"],
        timeout=600,
    )
    assert up_ck and up_ck.returncode == 0, "Failed to upload mb_v3_cedar_split_op_seed42.pt to cedar_v3_f0!"

    t2_script = os.path.join(ROOT, "scripts", "v3_theorem2_convergence_and_geometry.py")
    r_t2 = subprocess.run([COLAB, "exec", "-s", "cedar_v3_f0", "-f", t2_script, "--timeout", "1800"], text=True)
    assert r_t2.returncode == 0, "Theorem 1/2 convergence & geometry script failed!"

    t2_dst = os.path.join(ROOT, "results_v3", "nsd_v3_theorem2_and_geometry.json")
    dl_t2 = run_cmd([COLAB, "download", "-s", "cedar_v3_f0", "/content/cedar_results_v3/nsd_v3_theorem2_and_geometry.json", t2_dst], timeout=120)
    assert dl_t2 and dl_t2.returncode == 0 and os.path.exists(t2_dst), "Failed to download nsd_v3_theorem2_and_geometry.json!"

    print("=== Stopping ONLY our 4 sessions (cedar_v3_f0, cedar_v3_f1, cedar_v3_f2, cedar_v3_f3) ===", flush=True)
    for sess in SESSIONS:
        run_cmd([COLAB, "stop", "-s", sess], timeout=60)

    u = run_cmd([COLAB, "usage"], timeout=45)
    if u:
        print(u.stdout, flush=True)
    print("=== ALL 116 RUNS + CHECKPOINTS + THEOREM 1 GEOMETRY SYNCED & ALL GPUS STOPPED! ===", flush=True)


if __name__ == "__main__":
    main()
