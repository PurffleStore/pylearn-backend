import os
import sys
import subprocess
import threading
import traceback
from glob import glob
from flask import Blueprint, jsonify, request

ingest_trigger_bp = Blueprint("ingest_trigger_bp", __name__)

# Prevent concurrent executions
_ingest_lock = threading.Lock()

def _list_dir_sample(path: str, patterns=("*.pdf", "*.PDF"), limit: int = 10):
    try:
        files = []
        for pat in patterns:
            files.extend(glob(os.path.join(path, pat)))
        files.sort()
        total = len(files)
        sample = files[:limit]
        return {"exists": os.path.isdir(path), "total": total, "sample": sample}
    except Exception as e:
        return {"exists": False, "error": str(e)}

def _count_files(path: str, exts=(".json", ".sqlite3", ".parquet", ".bin", ".txt", ".lock")):
    try:
        if not os.path.isdir(path):
            return {"exists": False, "total": 0, "by_ext": {}}
        counts = {}
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                total += 1
                ext = os.path.splitext(f)[1].lower()
                counts[ext] = counts.get(ext, 0) + 1
        return {"exists": True, "total": total, "by_ext": counts}
    except Exception as e:
        return {"exists": False, "total": 0, "error": str(e)}

@ingest_trigger_bp.route("/ingest/ping", methods=["GET"])
def ingest_ping():
    info = {
        "ok": True,
        "cwd": os.getcwd(),
        "env": {
            "ENV": os.getenv("ENV"),
            "INGEST_MODULE": os.getenv("INGEST_MODULE", "ragg.ingest_all"),
            "INGEST_TIMEOUT_SEC": os.getenv("INGEST_TIMEOUT_SEC", "1800"),
            "CHROMA_DIR": os.getenv("CHROMA_DIR"),
            "CHROMA_ROOT": os.getenv("CHROMA_ROOT"),
            "HF_HOME": os.getenv("HF_HOME"),
        }
    }
    print("\n=== [PING] ===", flush=True)
    print(info, flush=True)
    return jsonify(info), 200

@ingest_trigger_bp.route("/ingest/debug", methods=["GET"])
def ingest_debug():
    import importlib
    check_paths = ["ragg/pdfs/low", "ragg/pdfs/mid", "ragg/pdfs/high",
                   "pdfs/low", "pdfs/mid", "pdfs/high"]
    paths_info = []
    for p in check_paths:
        sample = _list_dir_sample(p)
        paths_info.append({"path": p, **sample})

    mod_name = os.getenv("INGEST_MODULE", "ragg.ingest_all")
    import_ok, callable_ok, import_error = False, False, None
    try:
        mod = importlib.import_module(mod_name)
        import_ok = True
        callable_ok = hasattr(mod, "ingest_all_levels")
    except Exception as e:
        import_error = f"{type(e).__name__}: {e}"

    resp = {
        "cwd": os.getcwd(),
        "env": {
            "ENV": os.getenv("ENV"),
            "INGEST_MODULE": mod_name,
            "CHROMA_DIR": os.getenv("CHROMA_DIR"),
            "CHROMA_ROOT": os.getenv("CHROMA_ROOT"),
        },
        "paths": paths_info,
        "import_ok": import_ok,
        "callable_ok": callable_ok,
        "import_error": import_error
    }
    print("\n=== [INGEST DEBUG] ===", flush=True)
    print(resp, flush=True)
    return jsonify(resp), 200

@ingest_trigger_bp.route("/ingest/status", methods=["GET"])
def ingest_status():
    chroma_root = os.getenv("CHROMA_ROOT") or "/data/chroma"
    levels = ["low", "mid", "high"]
    out = {"chroma_root": chroma_root, "levels": {}}
    for lv in levels:
        out["levels"][lv] = _count_files(os.path.join(chroma_root, lv))
    print("\n=== [INGEST STATUS] ===", flush=True)
    print(out, flush=True)
    return jsonify(out), 200

# Inline run (explicit)
@ingest_trigger_bp.route("/ingest/run-inline", methods=["POST"])
def ingest_run_inline():
    import importlib
    try:
        print("\n=== [INGEST INLINE] ===", flush=True)
        mod_name = os.getenv("INGEST_MODULE", "ragg.ingest_all")
        print("Importing module:", mod_name, flush=True)
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, "ingest_all_levels"):
            return jsonify({"status": "error", "message": "ingest_all_levels() not found"}), 500

        # Quick preflight
        for p in ["ragg/pdfs/low", "ragg/pdfs/mid", "ragg/pdfs/high", "pdfs/low", "pdfs/mid", "pdfs/high"]:
            print(f"[INLINE] {p} -> {_list_dir_sample(p)}", flush=True)

        mod.ingest_all_levels()
        return jsonify({"status": "success", "message": "Ingest completed inline"}), 200
    except Exception as e:
        print("[ERROR][INLINE]", e, flush=True)
        print(traceback.format_exc(), flush=True)
        return jsonify({"status": "error", "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc()}), 500

# /ingest – inline by default; subprocess kept as optional fallback
@ingest_trigger_bp.route("/ingest", methods=["POST"])
def run_ingest():
    if not _ingest_lock.acquire(blocking=False):
        print("[DEBUG] Ingest lock already held → busy", flush=True)
        return jsonify({"status": "busy", "message": "Ingestion already in progress"}), 409

    try:
        use_subprocess = os.getenv("INGEST_USE_SUBPROCESS", "0") == "1"
        module_name = os.getenv("INGEST_MODULE", "ragg.ingest_all")
        timeout_sec = int(os.getenv("INGEST_TIMEOUT_SEC", "1800"))

        print("\n=== [INGEST TRIGGER] ===", flush=True)
        print("Trigger called", flush=True)
        print("Use subprocess:", use_subprocess, flush=True)
        print("Module:", module_name, flush=True)
        print("CWD:", os.getcwd(), flush=True)
        print("PYTHON:", sys.executable, flush=True)
        print("PYTHONPATH (len={}):".format(len(sys.path)), flush=True)
        print(sys.path, flush=True)
        print("ENV:", os.getenv("ENV"), flush=True)
        print("CHROMA_DIR:", os.getenv("CHROMA_DIR"), flush=True)
        print("CHROMA_ROOT:", os.getenv("CHROMA_ROOT"), flush=True)

        # Preflight: show expected PDF folders
        for b in ["ragg/pdfs/low", "ragg/pdfs/mid", "ragg/pdfs/high", "pdfs/low", "pdfs/mid", "pdfs/high"]:
            info = _list_dir_sample(b)
            print(f"  - {b}: exists={info.get('exists')} total={info.get('total', 0)} sample={info.get('sample', [])}", flush=True)

        if not use_subprocess:
            # Inline mode (default)
            import importlib
            print("[DEBUG] Running inline ingestion...", flush=True)
            mod = importlib.import_module(module_name)
            if not hasattr(mod, "ingest_all_levels"):
                return jsonify({"status": "error", "message": "ingest_all_levels() not found"}), 500
            mod.ingest_all_levels()
            return jsonify({"status": "success", "mode": "inline"}), 200

        # Subprocess fallback (set INGEST_USE_SUBPROCESS=1 to enable)
        cmd = [sys.executable, "-m", module_name]
        print("\n[DEBUG] Running subprocess:", cmd, flush=True)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=None,
            timeout=timeout_sec
        )
        print("\n[DEBUG] Subprocess completed.", flush=True)
        print("Return code:", result.returncode, flush=True)

        def _preview(label, text, head=30, tail=30):
            lines = (text or "").splitlines()
            print(f"\n----- {label} (total lines: {len(lines)}) -----", flush=True)
            for line in lines[:head]:
                print(line, flush=True)
            if len(lines) > head + tail:
                print("... [truncated] ...", flush=True)
            for line in lines[-tail:]:
                print(line, flush=True)
            print("----- END", label, "-----\n", flush=True)

        _preview("STDOUT", result.stdout)
        _preview("STDERR", result.stderr)

        payload = {
            "status": "success" if result.returncode == 0 else "error",
            "returncode": result.returncode,
            "stdout": (result.stdout or "")[-4000:],
            "stderr": (result.stderr or "")[-4000:],
            "mode": "subprocess"
        }
        status_code = 200 if result.returncode == 0 else 500
        return jsonify(payload), status_code

    except subprocess.TimeoutExpired:
        print("[ERROR] Ingestion timed out.", flush=True)
        return jsonify({"status": "timeout"}), 504

    except Exception as e:
        print("[ERROR] Exception during ingestion:", e, flush=True)
        print(traceback.format_exc(), flush=True)
        return jsonify({
            "status": "error",
            "message": "trigger crashed",
            "traceback": traceback.format_exc()
        }), 500

    finally:
        try:
            _ingest_lock.release()
        except Exception:
            pass
