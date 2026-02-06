import os
import time
import uuid
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from dotenv import load_dotenv, find_dotenv
from flask import Flask, Blueprint, request, jsonify, current_app, send_from_directory
from flask_cors import CORS

import requests
from TTS.api import TTS


try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
except Exception:
    boto3 = None
    NoCredentialsError = ClientError = Exception

# local imports (support running as a package or module)
try:
    from .rag_backend import IngestBody, ingest_documents, ingest_pdfs_from_folder
    from .rag_llm import (
        LLMBody,
        llm_generate,
        ExplainBody,
        llm_explain,
        FollowupBody,
        llm_followups,
        get_vectorstore,
        get_vectorstore_for,
    )
except ImportError:
    from rag_backend import IngestBody, ingest_documents, ingest_pdfs_from_folder
    from rag_llm import (
        LLMBody,
        llm_generate,
        ExplainBody,
        llm_explain,
        FollowupBody,
        llm_followups,
        get_vectorstore,
        get_vectorstore_for,
    )

from openai import OpenAI

load_dotenv(find_dotenv())
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# Configuration
S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1").strip()
S3_PREFIX = os.getenv("S3_PREFIX", "audio/").strip()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()

BASE_DIR = Path(__file__).resolve().parent.parent
MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", str(BASE_DIR / "media")))
AUDIO_DIR = MEDIA_ROOT / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
XTTS_REF_DIR = Path(os.getenv("XTTS_REF_DIR", str(BASE_DIR / "assets")))

DID_API_KEY = os.getenv("DID_API_KEY", "")
DID_SOURCE_IMAGE_URL = os.getenv("DID_SOURCE_IMAGE_URL", "")
DID_VOICE_ID = os.getenv("DID_VOICE_ID", "en-US-JennyNeural")
PDF_DEFAULT_FOLDER = os.getenv("RAG_PDF_DIR", "../assets/pdfs")

# init optional s3 client
_s3_client = None
if boto3 and S3_BUCKET and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    try:
        _s3_client = boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
    except Exception:
        _s3_client = None

rag_bp = Blueprint("rag", __name__)

REMOTE_API_URL = "https://kw6j9hcwmljvpa-5000.proxy.runpod.net/generate"

def _upload_to_s3(file_path: Union[str, Path]) -> Optional[str]:
    """Upload file to S3 and return presigned URL or None.""" 
    if not _s3_client or not S3_BUCKET:
        return None
    try:
        file_path = str(file_path)
        key = f"{S3_PREFIX}{Path(file_path).name}"
        _s3_client.upload_file(file_path, S3_BUCKET, key)
        return _s3_client.generate_presigned_url(
            "get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=24 * 3600
        )
    except (NoCredentialsError, ClientError) as e:
        try:
            current_app.logger.error("S3 upload failed: %s", e)
        except Exception:
            print("S3 upload failed:", e)
        return None


# XTTS helper (lazy-initializes the Coqui model)
def xtts_speak_to_file(
    text: str,
    out_file: Optional[Union[str, Path]] = None,
    reference_dir: Optional[Union[str, Path]] = "assets",
    reference_files: Optional[Sequence[Union[str, Path]]] = None,
    language: str = "en",
    patterns: Iterable[str] = ("*.wav", "*.mp3", "*.flac"),
) -> Path:
    speakers = []
    if reference_files:
        speakers.extend(str(Path(p)) for p in reference_files)

    if (not speakers) and reference_dir:
        vdir = Path(reference_dir)
        for pat in patterns:
            speakers.extend(str(p) for p in vdir.glob(pat))

    speakers = list(dict.fromkeys(speakers))
    if not speakers:
        raise FileNotFoundError(f"No reference audio files found: {reference_files or reference_dir}")

    if not hasattr(xtts_speak_to_file, "_model") or xtts_speak_to_file._model is None:
        import sys, builtins
        sys.stdin = open(os.devnull)
        builtins.input = lambda *a, **kw: ""
        os.environ["COQUI_TOS_AGREED"] = "1"

        # Best-effort registration for safe globals (if available)
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
            from TTS.config.shared_configs import BaseDatasetConfig
            import torch

            add_safe = getattr(torch.serialization, "add_safe_globals", None)
            if callable(add_safe):
                add_safe([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
        except Exception:
            pass

        xtts_speak_to_file._model = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            gpu=False,
            progress_bar=False,
        )

    tts = xtts_speak_to_file._model
    out_path = Path(out_file) if out_file else Path(f"xtts_{uuid.uuid4().hex}.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        tts.tts_to_file(text=text, speaker_wav=speakers, language=language, file_path=str(out_path))
    except Exception as e:
        raise RuntimeError(f"XTTS synthesis failed: {e}") from e

    return out_path


# Serve audio files from AUDIO_DIR
@rag_bp.route("/audio/<path:filename>", methods=["GET"])
def rag_serve_audio(filename: str):
    return send_from_directory(str(AUDIO_DIR), filename, mimetype="audio/wav", conditional=True)


# CORS for dev Angular origins
@rag_bp.after_app_request
def add_cors_headers(resp):
    origin = request.headers.get("Origin")
    if origin in ("http://localhost:4200", "http://127.0.0.1:4200"):
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-User"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


def user_to_db_level(username: Optional[str]) -> Optional[str]:
    if not username:
        return None
    u = username.strip().lower()
    return {"lowergrade": "low", "midgrade": "mid", "highergrade": "high"}.get(u)


def extract_username_from_request(req) -> Optional[str]:
    hdr = req.headers.get("X-User")
    if hdr:
        return hdr
    data = req.get_json(silent=True) or {}
    return data.get("username")


# D-ID helpers
def _did_create_talk(text: str):
    if not DID_API_KEY:
        return None, ("DID_API_KEY not set on the server", 500)
    if not DID_SOURCE_IMAGE_URL:
        return None, ("DID_SOURCE_IMAGE_URL not set on the server", 500)

    payload = {
        "script": {"type": "text", "input": text, "provider": {"type": "microsoft", "voice_id": DID_VOICE_ID}},
        "source_url": DID_SOURCE_IMAGE_URL,
        "config": {"fluent": True, "pad_audio": 0},
    }
    try:
        r = requests.post("https://api.d-id.com/talks", json=payload, auth=(DID_API_KEY, ""))
        if r.status_code not in (200, 201):
            return None, (f"D-ID create error: {r.text}", 502)
        talk_id = r.json().get("id")
        if not talk_id:
            return None, ("D-ID did not return a talk id", 502)
        return talk_id, None
    except Exception as e:
        current_app.logger.exception("D-ID create failed: %s", e)
        return None, ("D-ID create failed", 502)


def _did_poll_talk(talk_id: str, timeout_sec: int = 60, interval_sec: float = 2.0):
    deadline = time.time() + timeout_sec
    url = f"https://api.d-id.com/talks/{talk_id}"
    try:
        while time.time() < deadline:
            r = requests.get(url, auth=(DID_API_KEY, ""))
            if r.status_code != 200:
                return None, (f"D-ID poll error: {r.text}", 502)
            data = r.json()
            status = data.get("status")
            if status == "done":
                return data.get("result_url") or data.get("result", {}).get("url"), None
            if status == "error":
                return None, (f"D-ID generation failed: {data.get('error')}", 502)
            time.sleep(interval_sec)
        return None, ("Timed out waiting for the video", 504)
    except Exception as e:
        current_app.logger.exception("D-ID poll failed: %s", e)
        return None, ("D-ID poll failed", 502)


# New helper: generate KD Talker video from text (returns (video_url, None) or (None, (msg, status)))
def _generate_kd_video_from_text(text: str, language: str = "en"):
    image_path = Path(os.getenv("VIDEO_IMAGE_PATH", str(BASE_DIR / 'assets' / 'teacher.png')))
    if not image_path.exists():
        return None, ("Image file not found", 404)

    # 1) Synthesize audio from text -> save wav under AUDIO_DIR
    try:
        out_name = f"genvid_{uuid.uuid4().hex}.wav"
        wav_path = xtts_speak_to_file(
            text=text,
            out_file=AUDIO_DIR / out_name,
            reference_dir=XTTS_REF_DIR,
            reference_files=None,
            language=language
        )
    except FileNotFoundError as e:
        current_app.logger.error("XTTS references missing: %s", e)
        return None, ("XTTS reference audio files not found on server", 500)
    except Exception as e:
        current_app.logger.exception("XTTS synthesis failed: %s", e)
        return None, ("Audio synthesis failed", 500)

    # 2) Call GPU server with image + synthesized audio
    try:
        with image_path.open("rb") as img_file, Path(wav_path).open("rb") as audio_file:
            files = {
                "image": ("image", img_file),
                "audio": ("audio", audio_file),
            }
            data_form = {"text": text}
            response = requests.post(REMOTE_API_URL, files=files, data=data_form, timeout=120)

        if response.status_code != 200:
            return None, (f"GPU server error: {response.text}", 502)

        # Expect JSON { "video_url": "..." }
        try:
            payload = response.json()
            video_url = payload.get("video_url")
            if not video_url:
                return None, ("Video URL not found in GPU response", 502)
            return video_url, None
        except Exception as e:
            current_app.logger.exception("GPU response parse failed: %s", e)
            return None, ("Error parsing GPU response JSON", 500)

    except Exception as e:
        current_app.logger.exception("GPU server request failed: %s", e)
        return None, ("GPU server request failed", 500)


# Ingest endpoints
@rag_bp.route("/ingest", methods=["POST", "OPTIONS"])
def rag_ingest():
    if request.method == "OPTIONS":
        return ("", 204)
    body = IngestBody(**(request.json or {}))
    return jsonify(ingest_documents(body))


@rag_bp.route("/ingest-pdfs", methods=["POST", "OPTIONS"])
def rag_ingest_pdfs():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.json or {}
    folder = data.get("folder", PDF_DEFAULT_FOLDER)
    return jsonify(ingest_pdfs_from_folder(folder, subject=data.get("subject"), grade=data.get("grade"), chapter=data.get("chapter")))


@rag_bp.route("/generate-questions", methods=["POST", "OPTIONS"])
def rag_generate_questions():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.json or {}
    username = extract_username_from_request(request)
    mapped_level = user_to_db_level(username)
    if not data.get("db_level"):
        data["db_level"] = mapped_level
    body = LLMBody(**data)
    return jsonify(llm_generate(body))


@rag_bp.route("/explain-grammar", methods=["POST", "OPTIONS"])
def rag_explain_grammar():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.get_json(force=True) or {}
    username = extract_username_from_request(request)

    body = ExplainBody(
        question=(data.get("question") or "").strip(),
        model=data.get("model", "gpt-4o-mini"),
        db_level=user_to_db_level(username),
        source_ids=data.get("source_ids") or [],
    )

    result_raw = llm_explain(body)

    # normalize result
    try:
        if isinstance(result_raw, dict):
            result_dict = dict(result_raw)
        elif hasattr(result_raw, "model_dump"):
            result_dict = result_raw.model_dump()
        elif hasattr(result_raw, "dict"):
            result_dict = result_raw.dict()
        elif isinstance(result_raw, str):
            result_dict = {"answer": result_raw}
        else:
            result_dict = {"answer": str(result_raw)}
    except Exception as e:
        current_app.logger.exception("Failed to normalize llm_explain result: %s", e)
        return jsonify({"error": "Internal error normalizing LLM response"}), 500

    answer_text = (result_dict.get("answer") or result_dict.get("response") or result_dict.get("text") or "").strip()

    # optional audio synthesis
    if data.get("synthesize_audio"):
        try:
            out_name = f"explain_{uuid.uuid4().hex}.wav"
            wav_path = xtts_speak_to_file(
                text=answer_text or result_dict.get("answer", ""),
                out_file=AUDIO_DIR / out_name,
                reference_dir=XTTS_REF_DIR,
                reference_files=None,
                language=data.get("language", "en"),
            )
            base = request.host_url.rstrip("/")
            result_dict["audio_url"] = f"{base}/rag/audio/{wav_path.name}"
        except FileNotFoundError as e:
            current_app.logger.error("XTTS reference audio missing: %s", e)
        except Exception as e:
            current_app.logger.exception("XTTS synthesis during explain-grammar failed: %s", e)

    # optional video synthesis (D-ID or KD Talker)
    if data.get("synthesize_video"):
        # KD Talker path if frontend requested it (chatId === '2')
        if data.get("kdtalker") or data.get("use_kdtalker"):
            try:
                video_url, err = _generate_kd_video_from_text(answer_text or result_dict.get("answer", ""), data.get("language", "en"))
                if err:
                    try:
                        current_app.logger.error("KD Talker create error during explain-grammar: %s", err[0] if isinstance(err, tuple) else err)
                    except Exception:
                        print("KD Talker error:", err)
                elif video_url:
                    result_dict["video_url"] = video_url
            except Exception as e:
                current_app.logger.exception("KD Talker inline synthesis failed during explain-grammar: %s", e)
        else:
            # existing D-ID flow
            if not DID_API_KEY or not DID_SOURCE_IMAGE_URL:
                current_app.logger.error("D-ID not configured for inline explain-grammar video synthesis")
            else:
                try:
                    talk_id, err = _did_create_talk(answer_text or result_dict.get("answer", ""))
                    if err:
                        current_app.logger.error("D-ID create error during explain-grammar: %s", err[0] if isinstance(err, tuple) else err)
                    else:
                        video_url, err = _did_poll_talk(talk_id, timeout_sec=120, interval_sec=2.0)
                        if err:
                            current_app.logger.error("D-ID poll error during explain-grammar: %s", err[0] if isinstance(err, tuple) else err)
                        elif video_url:
                            result_dict["video_url"] = video_url
                except Exception as e:
                    current_app.logger.exception("D-ID inline synthesis failed during explain-grammar: %s", e)

    return jsonify(result_dict), 200


@rag_bp.route("/suggest-followups", methods=["POST", "OPTIONS"])
def rag_suggest_followups():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.get_json(force=True) or {}
    username = extract_username_from_request(request)
    body = FollowupBody(
        last_question=(data.get("last_question") or "").strip(),
        last_answer=(data.get("last_answer") or "").strip(),
        n=int(data.get("n", 5)),
        model=data.get("model", "gpt-4o-mini"),
        db_level=user_to_db_level(username),
        source_ids=data.get("source_ids") or [],
    )
    return jsonify(llm_followups(body))


@rag_bp.get("/_diag")
def rag_diag():
    # Vectorstore diagnostics + media & routing checks
    try:
        from .rag_llm import CHROMA_DIR, CHROMA_ROOT, get_vectorstore as gs, get_vectorstore_for as gvf
    except ImportError:
        from rag_llm import CHROMA_DIR, CHROMA_ROOT, get_vectorstore as gs, get_vectorstore_for as gvf

    def _count(vs):
        if vs is None:
            return None
        if hasattr(vs, "count") and callable(vs.count):
            try:
                return vs.count()
            except Exception:
                return None
        if hasattr(vs, "_collection"):
            try:
                return vs._collection.count()
            except Exception:
                try:
                    return vs._client.get_collection(vs._collection.name).count()
                except Exception:
                    return None
        return None

    low_vs = gvf("low")
    mid_vs = gvf("mid")
    high_vs = gvf("high")

    # media checks
    ref_dir_exists = XTTS_REF_DIR.exists() and XTTS_REF_DIR.is_dir()
    ref_files = []
    if ref_dir_exists:
        for ext in ("*.wav", "*.mp3", "*.flac"):
            ref_files.extend([str(p.name) for p in XTTS_REF_DIR.glob(ext)])
    audio_dir_exists = AUDIO_DIR.exists() and AUDIO_DIR.is_dir()
    audio_files = [p.name for p in AUDIO_DIR.glob("*.wav")] if audio_dir_exists else []

    # list registered routes beginning with /rag
    routes = [r.rule for r in current_app.url_map.iter_rules() if r.rule.startswith("/rag")]

    info = {
        "env_seen": {"CHROMA_DIR": CHROMA_DIR, "CHROMA_ROOT": CHROMA_ROOT},
        "low_dir": {"path": str(Path(CHROMA_ROOT) / "low"), "exists": Path(CHROMA_ROOT, "low").is_dir()},
        "counts_default": _count(gs()),
        "counts_low": _count(low_vs),
        "counts_mid": _count(mid_vs),
        "counts_high": _count(high_vs),
        "media": {
            "xtts_ref_dir": str(XTTS_REF_DIR),
            "xtts_ref_dir_exists": ref_dir_exists,
            "xtts_ref_files_sample": ref_files[:10],
            "audio_dir": str(AUDIO_DIR),
            "audio_dir_exists": audio_dir_exists,
            "audio_files_sample": audio_files[:20],
        },
        "routes": routes,
    }
    return jsonify(info), 200


@rag_bp.route("/search", methods=["POST", "OPTIONS"])
def rag_search():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.json or {}
    q = (data.get("q") or "").strip()
    if not q:
        return jsonify({"results": []})
    username = extract_username_from_request(request)
    db_level = data.get("db_level") or user_to_db_level(username)
    vs = get_vectorstore_for(db_level)
    hits = vs.similarity_search_with_score(q, k=5)
    out = []
    for doc, dist in hits:
        out.append(
            {
                "distance": float(dist),
                "snippet": doc.page_content[:200],
                "source_path": os.path.normpath(doc.metadata.get("source_path", "")),
                "page": doc.metadata.get("page_1based"),
            }
        )
    return jsonify({"results": out})


@rag_bp.route("/generate-questions-from-chroma", methods=["POST", "OPTIONS"])
def generate_questions_from_chroma():
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        vectorstore = get_vectorstore()
        query_text = "important content related to grammar"
        results = vectorstore.similarity_search_with_score(query_text, k=5)
        content = "\n".join([doc.page_content for doc, _ in results])
        if not content:
            return jsonify({"error": "No content retrieved from vectorstore. Please ingest PDFs first."}), 200
        prompt = f"Generate 5 important questions based on the following content: {content}"
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=150
        )
        generated = response.choices[0].message.content.strip()
    except Exception as e:
        generated = {"error": f"Failed to call OpenAI: {str(e)}"}
    return jsonify({"generated_questions": generated})


@rag_bp.get("/health")
def health():
    return {"status": "ok"}, 200


@rag_bp.route("/synthesize-audio", methods=["POST", "OPTIONS"])
def rag_synthesize_audio():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    language = (data.get("language") or "en").strip()
    reference_files = data.get("reference_files")

    # preflight checks
    try:
        if not reference_files:
            if not XTTS_REF_DIR.exists() or not XTTS_REF_DIR.is_dir():
                current_app.logger.error("XTTS_REF_DIR not found: %s", XTTS_REF_DIR)
                return jsonify({"error": "XTTS reference directory not found", "details": str(XTTS_REF_DIR)}), 500
            has_any = any(XTTS_REF_DIR.glob("*.wav")) or any(XTTS_REF_DIR.glob("*.mp3")) or any(XTTS_REF_DIR.glob("*.flac"))
            if not has_any:
                current_app.logger.error("No reference audio files in XTTS_REF_DIR: %s", XTTS_REF_DIR)
                return jsonify({"error": "XTTS reference audio files not found on server", "details": str(XTTS_REF_DIR)}), 500
        else:
            missing = [str(p) for p in reference_files if not Path(p).exists()]
            if missing:
                current_app.logger.error("Provided reference_files missing: %s", missing)
                return jsonify({"error": "One or more reference_files not found", "details": missing}), 400
    except Exception as pre_e:
        current_app.logger.exception("Preflight validation failed: %s", pre_e)
        return jsonify({"error": "Preflight validation failed", "details": str(pre_e)}), 500

    try:
        out_name = f"synth_{uuid.uuid4().hex}.wav"
        wav_path = xtts_speak_to_file(
            text=text, out_file=AUDIO_DIR / out_name, reference_dir=XTTS_REF_DIR, reference_files=reference_files, language=language
        )

        if "localhost" in request.host_url or "127.0.0.1" in request.host_url:
            base = request.host_url.rstrip("/")
            audio_url = f"{base}/rag/audio/{wav_path.name}"
        else:
            s3_url = _upload_to_s3(str(wav_path))
            if s3_url:
                audio_url = s3_url
            else:
                base = os.getenv("SPACE_URL", "https://majemaai-mj-learn-backend.hf.space")
                audio_url = f"{base}/rag/audio/{wav_path.name}"

        return jsonify({"audio_url": audio_url, "file": wav_path.name}), 200
    except FileNotFoundError as e:
        current_app.logger.error("XTTS references missing: %s", e)
        return jsonify({"error": "XTTS reference audio files not found on server", "details": str(e)}), 500
    except Exception as e:
        current_app.logger.exception("XTTS synthesis error: %s", e)
        return jsonify({"error": "Synthesis failed", "details": str(e)}), 500


@rag_bp.route("/synthesize-video", methods=["POST", "OPTIONS"])
def rag_synthesize_video():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    if not DID_API_KEY or not DID_SOURCE_IMAGE_URL:
        current_app.logger.error("D-ID not configured (DID_API_KEY or DID_SOURCE_IMAGE_URL missing)")
        return jsonify({"error": "D-ID not configured on server"}), 500
    try:
        talk_id, err = _did_create_talk(text)
        if err:
            return jsonify({"error": err[0]}), err[1]
        video_url, err = _did_poll_talk(talk_id, timeout_sec=120, interval_sec=2.0)
        if err:
            return jsonify({"error": err[0]}), err[1]
        if not video_url:
            return jsonify({"error": "D-ID did not return a video URL"}), 502
        return jsonify({"video_url": video_url}), 200
    except Exception as e:
        current_app.logger.exception("Unexpected error generating D-ID video: %s", e)
        return jsonify({"error": "Internal server error generating video"}), 500


@rag_bp.route("/generate-video-from-text", methods=["POST", "OPTIONS"])
def generate_video_from_text():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    language = data.get("language", "en")
    video_url, err = _generate_kd_video_from_text(text, language)
    if err:
        return jsonify({"error": err[0]}), err[1]
    return jsonify({"video_url": video_url}), 200


#KD Talker setup (helper already added above)

if __name__ == "__main__":
    app = Flask(__name__)
    CORS(
        app,
        resources={r"/rag/*": {"origins": ["http://localhost:4200", "http://127.0.0.1:4200"]}},
        supports_credentials=True,
        allow_headers=["Content-Type", "Authorization", "X-User"],
        methods=["GET", "POST", "OPTIONS"],
    )
    os.makedirs(os.getenv("CHROMA_DIR", "./chroma"), exist_ok=True)
    app.register_blueprint(rag_bp, url_prefix="/rag")
    app.run(host="0.0.0.0", port=7000, debug=True)
