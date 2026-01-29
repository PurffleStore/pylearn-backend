import os
import re
import glob
from typing import List, Optional, Dict, Any
from shutil import which

# Load .env early so TESSERACT_CMD/CHROMA_DIR are available in local runs
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Text splitter: LC 0.3 uses langchain_text_splitters; older uses langchain.text_splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # LC 0.3+
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter   # older LC

# Embedding backends (we'll select at runtime)
from langchain_community.vectorstores import Chroma
try:
    # prefer modern shim packages
    from langchain_openai import OpenAIEmbeddings
except Exception:
    OpenAIEmbeddings = None  # type: ignore

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    # fallback to older import path if needed
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
    except Exception:
        HuggingFaceEmbeddings = None  # type: ignore

try:
    from langchain_core.documents import Document   # LC >= 0.2
except Exception:
    from langchain.schema import Document

from pdf2image import convert_from_path
from PIL import Image  # noqa: F401  (used implicitly via pdf2image)
import pytesseract

# ---------------- Environment: Tesseract & Chroma ---------------- #

# 1) Tesseract binary path (env first; sensible OS default; strip quotes if present)
_tess_from_env = os.getenv("TESSERACT_CMD")
if _tess_from_env:
    pytesseract.pytesseract.tesseract_cmd = _tess_from_env.strip('"')
else:
    if os.name == "nt":
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    else:
        pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# 2) Chroma persistence dir
_is_hf = bool(os.getenv("HF_HOME") or os.getenv("SPACE_ID"))
_default_chroma = "/data/chroma" if _is_hf else "./chroma"
CHROMA_DIR = os.getenv("CHROMA_DIR", _default_chroma)

# 3) Embedding model controls
# If running on HF, default to OpenAI embeddings unless explicitly disabled.
USE_OPENAI_EMBEDDINGS = os.getenv(
    "USE_OPENAI_EMBEDDINGS",
    "true" if _is_hf else "false"
).lower() == "true"

# OpenAI model (when USE_OPENAI_EMBEDDINGS=true)
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# HF model (when USE_OPENAI_EMBEDDINGS=false)
HF_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_embeddings = None
_vectorstore = None

def _log_env_banner():
    try:
        import openai as _oa  # just to log version if present
        _oaver = getattr(_oa, "__version__", None)
    except Exception:
        _oaver = None
    print(
        f"[RAG] ENV -> CHROMA_DIR={CHROMA_DIR} | "
        f"USE_OPENAI_EMBEDDINGS={'true' if USE_OPENAI_EMBEDDINGS else 'false'} | "
        f"OPENAI_MODEL={OPENAI_EMBEDDING_MODEL if USE_OPENAI_EMBEDDINGS else '-'} | "
        f"HF_MODEL={HF_EMBEDDING_MODEL if not USE_OPENAI_EMBEDDINGS else '-'} | "
        f"openai_pkg={_oaver or 'n/a'}"
    )

# ---------------- Environment Check (cross-platform) ---------------- #
def verify_environment():
    print("\n🔧 Verifying OCR environment...")
    tess = pytesseract.pytesseract.tesseract_cmd
    print(f"• Tesseract cmd set to: {tess}")
    if not os.path.exists(tess):
        print("  ⚠️ Tesseract binary not found at that path. If OCR fails, set TESSERACT_CMD.")

    pdftoppm_path = which("pdftoppm")
    if pdftoppm_path:
        print(f"• Poppler 'pdftoppm' found at: {pdftoppm_path}")
    else:
        print("  ⚠️ 'pdftoppm' not found in PATH. On Windows, install Poppler and set poppler_path; on Linux, install poppler-utils.")

verify_environment()
_log_env_banner()

# ---------------- Vectorstore ---------------- #
def get_embeddings():
    """
    Selects the embedding backend:
    - OpenAI (default on HF) using text-embedding-3-small
    - HuggingFace (local/offline) using sentence-transformers/all-MiniLM-L6-v2
    """
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    if USE_OPENAI_EMBEDDINGS:
        if OpenAIEmbeddings is None:
            raise RuntimeError("OpenAIEmbeddings not available. Please add 'langchain-openai' to requirements.txt.")
        print(f"🔹 Using OpenAI embeddings: {OPENAI_EMBEDDING_MODEL}")
        _embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
        return _embeddings

    # HF fallback
    if HuggingFaceEmbeddings is None:
        raise RuntimeError(
            "HuggingFaceEmbeddings not available. Please add 'langchain-huggingface' and 'sentence-transformers' to requirements.txt."
        )
    print(f"🔹 Using Hugging Face embeddings: {HF_EMBEDDING_MODEL}")
    _embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
    return _embeddings

def _vs_count_safe(vs) -> Optional[int]:
    """Try to get a document count from a Chroma vectorstore safely."""
    try:
        return vs._collection.count()  # type: ignore[attr-defined]
    except Exception:
        try:
            return vs._client.get_collection(vs._collection.name).count()  # type: ignore[attr-defined]
        except Exception:
            return None

def get_vectorstore():
    """
    Returns a Chroma vectorstore that works in both local and Hugging Face environments.
    - Uses CHROMA_DIR if defined (e.g., /data/chroma/low)
    - Defaults to ./chroma when running locally
    - Monkey-patching from ingest_all.py can override this function to point to per-level dirs
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    # ensure directory
    os.makedirs(CHROMA_DIR, exist_ok=True)

    print(f"🔹 Loading Chroma vectorstore at: {CHROMA_DIR}")
    _vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings()
    )
    cnt = _vs_count_safe(_vectorstore)
    if cnt is not None:
        print(f"📦 Vectorstore currently has ~{cnt} chunks.")
    else:
        print("📦 Vectorstore count not available (skipping).")
    return _vectorstore

# ---------------- Text Splitter ---------------- #
def chunk_docs(docs: List[Document], chunk_size=1200, chunk_overlap=150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

# ---------------- Pydantic ---------------- #
class IngestBody(BaseModel):
    paths: List[str]
    subject: Optional[str] = None
    grade: Optional[str] = None
    chapter: Optional[str] = None

# ---------------- Chapter Detection ---------------- #
def detect_chapter(text: str, current_chapter: str) -> str:
    match = re.search(r"CHAPTER\s+\w+\s*[-:]?\s*(.+)", text, re.IGNORECASE)
    if match:
        current_chapter = match.group(1).strip().lower()
        print(f"📖 Detected new chapter: {current_chapter}")
        return current_chapter
    known = [
        "verb","noun","adjective","adverb","tense","article",
        "preposition","pronoun","conjunction","sentence",
        "clause","phrase","composition"
    ]
    for t in known:
        if re.search(rf"\b{t}\b", text, re.IGNORECASE):
            current_chapter = t
            break
    return current_chapter

# ---------------- OCR Engine ---------------- #
def ocr_pdf_to_text(pdf_path: str) -> str:
    """High-quality OCR extraction with 300 DPI and paragraph mode."""
    print(f"🔍 Performing OCR on {pdf_path}")

    # Windows-specific poppler locations (ignored on Linux/Mac)
    windows_poppler_paths = [
        r"C:\Users\DELL\Downloads\Release-25.07.0-0 (1)\poppler-25.07.0\Library\bin",
        r"C:\poppler\Library\bin",
        r"C:\Program Files\poppler-25.07.0\Library\bin"
    ]

    images = None
    tried = []

    # 1) Try system PATH first (Linux/Mac)
    try:
        images = convert_from_path(pdf_path, dpi=300, poppler_path=None)
        print("✅ Poppler working via system PATH")
    except Exception as e:
        tried.append(f"PATH: {e}")

    # 2) On Windows, try known folders
    if images is None and os.name == "nt":
        for path in windows_poppler_paths:
            try:
                images = convert_from_path(pdf_path, dpi=300, poppler_path=path)
                print(f"✅ Poppler working with: {path}")
                break
            except Exception as e:
                tried.append(f"{path}: {e}")

    if images is None:
        print("❌ All Poppler attempts failed.")
        for t in tried:
            print("   -", t)
        return ""

    full_text = []
    for i, img in enumerate(images, 1):
        print(f"📄 OCR page {i}/{len(images)}...")
        text = pytesseract.image_to_string(img, lang="eng", config="--oem 3 --psm 6")
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'Page\s*\d+', '', text, flags=re.IGNORECASE)
        if len(text.strip()) > 30:
            full_text.append(text.strip())
            print(f"🧾 Page {i} sample:\n{text[:300]}\n{'-'*60}")

    combined = "\n\n".join(full_text)
    if not combined.strip():
        print("⚠️ OCR produced no usable text.")
    return combined

# ---------------- Ingest Logic ---------------- #
def ingest_documents(body: IngestBody) -> Dict[str, Any]:
    docs: List[Document] = []

    for p in body.paths:
        print(f"\n📘 Processing {p}")
        if not os.path.exists(p):
            print("⚠️ Missing file:", p)
            continue

        current_chapter = "unknown"

        if p.lower().endswith(".pdf"):
            try:
                loader = PyPDFLoader(p)
                pages = loader.load()
            except Exception as e:
                print(f"❌ PyPDFLoader failed: {e}")
                pages = []

            if not pages or all(len(d.page_content.strip()) < 20 for d in pages):
                print("⚠️ PDF has no text layer; switching to OCR.")
                ocr_text = ocr_pdf_to_text(p)
                if ocr_text.strip():
                    current_chapter = detect_chapter(ocr_text, current_chapter)
                    docs.append(Document(
                        page_content=ocr_text,
                        metadata={
                            "subject": body.subject,
                            "grade": body.grade,
                            "chapter": current_chapter,
                            "source_path": p,
                            "ocr": True
                        }
                    ))
            else:
                for d in pages:
                    current_chapter = detect_chapter(d.page_content, current_chapter)
                    d.metadata = {
                        **d.metadata,
                        "subject": body.subject,
                        "grade": body.grade,
                        "chapter": current_chapter,
                        "source_path": d.metadata.get("source", p),
                        "page_1based": int(d.metadata.get("page", 0)) + 1,
                        "ocr": False
                    }
                docs.extend(pages)
        else:
            print(f"📝 Loading text file {p}")
            tl = TextLoader(p, encoding="utf-8").load()
            for d in tl:
                current_chapter = detect_chapter(d.page_content, current_chapter)
                d.metadata.update({
                    "subject": body.subject,
                    "grade": body.grade,
                    "chapter": current_chapter,
                    "source_path": p
                })
            docs.extend(tl)

    if not docs:
        return {"error": "No valid text extracted."}

    chunks = chunk_docs(docs)
    print(f"✅ Created {len(chunks)} chunks from {len(docs)} docs.")

    vs = get_vectorstore()
    vs.add_documents(chunks)
    # Explicit persist to ensure data is flushed to disk
    try:
        vs.persist()
    except Exception:
        pass
    print(f"💾 Ingestion complete — {len(docs)} pages, {len(chunks)} chunks saved.")
    return {"ingested_pages": len(docs), "ingested_chunks": len(chunks)}

# ---------------- Folder Ingestion ---------------- #
def ingest_pdfs_from_folder(folder_path: str, subject=None, grade=None, chapter=None) -> dict:
    pdfs = glob.glob(os.path.join(folder_path, "*.pdf"))
    print("📂 PDF files found:", pdfs)
    if not pdfs:
        return {"error": f"No PDF files found in {folder_path}"}
    body = IngestBody(paths=pdfs, subject=subject, grade=grade, chapter=chapter)
    return ingest_documents(body)
