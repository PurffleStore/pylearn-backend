# ragg/ingest_all.py
import os
import sys
from pathlib import Path
from langchain_community.vectorstores import Chroma

# Support both module and direct runs
if __package__ in (None, ""):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from ragg.rag_backend import ingest_pdfs_from_folder, get_embeddings
    import ragg.rag_backend as rag_backend
else:
    from ragg.rag_backend import ingest_pdfs_from_folder, get_embeddings
    import ragg.rag_backend as rag_backend

# Detect environment
IS_HF = bool(os.getenv("HF_HOME") or os.getenv("SPACE_ID"))
HERE = Path(__file__).resolve().parent

# PDF root auto-detect
PDFS_ROOT = (HERE / "assets" / "pdfs")
if not PDFS_ROOT.is_dir():
    PDFS_ROOT = (HERE.parent / "assets" / "pdfs")  # Works for /app/pdfs/*

# Chroma root auto-detect
CHROMA_BASE = Path(os.getenv("CHROMA_ROOT") or ("/data/chroma" if IS_HF else "./chroma"))
CHROMA_BASE.mkdir(parents=True, exist_ok=True)

def ingest_all_levels():
    print("\n🚀 Starting ingestion")
    print(f"📂 PDFs root: {PDFS_ROOT}")
    print(f"💾 Chroma base: {CHROMA_BASE}\n")

    for level in ["low", "mid", "high"]:
        folder_path = PDFS_ROOT / level
        if not folder_path.is_dir():
            print(f"⚠️ Skip {level}: {folder_path} not found")
            continue

        chroma_dir = CHROMA_BASE / level
        chroma_dir.mkdir(parents=True, exist_ok=True)
        collection_name = f"pylearn_{level}"

        # Monkey-patch vectorstore factory for this run
        def get_vectorstore_for_level():
            print(f"🔹 Using Chroma at: {chroma_dir}")
            print(f"🔹 Collection: {collection_name}")
            vs = Chroma(
                collection_name=collection_name,
                persist_directory=str(chroma_dir),
                embedding_function=get_embeddings(),
            )
            return vs

        rag_backend.get_vectorstore = get_vectorstore_for_level

        try:
            print(f"➡️ Ingesting {folder_path}")
            res = ingest_pdfs_from_folder(
                folder_path=str(folder_path),
                subject="English",
                grade="5",
                chapter=level,
            )
            # Persist the vectorstore to disk
            vs = rag_backend.get_vectorstore()
            if hasattr(vs, "persist"):
                vs.persist()
                print("📝 Called persist()")

            print(f"✅ {level}: {res}")
            print(f"📦 Stored in: {chroma_dir}\n")

        except Exception as e:
            import traceback
            print(f"❌ {level}: {e}")
            print(traceback.format_exc())

    print("🎯 Ingestion complete.\n")

if __name__ == "__main__":
    ingest_all_levels()
