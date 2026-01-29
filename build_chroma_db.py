import os
import json
import chromadb

# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = os.path.join(BASE_DIR, "assets/teacher_feedback_sentences_category.json")
CHROMA_DIR = os.path.join(BASE_DIR, "assets/chroma_db")
COLLECTION_NAME = "feedback"


def safe_float(x):
    """Convert '000.000' or 124.944 to float."""
    try:
        return float(x)
    except:
        return 0.0


def load_segments(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_chroma():
    segments = load_segments(JSON_FILE)

    # Create Chroma client
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # OPTIONAL: clear existing db (recommended if you already inserted wrong)
    existing = collection.get()
    existing_ids = existing.get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)
        print(f"✅ Deleted old entries: {len(existing_ids)}")

    ids = []
    documents = []
    metadatas = []

    for seg in segments:
        seg_id = seg.get("id")
        text = seg.get("text", "").strip()
        category = seg.get("category", "").strip()
        video_file = seg.get("video_file", "").strip()

        start = safe_float(seg.get("start"))
        end = safe_float(seg.get("end"))

        # metadata for chroma
        meta = {
            "category": category,
            "video_file": video_file,
            "start": start,
            "end": end,
        }

        # store phoneme only if exists
        if "phoneme" in seg and seg["phoneme"]:
            meta["phoneme"] = seg["phoneme"].strip()

        ids.append(seg_id)
        documents.append(text)
        metadatas.append(meta)

    # Insert into ChromaDB
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print("\n✅ ChromaDB created successfully!")
    print(f"Total inserted: {len(ids)}")

    # quick stats
    vowels = [m for m in metadatas if m["category"] == "vowel"]
    vowel_specific = [m for m in vowels if m.get("phoneme")]
    consonants = [m for m in metadatas if m["category"] == "consonant"]
    consonant_specific = [m for m in consonants if m.get("phoneme")]

    print(f"Vowel total: {len(vowels)} | vowel specific: {len(vowel_specific)}")
    print(f"Consonant total: {len(consonants)} | consonant specific: {len(consonant_specific)}")


if __name__ == "__main__":
    build_chroma()
