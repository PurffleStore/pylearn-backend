import os
import json
import re
from typing import List, Optional, Dict, Any, Tuple

from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import chromadb
from chromadb.utils import embedding_functions

# --- Constants ---
# CHROMA_DIR = "./chroma"
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHROMA_DIR   = os.getenv("CHROMA_DIR", "./chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_ROOT  = os.getenv("CHROMA_ROOT", CHROMA_DIR)
print(f"[RAG] ENV -> CHROMA_DIR={CHROMA_DIR} | CHROMA_ROOT={CHROMA_ROOT} | EMBEDDING_MODEL={EMBEDDING_MODEL}")

# Chroma distance: smaller is better. Keep docs with distance <= MAX_DISTANCE.
MAX_DISTANCE = 1.3

# Parent directory for low/mid/high (overridable via env)
CHROMA_ROOT = os.getenv("CHROMA_ROOT", "./chroma")

# --- Globals ---
_embeddings = None
_vectorstore = None
_vectorstores: Dict[str, Chroma] = {}
_client: Optional[OpenAI] = None


# ---------------------- Vector store & Client ---------------------- #
def get_embeddings():
    """Load or reuse the HuggingFace embedding model."""
    global _embeddings
    if _embeddings is None:
        print("🔹 Loading embeddings:", EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_vectorstore():
    """Backward-compatible default vectorstore (single store)."""
    global _vectorstore
    if _vectorstore is None:
        print("🔹 Loading Chroma vectorstore:", CHROMA_DIR)
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=get_embeddings(),
        )
    return _vectorstore
# new
# def get_vectorstore_for(db_level: Optional[str] = None):
#     key = (db_level or "").strip().lower()
#     if key in ("low", "mid", "high"):
#         persist_dir = os.path.join(CHROMA_ROOT, key)
#     else:
#         persist_dir = CHROMA_DIR

#     print(f"[RAG] Using Chroma from: {persist_dir}")

#     client = chromadb.PersistentClient(path=persist_dir)

#     # Show collections available
#     collections = client.list_collections()
#     print(f"Available collections: {[c.name for c in collections]}")

#     # Pick the default collection (first one)
#     if not collections:
#         print("❌ No collections found.")
#         return None
#     collection = client.get_collection(collections[0].name)
#     print(f"✅ Loaded Chroma collection: {collection.name}")
#     return collection


def get_vectorstore_for(db_level: Optional[str] = None):
    key = (db_level or "").strip().lower()
    persist_dir = os.path.join(CHROMA_ROOT, key) if key in ("low", "mid", "high") else CHROMA_DIR

    print(f"[RAG] Using LangChain Chroma at: {persist_dir}")
    os.makedirs(persist_dir, exist_ok=True)

    # Discover the existing collection name (created during ingest)
    client = chromadb.PersistentClient(path=persist_dir)
    cols = client.list_collections()
    collection_name = cols[0].name if cols else "langchain"   # fallback if empty

    print(f"[RAG] Attaching to collection: {collection_name}")

    # Return a LangChain vectorstore bound to that collection
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=get_embeddings(),
    )

def get_client():
    """Initialize and return a singleton OpenAI client (uses OPENAI_API_KEY)."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ---------------------- Utilities ---------------------- #
def extract_clean_sentences(text: str) -> str:
    """Extract usable text while keeping short list-style lines."""
    text = re.sub(r"\s+", " ", text or "")
    text = re.sub(r"Page\s*\d+", "", text, flags=re.IGNORECASE)
    # Remove only all-caps section headers (e.g., CHAPTER 1, CONTENTS)
    text = re.sub(r"\b([A-Z\s]{4,})\b", "", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    valid = []
    for s in sentences:
        s = s.strip()
        if len(s.split()) >= 3 or re.match(r"^\d+\.", s):
            valid.append(s)
    return " ".join(valid[:15])

# intergartion - 07-11
# --- NEW: Conceptual safety checker ---
def is_inappropriate(text: str) -> bool:
    """Checks for unprofessional or offensive language (conceptual)."""
    text_lower = text.lower()
    # Replace these with a more robust list or an API like OpenAI Moderation in production
    inappropriate_terms = ["stupid", "idiot", "badword", "offensiveword"] 
    
    if any(term in text_lower for term in inappropriate_terms):
        return True
    return False


# --- NEW: Conversational / polite-closing detector ---
def _polite_closing_reply(text: str) -> Optional[str]:
    """
    Detect short conversational closings like "thank you", "thanks", "ok thanks", "bye".
    Return a short conversational reply (not requiring the LLM) or None.
    """
    if not text:
        return None
    t = re.sub(r"[^\w\s]", "", text.lower()).strip()
    # Accept short polite closings up to a few words
    words = t.split()
    if len(words) > 6:
        return None

    # common thank-you variants
    if re.search(r"\b(thank|thanks|thx)\b", t):
        return "You're welcome."
    # brief goodbyes
    if t in ("bye", "goodbye", "see you", "see you later", "ok bye", "okay bye"):
        return "Goodbye."
    return None


# --- NEW: Follow-up question detector (short ambiguous follow-ups like "give more example?") ---
def _is_followup_question(text: str) -> bool:
    """
    Heuristic to detect short follow-up requests that ask for 'more examples' or similar.
    Returns True for short / context-dependent follow-ups so we can gather context using previous source_ids or last Q/A.
    """
    if not text:
        return False
    t = text.lower().strip()
    # short questions asking for examples or continuation
    if re.search(r"\b(example|examples|more examples|another example|more example|give example|give examples)\b", t):
        # avoid long free-form requests being misclassified
        if len(t.split()) <= 8:
            return True
    # very short continuations like "more?" or "more examples?"
    if t in ("more", "more?", "more examples?", "another?", "another example?"):
        return True
    return False



# ---------------------- Request Body Models ---------------------- #
class LLMBody(BaseModel):
    topic: Optional[str] = None
    n: Optional[int] = 5
    level: str = "easy"
    qtype: str = "FITB"          # FITB | MCQ | OPEN
    subject: Optional[str] = None
    grade: Optional[str] = None
    chapter: Optional[str] = None
    model: str = "gpt-4o-mini"
    allow_generate: bool = True
    db_level: Optional[str] = None


class ExplainBody(BaseModel):
    question: str
    subject: Optional[str] = None
    grade: Optional[str] = None
    chapter: Optional[str] = None
    model: str = "gpt-4o-mini"
    max_words: int = 120
    db_level: Optional[str] = None
    last_question: Optional[str] = None
    last_answer: Optional[str] = None
    source_ids: Optional[List[str]] = None   # ← NEW


class FollowupBody(BaseModel):
    last_question: str
    last_answer: str
    n: int = 5
    model: str = "gpt-4o-mini"
    db_level: Optional[str] = None
    source_ids: Optional[List[str]] = None


# ---------------------- Helpers for follow-ups ---------------------- #
_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by", "from",
    "that", "this", "these", "those", "it", "is", "are", "was", "were", "be", "being",
    "been", "as", "at", "if", "then", "than", "so", "such", "but", "not", "no", "do", "does",
    "did", "can", "could", "should", "would", "may", "might", "will", "shall", "i", "you",
    "he", "she", "we", "they", "them", "his", "her", "their", "our", "your", "my", "mine",
    "yours", "ours", "theirs"
}


def _extract_focus_terms(text: str, k: int = 6) -> List[str]:
    """Pick a few content words to keep follow-ups on-topic."""
    toks = re.findall(r"[a-z]{3,}", (text or "").lower())
    terms = [t for t in toks if t not in _STOPWORDS]
    seen, out = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= k:
            break
    return out


def _looks_like_definition(text: str) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in [" is a ", " is an ", " defined as ", " means ", " refers to "])


def _derive_next_step_terms(last_q: str, last_a: str) -> List[str]:
    """If the last answer looks like a definition, bias toward classification next."""
    base = ["examples", "identify", "usage"]
    if _looks_like_definition(last_a):
        return ["kinds", "types", "forms", "classification"] + base
    return base


def _parse_source_tag(tag: str) -> Tuple[str, Optional[int]]:
    """
    Parse '.../low.pdf#p3' → (path, 3) or '.../low.pdf' → (path, None).
    """
    if "#p" in tag:
        base, p = tag.split("#p", 1)
        try:
            return os.path.normpath(base), int(p)
        except ValueError:
            return os.path.normpath(base), None
    return os.path.normpath(tag), None


def _fetch_docs_for_followups(
    vs: Chroma,
    source_ids: Optional[List[str]],
    last_q: str,
    last_a: str
) -> List[Document]:
    """
    Try to keep follow-ups grounded in the same pages/section if we have page tags.
    Otherwise, fall back to similarity on last Q/A.
    """
    docs: List[Document] = []

    if source_ids:
        buckets: Dict[str, List[int]] = {}
        for tag in source_ids:
            sp, page = _parse_source_tag(tag)
            if not sp:
                continue
            buckets.setdefault(sp, [])
            if page is not None:
                buckets[sp].append(page)

        for sp, pages in buckets.items():
            if pages:
                lo = max(1, min(pages) - 1)
                hi = max(pages) + 1
                try:
                    res = vs.similarity_search_with_score(
                        query="grammar follow-up",
                        k=30,
                        filter={"source_path": sp, "page_1based": {"$gte": lo, "$lte": hi}},
                    )
                    docs.extend([doc for doc, _ in res])
                except Exception:
                    # If filters not supported, fetch many and filter in Python
                    res = vs.similarity_search_with_score("grammar follow-up", k=50)
                    for doc, _ in res:
                        sp2 = os.path.normpath(doc.metadata.get("source_path", ""))
                        pg = doc.metadata.get("page_1based")
                        if sp2 == sp and isinstance(pg, int) and lo <= pg <= hi:
                            docs.append(doc)

    if not docs:
        # Fallback: stick to the semantics of the last Q & A
        query = f"{last_q or ''} {last_a or ''}".strip() or "grammar"
        res = vs.similarity_search_with_score(query, k=20)
        docs = [doc for doc, _ in res]

    return docs[:30]


def _build_context_from_docs(docs: List[Document]) -> Dict[str, Any]:
    """Return context_text and source_ids from a list of Documents."""
    source_ids: List[str] = []
    context_blocks: List[str] = []
    for i, d in enumerate(docs[:10]):
        # Be robust to varied metadata keys
        sid = os.path.normpath(
            d.metadata.get("source_path")
            or d.metadata.get("source")
            or d.metadata.get("file_path")
            or f"doc-{i}"
        )
        page = d.metadata.get("page_1based")
        tag = f"{sid}#p{page}" if page else sid
        source_ids.append(tag)

        clean_text = extract_clean_sentences((d.page_content or "").strip())
        if len(clean_text) > 1200:
            clean_text = clean_text[:1200]
        context_blocks.append(f"[{tag}] {clean_text}")

    return {
        "context_text": "\n\n".join(context_blocks),
        "source_ids": list(dict.fromkeys(source_ids)),
    }


# ---------------------- Prompt Templates ---------------------- #
FITB_PROMPT = PromptTemplate.from_template("""
You are an English grammar teacher. Use ONLY the sentences in <CONTEXT>.
# Create {n} grammar questions about **{topic}** for Grade 5 students.
Create {n} fill-in-the-blank grammar questions about **{topic}**, based strictly on the content provided.

Goal:
- If the topic is 'Verb', underline the verb using Markdown underscores like: He __runs__ fast.
- If the topic is 'Noun', underline the noun(s), e.g.: The __cat__ sat on the mat.
- Use sentences EXACTLY from the context.
- Each question must contain at least one __underlined__ word.
- Output strict JSON:
{{
  "questions": [
    {{
      "question": "string with __underlined__ word",
      "answer": "string",
      "explanation": "string"
    }}
  ]
}}

<CONTEXT>
{context}
</CONTEXT>

If the context lacks valid sentences, return {"questions":[]}.
""")

MCQ_PROMPT = PromptTemplate.from_template("""
You are an English grammar teacher. Use ONLY the facts in <CONTEXT>.
# Create {n} multiple-choice questions about **{topic}**.

Rules:
- Exactly 4 options (A–D) and one correct answer.
- Use only sentences from the context.
- Output strict JSON:
{{
  "questions": [
    {{
      "question": "string",
      "options": ["A","B","C","D"],
      "answer": "A|B|C|D",
      "explanation": "string"
    }}
  ]
}}
<CONTEXT>
{context}
</CONTEXT>
If insufficient, return {"questions":[]}.
""")

# ---------------------- Prompt Templates ---------------------- #
# NOTE: Removed the context-checking instruction from ANSWER_PROMPT as it's now in the System Prompt
ANSWER_PROMPT = PromptTemplate.from_template("""
Answer the user's question clearly and completely, using only facts and examples from the context.

Rules:
- If the context defines or lists items, include all items mentioned.
- Include at least one example if present.
- Never add facts not in the context.

Output STRICT JSON only:
{{
  "answer": "string"
}}

User Question: "{question}"

<CONTEXT>
{context}
</CONTEXT>
""")

FITB_SYNTH_PROMPT = PromptTemplate.from_template("""
You are an English grammar teacher. Use ONLY the facts in <CONTEXT>.
# Create {n} fill-in-the-blank grammar questions about **{topic}**.

Rules:
- You may paraphrase briefly using the facts from context.
- Use a single blank as exactly 7 underscores: _______ .
- Output strict JSON:
{{
  "questions": [
    {{"question": "string with _______", "answer": "string", "explanation": "string"}}
  ]
}}

<CONTEXT>
{context}
</CONTEXT>
If insufficient, return {"questions":[]}.
""")

# ---------------------- Generation (OPEN questions) ---------------------- #
def llm_generate(body: LLMBody):
    vs = get_vectorstore_for(body.db_level)

    # Normalize topic and n
    raw_topic = (body.topic or "").strip()
    topic_is_empty = (raw_topic == "" or raw_topic == "*")
    n_questions = (body.n if body.n and body.n > 0 else 10) if topic_is_empty else (body.n or 5)

    # Retrieve documents
    docs: List[Document] = []
    if topic_is_empty:
        # No topic → diversified (MMR) retrieval with a neutral grammar query
        try:
            retriever = vs.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 24, "fetch_k": 80, "lambda_mult": 0.5}
            )
            docs = retriever.get_relevant_documents("English grammar")
        except Exception as e:
            print("⚠️ MMR retrieval failed; falling back to similarity:", e)
            docs_with_scores = vs.similarity_search_with_score("English grammar", k=24)
            docs = [doc for doc, _ in docs_with_scores]
    else:
        # Topic present → similarity with distance filter
        docs_with_scores = vs.similarity_search_with_score(raw_topic, k=20)
        docs = [doc for doc, dist in docs_with_scores if dist <= MAX_DISTANCE]
        if not docs:
            docs = [doc for doc, _ in docs_with_scores[:6]]

    # Build context and source ids
    built = _build_context_from_docs(docs)
    context_text = built["context_text"]
    source_ids = built["source_ids"]

    if body.qtype.upper() == "OPEN":
        topic_label = raw_topic if not topic_is_empty else "grammar concepts present in the textbook pages"

        system_prompt = (
            "You are a careful question writer for school students. "
            "Use only the provided textbook context. "
            "Your task is to produce GRAMMAR questions only: about definitions, rules, and usage that can be answered "
            "directly from the context (e.g., parts of speech, agreement, tense, clauses/phrases, voice, punctuation, etc.). "
            "Do not invent facts. "
            "Avoid questions about book metadata such as authors, editions, prefaces, publishers, anti-piracy notices, "
            "catalogs, prices, or acknowledgements. "
            "If the context contains only a small portion of grammar instruction, still ask questions only about that portion. "
            "If there is no instructional grammar in the context at all, return an empty list."
        )

        user_prompt = f"""
TOPIC (optional): {topic_label}

CONTEXT (verbatim excerpts from the textbook; may include headings and page tags):
{context_text}

TASK:
- Write {n_questions} open-ended STUDY QUESTIONS that a student can answer using ONLY the grammar teaching present in the CONTEXT.
- Focus on grammar understanding: definitions, rules, and how to use them in sentences (with examples when the context provides them).
- STRICTLY AVOID questions about book metadata (authors, editions, prefaces, publishers, anti-piracy notes, acknowledgements, prices, catalogs).
- If the context contains only a small amount of grammar, write questions about that small part; if none, output an empty list.

OUTPUT (strict JSON, no extra text):
{{
  "questions": [
    {{
      "question": "<grammar-only question answerable from the context>",
      "rationale": "<why this is a good grammar question based on the context>",
      "source_ids": {source_ids}
    }}
  ]
}}
"""

        client = get_client()
        try:
            resp = client.chat.completions.create(
                model=body.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            raw = resp.choices[0].message.content or "{}"
            payload = json.loads(raw)
        except Exception as e:
            return {"questions": [], "note": f"Error while generating questions: {str(e)}"}

        out = payload if isinstance(payload, dict) and "questions" in payload else {"questions": []}
        for q in out.get("questions", []):
            q.setdefault("source_ids", source_ids)
        return out

    return {"questions": [], "note": "Unsupported qtype. Use OPEN for concept questions."}


# ---------------------- Answer (Explain) ---------------------- #
def llm_explain(body: ExplainBody) -> Dict[str, Any]:
    vs = get_vectorstore_for(body.db_level)

    query_text = (body.question or "").strip()
    if not query_text:
        return {"answer": "", "source_ids": [], "note": "No question provided."}

    # --- NEW: retrieval prefers provided source_ids first ---
    if body.source_ids:
        docs = _fetch_docs_for_followups(vs, body.source_ids, body.question, "")
        if not docs:
            # fallback to similarity search if nothing found in same pages
            docs_with_scores = vs.similarity_search_with_score(query_text, k=20)
            docs = [doc for doc, dist in docs_with_scores if dist <= MAX_DISTANCE] or [doc for doc, _ in docs_with_scores[:6]]
        # 🛑 SCENARIO 3: Handle Inappropriate/Unprofessional Language (Early Exit)
    if is_inappropriate(query_text):
        return {
            "answer": "I cannot fulfill this request due to the use of inappropriate language. Please ask a polite and professional question.",
            "source_ids": [],
            "note": "Request blocked by safety filter."
        }

    # NEW: Conversational short reply handling (e.g., "thank you", "okay thank you")
    polite_reply = _polite_closing_reply(query_text)
    if polite_reply:
        return {
            "answer": polite_reply,
            "source_ids": [],
            "note": "Conversational reply (short closing) handled locally."
        }

    # NEW: If the incoming question looks like a short follow-up (e.g., "give more example?")
    # prefer to fetch docs grounded to previous source_ids / last Q&A if provided by the client.
    followup_docs: Optional[List[Document]] = None
    last_q = (body.last_question or "").strip()
    last_a = (body.last_answer or "").strip()
    src_ids = body.source_ids or None

    if _is_followup_question(query_text):
        if last_q or last_a or src_ids:
            try:
                followup_docs = _fetch_docs_for_followups(vs, src_ids, last_q, last_a)
            except Exception as e:
                print("⚠️ _fetch_docs_for_followups failed:", e)
                followup_docs = None

    # Retrieve relevant chunks (either from followup_docs or standard similarity)
    # Retrieve relevant chunks (either from followup_docs or standard similarity)
    if followup_docs:
        docs = followup_docs
        docs_with_scores = []  # not used in this branch
    else:
        docs_with_scores = vs.similarity_search_with_score(query_text, k=20)
        docs = [doc for doc, dist in docs_with_scores if dist <= MAX_DISTANCE]

    # ---- DEBUG LOGS ----
    print(f"[RAG DEBUG] incoming query: {query_text!r}")
    print(f"[RAG DEBUG] last_question={last_q!r} last_answer={last_a!r} source_ids={src_ids!r}")
    if followup_docs is not None:
        print(f"[RAG DEBUG] followup_docs_count={len(followup_docs)}")
    elif 'docs_with_scores' in locals():
        print(f"[RAG DEBUG] docs_with_scores_count={len(docs_with_scores)}")
    print(f"[RAG DEBUG] docs_used_count={len(docs)}")
    for i, d in enumerate(docs[:5]):
        snip = (d.page_content or "")[:160].replace("\n", " ")
        src = os.path.normpath(d.metadata.get("source_path", "") or d.metadata.get("source", "") or d.metadata.get("file_path", ""))
        pg = d.metadata.get("page_1based")
        print(f"  DOC{i+1}: src={src} page={pg} snippet={snip!r}")
    # ---------------------

    # Fallback if nothing passes the threshold
    if not docs:
        if docs_with_scores:
            docs = [doc for doc, _ in docs_with_scores[:6]]
            print(f"ℹ️ Fallback engaged (QA): using top {len(docs)} docs without distance filter.")
        else:
            print("ℹ️ No docs found for follow-up or similarity.")
            docs = []

    if docs_with_scores:
        print(f"🔎 QA retrieved {len(docs_with_scores)} raw, {len(docs)} kept (≤ {MAX_DISTANCE})")
        for i, (doc, dist) in enumerate(docs_with_scores[:5]):
            snippet = (doc.page_content or "")[:100].replace("\n", " ")
            print(f"    QA DOC {i+1} distance={dist:.3f} | {snippet}...")
  
    # Build compact context
    source_ids: List[str] = []
    parts = []
    for i, d in enumerate(docs[:10]):
        sid = os.path.normpath(
            d.metadata.get("source_path")
            or d.metadata.get("source")
            or d.metadata.get("file_path")
            or f"doc-{i}"
        )
        page = d.metadata.get("page_1based")
        tag = f"{sid}#p{page}" if page else sid
        source_ids.append(tag)

        clean_text = extract_clean_sentences(d.page_content.strip())
        if len(clean_text) > 1200:
            clean_text = clean_text[:1200]
        parts.append(f"[{tag}] {clean_text}")

    context = "\n\n".join(parts)
    print("\n🧾 QA Context to LLM (first 800 chars):")
    print(context[:800])
    print("--------------------------------------------------")

    # 🛑 SCENARIO 1 & 2: Modified System Prompt for Scope and Failure Responses
    system_prompt = """
You are an English Grammar tutor for students. 
Your task is to answer the user's question clearly and completely, using **only facts and examples from the provided context**.

**STRICT RULES for Response:**
1. **Focus on Grammar:** Only answer questions strictly related to **English grammar** (parts of speech, usage, rules, tense, etc.).
2. **Handle Non-Grammar:** If the question is clearly **NOT about grammar** (e.g., general knowledge, history, metadata, or other subjects), you must return the specific answer: "This query is outside the scope of the provided English Grammar textbook content."
3. **Handle Out-of-Context:** If the question is about grammar but the answer is **NOT present** in the context, you must return the specific answer: "No information available in the provided textbook content."
4. **Output Format:** You must always output STRICT JSON.
"""

    prompt = ANSWER_PROMPT.format(question=body.question, context=context)

    client = get_client()
    try:
        resp = client.chat.completions.create(
            model=body.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
    except Exception as e:
        print("❌ OpenAI API call failed (QA):", e)
        return {"answer": "", "source_ids": [], "note": f"Error while generating answer: {str(e)}"}

    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except Exception:
        data = {"answer": ""}

    answer = (data.get("answer") or "").strip()
    answer_low = answer.lower()
    
    # Check against the specific failure phrases returned by the LLM
    if "outside the scope" in answer_low:
        return {
            "answer": answer, 
            "source_ids": [],
            "note": "The query is outside the scope of the provided English Grammar textbook content."
        }
    
    if "no information available" in answer_low:
        return {
            "answer": answer,
            "source_ids": list(dict.fromkeys(source_ids))[:3],
            "note": "The requested information was not found in the provided material."
        }

    # Fallback for empty or unrecognized failure answer
    if not answer or answer_low.startswith("i cannot find"):
    
        return {
            "answer": "",
            "source_ids": list(dict.fromkeys(source_ids))[:3],
            "note": "The requested information was not found in the provided material."
        }

    return {
        "answer": answer[: body.max_words * 8],
        "source_ids": list(dict.fromkeys(source_ids))[:3]
    }



# ---------------------- Follow-up Suggestions ---------------------- #
def _verify_answerable(question: str, *, db_level: Optional[str], model: str, max_words: int = 60) -> Tuple[bool, List[str]]:
    """
    Check quickly if a follow-up question is answerable from the PDFs for the given db_level
    by calling llm_explain with a short answer budget.
    Returns (ok, source_ids).
    """
    try:
        res = llm_explain(ExplainBody(
            question=question,
            model=model,
            max_words=max_words,
            db_level=db_level,
            source_ids=None   # verification uses full retrieval
        ))
    except Exception:
        return (False, [])

    ans_text = (res.get("answer") or "").strip() if isinstance(res, dict) else (str(res).strip())
    src_ids: List[str] = []
    if isinstance(res, dict):
        val = res.get("source_ids")
        if isinstance(val, list):
            src_ids = val

    if not ans_text:
        return (False, src_ids)

    low = ans_text.lower()
    if ("no information available" in low) or ("not found" in low) or ("insufficient" in low):
        return (False, src_ids)

    return (True, src_ids)


def llm_followups(body: FollowupBody) -> Dict[str, Any]:
    """
    Suggest follow-up grammar questions based on the user's last question and the answer just given.
    Ground suggestions in the same textbook context (Chroma) used for the answer,
    and KEEP ONLY those that are actually answerable from the PDFs (verified via llm_explain).
    """
    vs = get_vectorstore_for(body.db_level)

    q = (body.last_question or "").strip()
    a = (body.last_answer or "").strip()
    if not q or not a:
        return {"suggestions": [], "note": "Both last_question and last_answer are required."}

    # Prefer same section/pages if source_ids available
    docs = _fetch_docs_for_followups(vs, body.source_ids, q, a)
    built = _build_context_from_docs(docs)
    context_text = built["context_text"]
    source_ids = built["source_ids"]

    # Focus & next steps
    focus_terms = _extract_focus_terms(f"{q} {a}") or ["grammar"]
    next_step_terms = _derive_next_step_terms(q, a)

    system_prompt = (
        "You are an English grammar tutor. Use ONLY the provided textbook context.\n"
        "Generate follow-up QUESTIONS that build directly on the student's LAST QUESTION and the given ANSWER.\n"
        "Stay strictly on the SAME concept/terminology (focus terms below). Do not switch topics.\n"
        "Allowed: parts of speech, agreement, tense/aspect, clauses/phrases, voice, sentence elements, punctuation, definitions, usage.\n"
        "FORBIDDEN: author/publisher/preface/editions/piracy/contents pages and any non-instructional metadata.\n"
        "If the context does not continue the topic, return an empty list."
    )

    user_prompt = f"""
LAST QUESTION: {q}

LAST ANSWER (authoritative): {a}

FOCUS TERMS (stay on these): {focus_terms}

NEXT-STEP TERMS (prefer questions that use one of these): {next_step_terms}

PROGRESSION LADDER (move just one step deeper than the last answer):
1. Definition → 2. Classification/Types → 3. Examples → 4. Identification (in given sentences)
→ 5. Application/Transformation → 6. Contrast/Edge cases

CONTEXT (verbatim textbook snippets from the same section/pages if available):
{context_text}

TASK:
- Propose {max(1, body.n)} short follow-up questions that deepen understanding of EXACTLY the same concept.
- If the last answer is a definition, prefer classification (e.g., kinds/types) as the next step.
- Otherwise, advance by ONE rung on the ladder (e.g., from types → examples; from examples → identification).
- Each question must be answerable from this CONTEXT and must mention at least one FOCUS TERM.
- Do NOT repeat the last question, and do NOT drift to unrelated topics.

OUTPUT (strict JSON only):
{{
  "suggestions": ["<q1>", "<q2>", "..."]
}}
"""

    client = get_client()
    try:
        resp = client.chat.completions.create(
            model=body.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        suggestions = data.get("suggestions", [])
    except Exception as e:
        return {"suggestions": [], "source_ids": source_ids, "note": f"follow-ups error: {str(e)}"}

    # Light post-filters: keep on-topic, avoid near-duplicates
    def _similar(a_text: str, b_text: str) -> float:
        sa = set(re.findall(r"[a-z]+", (a_text or "").lower()))
        sb = set(re.findall(r"[a-z]+", (b_text or "").lower()))
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    ft_lower = [t.lower() for t in focus_terms]
    nst_lower = [t.lower() for t in next_step_terms]

    def _on_topic(s: str) -> bool:
        s_low = (s or "").lower()
        return any(t in s_low for t in ft_lower)

    def _prefers_next_step(s: str) -> bool:
        s_low = (s or "").lower()
        return any(t in s_low for t in nst_lower)

    filtered = []
    for s in suggestions:
        if _similar(s, q) >= 0.65:
            continue  # too close to previous question
        if not _on_topic(s):
            continue
        filtered.append(s)

    if _looks_like_definition(a):
        preferred = [s for s in filtered if _prefers_next_step(s)]
        if preferred:
            filtered = preferred

    # ---- Verification: keep only answerable follow-ups ----
    verified: List[str] = []
    for s in filtered:
        ok, _ = _verify_answerable(
            s,
            db_level=body.db_level,
            model=(body.model or "gpt-4o-mini"),
            max_words=60,
        )
        if ok:
            verified.append(s)
        if len(verified) >= max(1, body.n):
            break

    note = f"kept {len(verified)} of {len(filtered)} after verification" if filtered else "no candidates"

    return {
        "suggestions": verified[: max(1, body.n)],
        "source_ids": source_ids,
        "note": note
    }
