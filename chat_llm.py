import json
import os
import time
import hashlib
import re
from typing import Optional, List, Dict, Any, Tuple
from flask import Flask, Blueprint, request, jsonify, session
from flask_cors import CORS
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import tempfile
from datetime import datetime


try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================================================
# Flask
# ============================================================
chat_llm_bp = Blueprint("chat_llm", __name__)


# ============================================================
# Config
# ============================================================
QA_FILE = "assets/qa.json"
EMBED_CACHE_FILE = "tense_qa_embeddings_cache.json"

# Matching only (NO answer generation)
USE_OPENAI_EMBEDDINGS = True
EMBED_MODEL = "text-embedding-3-small"

# Moderation (optional)
USE_OPENAI_MODERATION = True
MODERATION_MODEL = "omni-moderation-latest"
# Defer heavy optional import (whisper) to optional load so import-time does not crash app
MODEL_NAME = "base"
model = None
MODEL_AVAILABLE = False
try:
    import whisper
    try:
        model = whisper.load_model(MODEL_NAME)
        MODEL_AVAILABLE = True
        print(f"Whisper model '{MODEL_NAME}' loaded successfully")
    except Exception as ex:
        print(f"Whisper installed but failed to load model '{MODEL_NAME}': {ex}")
        model = None
        MODEL_AVAILABLE = False
except Exception as ex:
    print(f"Whisper not available: {ex}")
    model = None
    MODEL_AVAILABLE = False


# ============================================================
# OpenAI client (ENV only) ✅ DO NOT hardcode keys
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
client = None
if USE_OPENAI_EMBEDDINGS or USE_OPENAI_MODERATION:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing. Please set it as an environment variable.")
    if OpenAI is None:
        raise RuntimeError("openai package not found. Please install it: pip install openai")
    client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
# Scenario media (frontend asset paths)
# ============================================================
SCENARIO_MEDIA = {
    "GREETING": {"video_url": "assets/staticchat/intro.mp4"},
    "THANKS": {"video_url": "assets/staticchat/you_are_welcome.mp4"},
    "BYE": {"video_url": "assets/staticchat/bye.mp4"},
    "PRAISE": {"video_url": "assets/staticchat/praise.mp4"},
    "SLEEPY": {"video_url": "assets/staticchat/sleepy.mp4"},
    "OUT_OF_TOPIC": {"video_url": "assets/staticchat/out_of_topic.mp4"},
    "NOT_UNDERSTANDABLE": {"video_url": "assets/staticchat/not_understand.mp4"},
    "BAD_LANGUAGE": {"video_url": "assets/staticchat/bad_lang.mp4"},
    "NOT_AVAILABLE": {"video_url": "assets/staticchat/no_db.mp4"},
}


# ============================================================
# Load Q&A JSON
# ============================================================
def load_qa_data() -> List[Dict[str, Any]]:
    with open(QA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Helpers: pick optional fields safely
# ============================================================
QA_RETURN_FIELDS = [
    "sno", "question", "answer",
    "audio_url", "video_url",
    "detail_url", "story_url", "example_url",
    "detail_text", "story_text", "example_text",
    "keyword",
]

def pick_qa_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k in QA_RETURN_FIELDS:
        if k in item and item[k] is not None:
            out[k] = item[k]
    return out


# ============================================================
# Session context
# ============================================================
conversation_context: Dict[str, Dict[str, Any]] = {}
CONTEXT_TTL_SECONDS = 60 * 30  # 30 minutes

def _now() -> int:
    return int(time.time())

def get_session_id(data: dict) -> str:
    sid = (data.get("session_id") or "").strip()
    if sid:
        return sid
    sid = (request.headers.get("X-Session-Id") or "").strip()
    if sid:
        return sid
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "")
    ua = request.headers.get("User-Agent", "")
    raw = f"{ip}|{ua}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def cleanup_old_context():
    now = _now()
    to_delete = []
    for sid, ctx in conversation_context.items():
        ts = int(ctx.get("updated_at", 0))
        if ts and (now - ts) > CONTEXT_TTL_SECONDS:
            to_delete.append(sid)
    for sid in to_delete:
        del conversation_context[sid]


# ============================================================
# Scenario replies
# ============================================================
def scenario_reply_text(scenario: str) -> str:
    replies = {
        "GREETING": "Good morning! Let's begin our lesson on tenses. You can ask me any question about tenses",
        "BYE": "Goodbye! Keep practicing your English tenses. Remember, practice makes perfect!",
        "THANKS": "You're welcome! Do you have any other questions?",
        "PRAISE": "Thank you. Please ask your question about tenses.",
        "SLEEPY": "Please take a short break. When you are ready, ask a tense question.",
        "OUT_OF_TOPIC": "That's not part of our tense lesson. Let's stay on our topic.",
        "NOT_UNDERSTANDABLE": "I don't understand your question. Can you ask it again more simply?",
        "BAD_LANGUAGE": "Please use polite words. Ask your question in a respectful way.",
        "NOT_AVAILABLE": "I don't have the answer for that. Let's not available in my lesson today.",
    }
    return replies.get(scenario, "Please ask a tense question.")

def scenario_payload(scenario: str, session_id: str) -> Dict[str, Any]:
    payload = {
        "scenario": scenario,
        "answer": scenario_reply_text(scenario),
        "session_id": session_id,
    }
    payload.update(SCENARIO_MEDIA.get(scenario, {}))
    return payload

def local_scenario_override(text: str) -> Optional[str]:
    t = text.strip().lower()

    if re.search(r"^(hi|hello|hey|hihi|hai|good morning|good afternoon|good evening)\b", t):
        return "GREETING"

    # NOTE: if message contains both bye and thanks, BYE should win (more logical end)
    if re.search(r"\b(bye|goodbye|see you|see u|cya)\b", t):
        return "BYE"

    if re.search(r"\b(thanks|thank you|tnx|thx|appreciate)\b", t):
        return "THANKS"

    if re.search(r"\b(super|good|love you|luv u|nice class|nice|sweet|cute|good teacher|great teacher|you are great|awesome)\b", t):
        return "PRAISE"

    if re.search(r"\b(sleepy|tired|feeling tired|bored|boring|not interested|break)\b", t):
        return "SLEEPY"

    return None


# ============================================================
# Moderation (optional)
# ============================================================
def is_bad_language(text: str) -> bool:
    if not USE_OPENAI_MODERATION or client is None:
        return False
    try:
        r = client.moderations.create(model=MODERATION_MODEL, input=text)
        return bool(r.results[0].flagged)
    except Exception:
        return False


# ============================================================
# Question / intent helpers (NO LLM)
# ============================================================
def looks_like_question(text: str) -> bool:
    t = text.strip().lower()
    if "?" in t:
        return True
    if re.search(r"^(what|why|how|when|where|who|which|give|define|explain|difference|compare)\b", t):
        return True
    return False

def is_compare_request(text: str) -> bool:
    t = text.lower()
    return bool(re.search(r"\b(difference between|compare|vs|versus)\b", t))

def extract_two_topics_from_compare(text: str) -> Optional[Tuple[str, str]]:
    t = text.strip()

    m = re.search(r"difference\s+between\s+(.*?)\s+and\s+(.*)$", t, flags=re.I)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    m = re.search(r"compare\s+(.*?)\s+and\s+(.*)$", t, flags=re.I)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    m = re.search(r"(.+?)\s+(vs|versus)\s+(.+)$", t, flags=re.I)
    if m:
        return m.group(1).strip(), m.group(3).strip()

    return None

def clean_topic_piece(p: str) -> str:
    p = p.strip()
    # remove leading articles: "a past" -> "past"
    p = re.sub(r"^(a|an|the)\s+", "", p, flags=re.I).strip()
    return p

def split_topic_only_text(full_text: str) -> List[str]:
    """
    Example:
    - "past and future tense" -> ["past tense", "future tense"]
    - "a past and future tense" -> ["past tense", "future tense"]
    - "past, present, future tense" -> ["past tense", "present tense", "future tense"]
    """
    raw = full_text.strip()
    parts = re.split(r"\s*(?:,| and )\s*", raw, flags=re.I)
    parts = [clean_topic_piece(x) for x in parts if clean_topic_piece(x)]

    if not parts:
        return [raw]

    # If full input contains "tense", and a part does NOT contain "tense",
    # then append "tense" to it.
    full_has_tense = bool(re.search(r"\btense\b", raw, flags=re.I))
    if full_has_tense:
        fixed = []
        for p in parts:
            if re.search(r"\btense\b", p, flags=re.I):
                fixed.append(p)
            else:
                fixed.append(p + " tense")
        parts = fixed

    return parts

def split_compound_what_is_question(text: str) -> Optional[List[str]]:
    """
    Split only this type:
    - "what is past and present tense"
    - "what is past and future tense"
    into two questions.

    It will NOT split normal long sentences.
    """
    t = text.strip().rstrip("?").strip()

    # must start with "what is"
    if not re.match(r"^what\s+is\s+", t, flags=re.I):
        return None

    # must contain " and "
    if " and " not in t.lower():
        return None

    # capture: what is <A> and <B>
    m = re.match(r"^what\s+is\s+(.+?)\s+and\s+(.+)$", t, flags=re.I)
    if not m:
        return None

    a = clean_topic_piece(m.group(1))
    b = clean_topic_piece(m.group(2))

    # if the sentence ends with "tense", and A does not have "tense", add it
    # example: "past and present tense" -> A="past", B="present tense"
    if re.search(r"\btense\b$", t, flags=re.I):
        if not re.search(r"\btense\b", a, flags=re.I):
            a = a + " tense"
        if not re.search(r"\btense\b", b, flags=re.I):
            b = b + " tense"

    return [f"What is {a}?", f"What is {b}?"]

def normalize_topic_phrase(topic: str) -> str:
    topic = topic.strip().rstrip("?").strip()
    if not topic:
        return topic
    if looks_like_question(topic):
        return topic
    topic = clean_topic_piece(topic)
    return f"What is {topic}?"


def followup_rewrite_no_llm(user_text: str, session_ctx: dict) -> List[str]:
    """
    Handle:
    - "give example for both"
    - "give example"
    Using previous matched topics.
    Now with improved example detection
    """
    t = user_text.strip().lower()
    last_topics = session_ctx.get("last_topics", []) if session_ctx else []
    if not isinstance(last_topics, list):
        last_topics = []
    
    # Get the last matched tense for context
    last_tense = session_ctx.get("last_tense", "") if session_ctx else ""
    
    # Check for example requests
    is_example_request = (
        re.search(r"\b(give|show|tell).*?\bexample", t) or 
        re.search(r"\bexample\b", t) or
        t.strip() == "give example" or
        t.strip() == "example"
    )
    
    if is_example_request:
        if last_topics and len(last_topics) >= 1:
            # Create specific example queries for the last topic
            last_topic = last_topics[-1]
            
            # Clean up the topic
            topic_clean = re.sub(r"^(what is|explain|tell me about)\s+", "", last_topic, flags=re.I).strip()
            topic_clean = re.sub(r"\?$", "", topic_clean).strip()
            
            # Create specific example questions
            example_queries = [
                f"Give examples for {topic_clean}",
                f"Examples of {topic_clean}",
                f"Example sentences for {topic_clean}",
                f"Show me examples of {topic_clean}"
            ]
            
            # If we have the specific tense, also try with "tense" suffix
            if last_tense and "tense" not in topic_clean.lower():
                example_queries.append(f"Examples of {last_tense} tense")
            
            return example_queries
        return [user_text]
    
    if re.search(r"\bgive\s+example\s+for\s+both\b", t) or re.search(r"\bgive\s+examples\s+for\s+both\b", t):
        if len(last_topics) >= 2:
            examples = []
            for topic in last_topics:
                topic_clean = re.sub(r"^(what is|explain|tell me about)\s+", "", topic, flags=re.I).strip()
                topic_clean = re.sub(r"\?$", "", topic_clean).strip()
                examples.append(f"Give examples for {topic_clean}")
            return examples
        if len(last_topics) == 1:
            topic_clean = re.sub(r"^(what is|explain|tell me about)\s+", "", last_topics[0], flags=re.I).strip()
            topic_clean = re.sub(r"\?$", "", topic_clean).strip()
            return [f"Give examples for {topic_clean}"]
        return [user_text]

    return [user_text]


# ============================================================
# TF-IDF search
# ============================================================
class GenericQASearchEngine:
    def __init__(self, qa_data: List[Dict[str, Any]]):
        self.items = qa_data
        self.questions = [str(item.get("question", "")).strip() for item in qa_data]

        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        self.question_vectors = None
        if self.questions:
            self.question_vectors = self.vectorizer.fit_transform(self.questions)

    def best_match(self, query: str, threshold: float = 0.35) -> Optional[Dict[str, Any]]:
        if not self.questions or self.question_vectors is None:
            return None
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.question_vectors).flatten()
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score < threshold:
            return None
        item = self.items[best_idx]
        return {
            **pick_qa_fields(item),
            "matched_question": item.get("question", ""),
            "score": best_score,
            "method": "tfidf",
            "confidence": "medium",
        }


# ============================================================
# Embedding index (matching + followups only)
# ============================================================
_embedding_state = {
    "qa_hash": "",
    "questions": [],
    "snos": [],
    "vectors": None,
}

def _hash_qa(qa_data: List[Dict[str, Any]]) -> str:
    s = "\n".join([str(x.get("question", "")).strip() for x in qa_data])
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def _normalize_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return v / n

def _load_embed_cache() -> Optional[dict]:
    if not os.path.exists(EMBED_CACHE_FILE):
        return None
    try:
        with open(EMBED_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _save_embed_cache(payload: dict):
    try:
        with open(EMBED_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass

def _embed_texts(texts: List[str]) -> np.ndarray:
    if not USE_OPENAI_EMBEDDINGS or client is None:
        raise RuntimeError("Embeddings disabled or OpenAI client not available.")
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
    return np.vstack(vectors)

def ensure_embedding_index(qa_data: List[Dict[str, Any]]):
    if not USE_OPENAI_EMBEDDINGS:
        return

    qa_hash = _hash_qa(qa_data)
    if _embedding_state["qa_hash"] == qa_hash and _embedding_state["vectors"] is not None:
        return

    questions = [str(x.get("question", "")).strip() for x in qa_data]
    snos = [int(x.get("sno", 0)) for x in qa_data]

    cache = _load_embed_cache()
    if cache and cache.get("qa_hash") == qa_hash and cache.get("questions") == questions:
        vecs = np.array(cache.get("vectors"), dtype=np.float32)
        vecs = _normalize_rows(vecs)
        _embedding_state.update({
            "qa_hash": qa_hash,
            "questions": questions,
            "snos": snos,
            "vectors": vecs,
        })
        return

    vecs = _embed_texts(questions)
    vecs = _normalize_rows(vecs)
    _save_embed_cache({"qa_hash": qa_hash, "questions": questions, "vectors": vecs.tolist()})

    _embedding_state.update({
        "qa_hash": qa_hash,
        "questions": questions,
        "snos": snos,
        "vectors": vecs,
    })

def embed_best_match(query: str, qa_data: List[Dict[str, Any]], threshold: float = 0.55) -> Optional[Dict[str, Any]]:
    if not USE_OPENAI_EMBEDDINGS:
        return None
    if _embedding_state["vectors"] is None:
        return None

    qv = _embed_texts([query])[0]
    qv = _normalize_vec(qv)

    sims = _embedding_state["vectors"] @ qv
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    if best_score < threshold:
        return None

    matched_question = _embedding_state["questions"][best_idx]
    for item in qa_data:
        if str(item.get("question", "")).strip() == matched_question:
            return {
                **pick_qa_fields(item),
                "matched_question": matched_question,
                "score": best_score,
                "method": "embedding",
                "confidence": "high",
            }
    return None

def suggest_followups_from_question(base_question: str, k: int = 5, exclude: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if not USE_OPENAI_EMBEDDINGS or _embedding_state["vectors"] is None:
        return []

    base_question = (base_question or "").strip()
    if not base_question:
        return []

    exclude_set = set([x.strip() for x in (exclude or []) if x.strip()])

    base_vec = _embed_texts([base_question])[0]
    base_vec = _normalize_vec(base_vec)

    sims = _embedding_state["vectors"] @ base_vec
    ranked = np.argsort(sims)[::-1]

    out = []
    for idx in ranked:
        q = _embedding_state["questions"][int(idx)]
        if not q:
            continue
        if q.strip() == base_question.strip():
            continue
        if q in exclude_set:
            continue
        out.append({"sno": int(_embedding_state["snos"][int(idx)]), "question": q, "score": float(sims[int(idx)])})
        if len(out) >= k:
            break
    return out


# ============================================================
# Suggestions endpoint
# ============================================================
def suggest_first_questions(qa_data: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for item in qa_data[:k]:
        q = str(item.get("question", "")).strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"sno": int(item.get("sno", 0)), "question": q})
    return out

@chat_llm_bp.route("/suggestions", methods=["GET"])
def api_suggestions():
    qa_data = load_qa_data()
    return jsonify({"suggestions": suggest_first_questions(qa_data, k=5)})

@chat_llm_bp.route("/questions", methods=["GET"])
def get_all_questions():
    qa = load_qa_data()
    return jsonify({"success": True, "questions": qa, "count": len(qa)})


# ============================================================
# MAIN ASK ROUTE
# - Answers ONLY from JSON
# - Fix multiple questions splitting
# - Fixed example follow-up issue
# ============================================================
@chat_llm_bp.route("/ask", methods=["POST"])
def ask_question():
    try:
        cleanup_old_context()

        data = request.json or {}
        user_text = (data.get("question") or "").strip()
        if not user_text:
            return jsonify({"error": "Question cannot be empty"}), 400

        session_id = get_session_id(data)
        session_ctx = conversation_context.get(session_id, {}) or {}

        # 1) Moderation
        if is_bad_language(user_text):
            return jsonify(scenario_payload("BAD_LANGUAGE", session_id))

        # 2) Scenario overrides
        override = local_scenario_override(user_text)
        if override:
            conversation_context[session_id] = {
                "updated_at": _now(),
                "last_topics": session_ctx.get("last_topics", []),
                "last_tense": session_ctx.get("last_tense", ""),
            }
            return jsonify(scenario_payload(override, session_id))

        # 3) Load DB
        qa_data = load_qa_data()
        if not qa_data:
            return jsonify({"error": "No questions available in database"}), 500

        ensure_embedding_index(qa_data)
        tfidf_engine = GenericQASearchEngine(qa_data)

        # 4) Build final queries (NO LLM)
        final_queries: List[str] = []

        # Compare request -> extract two topics
        if is_compare_request(user_text):
            pair = extract_two_topics_from_compare(user_text)
            if pair:
                a, b = pair
                final_queries = [normalize_topic_phrase(a), normalize_topic_phrase(b)]
            else:
                final_queries = [user_text]

        else:
            # Special fix: split "what is A and B tense"
            compound = split_compound_what_is_question(user_text)
            if compound:
                final_queries.extend(compound)
            else:
                # Follow-up handling
                rewritten_list = followup_rewrite_no_llm(user_text, session_ctx)

                for x in rewritten_list:
                    # If it is NOT a question, treat as topic-only and split properly
                    if not looks_like_question(x):
                        parts = split_topic_only_text(x)
                        for p in parts:
                            final_queries.append(normalize_topic_phrase(p))
                    else:
                        final_queries.append(x)

        # 5) Match each query with improved example handling
        resolved_items = []
        matches = []
        answers = []
        matched_topics_for_context: List[str] = []
        
        # Track matched questions to avoid duplicates
        matched_questions_set = set()

        # Check if this is an example request
        is_example_query = any(word in user_text.lower() for word in ['example', 'examples'])

        for q in final_queries:
            # Skip empty queries
            if not q:
                continue
                
            q_lower = q.lower()
            
            # Check if this specific query is about examples
            this_is_example = 'example' in q_lower or 'examples' in q_lower
            
            # Detect tense type
            is_simple_past = (
                re.search(r'\b(past|simple past|past simple)\b', q_lower) and 
                not re.search(r'\b(perfect|continuous|progressive|perfect continuous|perfect progressive)\b', q_lower)
            )
            
            is_simple_present = (
                re.search(r'\b(present|simple present|present simple)\b', q_lower) and 
                not re.search(r'\b(perfect|continuous|progressive|perfect continuous|perfect progressive)\b', q_lower)
            )
            
            is_simple_future = (
                re.search(r'\b(future|simple future|future simple)\b', q_lower) and 
                not re.search(r'\b(perfect|continuous|progressive|perfect continuous|perfect progressive)\b', q_lower)
            )
            
            # Adjust thresholds and matching strategy
            if is_simple_past or is_simple_present or is_simple_future:
                # For simple tenses, we want to EXCLUDE perfect and continuous tenses
                matched = embed_best_match(q, qa_data, threshold=0.70)
                
                # If embedding match found, verify it's actually the correct tense
                if matched:
                    matched_q = matched.get("matched_question", "").lower()
                    
                    # For example queries, check if it's an example content
                    if this_is_example:
                        # Look for example content specifically
                        if 'example' not in matched_q and 'examples' not in matched_q:
                            matched = None
                    else:
                        # For definition queries, reject if it has perfect/continuous
                        if re.search(r'\b(perfect|continuous|progressive)\b', matched_q):
                            matched = None
                
                # If no good embedding match, try TF-IDF with filtering
                if not matched:
                    # Get all potential matches and filter manually
                    if tfidf_engine.questions and tfidf_engine.question_vectors is not None:
                        qv = tfidf_engine.vectorizer.transform([q])
                        sims = cosine_similarity(qv, tfidf_engine.question_vectors).flatten()
                        
                        # Get top 10 indices
                        top_indices = np.argsort(sims)[-10:][::-1]
                        
                        # Find the first one that matches our criteria
                        for idx in top_indices:
                            if sims[idx] < 0.40:  # Lower threshold for TF-IDF
                                continue
                                
                            candidate_q = tfidf_engine.questions[idx].lower()
                            candidate_item = tfidf_engine.items[idx]
                            
                            # For example queries, prioritize example content
                            if this_is_example:
                                if 'example' in candidate_q or 'examples' in candidate_q:
                                    # Check if it's about the right tense
                                    if is_simple_present and 'present' in candidate_q:
                                        matched = {
                                            **pick_qa_fields(candidate_item),
                                            "matched_question": candidate_item.get("question", ""),
                                            "score": float(sims[idx]),
                                            "method": "tfidf_filtered",
                                            "confidence": "medium",
                                        }
                                        break
                                    elif is_simple_past and 'past' in candidate_q and not re.search(r'\b(perfect|continuous)\b', candidate_q):
                                        matched = {
                                            **pick_qa_fields(candidate_item),
                                            "matched_question": candidate_item.get("question", ""),
                                            "score": float(sims[idx]),
                                            "method": "tfidf_filtered",
                                            "confidence": "medium",
                                        }
                                        break
                                    elif is_simple_future and 'future' in candidate_q:
                                        matched = {
                                            **pick_qa_fields(candidate_item),
                                            "matched_question": candidate_item.get("question", ""),
                                            "score": float(sims[idx]),
                                            "method": "tfidf_filtered",
                                            "confidence": "medium",
                                        }
                                        break
                            else:
                                # For definition queries
                                if (re.search(r'\b(simple|tense)\b', candidate_q) and 
                                    not re.search(r'\b(perfect|continuous|progressive)\b', candidate_q)):
                                    # Check if it's about the right tense
                                    if is_simple_present and 'present' in candidate_q:
                                        matched = {
                                            **pick_qa_fields(candidate_item),
                                            "matched_question": candidate_item.get("question", ""),
                                            "score": float(sims[idx]),
                                            "method": "tfidf_filtered",
                                            "confidence": "medium",
                                        }
                                        break
                                    elif is_simple_past and 'past' in candidate_q:
                                        matched = {
                                            **pick_qa_fields(candidate_item),
                                            "matched_question": candidate_item.get("question", ""),
                                            "score": float(sims[idx]),
                                            "method": "tfidf_filtered",
                                            "confidence": "medium",
                                        }
                                        break
                                    elif is_simple_future and 'future' in candidate_q:
                                        matched = {
                                            **pick_qa_fields(candidate_item),
                                            "matched_question": candidate_item.get("question", ""),
                                            "score": float(sims[idx]),
                                            "method": "tfidf_filtered",
                                            "confidence": "medium",
                                        }
                                        break
            else:
                # For other queries, use standard matching
                embed_threshold = 0.65 if re.search(r'\b(perfect|continuous|progressive)\b', q_lower) else 0.55
                matched = embed_best_match(q, qa_data, threshold=embed_threshold)
                if not matched:
                    matched = tfidf_engine.best_match(q, threshold=0.45 if embed_threshold > 0.55 else 0.35)

            if not matched:
                continue
                
            # Check for duplicate matched question
            matched_q = matched.get("matched_question", "")
            if matched_q in matched_questions_set:
                continue
            matched_questions_set.add(matched_q)

            ans = str(matched.get("answer", "")).strip()
            if ans:
                answers.append(ans)

            matches.append({
                "effective_question": q,
                "matched_question": matched_q,
                "method": matched.get("method", ""),
                "confidence": matched.get("confidence", ""),
                "score": matched.get("score", None),
            })

            # Store full matched item
            resolved_items.append(pick_qa_fields(matched))

            mq = matched_q
            topic = re.sub(r"^(what is|give examples for|give example for|examples of|show me examples of)\s+", "", mq, flags=re.I).strip()
            topic = topic.rstrip("?").strip()
            if topic:
                matched_topics_for_context.append(topic)

        # 6) If matched, return combined JSON answers with media arrays
        if answers:
            # Dedupe answers
            seen = set()
            deduped_answers = []
            for a in answers:
                if a not in seen:
                    seen.add(a)
                    deduped_answers.append(a)

            # For compare: do NOT generate explanation; show two answers
            if is_compare_request(user_text) and len(deduped_answers) >= 2:
                final_answer = f"1) {deduped_answers[0]}\n\n2) {deduped_answers[1]}"
            elif len(deduped_answers) == 1:
                final_answer = deduped_answers[0]
            else:
                final_answer = "\n\n".join([f"{i+1}) {a}" for i, a in enumerate(deduped_answers)])

            # Save context with tense information
            if matched_topics_for_context:
                # Detect which tense was discussed
                last_topic = matched_topics_for_context[-1].lower()
                last_tense = ""
                if 'present' in last_topic:
                    last_tense = 'present'
                elif 'past' in last_topic:
                    last_tense = 'past'
                elif 'future' in last_topic:
                    last_tense = 'future'
                
                conversation_context[session_id] = {
                    "updated_at": _now(),
                    "last_topics": matched_topics_for_context[-2:] if matched_topics_for_context else session_ctx.get("last_topics", []),
                    "last_tense": last_tense,
                    "last_matched_question": (matches[-1].get("matched_question") if matches else ""),
                }
            else:
                conversation_context[session_id] = {
                    "updated_at": _now(),
                    "last_topics": session_ctx.get("last_topics", []),
                    "last_tense": session_ctx.get("last_tense", ""),
                }

            base_for_followups = (matches[-1].get("matched_question") if matches else user_text) or user_text
            followups = suggest_followups_from_question(base_for_followups, k=5, exclude=[base_for_followups, user_text])

            # ===== COMBINE ALL MEDIA FROM ALL MATCHED ITEMS INTO ARRAYS =====
            combined_media = {
                "audio_urls": [],
                "video_urls": [],
                "detail_urls": [],
                "story_urls": [],
                "example_urls": [],
                "detail_texts": [],
                "story_texts": [],
                "example_texts": [],
                "keywords": [],
            }
            
            # Collect all media from resolved items
            for item in resolved_items:
                if item.get("audio_url"):
                    combined_media["audio_urls"].append(item["audio_url"])
                if item.get("video_url"):
                    combined_media["video_urls"].append(item["video_url"])
                if item.get("detail_url"):
                    combined_media["detail_urls"].append(item["detail_url"])
                if item.get("story_url"):
                    combined_media["story_urls"].append(item["story_url"])
                if item.get("example_url"):
                    combined_media["example_urls"].append(item["example_url"])
                if item.get("detail_text"):
                    combined_media["detail_texts"].append(item["detail_text"])
                if item.get("story_text"):
                    combined_media["story_texts"].append(item["story_text"])
                if item.get("example_text"):
                    combined_media["example_texts"].append(item["example_text"])
                if item.get("keyword"):
                    combined_media["keywords"].append(item["keyword"])
            
            # main item for backward compatibility (first item)
            main_item = resolved_items[0] if resolved_items else {}
            main_item_safe = dict(main_item)
            for k in ["answer", "question", "sno"]:
                main_item_safe.pop(k, None)

            return jsonify({
                "scenario": "TENSE_QUESTION",
                "question": user_text,
                "answer": final_answer,
                "session_id": session_id,
                "matches": matches,
                "followups": followups,
                "resolved_items": resolved_items,
                # Combined media arrays
                "audio_urls": combined_media["audio_urls"],
                "video_urls": combined_media["video_urls"],
                "detail_urls": combined_media["detail_urls"],
                "story_urls": combined_media["story_urls"],
                "example_urls": combined_media["example_urls"],
                "detail_texts": combined_media["detail_texts"],
                "story_texts": combined_media["story_texts"],
                "example_texts": combined_media["example_texts"],
                "keywords": combined_media["keywords"],
                # Keep single fields for backward compatibility (first item)
                **main_item_safe
            })

        # 7) No match -> scenario
        short = len(user_text.split()) <= 3
        if short and not looks_like_question(user_text):
            scenario = "NOT_UNDERSTANDABLE"
        else:
            scenario = "OUT_OF_TOPIC"

        conversation_context[session_id] = {
            "updated_at": _now(),
            "last_topics": session_ctx.get("last_topics", []),
            "last_tense": session_ctx.get("last_tense", ""),
        }
        return jsonify(scenario_payload(scenario, session_id))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@chat_llm_bp.route('/transcribe', methods=['POST'])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file field named 'file'"}), 400

    f = request.files["file"]
    if not f:
        return jsonify({"error": "No file uploaded"}), 400

    language = request.form.get("language")

    tmp_path = None
    try:
        suffix = os.path.splitext(f.filename or "")[1].lower()
        if not suffix:
            suffix = ".webm"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            f.save(tmp_path)

        result = model.transcribe(
            tmp_path,
            language=language if language else None,
            fp16=False
        )

        text = (result.get("text") or "").strip()
        return jsonify({"text": text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

@chat_llm_bp.route("/clear_context", methods=["POST"])
def clear_context():
    data = request.json or {}
    session_id = get_session_id(data)
    if session_id in conversation_context:
        del conversation_context[session_id]
    return jsonify({"status": "context cleared", "session_id": session_id})

