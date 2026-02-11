"""
Generic Context Manager — domain-agnostic conversation context tracking.

Works by extracting key noun phrases / entities from user questions using
lightweight NLP (TF-IDF keyword extraction + POS tagging when NLTK is
available).  No hardcoded topics, tense lists, or domain keywords.
"""

from datetime import datetime
import re
import string

# Try to import NLP tools (graceful fallback)
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag

    for resource in ['tokenizers/punkt', 'taggers/averaged_perceptron_tagger',
                     'taggers/averaged_perceptron_tagger_eng']:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split('/')[-1], quiet=True)
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False

# ---------------------------------------------------------------------------
# In-memory store keyed by user_id
# ---------------------------------------------------------------------------
CONTEXT_STORE: dict[str, dict] = {}

MAX_HISTORY = 10          # rolling window of recent exchanges
FOLLOW_UP_MAX_WORDS = 5   # questions this short are likely follow-ups


# ---------------------------------------------------------------------------
# Lightweight keyword / topic extraction (no domain knowledge required)
# ---------------------------------------------------------------------------

# Common English stopwords (small set so we don't need NLTK corpus)
_STOPWORDS = frozenset(
    "i me my myself we our ours ourselves you your yours yourself yourselves "
    "he him his himself she her hers herself it its itself they them their "
    "theirs themselves what which who whom this that these those am is are was "
    "were be been being have has had having do does did doing a an the and but "
    "if or because as until while of at by for with about against between into "
    "through during before after above below to from up down in out on off over "
    "under again further then once here there when where why how all both each "
    "few more most other some such no nor not only own same so than too very "
    "can will just don should now would could shall may might".split()
)

# Pronouns that signal a follow-up (the user refers to something earlier)
_PRONOUNS = frozenset(
    "it that this they them those these he she his her".split()
)

# Phrases that signal the user wants more about the same topic
_FOLLOWUP_PHRASES = [
    "tell me more", "explain more", "more about", "give example",
    "give examples", "show me", "elaborate", "go on", "continue",
    "what about", "how about", "another example", "more details",
    "can you explain", "why is that", "what do you mean",
]


def extract_keywords(text: str) -> list[str]:
    """
    Extract meaningful keywords / noun-phrases from *text*.

    When NLTK is available we use POS tagging to pull nouns, verbs and
    adjectives.  Otherwise we fall back to simple stop-word removal.
    """
    if not text:
        return []

    text = text.lower().strip()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()

    if _NLTK_AVAILABLE:
        try:
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)
            # Keep nouns (NN*), verbs (VB*), adjectives (JJ*), and foreign words (FW)
            keywords = [
                word for word, tag in tagged
                if tag[:2] in ('NN', 'VB', 'JJ', 'FW')
                and word not in _STOPWORDS
                and len(word) > 1
            ]
            return keywords
        except Exception:
            pass  # fall through to simple mode

    # Simple fallback — remove stopwords
    return [w for w in tokens if w not in _STOPWORDS and len(w) > 1]


def _build_topic_summary(keywords: list[str]) -> str:
    """Build a short readable topic string from extracted keywords."""
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return " ".join(unique[:6])  # keep it short


# ---------------------------------------------------------------------------
# Follow-up detection (generic, no domain knowledge)
# ---------------------------------------------------------------------------

def is_follow_up(question: str, context: dict) -> bool:
    """
    Decide whether *question* is a follow-up to the running conversation
    stored in *context*.  Uses structural cues only — no domain words.
    """
    if not question or not context:
        return False

    q = question.lower().strip()
    words = q.split()
    history = context.get("conversation_history", [])

    # Nothing to follow up on
    if not history:
        return False

    # Contains an explicit follow-up phrase
    if any(phrase in q for phrase in _FOLLOWUP_PHRASES):
        return True

    # Starts with a pronoun that likely refers to prior context
    if words and words[0] in _PRONOUNS:
        return True

    # Short queries: only treat as follow-up if they have NO meaningful
    # content words (e.g. "yes", "ok", "and?").  Queries like "past tense"
    # or "tense" carry their own meaning and should NOT be altered.
    if len(words) <= FOLLOW_UP_MAX_WORDS:
        content_words = [w for w in words if w not in _STOPWORDS and len(w) > 1]
        # If the short query has at least one real content word, treat it as
        # a standalone query so the matcher can use it directly.
        if content_words:
            return False
        # No content words at all (e.g. "ok", "yes") → follow-up
        return True

    # Question has very few new keywords compared to last exchange
    prev_keywords = set(context.get("current_keywords", []))
    cur_keywords = set(extract_keywords(question))
    if prev_keywords and cur_keywords:
        overlap = cur_keywords & prev_keywords
        new_words = cur_keywords - prev_keywords
        if len(overlap) >= 1 and len(new_words) <= 2:
            return True

    return False


def enhance_with_context(question: str, context: dict) -> str:
    """
    If the question is a follow-up, prepend / append context so downstream
    matching has enough information.

    Example flow:
        Q1: "What is photosynthesis?"       → topic = "photosynthesis"
        Q2: "Give example"                  → enhanced to "Give example of photosynthesis"
    """
    if not is_follow_up(question, context):
        return question

    topic = context.get("current_topic", "")
    if not topic:
        return question

    q_lower = question.lower()
    # Don't duplicate if topic words are already present
    topic_words = set(topic.lower().split())
    question_words = set(q_lower.split())
    if topic_words.issubset(question_words):
        return question

    # Append topic context
    enhanced = f"{question} of {topic}"
    return enhanced


# ---------------------------------------------------------------------------
# Context CRUD
# ---------------------------------------------------------------------------

def _blank_context() -> dict:
    return {
        "conversation_history": [],
        "current_topic": None,
        "current_keywords": [],
        "last_question": None,
        "last_answer": None,
        "follow_up_count": 0,
        "session_start": datetime.now().isoformat(),
    }


def get_context(user_id: str) -> dict:
    if user_id not in CONTEXT_STORE:
        CONTEXT_STORE[user_id] = _blank_context()
    return CONTEXT_STORE[user_id]


def clear_context(user_id: str) -> None:
    CONTEXT_STORE.pop(user_id, None)


def record_exchange(
    user_id: str,
    user_question: str,
    bot_response: str,
    matched_question: str | None = None,
) -> dict:
    """
    Record a Q-A exchange and auto-update topic / keywords.

    Returns the updated context dict.
    """
    ctx = get_context(user_id)

    # --- extract keywords from the question (+ matched question if any) ---
    combined_text = user_question
    if matched_question:
        combined_text = f"{user_question} {matched_question}"

    keywords = extract_keywords(combined_text)

    # --- decide if follow-up ---
    followup = is_follow_up(user_question, ctx)

    if followup:
        # Merge new keywords into existing set (topic stays)
        existing = ctx.get("current_keywords", [])
        merged = list(dict.fromkeys(existing + keywords))  # dedupe, order preserved
        ctx["current_keywords"] = merged[:15]
        ctx["follow_up_count"] = ctx.get("follow_up_count", 0) + 1
    else:
        # New topic
        ctx["current_keywords"] = keywords
        ctx["current_topic"] = _build_topic_summary(keywords)
        ctx["follow_up_count"] = 0

    # --- append to rolling history ---
    ctx["conversation_history"].append({
        "timestamp": datetime.now().isoformat(),
        "user_question": user_question,
        "bot_response": bot_response[:300],  # truncate to save memory
        "matched_question": matched_question,
        "is_follow_up": followup,
    })
    if len(ctx["conversation_history"]) > MAX_HISTORY:
        ctx["conversation_history"] = ctx["conversation_history"][-MAX_HISTORY:]

    ctx["last_question"] = user_question
    ctx["last_answer"] = bot_response

    return ctx