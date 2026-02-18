from flask import Blueprint, request, jsonify, session
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import string
import tempfile
from datetime import datetime

# Import the generic context manager
from context_manager import (
    get_context, clear_context, record_exchange,
    is_follow_up, enhance_with_context, extract_keywords,
)

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

# Add SymSpell for spell checking
try:
    from symspellpy import SymSpell, Verbosity
    import pkg_resources
    SYMSPELL_AVAILABLE = True
except ImportError:
    print("SymSpell not available. Please install: pip install symspellpy")
    SYMSPELL_AVAILABLE = False


staticchat_bp = Blueprint("staticchat", __name__)

# Initialize SymSpell if available
sym_spell = None
if SYMSPELL_AVAILABLE:
    try:
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        bigram_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
        )
        
        # Load dictionaries
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=1)
        print("SymSpell spell checker initialized successfully")
    except Exception as e:
        print(f"Failed to initialize SymSpell: {e}")
        SYMSPELL_AVAILABLE = False

# Try to import NLTK with fallback
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    
    # Download required NLTK resources
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    NLTK_AVAILABLE = True
except Exception as e:
    print(f"NLTK not available, using simple text processing: {e}")
    NLTK_AVAILABLE = False

# Scenario configurations — these are conversational UX patterns,
# NOT domain-specific topics.  They handle greetings, thanks, etc.
SCENARIOS = {
    "greeting": {
        "keywords": [
            "good morning", "good afternoon", "good evening",
            "hello", "hi", "hey", "greetings", "hai", "hii",
            "good day", "howdy", "namaste", "hola",
        ],
        "message": {
            "morning": "Good morning! Let's begin our lesson on tenses. You can ask me any question about tenses",
            "afternoon": "Good afternoon! Let's begin our lesson on tenses. You can ask me any question about tenses",
            "evening": "Good evening! Let's begin our lesson on tenses. You can ask me any question about tenses",
            "general": "Hello! Welcome to the English Tenses Learning Assistant. How can I help you with tenses today?"
        },
        "audio_url": {
            "morning": "assets/staticchat/intro.mp3",
            "afternoon": "assets/staticchat/intro.mp3",
            "evening": "assets/staticchat/intro.mp3",
            "general": "assets/staticchat/intro.mp3"
        },
        "video_url": {
            "morning": "assets/staticchat/intro.mp4",
            "afternoon": "assets/staticchat/intro.mp4",
            "evening": "assets/staticchat/intro.mp4",
            "general": "assets/staticchat/intro.mp4"
        },
        "story_url": "",
        "detail_url": "",
        "example_url": "",
        "story_text": "",
        "detail_text": "",
        "example_text": "",
        "type": "scenario"
    },
    "thanks": {
        "keywords": [
            "thank you", "thanks", "thank you very much", "appreciate it",
            "thanks a lot", "thank u", "thanku", "tq", "ty",
            "this class was very nice", "this class was great",
            "this was helpful", "very helpful", "great class",
            "nice class", "wonderful class", "awesome class",
            "loved this class", "enjoyed this class",
            "this session was good", "good session",
            "nice session", "great session", "amazing",
            "fantastic", "excellent", "good job", "well done",
        ],
        "message": "You're welcome! Do you have any other questions?",
        "audio_url": "assets/staticchat/you_are_welcome.mp3",
        "video_url": "assets/staticchat/you_are_welcome.mp4",
        "story_url": "",
        "detail_url": "",
        "example_url": "",
        "story_text": "",
        "detail_text": "",
        "example_text": "",
        "type": "scenario"
    },
    "farewell": {
        "keywords": [
            "bye", "goodbye", "see you", "farewell", "take care", "bye bye",
            "bye mam", "bye ma'am", "bye teacher", "bye miss", "bye sir",
            "close this session", "end this session", "end session",
            "i will close", "i am leaving", "i'm leaving",
            "got to go", "gotta go", "see you later", "see ya",
            "good night", "goodnight", "signing off", "log off",
            "that's all", "thats all", "no more questions",
            "i am done", "i'm done", "im done", "end class",
            "close class", "finish", "finished", "complete",
        ],
        "message": "Goodbye! Keep practicing your English tenses. Remember, practice makes perfect!",
        "audio_url": "assets/staticchat/bye.mp3",
        "video_url": "assets/staticchat/bye.mp4",
        "story_url": "",
        "detail_url": "",
        "example_url": "",
        "story_text": "",
        "detail_text": "",
        "example_text": "",
        "type": "scenario"
    },
    # SEPARATED: Rude/inappropriate language
    "rude_language": {
        "keywords": [
            # Rude/offensive words
            "stupid", "idiot", "dumb", "shut up", "hate you",
            "fool", "useless", "waste", "sucks",
            "damn", "hell", "crap", "rubbish", "nonsense",
            "fuck", "shit", "bastard", "asshole", "bitch",
        ],
        "message": (
            "Please use respectful language. 😊 We're here to learn together. "
            "Let's focus on tenses. Would you like to ask a question about English tenses?"
        ),
        "suggestions": [
            "What is a tense?",
            "How many types of tenses are there?",
            "Give an example of Present Tense",
        ],
        "audio_url": "assets/staticchat/blink.mp3",
        "video_url": "assets/staticchat/bad_lang.mp4",
        "story_url": "",
        "detail_url": "",
        "example_url": "",
        "story_text": "",
        "detail_text": "",
        "example_text": "",
        "type": "scenario"
    },
    # SEPARATED: Tired/sleepy/bored (disruptive but polite)
    "disruptive_behavior": {
        "keywords": [
            # Tired/sleepy phrases
            "sleeping", "i am sleeping", "i'm sleeping",
            "iam getting sleeping", "i am getting sleeping",
            "i'm getting sleeping", "im getting sleeping",
            "feeling sleepy", "want to sleep", "tired to study",
            "sleepy now", "feeling drowsy", "can't focus",
            "too tired", "exhausted", "need rest",
            
            # Bored phrases
            "i am bored", "i'm bored", "so bored", "very bored",
            "boring", "this is boring", "class is boring",
            "not interested", "no interest",
            
            # Distracted phrases
            "i don't want to study", "i dont want to study",
            "i want to play", "let me play",
            "i want to sleep", "let me sleep",
            "i don't care", "i dont care", "whatever",
            "i don't like this", "i dont like this",
            "this is waste", "waste of time", "time waste",
            "i hate this", "i hate english", "i hate tenses",
        ],
        "message": (
            "I understand you might be feeling a bit tired or distracted. 😊 "
            "But learning tenses is really important and will help you a lot! "
            "Let's try to focus — you can do this! "
            "Would you like to start with something simple? Try asking: \"What is a tense?\""
        ),
        "suggestions": [
            "What is a tense?",
            "How many types of tenses are there?",
            "Give an example of Present Tense",
        ],
        "audio_url": "assets/staticchat/blink.mp3",
        "video_url": "assets/staticchat/sleepy.mp4",
        "story_url": "",
        "detail_url": "",
        "example_url": "",
        "story_text": "",
        "detail_text": "",
        "example_text": "",
        "type": "scenario"
    },
    "not_understandable": {
        "message": "I don't understand your question. Can you ask it again more simply?",
        "suggestions": [
            "Try using simpler words",
            "Be more specific about what you want to know",
            "Example: What is present tense?"
        ],
        "audio_url": "assets/staticchat/not_understand.mp3",
        "video_url": "assets/staticchat/not_understand.mp4",
        "story_url": "",
        "detail_url": "",
        "example_url": "",
        "story_text": "",
        "detail_text": "",
        "example_text": "",
        "type": "scenario"
    },
    "out_of_topic": {
        "message": (
            "That's not part of our tense lesson. Let's stay on our topic."
            
        ),
        "suggestions": [
            "What is a tense?",
            "How many types of tenses are there?",
            "What is Simple Present Tense?",
        ],
        "audio_url": "assets/staticchat/out_of_topic.mp3",
        "video_url": "assets/staticchat/out_of_topic.mp4",
        "story_url": "",
        "detail_url": "",
        "example_url": "",
        "story_text": "",
        "detail_text": "",
        "example_text": "",
        "type": "scenario"
    },
    "not_available": {
        "message": "I don't have the answer for that. Let's not available in my lesson today.",
        "suggestions": [
            "Try asking your question in a different way",
            "Ask about a more specific tense type",
            "Example: What is Future Perfect Tense?"
        ],
        "audio_url": "assets/staticchat/no_db.mp3",
        "video_url": "assets/staticchat/no_db.mp4",
        "story_url": "",
        "detail_url": "",
        "example_url": "",
        "story_text": "",
        "detail_text": "",
        "example_text": "",
        "type": "scenario"
    }
}

# ============================================
# ORIGINAL FUNCTIONS (with generic context)
# ============================================

# Load questions from JSON file
def load_questions():
    try:
        with open('assets/qa.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} questions from qa.json")
        return data
    except FileNotFoundError:
        print("Error: qa.json not found")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing qa.json: {e}")
        return []

# Spell correction function
def correct_spelling(text):
    """Correct spelling using SymSpell"""
    if not SYMSPELL_AVAILABLE or sym_spell is None:
        return text
    
    try:
        words = text.split()
        corrected_words = []
        
        for word in words:
            if len(word) <= 2:
                corrected_words.append(word)
                continue
            
            suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions and suggestions[0].term != word:
                corrected_words.append(suggestions[0].term)
                print(f"Corrected '{word}' to '{suggestions[0].term}'")
            else:
                corrected_words.append(word)
        
        corrected_text = ' '.join(corrected_words)
        
        bigram_suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
        if bigram_suggestions and bigram_suggestions[0].term != corrected_text:
            print(f"Bigram correction: '{text}' -> '{bigram_suggestions[0].term}'")
            return bigram_suggestions[0].term
        
        return corrected_text
    except Exception as e:
        print(f"Spell correction error: {e}")
        return text

# Enhanced text preprocessing
def preprocess_text(text):
    """Preprocess text with spelling correction and enhanced NLP"""
    if not text:
        return ""
    
    if SYMSPELL_AVAILABLE:
        text = correct_spelling(text)
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    
    if NLTK_AVAILABLE:
        try:
            tokens = word_tokenize(text)
            
            stop_words = set(stopwords.words('english'))
            # Do NOT keep auxiliary verbs — they are question-structure words
            # (e.g. "is", "does") that hurt TF-IDF discrimination between
            # "What is future tense?" and "What is future perfect tense?"
            
            tokens = [word for word in tokens if word not in stop_words]
            
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
            
            return ' '.join(tokens)
        except Exception as e:
            print(f"Error in NLP processing: {e}")
            return text
    else:
        # Simple stopword removal
        basic_stopwords = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'by', 'for',
                          'with', 'about', 'against', 'between', 'into', 'through',
                          'during', 'before', 'after', 'above', 'below', 'to', 'from',
                          'up', 'down', 'out', 'off', 'over', 'under', 'again',
                          'further', 'then', 'once', 'here', 'there', 'when', 'where',
                          'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                          'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                          'own', 'same', 'so', 'than', 'too', 'very'}
        
        words = text.split()
        return ' '.join(w for w in words if w not in basic_stopwords)

# Filler words to strip when isolating the user's real subject.
_FILLER_WORDS = frozenset({
    'what', 'is', 'are', 'does', 'do', 'did', 'was', 'were',
    'explain', 'describe', 'give', 'tell', 'me', 'about',
    'define', 'mean', 'meaning', 'please', 'can', 'you',
    'the', 'a', 'an', 'how', 'why', 'which', 'where', 'when',
    'who', 'have', 'has', 'had', 'will', 'shall', 'would',
    'could', 'should', 'may', 'might', 'mam', 'sir', 'miss', 'teacher',
})


def _extract_content_words(preprocessed_text):
    """Extract content words from a preprocessed string, removing filler words."""
    return set(preprocessed_text.split()) - _FILLER_WORDS


# Initialize questions data
questions_data = load_questions()
question_texts = [item['question'] for item in questions_data]
preprocessed_questions = [preprocess_text(q) for q in question_texts]

# Pre-process keyword fields once at startup for consistent comparison
preprocessed_keywords = [preprocess_text(item.get('keyword', '')) for item in questions_data]

# Build a set of all known content words from the database ONCE at startup
_ALL_DB_CONTENT_WORDS = set()
for _item in questions_data:
    _kw = _item.get('keyword', '')
    if _kw:
        _ALL_DB_CONTENT_WORDS.update(preprocess_text(_kw).split())
for _pq in preprocessed_questions:
    _ALL_DB_CONTENT_WORDS.update(set(_pq.split()) - _FILLER_WORDS)
# Also add the raw (lowercased) keyword words WITHOUT spell-correction
for _item in questions_data:
    _kw = _item.get('keyword', '')
    if _kw:
        _ALL_DB_CONTENT_WORDS.update(_kw.lower().split())
print(f"📚 Known DB content words: {sorted(_ALL_DB_CONTENT_WORDS)}")

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
if preprocessed_questions:
    tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
else:
    tfidf_matrix = None


def _extract_raw_content_words(text):
    """Extract content words from RAW text (no spell correction) for syllabus check."""
    if not text:
        return set()
    t = text.lower()
    t = re.sub(r'[^\w\s]', ' ', t)
    words = t.split()
    return set(words) - _FILLER_WORDS


def _is_out_of_syllabus(user_question):
    """Check if the user's question is completely outside the syllabus."""
    user_content = _extract_raw_content_words(user_question)
    if not user_content:
        return False
    
    overlap = user_content & _ALL_DB_CONTENT_WORDS
    print(f"🔍 Syllabus check: user_content={user_content}, overlap={overlap}")
    return len(overlap) == 0


def _is_gibberish(text):
    """Improved gibberish detection that catches random keyboard mashing."""
    if not text:
        return True
    
    clean = re.sub(r'[^\w\s]', '', text.lower()).strip()
    
    # Very short after cleaning
    if len(clean) < 2:
        return True
    
    # Check for random keyboard mashing (like "skjldflkjsdlf")
    # If it's a single "word" with more than 5 characters and no vowels, it's gibberish
    words = clean.split()
    
    # Single long word with no vowels or mostly consonants
    if len(words) == 1:
        word = words[0]
        if len(word) > 5:
            # Count vowels
            vowels = sum(1 for char in word if char in 'aeiou')
            vowel_ratio = vowels / len(word)
            
            # If no vowels or very few vowels, likely gibberish
            if vowel_ratio < 0.2:
                print(f"🔤 No vowels detected in '{word}', marking as gibberish")
                return True
            
            # Check for repeated consonant patterns (like "skjldflkj")
            if re.search(r'(.)\1{2,}', word):  # Same character repeated 3+ times
                return True
    
    # Check for random character sequences (no real words)
    # A word is "real" if it has at least one vowel and reasonable consonant-vowel pattern
    real_word_count = 0
    for word in words:
        if len(word) <= 2:
            # Very short words might be real ("am", "is", "in", etc.)
            if word in {'am', 'is', 'in', 'on', 'at', 'by', 'my', 'me', 'we', 'he', 'an', 'as', 'it', 'of', 'or', 'so', 'to', 'up', 'us'}:
                real_word_count += 1
        else:
            # Longer word: check if it has vowels and reasonable pattern
            has_vowel = any(char in 'aeiou' for char in word)
            if has_vowel:
                real_word_count += 1
    
    # If no real words found, it's gibberish
    if real_word_count == 0:
        print(f"🤷 No real words detected in '{text}', marking as gibberish")
        return True
    
    # Single filler word only
    if len(words) == 1 and words[0] in _FILLER_WORDS:
        return True
    
    # Check for repetitive characters
    if len(words) == 1 and len(set(clean)) <= 2:
        return True
    
    # Common single words that are incomplete questions
    common_single_words = {'what', 'who', 'why', 'when', 'where', 'how', 'which', 'whose', 'whom'}
    if len(words) == 1 and words[0] in common_single_words:
        return True
    
    return False


def _is_greeting(text):
    """Check if text is a greeting with proper word boundary matching."""
    if not text:
        return False, "general"
    
    text_lower = text.lower().strip()
    
    # First check for exact greeting phrases
    greeting_phrases = [
        ("good morning", "morning"),
        ("good afternoon", "afternoon"),
        ("good evening", "evening"),
        ("hello", "general"),
        ("hi", "general"),
        ("hey", "general"),
        ("greetings", "general"),
        ("hai", "general"),
        ("hii", "general"),
        ("good day", "general"),
        ("howdy", "general"),
        ("namaste", "general"),
        ("hola", "general"),
    ]
    
    for phrase, greeting_type in greeting_phrases:
        # Use word boundary or exact match
        if phrase in text_lower:
            # Make sure it's not part of another word
            if re.search(r'\b' + re.escape(phrase) + r'\b', text_lower):
                current_hour = datetime.now().hour
                if greeting_type == "general":
                    if current_hour < 12:
                        return True, "morning"
                    elif current_hour < 17:
                        return True, "afternoon"
                    else:
                        return True, "evening"
                else:
                    return True, greeting_type
    
    return False, "general"


def _is_thanks(text):
    """Check if text expresses gratitude."""
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # Check for thank you patterns
    thank_patterns = [
        r'thank(?:s| you)(?:\s+(?:you|very much|a lot))?',
        r'appreciate(?:\s+it)?',
        r'(?:very|really|so)\s+(?:nice|good|great|helpful|wonderful|awesome|amazing|fantastic|excellent)',
        r'(?:loved|enjoyed|liked)\s+(?:this|the|class|session)',
        r'good\s+(?:class|session|job|work)',
        r'well\s+done',
        r'excellent\s+work',
        r'this\s+was\s+(?:very\s+)?helpful',
        r'this\s+class\s+was\s+(?:very\s+)?(?:nice|good|great)',
    ]
    
    for pattern in thank_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False


def _is_farewell(text):
    """Check if text is a farewell message."""
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # Check for farewell patterns
    farewell_patterns = [
        r'bye(?:\s+(?:bye|mam|ma\'am|teacher|miss|sir))?',
        r'goodbye',
        r'see\s+(?:you|ya)(?:\s+(?:later|tomorrow|soon))?',
        r'take\s+care',
        r'farewell',
        r'signing\s+off',
        r'logging\s+off',
        r'close\s+(?:this\s+)?(?:session|class)',
        r'end\s+(?:this\s+)?(?:session|class)',
        r'i(?:\'m|\s+am)\s+(?:leaving|done|finished|complete)',
        r'that(?:\'s|s)\s+all',
        r'no\s+more\s+questions',
        r'got(?:ta)?\s+go',
        r'good\s+night',
    ]
    
    for pattern in farewell_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False


def _is_rude_language(text):
    """Check if text contains rude/offensive language."""
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # List of rude words/phrases
    rude_words = [
        r'\bstupid\b',
        r'\bidiot\b',
        r'\bdumb\b',
        r'\bshut\s+up\b',
        r'\bhate\s+you\b',
        r'\bfool\b',
        r'\buseless\b',
        r'\bwaste\b',
        r'\bsucks\b',
        r'\bdamn\b',
        r'\bhell\b',
        r'\bcrap\b',
        r'\brubbish\b',
        r'\bnonsense\b',
        r'\bfuck\b',
        r'\bshit\b',
        r'\bbastard\b',
        r'\basshole\b',
        r'\bbitch\b',
    ]
    
    for pattern in rude_words:
        if re.search(pattern, text_lower):
            return True
    
    return False


def _is_disruptive_behavior(text):
    """Check if text indicates tiredness, boredom, or distraction."""
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # Patterns for disruptive behavior
    disruptive_patterns = [
        # Sleeping/tired
        r'\bsleeping\b',
        r'i(?:\'m|\s+am)\s+sleeping',
        r'getting\s+sleepy',
        r'feeling\s+sleepy',
        r'want\s+to\s+sleep',
        r'tired\s+to\s+study',
        r'too\s+tired',
        r'exhausted',
        r'need\s+rest',
        r'can\'?t\s+focus',
        r'feeling\s+drowsy',
        
        # Bored
        r'i(?:\'m|\s+am)\s+bored',
        r'so\s+bored',
        r'very\s+bored',
        r'\bboring\b',
        r'class\s+is\s+boring',
        r'this\s+is\s+boring',
        r'not\s+interested',
        r'no\s+interest',
        
        # Distracted/want to do something else
        r'i\s+don\'?t\s+want\s+to\s+study',
        r'i\s+want\s+to\s+play',
        r'let\s+me\s+play',
        r'i\s+want\s+to\s+sleep',
        r'let\s+me\s+sleep',
        r'i\s+don\'?t\s+care',
        r'whatever',
        r'i\s+don\'?t\s+like\s+this',
        r'this\s+is\s+waste',
        r'waste\s+of\s+time',
        r'time\s+waste',
        r'i\s+hate\s+this',
        r'i\s+hate\s+english',
        r'i\s+hate\s+tenses',
    ]
    
    for pattern in disruptive_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False


def _split_multi_question(text):
    """Detect and split multi-part questions."""
    if not text:
        return None
    
    text_lower = text.lower().strip()
    
    # Pattern 1: "difference between X and Y"
    diff_match = re.match(
        r'(?:what\s+is\s+the\s+)?(?:difference|diff)\s+between\s+(.+?)\s+and\s+(.+?)[\?\.]?\s*$',
        text_lower
    )
    if diff_match:
        return [diff_match.group(1).strip(), diff_match.group(2).strip()]
    
    # Pattern 2: "what is X and what is Y"
    double_what = re.match(
        r'what\s+is\s+(.+?)\s+and\s+what\s+is\s+(.+?)[\?\.]?\s*$',
        text_lower
    )
    if double_what:
        return [double_what.group(1).strip(), double_what.group(2).strip()]
    
    # Pattern 3: "what is X and Y" where both are tense-related
    what_and = re.match(
        r'(?:what\s+is|explain|describe|tell\s+me\s+about)\s+(.+?)\s+and\s+(.+?)[\?\.]?\s*$',
        text_lower
    )
    if what_and:
        part1 = what_and.group(1).strip()
        part2 = what_and.group(2).strip()
        if 'tense' in part1 and 'tense' in part2:
            return [part1, part2]
        if 'tense' in part2 and 'tense' not in part1:
            return [part1 + ' tense', part2]
    
    # Pattern 4: bare "X and Y" where both contain "tense"
    bare_and = re.match(
        r'(.+?)\s+and\s+(.+?)[\?\.]?\s*$',
        text_lower
    )
    if bare_and:
        part1 = bare_and.group(1).strip()
        part2 = bare_and.group(2).strip()
        if 'tense' in part1 and 'tense' in part2:
            return [part1, part2]
        if 'tense' in part2 and 'tense' not in part1:
            return [part1 + ' tense', part2]
    
    return None


def _build_scenario_response(scenario_key):
    """Build a standard scenario response dict from a SCENARIOS key."""
    sc = SCENARIOS[scenario_key]
    result = {
        "scenario": scenario_key,
        "message": sc["message"] if isinstance(sc["message"], str) else sc["message"]["general"],
        "audio_url": sc.get("audio_url", ""),
        "video_url": sc.get("video_url", ""),
        "story_url": sc.get("story_url", ""),
        "detail_url": sc.get("detail_url", ""),
        "example_url": sc.get("example_url", ""),
        "story_text": sc.get("story_text", ""),
        "detail_text": sc.get("detail_text", ""),
        "example_text": sc.get("example_text", ""),
    }
    if "suggestions" in sc:
        result["suggestions"] = sc["suggestions"]
    return result


def _build_greeting_response(greeting_type):
    """Build greeting response with time-specific audio/video."""
    resp = {
        "scenario": "greeting",
        "message": SCENARIOS["greeting"]["message"][greeting_type],
        "audio_url": SCENARIOS["greeting"]["audio_url"][greeting_type],
        "video_url": SCENARIOS["greeting"]["video_url"][greeting_type],
        "story_url": "",
        "detail_url": "",
        "example_url": "",
        "story_text": "",
        "detail_text": "",
        "example_text": "",
        "type": "scenario"
    }
    return resp


def _build_json_response(scenario_response, original_question, ctx, followup):
    """Helper to build consistent JSON response for scenarios."""
    return jsonify({
        'success': True,
        'scenario': scenario_response['scenario'],
        'message': scenario_response['message'],
        'audio_url': scenario_response.get('audio_url', ''),
        'video_url': scenario_response.get('video_url', ''),
        'story_url': scenario_response.get('story_url', ''),
        'detail_url': scenario_response.get('detail_url', ''),
        'example_url': scenario_response.get('example_url', ''),
        'story_text': scenario_response.get('story_text', ''),
        'detail_text': scenario_response.get('detail_text', ''),
        'example_text': scenario_response.get('example_text', ''),
        'suggestions': scenario_response.get('suggestions', []),
        'user_question': original_question,
        'matching_method': 'scenario',
        'conversation_context': {
            'is_follow_up': followup,
            'current_topic': ctx.get('current_topic'),
            'follow_up_count': ctx.get('follow_up_count', 0)
        }
    })


def calculate_similarity(user_question):
    """Calculate similarity between user question and stored questions using TF-IDF + keyword boost"""
    if not preprocessed_questions:
        return np.array([])
    
    preprocessed_user_q = preprocess_text(user_question)
    user_vector = vectorizer.transform([preprocessed_user_q])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)[0]
    
    user_words = set(preprocessed_user_q.split())
    user_content_words = _extract_content_words(preprocessed_user_q)
    
    exact_keyword_idx = None
    
    for i, q_data in enumerate(questions_data):
        q_words = set(preprocessed_questions[i].split())
        if not q_words or not user_words:
            continue
        
        # 1. Coverage penalty
        matched_words = user_words & q_words
        if matched_words:
            coverage = len(matched_words) / len(q_words)
            penalty = 0.7 + 0.3 * coverage
            similarity_scores[i] *= penalty
        
        # 2. Keyword field matching
        if not user_content_words:
            continue
        
        kw_words = set(preprocessed_keywords[i].split()) if preprocessed_keywords[i] else set()
        if not kw_words:
            continue
        
        if user_content_words == kw_words:
            similarity_scores[i] += 0.6
            exact_keyword_idx = i
        elif kw_words.issubset(user_content_words):
            ratio = len(kw_words) / len(user_content_words)
            similarity_scores[i] += 0.4 * ratio
        elif user_content_words.issubset(kw_words):
            ratio = len(user_content_words) / len(kw_words)
            similarity_scores[i] += 0.3 * ratio
            extra_words = kw_words - user_content_words
            similarity_scores[i] -= 0.15 * len(extra_words)
        else:
            overlap = user_content_words & kw_words
            if overlap:
                ratio = len(overlap) / max(len(user_content_words), len(kw_words))
                similarity_scores[i] += 0.2 * ratio
    
    if exact_keyword_idx is not None:
        current_max = similarity_scores.max()
        if similarity_scores[exact_keyword_idx] < current_max:
            similarity_scores[exact_keyword_idx] = current_max + 0.1
            print(f"⬆️ Forced exact keyword match (index {exact_keyword_idx}) to top")
    
    return similarity_scores


def keyword_match(user_question, questions):
    """Fallback keyword matching — checks both question text AND keyword field"""
    user_words = set(preprocess_text(user_question).split())
    user_content_words = user_words - _FILLER_WORDS
    matches = []
    
    for i, q_data in enumerate(questions):
        question_words = set(preprocess_text(q_data['question']).split())
        kw_words = set(preprocessed_keywords[i].split()) if preprocessed_keywords[i] else set()
        
        all_target_words = question_words | kw_words
        common_content = user_content_words.intersection(all_target_words - _FILLER_WORDS)
        
        if common_content:
            target_content = all_target_words - _FILLER_WORDS
            score = len(common_content) / max(len(user_content_words), len(target_content)) if user_content_words and target_content else 0
            
            if user_content_words and kw_words and user_content_words == kw_words:
                score += 0.4
            elif user_content_words and kw_words and kw_words.issubset(user_content_words):
                score += 0.25
            
            matches.append({
                'index': i,
                'score': score,
                'common_words': list(common_content)
            })
    
    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches


def _search_single_question(question_text):
    """Search for a single question against the Q&A database."""
    if not preprocessed_questions:
        return None
    
    similarity_scores = calculate_similarity(question_text)
    if len(similarity_scores) == 0:
        return None
    
    best_match_idx = similarity_scores.argmax()
    best_score = similarity_scores[best_match_idx]
    matched_obj = questions_data[best_match_idx]
    
    tfidf_threshold = 0.35
    keyword_threshold = 0.25
    
    if best_score > tfidf_threshold:
        return {
            'matched_question': matched_obj['question'],
            'answer': matched_obj['answer'],
            'sno': matched_obj['sno'],
            'audio_url': matched_obj.get('audio_url', ''),
            'video_url': matched_obj.get('video_url', ''),
            'story_url': matched_obj.get('story_url', ''),
            'detail_url': matched_obj.get('detail_url', ''),
            'example_url': matched_obj.get('example_url', ''),
            'story_text': matched_obj.get('story_text', ''),
            'detail_text': matched_obj.get('detail_text', ''),
            'example_text': matched_obj.get('example_text', ''),
            'confidence_score': float(best_score),
            'matching_method': 'tfidf',
        }
    
    # Keyword fallback
    keyword_matches = keyword_match(question_text, questions_data)
    if keyword_matches and keyword_matches[0]['score'] > keyword_threshold:
        best_kw = keyword_matches[0]
        matched_obj = questions_data[best_kw['index']]
        return {
            'matched_question': matched_obj['question'],
            'answer': matched_obj['answer'],
            'sno': matched_obj['sno'],
            'audio_url': matched_obj.get('audio_url', ''),
            'video_url': matched_obj.get('video_url', ''),
            'story_url': matched_obj.get('story_url', ''),
            'detail_url': matched_obj.get('detail_url', ''),
            'example_url': matched_obj.get('example_url', ''),
            'story_text': matched_obj.get('story_text', ''),
            'detail_text': matched_obj.get('detail_text', ''),
            'example_text': matched_obj.get('example_text', ''),
            'confidence_score': float(best_kw['score']),
            'matching_method': 'keyword',
        }
    
    return None


# ============================================
# ROUTES
# ============================================

def _get_user_id() -> str:
    """Extract user_id from request, fallback to session-based id."""
    data = request.get_json(silent=True) or {}
    uid = data.get("user_id")
    if not uid:
        if "anon_uid" not in session:
            import uuid
            session["anon_uid"] = str(uuid.uuid4())
            session.modified = True
        uid = session["anon_uid"]
    return uid


@staticchat_bp.route('/search', methods=['POST'])
def search_question():
    try:
        data = request.get_json() or {}
        original_question = (data.get('question') or '').strip()
        user_id = _get_user_id()

        if not original_question:
            return jsonify({'success': False, 'message': 'Please provide a question'}), 400

        print(f"\n{'='*50}")
        print(f"📝 Original question: '{original_question}' (user_id: {user_id})")

        # --- Get generic context for this user ---
        ctx = get_context(user_id)
        print(f"📊 Current topic: {ctx.get('current_topic', 'None')}")
        print(f"📖 History: {len(ctx.get('conversation_history', []))} exchanges")

        # --- Enhance question with context ---
        enhanced_question = enhance_with_context(original_question, ctx)
        followup = is_follow_up(original_question, ctx)

        if enhanced_question != original_question:
            print(f"🔧 Enhanced to: '{enhanced_question}'")

        question_lower = original_question.lower().strip()
        
        # ============================================
        # INTELLIGENT SCENARIO DETECTION
        # ============================================

        # 1. Check for GIBBERISH / NOT UNDERSTANDABLE (IMPROVED)
        if _is_gibberish(original_question):
            print(f"🤷 Gibberish detected: '{original_question}'")
            scenario_response = _build_scenario_response("not_understandable")
            record_exchange(user_id, original_question, scenario_response['message'])
            return _build_json_response(scenario_response, original_question, ctx, followup)

        # 2. Check for RUDE LANGUAGE
        if _is_rude_language(question_lower):
            print(f"🚫 Rude language detected: '{original_question}'")
            scenario_response = _build_scenario_response("rude_language")
            record_exchange(user_id, original_question, scenario_response['message'])
            return _build_json_response(scenario_response, original_question, ctx, followup)

        # 3. Check for DISRUPTIVE BEHAVIOR
        if _is_disruptive_behavior(question_lower):
            print(f"😴 Disruptive behavior detected: '{original_question}'")
            scenario_response = _build_scenario_response("disruptive_behavior")
            record_exchange(user_id, original_question, scenario_response['message'])
            return _build_json_response(scenario_response, original_question, ctx, followup)

        # 4. Check for GREETING (IMPROVED - word boundary checking)
        greeting_detected, greeting_type = _is_greeting(question_lower)
        if greeting_detected:
            print(f"👋 Greeting detected ({greeting_type}): '{original_question}'")
            scenario_response = _build_greeting_response(greeting_type)
            record_exchange(user_id, original_question, scenario_response['message'])
            return _build_json_response(scenario_response, original_question, ctx, followup)

        # 5. Check for THANKS
        if _is_thanks(question_lower):
            print(f"🙏 Thanks detected: '{original_question}'")
            scenario_response = _build_scenario_response("thanks")
            record_exchange(user_id, original_question, scenario_response['message'])
            return _build_json_response(scenario_response, original_question, ctx, followup)

        # 6. Check for FAREWELL
        if _is_farewell(question_lower):
            print(f"👋 Farewell detected: '{original_question}'")
            scenario_response = _build_scenario_response("farewell")
            record_exchange(user_id, original_question, scenario_response['message'])
            return _build_json_response(scenario_response, original_question, ctx, followup)

        # 7. Check for MULTI-PART QUESTIONS
        parts = _split_multi_question(enhanced_question)
        if parts and len(parts) >= 2:
            print(f"📋 Multi-question detected: {parts}")
            
            combined_answers = []
            combined_questions = []
            first_match = None
            
            for part in parts:
                result = _search_single_question(part)
                if result:
                    if first_match is None:
                        first_match = result
                    combined_questions.append(result['matched_question'])
                    combined_answers.append(result['answer'])
                else:
                    combined_questions.append(part)
                    combined_answers.append(f"I don't have information about \"{part}\" right now.")
            
            combined_answer = "\n\n".join(
                f"📌 {combined_questions[i]}:\n{combined_answers[i]}"
                for i in range(len(combined_answers))
            )
            
            updated_ctx = record_exchange(
                user_id, original_question, combined_answer,
                matched_question=" & ".join(combined_questions)
            )
            
            return jsonify({
                'success': True,
                'matched_question': " & ".join(combined_questions),
                'answer': combined_answer,
                'sno': first_match['sno'] if first_match else 0,
                'audio_url': first_match.get('audio_url', '') if first_match else '',
                'video_url': first_match.get('video_url', '') if first_match else '',
                'story_url': first_match.get('story_url', '') if first_match else '',
                'detail_url': first_match.get('detail_url', '') if first_match else '',
                'example_url': first_match.get('example_url', '') if first_match else '',
                'story_text': first_match.get('story_text', '') if first_match else '',
                'detail_text': first_match.get('detail_text', '') if first_match else '',
                'example_text': first_match.get('example_text', '') if first_match else '',
                'confidence_score': first_match.get('confidence_score', 0) if first_match else 0,
                'user_question': original_question,
                'matching_method': 'multi_question',
                'conversation_context': {
                    'is_follow_up': followup,
                    'current_topic': updated_ctx.get('current_topic'),
                    'follow_up_count': updated_ctx.get('follow_up_count', 0),
                    'enhanced_question': enhanced_question if enhanced_question != original_question else None
                }
            })

        # 8. Check for OUT OF TOPIC
        if _is_out_of_syllabus(enhanced_question):
            print(f"📭 Out of topic detected: '{enhanced_question}'")
            scenario_response = _build_scenario_response("out_of_topic")
            record_exchange(user_id, original_question, scenario_response['message'])
            return _build_json_response(scenario_response, original_question, ctx, followup)

        # ============================================
        # Q&A MATCHING (TENSE-RELATED QUESTIONS)
        # ============================================

        # --- TF-IDF similarity matching ---
        if not preprocessed_questions:
            msg = SCENARIOS['not_available']['message']
            record_exchange(user_id, original_question, msg)
            scenario_response = _build_scenario_response("not_available")
            return _build_json_response(scenario_response, original_question, ctx, followup)

        similarity_scores = calculate_similarity(enhanced_question)

        if len(similarity_scores) == 0:
            msg = SCENARIOS['not_available']['message']
            record_exchange(user_id, original_question, msg)
            scenario_response = _build_scenario_response("not_available")
            return _build_json_response(scenario_response, original_question, ctx, followup)

        best_match_idx = similarity_scores.argmax()
        best_score = similarity_scores[best_match_idx]
        matched_obj = questions_data[best_match_idx]

        print(f"🎯 Best TF-IDF score: {best_score:.3f}")
        print(f"🔗 Matched to: {matched_obj['question']}")

        tfidf_threshold = 0.35
        keyword_threshold = 0.25

        # --- TF-IDF Match Found ---
        if best_score > tfidf_threshold:
            updated_ctx = record_exchange(
                user_id, original_question, matched_obj['answer'],
                matched_question=matched_obj['question']
            )

            return jsonify({
                'success': True,
                'matched_question': matched_obj['question'],
                'answer': matched_obj['answer'],
                'sno': matched_obj['sno'],
                'audio_url': matched_obj.get('audio_url', ''),
                'video_url': matched_obj.get('video_url', ''),
                'story_url': matched_obj.get('story_url', ''),
                'detail_url': matched_obj.get('detail_url', ''),
                'example_url': matched_obj.get('example_url', ''),
                'story_text': matched_obj.get('story_text', ''),
                'detail_text': matched_obj.get('detail_text', ''),
                'example_text': matched_obj.get('example_text', ''),
                'confidence_score': float(best_score),
                'user_question': original_question,
                'matching_method': 'tfidf',
                'spell_corrected': original_question if SYMSPELL_AVAILABLE else 'not_available',
                'conversation_context': {
                    'is_follow_up': followup,
                    'current_topic': updated_ctx.get('current_topic'),
                    'follow_up_count': updated_ctx.get('follow_up_count', 0),
                    'enhanced_question': enhanced_question if enhanced_question != original_question else None
                }
            })

        # --- Keyword matching fallback ---
        keyword_matches = keyword_match(enhanced_question, questions_data)

        print(f"🔑 Keyword matches found: {len(keyword_matches)}")
        if keyword_matches:
            print(f"🏆 Best keyword score: {keyword_matches[0]['score']:.3f}")

        if keyword_matches and keyword_matches[0]['score'] > keyword_threshold:
            best_kw = keyword_matches[0]
            matched_obj = questions_data[best_kw['index']]

            updated_ctx = record_exchange(
                user_id, original_question, matched_obj['answer'],
                matched_question=matched_obj['question']
            )

            return jsonify({
                'success': True,
                'matched_question': matched_obj['question'],
                'answer': matched_obj['answer'],
                'sno': matched_obj['sno'],
                'audio_url': matched_obj.get('audio_url', ''),
                'video_url': matched_obj.get('video_url', ''),
                'story_url': matched_obj.get('story_url', ''),
                'detail_url': matched_obj.get('detail_url', ''),
                'example_url': matched_obj.get('example_url', ''),
                'story_text': matched_obj.get('story_text', ''),
                'detail_text': matched_obj.get('detail_text', ''),
                'example_text': matched_obj.get('example_text', ''),
                'confidence_score': float(best_kw['score']),
                'user_question': original_question,
                'matching_method': 'keyword',
                'common_words': best_kw['common_words'],
                'conversation_context': {
                    'is_follow_up': followup,
                    'current_topic': updated_ctx.get('current_topic'),
                    'follow_up_count': updated_ctx.get('follow_up_count', 0),
                    'enhanced_question': enhanced_question if enhanced_question != original_question else None
                }
            })

        # --- NOT AVAILABLE ---
        print(f"❌ No match found for: '{enhanced_question}'")
        response_message = SCENARIOS['not_available']['message']

        current_topic = ctx.get('current_topic')
        if current_topic and followup:
            response_message = (
                f"I don't have more details on \"{current_topic}\" right now. "
                "Could you try asking in a different way?"
            )

        record_exchange(user_id, original_question, response_message)
        
        scenario_response = _build_scenario_response("not_available")
        return _build_json_response(scenario_response, original_question, ctx, followup)

    except Exception as e:
        print(f"❌ Error in search_question: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error processing request: {str(e)}'
        }), 500


# Keep all other routes the same as before...

@staticchat_bp.route('/clear-context', methods=['POST'])
def clear_context_route():
    """Clear conversation context (call this when page loads/refreshes)"""
    try:
        uid = _get_user_id()
        clear_context(uid)
        print("🧹 Context cleared")
        return jsonify({'success': True, 'message': 'Conversation context cleared'})
    except Exception as e:
        print(f"Error clearing context: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@staticchat_bp.route('/context/<user_id>/clear', methods=['POST'])
def clear_context_by_user(user_id):
    """Clear conversation context for a specific user (called by frontend)"""
    try:
        clear_context(user_id)
        print(f"🧹 Context cleared for user {user_id}")
        return jsonify({'success': True, 'message': 'Conversation context cleared'})
    except Exception as e:
        print(f"Error clearing context: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@staticchat_bp.route('/context-info', methods=['GET'])
def get_context_info():
    """Get current conversation context (for debugging)"""
    try:
        uid = _get_user_id()
        ctx = get_context(uid)
        return jsonify({'success': True, 'context': ctx})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@staticchat_bp.route('/context/<user_id>', methods=['GET'])
def get_context_by_user(user_id):
    """Get conversation context for a specific user (called by frontend)"""
    try:
        ctx = get_context(user_id)
        return jsonify({'success': True, 'context': ctx})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@staticchat_bp.route('/suggest-followup', methods=['GET'])
def suggest_followup():
    """Suggest follow-up questions based on current context keywords"""
    try:
        uid = _get_user_id()
        ctx = get_context(uid)
        current_topic = ctx.get('current_topic')
        keywords = ctx.get('current_keywords', [])

        if not current_topic and not keywords:
            return get_suggestions()

        topic_str = current_topic or " ".join(keywords[:3])
        follow_up_suggestions = [
            f"Give me examples of {topic_str}",
            f"Explain {topic_str} in more detail",
            f"What are the rules for {topic_str}?",
            f"How do I use {topic_str}?",
            f"What is the difference between types of {topic_str}?",
        ]

        return jsonify({
            'success': True,
            'suggestions': follow_up_suggestions[:5],
            'based_on_topic': current_topic,
            'is_follow_up': True
        })

    except Exception as e:
        print(f"Error in suggest_followup: {e}")
        return jsonify({'success': False, 'message': str(e), 'suggestions': []}), 500


@staticchat_bp.route('/context/suggestions/<user_id>', methods=['GET'])
def suggest_followup_by_user(user_id):
    """Context-aware follow-up suggestions for a specific user (called by frontend)"""
    try:
        ctx = get_context(user_id)
        current_topic = ctx.get('current_topic')
        keywords = ctx.get('current_keywords', [])

        if not current_topic and not keywords:
            return get_suggestions()

        topic_str = current_topic or " ".join(keywords[:3])
        follow_up_suggestions = [
            f"Give me examples of {topic_str}",
            f"Explain {topic_str} in more detail",
            f"What are the rules for {topic_str}?",
            f"How do I use {topic_str}?",
            f"What is the difference between types of {topic_str}?",
        ]

        return jsonify({
            'success': True,
            'suggestions': follow_up_suggestions[:5],
            'current_topic': current_topic,
        })

    except Exception as e:
        print(f"Error in suggest_followup_by_user: {e}")
        return jsonify({'success': False, 'message': str(e), 'suggestions': []}), 500


@staticchat_bp.route('/questions', methods=['GET'])
def get_all_questions():
    """Get all questions for reference"""
    try:
        questions = load_questions()
        question_list = [{'sno': q['sno'], 'question': q['question']} for q in questions]
        return jsonify({
            'success': True,
            'questions': question_list,
            'count': len(question_list)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@staticchat_bp.route('/question/<int:sno>', methods=['GET'])
def get_question_by_sno(sno):
    """Get specific question by serial number"""
    try:
        questions = load_questions()
        question = next((q for q in questions if q['sno'] == sno), None)
        
        if question:
            return jsonify({'success': True, 'question': question})
        else:
            return jsonify({'success': False, 'message': f'Question with SNO {sno} not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@staticchat_bp.route('/suggestions', methods=['GET'])
def get_suggestions():
    """Get random suggestions from the database"""
    try:
        if not questions_data:
            return jsonify({'success': False, 'message': "No questions available.", 'suggestions': []})
        
        count = request.args.get('count', default=5, type=int)
        
        import random
        random_questions = random.sample(questions_data, min(count, len(questions_data)))
        suggestions = [q['question'] for q in random_questions]
        
        return jsonify({
            'success': True,
            'suggestions': suggestions,
            'count': len(suggestions)
        })
    except Exception as e:
        print(f"Error in get_suggestions: {str(e)}")
        return jsonify({'success': False, 'message': str(e), 'suggestions': []}), 500


@staticchat_bp.route('/scenarios', methods=['GET'])
def get_scenarios():
    """Get information about available scenarios"""
    try:
        scenarios_info = {}
        for scenario_name, scenario_data in SCENARIOS.items():
            scenarios_info[scenario_name] = {
                "type": scenario_data.get("type", "scenario"),
                "has_audio": bool(scenario_data.get("audio_url")),
                "has_video": bool(scenario_data.get("video_url")),
                "keywords": scenario_data.get("keywords", [])
            }
        
        return jsonify({
            'success': True,
            'scenarios': scenarios_info,
            'count': len(scenarios_info)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@staticchat_bp.route('/transcribe', methods=['POST'])
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


# Initialize
print("="*60)
print("🤖 Intelligent Q&A Chatbot with Smart Scenario Detection")
print("="*60)
print("Features:")
print("• IMPROVED gibberish detection (catches random keyboard mashing)")
print("• Word boundary checking for greetings (fixes 'super mam' issue)")
print("• Vowel detection for gibberish (catches 'skjldflkjsdlf')")
print("• Separate handling for rude language vs disruptive behavior")
print("• Time-based greetings with different audio/video")
print("• Intelligent thanks/farewell detection with patterns")
print("• Multi-question splitting and combining answers")
print("• Out-of-topic detection for non-tense questions")
print("• Not-available for tense questions not in database")
print("="*60)