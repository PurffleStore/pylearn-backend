from flask import Blueprint, request, jsonify, current_app
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import string
import tempfile
from datetime import datetime

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

# NOTE: Blueprints do not have a config dict. MAX_CONTENT_LENGTH must be set on the Flask app.
# If you want to enforce max content size, set app.config["MAX_CONTENT_LENGTH"] when creating the Flask app.

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

# Enhanced Scenario configurations
SCENARIOS = {
    "greeting": {
        "keywords": ["good morning", "good afternoon", "good evening", "hello", "hi", "hey", "greetings"],
        "message": {
            "morning": "Good morning! Let's begin our lesson on tenses. You can ask me any question about tenses",
            "afternoon": "Good afternoon! Let's begin our lesson on tenses. You can ask me any question about tenses",
            "evening": "Good evening! Let's begin our lesson on tenses. You can ask me any question about tenses",
            "general": "Hello! Welcome to the English Tenses Learning Assistant. How can I help you with tenses today?"
        },
        "audio_url": "assets/staticchat/intro.mp3",
        "video_url": "assets/staticchat/intro.mp4",
        "story_url": "",
        "detail_url": "",
        "example_url": "",
        "type": "scenario"
    },
    "thanks": {
        "keywords": ["thank you", "thanks", "thank you very much", "appreciate it", "thanks a lot"],
        "message": "You're welcome! Do you have any other questions?",
        "audio_url": "assets/staticchat/you_are_welcome.mp3",
        "video_url": "assets/staticchat/you_are_welcome.mp4",
        "story_url": "",
        "detail_url": "",
        "example_url": "",
        "type": "scenario"
    },
    "farewell": {
        "keywords": ["bye", "goodbye", "see you", "farewell", "take care", "bye bye"],
        "message": "Goodbye! Keep practicing your English tenses. Remember, practice makes perfect!",
        "audio_url": "assets/staticchat/bye.mp3",
        "video_url": "assets/staticchat/bye.mp4",
        "story_url": "",
        "detail_url": "",
        "example_url": "",
        "type": "scenario"
    },
    "not_available": {
        "message": "I don't have the answer for that. Let's not available in my lesson today.",
        "suggestions": [
            "Try asking about common tenses like present simple or past perfect",
            "Ask me about tense structures or examples",
            "Check if your question is specifically about English verb tenses"
        ],
        "audio_url": "assets/staticchat/no_db.mp3",
        "video_url": "assets/staticchat/no_db.mp4",
        "story_url": "",
        "detail_url": "",
        "example_url": "",
        "type": "scenario"
    },
    "out_of_syllabus": {
        "keywords": [
            # sports
            "sports", "sport", "cricket", "ipl", "match", "score", "wicket", "runs", "bat", "bowling",
            "football", "basketball", "tennis", "hockey",
            # other non-tense topics
            "weather", "rain", "sunny", "temperature",
            "food", "pizza", "burger", "restaurant", "cooking",
            "movie", "music", "song", "artist", "film",
            "history", "science", "math", "politics", "geography", "economics", "physics",
            # general grammar (NOT tenses)
            "noun", "pronoun", "adjective", "adverb", "preposition", "conjunction",
            "punctuation", "comma", "full stop", "spelling", "vocabulary", "synonym", "antonym",
            "phonetics", "pronunciation"
        ],
        "message": "That's not part of our tense lesson. Let's stay on our topic.",
        "audio_url": "assets/staticchat/out_of_topic.mp3",
        "video_url": "assets/staticchat/out_of_topic.mp4",
        "story_url": "",
        "detail_url": "",
        "example_url": "",
        "type": "scenario"
    },
    "not_understandable": {
        "message": "I don't understand your question. Can you ask it again more simply?",
        "suggestions": [
            "Try using simpler words",
            "Ask about specific tenses like 'What is present tense?'",
            "Ask for examples of tenses",
            "Check your spelling and grammar"
        ],
        "audio_url": "assets/staticchat/not_understand.mp3",
        "video_url": "assets/staticchat/not_understand.mp4",
        "story_url": "",
        "detail_url": "",
        "example_url": "",
        "type": "scenario"
    }
}

# Load questions from JSON file
def load_questions():
    try:
        with open('assets/qa.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} questions from qa.json")
        
        # Debug: Print question categories
        tense_categories = {}
        for item in data:
            q = item['question'].lower()
            if 'present' in q:
                if 'continuous' in q or 'progressive' in q:
                    tense_categories['present_continuous'] = tense_categories.get('present_continuous', 0) + 1
                elif 'perfect' in q:
                    tense_categories['present_perfect'] = tense_categories.get('present_perfect', 0) + 1
                elif 'simple' in q:
                    tense_categories['present_simple'] = tense_categories.get('present_simple', 0) + 1
                else:
                    tense_categories['present_general'] = tense_categories.get('present_general', 0) + 1
        
        print(f"Tense categories in database: {tense_categories}")
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
        # Split into words and correct each
        words = text.split()
        corrected_words = []
        
        for word in words:
            if len(word) <= 2:  # Don't correct very short words
                corrected_words.append(word)
                continue
            
            # Check if word needs correction
            suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions and suggestions[0].term != word:
                corrected_words.append(suggestions[0].term)
                print(f"Corrected '{word}' to '{suggestions[0].term}'")
            else:
                corrected_words.append(word)
        
        corrected_text = ' '.join(corrected_words)
        
        # Also check for common bigram errors
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
    # Correct spelling first
    if SYMSPELL_AVAILABLE:
        text = correct_spelling(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    if NLTK_AVAILABLE:
        try:
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            # Keep important tense-related words that might be in stopwords
            important_words = {'am', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 
                              'do', 'does', 'did', 'will', 'shall', 'would', 'could', 'should'}
            stop_words = stop_words - important_words
            
            tokens = [word for word in tokens if word not in stop_words]
            
            # Lemmatize
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]  # Lemmatize as verbs
            
            return ' '.join(tokens)
        except Exception as e:
            print(f"Error in NLP processing: {e}")
            # Fallback to simple processing
            return text
    else:
        # Enhanced simple processing
        # Keep important tense-related words
        important_words = {'tense', 'tenses', 'present', 'past', 'future', 
                          'continuous', 'perfect', 'simple', 'progressive',
                          'am', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
                          'do', 'does', 'did', 'will', 'shall', 'would', 'could', 'should'}
        
        # Basic stopwords to remove
        basic_stopwords = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'by', 'for', 
                          'with', 'about', 'against', 'between', 'into', 'through',
                          'during', 'before', 'after', 'above', 'below', 'to', 'from',
                          'up', 'down', 'out', 'off', 'over', 'under', 'again', 
                          'further', 'then', 'once', 'here', 'there', 'when', 'where',
                          'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                          'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                          'own', 'same', 'so', 'than', 'too', 'very', 'can', 'may',
                          'might', 'must', 'ought', 'shall', 'should', 'will', 'would'}
        
        # Remove stopwords but keep important tense words
        words = text.split()
        filtered_words = []
        for word in words:
            if word in important_words:
                filtered_words.append(word)
            elif word not in basic_stopwords:
                filtered_words.append(word)
        
        return ' '.join(filtered_words)

def detect_scenario(user_question):
    """Detect if the user input matches any special scenario"""
    question_lower = user_question.lower().strip()
    
    # First, check for greetings, thanks, and farewell (these have highest priority)
    # Check for greetings
    for greeting_keyword in SCENARIOS["greeting"]["keywords"]:
        if greeting_keyword in question_lower:
            current_hour = datetime.now().hour
            if current_hour < 12:
                greeting_type = "morning"
            elif current_hour < 17:
                greeting_type = "afternoon"
            else:
                greeting_type = "evening"
            
            return {
                "scenario": "greeting",
                "message": SCENARIOS["greeting"]["message"][greeting_type],
                "audio_url": SCENARIOS["greeting"]["audio_url"],
                "video_url": SCENARIOS["greeting"]["video_url"],
                "story_url": SCENARIOS["greeting"].get("story_url", ""),
                "detail_url": SCENARIOS["greeting"].get("detail_url", ""),
                "example_url": SCENARIOS["greeting"].get("example_url", "")
            }
    
    # Check for thanks
    for thanks_keyword in SCENARIOS["thanks"]["keywords"]:
        if thanks_keyword in question_lower:
            return {
                "scenario": "thanks",
                "message": SCENARIOS["thanks"]["message"],
                "audio_url": SCENARIOS["thanks"]["audio_url"],
                "video_url": SCENARIOS["thanks"]["video_url"],
                "story_url": SCENARIOS["thanks"].get("story_url", ""),
                "detail_url": SCENARIOS["thanks"].get("detail_url", ""),
                "example_url": SCENARIOS["thanks"].get("example_url", "")
            }
    
    # Check for farewell
    for farewell_keyword in SCENARIOS["farewell"]["keywords"]:
        if farewell_keyword in question_lower:
            return {
                "scenario": "farewell",
                "message": SCENARIOS["farewell"]["message"],
                "audio_url": SCENARIOS["farewell"]["audio_url"],
                "video_url": SCENARIOS["farewell"]["video_url"],
                "story_url": SCENARIOS["farewell"].get("story_url", ""),
                "detail_url": SCENARIOS["farewell"].get("detail_url", ""),
                "example_url": SCENARIOS["farewell"].get("example_url", "")
            }
    
    # Check for out of syllabus topics
    # Only trigger if question contains out-of-syllabus keywords AND no tense keywords
    question_words = set(question_lower.split())
    out_of_syllabus_keywords = set(SCENARIOS["out_of_syllabus"]["keywords"])
    
    # Check if question contains any out-of-syllabus keyword
    contains_out_of_syllabus = any(keyword in question_lower for keyword in out_of_syllabus_keywords)
    
    if contains_out_of_syllabus:
        # Check if it also contains tense-related keywords
        tense_keywords = ['tense', 'tenses', 'present', 'past', 'future', 
                         'continuous', 'perfect', 'simple', 'progressive', 
                         'verb', 'verbs', 'grammar', 'am', 'is', 'are', 
                         'was', 'were', 'have', 'has', 'had']
        
        contains_tense_keyword = any(tense_word in question_lower for tense_word in tense_keywords)
        
        # If it contains both, check if tense keyword is more dominant
        if contains_tense_keyword:
            # Count tense words vs out-of-syllabus words
            tense_count = sum(1 for word in tense_keywords if word in question_lower)
            out_count = sum(1 for word in out_of_syllabus_keywords if word in question_lower)
            
            # If more tense-related words, treat as tense question
            if tense_count >= out_count:
                return None
        
        # If no tense keywords or fewer tense words, it's out of syllabus
        return {
            "scenario": "out_of_syllabus",
            "message": SCENARIOS["out_of_syllabus"]["message"],
            "audio_url": SCENARIOS["out_of_syllabus"]["audio_url"],
            "video_url": SCENARIOS["out_of_syllabus"]["video_url"],
            "story_url": SCENARIOS["out_of_syllabus"].get("story_url", ""),
            "detail_url": SCENARIOS["out_of_syllabus"].get("detail_url", ""),
            "example_url": SCENARIOS["out_of_syllabus"].get("example_url", "")
        }
    
    # Check for not understandable
    # Clean text for length check
    clean_text = re.sub(r'[^\w\s]', '', question_lower)
    
    if len(clean_text.strip()) < 2:
        return {
            "scenario": "not_understandable",
            "message": SCENARIOS["not_understandable"]["message"],
            "audio_url": SCENARIOS["not_understandable"]["audio_url"],
            "video_url": SCENARIOS["not_understandable"]["video_url"],
            "story_url": SCENARIOS["not_understandable"].get("story_url", ""),
            "detail_url": SCENARIOS["not_understandable"].get("detail_url", ""),
            "example_url": SCENARIOS["not_understandable"].get("example_url", "")
        }
    
    # Check for gibberish
    words = clean_text.split()
    if words:
        avg_word_len = sum(len(word) for word in words) / len(words)
        if avg_word_len > 15:  # Very long words might be gibberish
            return {
                "scenario": "not_understandable",
                "message": SCENARIOS["not_understandable"]["message"],
                "audio_url": SCENARIOS["not_understandable"]["audio_url"],
                "video_url": SCENARIOS["not_understandable"]["video_url"],
                "story_url": SCENARIOS["not_understandable"].get("story_url", ""),
                "detail_url": SCENARIOS["not_understandable"].get("detail_url", ""),
                "example_url": SCENARIOS["not_understandable"].get("example_url", "")
            }
    
    return None

def check_topic_relevance(user_question):
    """Return True only if the question is about English tenses (not general topics)."""
    q = user_question.lower().strip()

    # If the question clearly contains out-of-topic words AND does not say "tense",
    # treat it as out of syllabus.
    out_words = SCENARIOS["out_of_syllabus"].get("keywords", [])
    if any(re.search(rf"\b{re.escape(w)}\b", q) for w in out_words):
        if not re.search(r"\btense(s)?\b", q):
            return False

    # Strong tense intent words
    if re.search(r"\btense(s)?\b", q):
        return True

    # Common tense names (phrases)
    tense_phrases = [
        "present simple", "past simple", "future simple",
        "present continuous", "past continuous", "future continuous",
        "present perfect", "past perfect", "future perfect",
        "present perfect continuous", "past perfect continuous", "future perfect continuous",
    ]
    if any(p in q for p in tense_phrases):
        return True

    # If user mentions time-words + aspect-words together, likely a tense question
    time_words = ["present", "past", "future"]
    aspect_words = ["simple", "continuous", "perfect", "progressive"]
    if any(re.search(rf"\b{w}\b", q) for w in time_words) and any(re.search(rf"\b{w}\b", q) for w in aspect_words):
        return True

    # If user asks usage/rules/structure about helping verbs, allow it (still tense-related)
    helpers = ["am", "is", "are", "was", "were", "have", "has", "had", "do", "does", "did", "will", "shall", "would", "could", "should"]
    intent_words = ["use", "using", "when", "rule", "rules", "structure", "form", "difference", "between", "meaning", "example", "examples"]
    if any(re.search(rf"\b{h}\b", q) for h in helpers) and any(re.search(rf"\b{i}\b", q) for i in intent_words):
        return True

    # Otherwise, not a tense question
    return False

# Initialize questions data
questions_data = load_questions()
question_texts = [item['question'] for item in questions_data]
preprocessed_questions = [preprocess_text(q) for q in question_texts]

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Use unigrams and bigrams
if preprocessed_questions:  # Only fit if we have questions
    tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
else:
    tfidf_matrix = None

def calculate_similarity(user_question):
    """Calculate similarity between user question and stored questions"""
    if not preprocessed_questions:  # No questions loaded
        return np.array([])
    
    # Preprocess user question
    preprocessed_user_q = preprocess_text(user_question)
    
    # Vectorize user question
    user_vector = vectorizer.transform([preprocessed_user_q])
    
    # Calculate similarity scores
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)
    
    return similarity_scores[0]

def keyword_match(user_question, questions):
    """Fallback keyword matching - IMPROVED"""
    user_words = set(preprocess_text(user_question).split())
    matches = []
    
    for i, q_data in enumerate(questions):
        question_words = set(preprocess_text(q_data['question']).split())
        common_words = user_words.intersection(question_words)
        
        if common_words:
            # Calculate score based on common words and length
            score = len(common_words) / max(len(user_words), len(question_words))
            matches.append({
                'index': i,
                'score': score,
                'common_words': list(common_words)
            })
    
    # Sort by score
    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches

def verify_match_relevance(user_q, matched_q, matched_answer):
    """Verify if the match is actually relevant - IMPROVED VERSION"""
    user_q_lower = user_q.lower()
    matched_q_lower = matched_q.lower()
    matched_answer_lower = matched_answer.lower()

    # Extract key terms from user question
    user_terms = set(preprocess_text(user_q).split())

    # Extract key terms from matched question
    matched_terms = set(preprocess_text(matched_q).split())

    # Check for important keywords in user question
    important_keywords = ['difference', 'compare', 'between', 'versus', 'vs', 
                         'how to', 'how do i', 'explain', 'when to',
                         'conditional', 'subjunctive', 'passive', 'modal',
                         'reported speech', 'used to', 'mixed', 'perfect']

    # Group similar question starters
    question_starters = {
        'what': ['what is', 'what are', 'what does', 'what do'],
        'how': ['how to', 'how do', 'how does'],
        'when': ['when to', 'when do', 'when does'],
        'why': ['why do', 'why does', 'why is']
    }

    # Check if user and match have similar question starters
    user_starter = None
    matched_starter = None

    for starter_type, starters in question_starters.items():
        for starter in starters:
            if starter in user_q_lower:
                user_starter = starter_type
            if starter in matched_q_lower:
                matched_starter = starter_type

    # If both are asking "what" questions, it's likely a match even if wording differs
    if user_starter and matched_starter and user_starter == matched_starter:
        # Both are the same type of question (e.g., both "what" questions)
        print(f"Both are {user_starter} questions - accepting match")
        # Continue with other checks but don't reject just because wording differs

    # Check for important keywords that MUST be in the answer
    must_have_keywords = []
    for keyword in important_keywords:
        if keyword in user_q_lower:
            must_have_keywords.append(keyword)

    # If user asks for differences but answer doesn't compare, reject
    if 'difference' in user_q_lower or 'compare' in user_q_lower or 'versus' in user_q_lower:
        if not ('difference' in matched_answer_lower or 'compare' in matched_answer_lower or 'vs' in matched_answer_lower):
            print("User asked for differences but answer doesn't compare - rejecting")
            return False

    # If user asks "how to" but answer is just definition
    if ('how to' in user_q_lower or 'how do' in user_q_lower) and 'how' not in matched_answer_lower.lower():
        # Check if answer contains instructions/steps
        instruction_words = ['step', 'first', 'second', 'then', 'next', 'finally', 'process']
        if not any(word in matched_answer_lower for word in instruction_words):
            print("User asked 'how to' but answer is not instructional - rejecting")
            return False

    # Check if the match is just generic when user asks for specific
    generic_questions = ['what is', 'what are', 'what does', 'what do']
    specific_questions = ['difference between', 'how to use', 'when to use', 
                         'compare', 'explain the difference', 'give example of']

    user_is_specific = any(phrase in user_q_lower for phrase in specific_questions)
    match_is_generic = any(phrase in matched_q_lower for phrase in generic_questions)

    if user_is_specific and match_is_generic:
        # Check if the generic answer actually addresses the specific question
        user_specific_terms = []
        for phrase in specific_questions:
            if phrase in user_q_lower:
                # Get the terms after the phrase
                idx = user_q_lower.find(phrase) + len(phrase)
                user_specific_terms = user_q_lower[idx:].strip().split()[:3]
                break

        if user_specific_terms:
            # Check if these specific terms are in the answer
            if not any(term in matched_answer_lower for term in user_specific_terms if len(term) > 2):
                print("User asked specific, match is generic - likely wrong")
                return False

    # Check for core topic overlap
    user_words = set(user_q_lower.split())
    matched_words = set(matched_q_lower.split())
    common_core = user_words.intersection(matched_words)

    # Remove common stopwords
    stopwords_set = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    common_core = {word for word in common_core if word not in stopwords_set and len(word) > 2}

    if len(common_core) >= 2:  # At least 2 meaningful words in common
        print(f"Common core words: {common_core} - accepting match")
        return True

    # If TF-IDF score was high and we got here, it's probably OK
    return True

def verify_tense_specificity(user_q, matched_q, matched_answer):
    """Ensure we return the correct specificity for tense questions"""
    user_q_lower = user_q.lower()
    matched_q_lower = matched_q.lower()
    
    # Check if user is asking about general tense vs specific tense
    if 'present tense' in user_q_lower and ('continuous' not in user_q_lower and 'perfect' not in user_q_lower):
        # User is asking about present tense in general
        if 'present continuous' in matched_q_lower or 'present perfect' in matched_q_lower:
            # They got a specific tense instead of general
            # Check if we have a general present tense question
            for i, q_data in enumerate(questions_data):
                q_text = q_data['question'].lower()
                if 'present tense' in q_text and 'continuous' not in q_text and 'perfect' not in q_text:
                    return i  # Return index of general present tense
    
    elif 'past tense' in user_q_lower and ('continuous' not in user_q_lower and 'perfect' not in user_q_lower):
        if 'past continuous' in matched_q_lower or 'past perfect' in matched_q_lower:
            for i, q_data in enumerate(questions_data):
                q_text = q_data['question'].lower()
                if 'past tense' in q_text and 'continuous' not in q_text and 'perfect' not in q_text:
                    return i
    
    elif 'future tense' in user_q_lower and ('continuous' not in user_q_lower and 'perfect' not in user_q_lower):
        if 'future continuous' in matched_q_lower or 'future perfect' in matched_q_lower:
            for i, q_data in enumerate(questions_data):
                q_text = q_data['question'].lower()
                if 'future tense' in q_text and 'continuous' not in q_text and 'perfect' not in q_text:
                    return i
    
    return None  # No need to override

@staticchat_bp.route('/search', methods=['POST'])
def search_question():
    try:
        data = request.get_json()
        original_question = data.get('question', '').strip()
        
        if not original_question:
            return jsonify({
                'success': False,
                'message': 'Please provide a question'
            }), 400
        
        print(f"\n=== Processing: '{original_question}' ===")
        
        # First, check for special scenarios
        scenario_result = detect_scenario(original_question)
        if scenario_result:
            print(f"Detected scenario: {scenario_result['scenario']}")  # Debug log
            return jsonify({
                'success': True,
                'scenario': scenario_result['scenario'],
                'message': scenario_result['message'],
                'audio_url': scenario_result.get('audio_url', ''),
                'video_url': scenario_result.get('video_url', ''),
                'story_url': scenario_result.get('story_url', ''),
                'detail_url': scenario_result.get('detail_url', ''),
                'example_url': scenario_result.get('example_url', ''),
                'user_question': original_question,
                'matching_method': 'scenario'
            })
        
        print("No scenario detected, checking topic relevance...")  # Debug log
        
        # Check if question is related to tenses
        is_topic_relevant = check_topic_relevance(original_question)
        print(f"Topic relevant: {is_topic_relevant}")  # Debug log
        
        if not is_topic_relevant:
            # If not relevant and not caught by out_of_syllabus scenario
            return jsonify({
                'success': True,
                'scenario': 'out_of_syllabus',
                'message': SCENARIOS['out_of_syllabus']['message'],
                'audio_url': SCENARIOS['out_of_syllabus']['audio_url'],
                'video_url': SCENARIOS['out_of_syllabus']['video_url'],
                'story_url': SCENARIOS['out_of_syllabus'].get('story_url', ''),
                'detail_url': SCENARIOS['out_of_syllabus'].get('detail_url', ''),
                'example_url': SCENARIOS['out_of_syllabus'].get('example_url', ''),
                'user_question': original_question,
                'matching_method': 'scenario'
            })
        
        # Calculate similarity if we have questions
        if not preprocessed_questions:
            return jsonify({
                'success': True,
                'scenario': 'not_available',
                'message': SCENARIOS['not_available']['message'],
                'suggestions': SCENARIOS['not_available']['suggestions'],
                'audio_url': SCENARIOS['not_available']['audio_url'],
                'video_url': SCENARIOS['not_available']['video_url'],
                'story_url': SCENARIOS['not_available'].get('story_url', ''),
                'detail_url': SCENARIOS['not_available'].get('detail_url', ''),
                'example_url': SCENARIOS['not_available'].get('example_url', ''),
                'user_question': original_question,
                'matching_method': 'scenario'
            })
        
        similarity_scores = calculate_similarity(original_question)
        
        if len(similarity_scores) == 0:  # No questions loaded
            return jsonify({
                'success': True,
                'scenario': 'not_available',
                'message': SCENARIOS['not_available']['message'],
                'suggestions': SCENARIOS['not_available']['suggestions'],
                'audio_url': SCENARIOS['not_available']['audio_url'],
                'video_url': SCENARIOS['not_available']['video_url'],
                'story_url': SCENARIOS['not_available'].get('story_url', ''),
                'detail_url': SCENARIOS['not_available'].get('detail_url', ''),
                'example_url': SCENARIOS['not_available'].get('example_url', ''),
                'user_question': original_question,
                'matching_method': 'scenario'
            })
        
        # Get the best match
        best_match_idx = similarity_scores.argmax()
        best_score = similarity_scores[best_match_idx]
        
        print(f"Best TF-IDF score: {best_score:.3f}")  # Debug log
        print(f"Matched to question #{best_match_idx + 1}: {questions_data[best_match_idx]['question']}")  # Debug log
        
        # Check if we need to override for tense specificity
        override_idx = verify_tense_specificity(
            original_question,
            questions_data[best_match_idx]['question'],
            questions_data[best_match_idx]['answer']
        )
        
        if override_idx is not None:
            best_match_idx = override_idx
            best_score = 0.9  # Set high score for exact match
            print(f"Overriding to general tense question: {questions_data[best_match_idx]['question']}")
        
        # Set higher threshold for matching - INCREASED to prevent wrong matches
        tfidf_threshold = 0.35  # Increased from 0.2 to 0.35
        keyword_threshold = 0.25  # Increased from 0.1 to 0.25
        
        if best_score > tfidf_threshold:
            # Verify the match is actually relevant
            matched_question = questions_data[best_match_idx]
            is_relevant = verify_match_relevance(original_question, 
                                                matched_question['question'],
                                                matched_question['answer'])
            
            if is_relevant:
                # Good match found with TF-IDF
                return jsonify({
                    'success': True,
                    'matched_question': matched_question['question'],
                    'answer': matched_question['answer'],
                    'sno': matched_question['sno'],
                    'audio_url': matched_question.get('audio_url', ''),
                    'video_url': matched_question.get('video_url', ''),
                    'story_url': matched_question.get('story_url', ''),
                    'detail_url': matched_question.get('detail_url', ''),
                    'example_url': matched_question.get('example_url', ''),
                    'confidence_score': float(best_score),
                    'user_question': original_question,
                    'matching_method': 'tfidf',
                    'spell_corrected': original_question if SYMSPELL_AVAILABLE else 'not_available'
                })
            else:
                # Match is not actually relevant
                print(f"Match verification failed. Score: {best_score:.3f}")
                # Fall through to not_available
        else:
            # Score below threshold
            print(f"Score below threshold. Score: {best_score:.3f}, Threshold: {tfidf_threshold}")
        
        # Try keyword matching as fallback (with higher threshold)
        keyword_matches = keyword_match(original_question, questions_data)
        
        print(f"Keyword matches found: {len(keyword_matches)}")  # Debug log
        if keyword_matches:
            print(f"Best keyword score: {keyword_matches[0]['score']:.3f}")  # Debug log
        
        if keyword_matches and keyword_matches[0]['score'] > keyword_threshold:
            best_keyword_match = keyword_matches[0]
            matched_question = questions_data[best_keyword_match['index']]
            
            # Verify keyword match too
            is_relevant = verify_match_relevance(original_question,
                                                matched_question['question'],
                                                matched_question['answer'])
            
            if is_relevant:
                return jsonify({
                    'success': True,
                    'matched_question': matched_question['question'],
                    'answer': matched_question['answer'],
                    'sno': matched_question['sno'],
                    'audio_url': matched_question.get('audio_url', ''),
                    'video_url': matched_question.get('video_url', ''),
                    'story_url': matched_question.get('story_url', ''),
                    'detail_url': matched_question.get('detail_url', ''),
                    'example_url': matched_question.get('example_url', ''),
                    'confidence_score': float(best_keyword_match['score']),
                    'user_question': original_question,
                    'matching_method': 'keyword',
                    'common_words': best_keyword_match['common_words']
                })
            else:
                print("Keyword match verification failed")
        
        # No good match found but question is tense-related
        return jsonify({
            'success': True,
            'scenario': 'not_available',
            'message': SCENARIOS['not_available']['message'],
            'suggestions': SCENARIOS['not_available']['suggestions'],
            'audio_url': SCENARIOS['not_available']['audio_url'],
            'video_url': SCENARIOS['not_available']['video_url'],
            'story_url': SCENARIOS['not_available'].get('story_url', ''),
            'detail_url': SCENARIOS['not_available'].get('detail_url', ''),
            'example_url': SCENARIOS['not_available'].get('example_url', ''),
            'user_question': original_question,
            'matching_method': 'scenario',
            'debug_info': {
                'best_tfidf_score': float(best_score) if len(similarity_scores) > 0 else 0,
                'best_keyword_score': keyword_matches[0]['score'] if keyword_matches else 0
            }
        })
                
    except Exception as e:
        print(f"Error in search_question: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error processing request: {str(e)}'
        }), 500

@staticchat_bp.route('/questions', methods=['GET'])
def get_all_questions():
    """Get all questions for reference"""
    try:
        questions = load_questions()
        # Return only question text for autocomplete
        question_list = [{'sno': q['sno'], 'question': q['question']} for q in questions]
        return jsonify({
            'success': True,
            'questions': question_list,
            'count': len(question_list)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@staticchat_bp.route('/question/<int:sno>', methods=['GET'])
def get_question_by_sno(sno):
    """Get specific question by serial number"""
    try:
        questions = load_questions()
        question = next((q for q in questions if q['sno'] == sno), None)
        
        if question:
            return jsonify({
                'success': True,
                'question': question
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Question with SNO {sno} not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@staticchat_bp.route('/suggestions', methods=['GET'])
def get_suggestions():
    """Get random suggestions from the database"""
    try:
        if not questions_data:
            return jsonify({
                'success': False,
                'message': "No questions available.",
                'suggestions': []
            })
        
        # Get parameter for number of suggestions
        count = request.args.get('count', default=5, type=int)
        
        # Get random questions for suggestions
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
        return jsonify({
            'success': False,
            'message': str(e),
            'suggestions': []
        }), 500

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
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@staticchat_bp.route('/transcribe', methods=['POST'])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file field named 'file'"}), 400

    f = request.files["file"]
    if not f:
        return jsonify({"error": "No file uploaded"}), 400

    # Optional language from client: en / hi / ta
    language = request.form.get("language")  # may be None

    tmp_path = None
    try:
        # Keep a suffix so ffmpeg/whisper detects it better
        suffix = os.path.splitext(f.filename or "")[1].lower()
        if not suffix:
            suffix = ".webm"  # safe default for browser uploads

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            f.save(tmp_path)

        # Run local whisper
        result = model.transcribe(
            tmp_path,
            language=language if language else None,
            fp16=False  # CPU-only: must be False
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


