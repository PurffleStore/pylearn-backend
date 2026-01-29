import os
import re
import torch
import tempfile
import subprocess
import soundfile as sf
import numpy as np
import base64
import random
import chromadb
import eng_to_ipa as ipa
from flask import Blueprint
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

pronunciation_bp = Blueprint("pronunciation", __name__)

# ==================================================
# 1. SETUP & CONFIG
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("BASE_DIR:", BASE_DIR)
VIDEO_PATH = os.path.join(BASE_DIR, "assets/feedback.mp4")
CHROMA_DIR = os.path.join(BASE_DIR, "assets/chroma_db")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "moxeeeem/wav2vec2-finetuned-pronunciation-correction"

print(f"Loading model to {DEVICE}...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

# ==================================================
# 2. CHROMA DB INITIALIZATION
# ==================================================
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("feedback")

# ==================================================
# 3. UK ENGLISH PRONUNCIATION SYSTEM
# ==================================================
UK_PHONEME_DB = {
    "ɪ": {"name": "KIT vowel", "example": "sit", "tip": "Short front vowel", "type": "vowel"},
    "iː": {"name": "FLEECE vowel", "example": "see", "tip": "Long front vowel", "type": "vowel"},
    "ʊ": {"name": "FOOT vowel", "example": "put", "tip": "Short rounded back vowel", "type": "vowel"},
    "uː": {"name": "GOOSE vowel", "example": "too", "tip": "Long rounded back vowel", "type": "vowel"},
    "e": {"name": "DRESS vowel", "example": "bed", "tip": "Short mid front vowel", "type": "vowel"},
    "ə": {"name": "SCHWA", "example": "about", "tip": "Relaxed central vowel", "type": "vowel"},
    "ɜː": {"name": "NURSE vowel", "example": "bird", "tip": "Long central vowel", "type": "vowel"},
    "ɔː": {"name": "THOUGHT vowel", "example": "law", "tip": "Long open-mid back vowel", "type": "vowel"},
    "æ": {"name": "TRAP vowel", "example": "cat", "tip": "Short open front vowel", "type": "vowel"},
    "ʌ": {"name": "STRUT vowel", "example": "cup", "tip": "Short mid back vowel", "type": "vowel"},
    "ɑː": {"name": "BATH vowel", "example": "father", "tip": "Long open back vowel", "type": "vowel"},
    "ɒ": {"name": "LOT vowel", "example": "hot", "tip": "Short open back rounded vowel", "type": "vowel"},
    "eɪ": {"name": "FACE diphthong", "example": "day", "tip": "Glide from e to ɪ", "type": "diphthong"},
    "aɪ": {"name": "PRICE diphthong", "example": "eye", "tip": "Glide from a to ɪ", "type": "diphthong"},
    "ɔɪ": {"name": "CHOICE diphthong", "example": "boy", "tip": "Glide from ɔ to ɪ", "type": "diphthong"},
    "aʊ": {"name": "MOUTH diphthong", "example": "now", "tip": "Glide from a to ʊ", "type": "diphthong"},
    "əʊ": {"name": "GOAT diphthong", "example": "go", "tip": "Glide from ə to ʊ", "type": "diphthong"},
    "p": {"name": "voiceless bilabial plosive", "example": "pen", "tip": "Explosive 'p' sound", "type": "consonant"},
    "b": {"name": "voiced bilabial plosive", "example": "bad", "tip": "Voiced 'b' with vibration", "type": "consonant"},
    "t": {"name": "voiceless alveolar plosive", "example": "tea", "tip": "Tongue tip on alveolar ridge", "type": "consonant"},
    "d": {"name": "voiced alveolar plosive", "example": "did", "tip": "Voiced 'd' with vibration", "type": "consonant"},
    "k": {"name": "voiceless velar plosive", "example": "cat", "tip": "Back of tongue on soft palate", "type": "consonant"},
    "ɡ": {"name": "voiced velar plosive", "example": "get", "tip": "Voiced 'g' with vibration", "type": "consonant"},
    "tʃ": {"name": "voiceless palato-alveolar affricate", "example": "chin", "tip": "Combination of 't' and 'ʃ'", "type": "consonant"},
    "dʒ": {"name": "voiced palato-alveolar affricate", "example": "jam", "tip": "Combination of 'd' and 'ʒ'", "type": "consonant"},
    "f": {"name": "voiceless labiodental fricative", "example": "fall", "tip": "Upper teeth on lower lip", "type": "consonant"},
    "v": {"name": "voiced labiodental fricative", "example": "van", "tip": "Voiced version of 'f'", "type": "consonant"},
    "θ": {"name": "voiceless dental fricative", "example": "thin", "tip": "Tongue between teeth, no vibration", "type": "consonant"},
    "ð": {"name": "voiced dental fricative", "example": "then", "tip": "Tongue between teeth, with vibration", "type": "consonant"},
    "s": {"name": "voiceless alveolar fricative", "example": "see", "tip": "Hissing 's' sound", "type": "consonant"},
    "z": {"name": "voiced alveolar fricative", "example": "zoo", "tip": "Voiced 'z' sound", "type": "consonant"},
    "ʃ": {"name": "voiceless palato-alveolar fricative", "example": "she", "tip": "'Sh' sound, tongue raised", "type": "consonant"},
    "ʒ": {"name": "voiced palato-alveolar fricative", "example": "pleasure", "tip": "Voiced 'zh' sound", "type": "consonant"},
    "h": {"name": "voiceless glottal fricative", "example": "hot", "tip": "Breathy 'h' from throat", "type": "consonant"},
    "m": {"name": "bilabial nasal", "example": "man", "tip": "Humming 'm' with lips closed", "type": "consonant"},
    "n": {"name": "alveolar nasal", "example": "no", "tip": "Tongue on alveolar ridge", "type": "consonant"},
    "ŋ": {"name": "velar nasal", "example": "sing", "tip": "'Ng' sound, back of tongue up", "type": "consonant"},
    "l": {"name": "alveolar lateral approximant", "example": "let", "tip": "Tongue tip on alveolar ridge", "type": "consonant"},
    "r": {"name": "alveolar approximant", "example": "red", "tip": "UK 'r' is soft", "type": "consonant"},
    "j": {"name": "palatal approximant", "example": "yes", "tip": "'Y' sound", "type": "consonant"},
    "w": {"name": "labio-velar approximant", "example": "we", "tip": "Round lips", "type": "consonant"},
}

def get_uk_pronunciation(word):
    word_lower = word.lower().strip()
    
    try:
        ipa_str = ipa.convert(word)
        clean_ipa = re.sub(r'[ˈˌː]', '', ipa_str)
        
        phonemes = []
        i = 0
        while i < len(clean_ipa):
            if i + 1 < len(clean_ipa):
                two_char = clean_ipa[i:i+2]
                if two_char in ['eɪ', 'aɪ', 'ɔɪ', 'aʊ', 'əʊ', 'tʃ', 'dʒ']:
                    phonemes.append(two_char)
                    i += 2
                    continue
            phonemes.append(clean_ipa[i])
            i += 1
        
        return phonemes
    except Exception:
        # Simple fallback for basic words
        phonemes = []
        for char in word_lower:
            if char in 'aeiou':
                vowel_map = {'a': 'æ', 'e': 'ɛ', 'i': 'ɪ', 'o': 'ɒ', 'u': 'ʌ'}
                phonemes.append(vowel_map.get(char, char))
            elif char == 'g':
                phonemes.append('ɡ')
            else:
                phonemes.append(char)
        return phonemes

def get_word_info(word):
    phonemes = get_uk_pronunciation(word)
    vowel_count = sum(1 for p in phonemes 
                     if UK_PHONEME_DB.get(p, {}).get('type') in ['vowel', 'diphthong'])
    
    if vowel_count == 1:
        stress = "only"
    elif vowel_count == 2:
        stress = "first"
    else:
        stress = "second"
    
    return {
        "syllables": vowel_count,
        "stress": stress
    }

# ==================================================
# 4. CORRECTED PHONEME ANALYSIS
# ==================================================
def is_exact_phoneme_match(ref, stu):
    if not stu:
        return False
    
    ref_norm = ref.replace('ː', '')
    stu_norm = stu.replace('ː', '')
    
    if ref_norm == stu_norm:
        return True
    
    uk_variations = {
        'ɒ': ['ɔ'], 'ɔ': ['ɒ'],
        'ɪ': ['i'], 'ɛ': ['e'],
        'ɡ': ['g'], 'æ': ['a'],
    }
    
    if ref_norm in uk_variations and stu_norm in uk_variations[ref_norm]:
        return 0.5
    
    return False

def analyze_pronunciation_strict(student_phonemes, reference_phonemes):
    if not student_phonemes:
        return {
            "score": 0,
            "errors": [],
            "exact_correct": 0,
            "partial_correct": 0,
            "total_expected": len(reference_phonemes) if reference_phonemes else 0,
            "accuracy_percentage": 0,
        }
    
    min_len = min(len(student_phonemes), len(reference_phonemes))
    exact_correct = 0
    partial_correct = 0
    errors = []
    
    for i in range(min_len):
        ref = reference_phonemes[i]
        stu = student_phonemes[i]
        match_result = is_exact_phoneme_match(ref, stu)
        
        if match_result == True:
            exact_correct += 1
        elif match_result == 0.5:
            partial_correct += 0.5
        else:
            errors.append({
                "position": i + 1,
                "expected": ref,
                "said": stu,
                "type": UK_PHONEME_DB.get(ref, {}).get("type", "unknown"),
            })
    
    total_expected = len(reference_phonemes) if reference_phonemes else 0
    if total_expected == 0:
        score = 0
    else:
        base_score = (exact_correct + partial_correct) / total_expected * 100
        
        if len(student_phonemes) < len(reference_phonemes):
            missing_penalty = (len(reference_phonemes) - len(student_phonemes)) / len(reference_phonemes) * 30
            base_score = max(0, base_score - missing_penalty)
        
        if len(student_phonemes) > len(reference_phonemes):
            extra_penalty = (len(student_phonemes) - len(reference_phonemes)) / len(reference_phonemes) * 20
            base_score = max(0, base_score - extra_penalty)
        
        score = round(max(0, min(100, base_score)), 1)
    
    accuracy_percentage = round((exact_correct + partial_correct) / total_expected * 100, 1) if total_expected > 0 else 0
    
    return {
        "score": score,
        "errors": errors,
        "exact_correct": exact_correct,
        "partial_correct": partial_correct,
        "total_expected": total_expected,
        "accuracy_percentage": accuracy_percentage,
    }

# ==================================================
# 5. SCENARIO DETECTION
# ==================================================
class ScenarioDetector:
    @staticmethod
    def detect_silence(student_phonemes, audio_error=None):
        if audio_error:
            error_lower = audio_error.lower()
            if any(x in error_lower for x in ['silence', 'quiet', 'empty']):
                return {
                    'scenario': 'silence',
                    'category': 'silence',
                    'confidence': 1.0,
                    'feedback': "I couldn't hear anything. Please speak louder.",
                    'action': "increase_volume"
                }
        
        if not student_phonemes or len(student_phonemes) == 0:
            return {
                'scenario': 'silence',
                'category': 'silence',
                'confidence': 0.9,
                'feedback': "No speech detected.",
                'action': "check_microphone"
            }
        
        return None
    
    @staticmethod
    def detect_multiple_words(student_phonemes, reference_phonemes):
        if not student_phonemes:
            return None
        
        if len(student_phonemes) > len(reference_phonemes) * 2:
            return {
                'scenario': 'multiple_words',
                'category': 'multiple_words',
                'confidence': 0.8,
                'feedback': "I heard multiple words. Please say only one word.",
                'action': "speak_single_word"
            }
        
        return None
    
    @staticmethod
    def detect_wrong_word(student_phonemes, reference_phonemes, word):
        if not student_phonemes or not reference_phonemes:
            return None
        
        min_len = min(len(student_phonemes), len(reference_phonemes))
        if min_len == 0:
            return None
        
        matches = 0
        for i in range(min_len):
            ref = reference_phonemes[i]
            stu = student_phonemes[i]
            if is_exact_phoneme_match(ref, stu):
                matches += 1
        
        similarity = matches / len(reference_phonemes) if len(reference_phonemes) > 0 else 0
        
        if similarity < 0.3:
            return {
                'scenario': 'wrong_word',
                'category': 'wrong_word',
                'confidence': 0.9,
                'feedback': f"That doesn't sound like '{word}'.",
                'action': "repeat_target_word"
            }
        
        return None
    
    @staticmethod
    def detect_syllable_issues(student_phonemes, reference_phonemes, word):
        if not student_phonemes or not reference_phonemes:
            return None
        
        word_info = get_word_info(word)
        ref_syllables = word_info["syllables"]
        
        stu_vowels = sum(1 for p in student_phonemes 
                        if UK_PHONEME_DB.get(p, {}).get('type') in ['vowel', 'diphthong'])
        
        if stu_vowels == 0 and len(student_phonemes) > 0:
            return {
                'scenario': 'syllable',
                'category': 'syllable',
                'confidence': 0.9,
                'feedback': f"Missing vowel sounds. '{word}' needs vowel pronunciation.",
                'action': "add_vowel_sounds"
            }
        
        if ref_syllables >= 2 and abs(stu_vowels - ref_syllables) >= 1:
            missing_count = len(reference_phonemes) - len(student_phonemes)
            if missing_count >= 2 and stu_vowels < ref_syllables:
                return {
                    'scenario': 'syllable',
                    'category': 'syllable',
                    'confidence': 0.8,
                    'feedback': f"'{word}' has {ref_syllables} syllable(s). You're missing a syllable.",
                    'action': "add_syllables"
                }
            elif stu_vowels > ref_syllables:
                return {
                    'scenario': 'syllable',
                    'category': 'syllable',
                    'confidence': 0.7,
                    'feedback': f"'{word}' has {ref_syllables} syllable(s). You added extra sounds.",
                    'action': "reduce_syllables"
                }
        
        return None
    
    @staticmethod
    def detect_ending_issues(student_phonemes, reference_phonemes):
        if not student_phonemes or not reference_phonemes:
            return None
        
        if len(student_phonemes) < len(reference_phonemes):
            missing_count = len(reference_phonemes) - len(student_phonemes)
            if missing_count == 1:
                missing_sound = reference_phonemes[-1]
                return {
                    'scenario': 'ending',
                    'category': 'ending',
                    'confidence': 0.8,
                    'feedback': f"You're missing the final sound: '{missing_sound}'.",
                    'action': "complete_ending",
                    'target_phoneme': missing_sound
                }
            elif missing_count > 1:
                missing_part = reference_phonemes[-missing_count:]
                missing_vowels = sum(1 for p in missing_part 
                                   if UK_PHONEME_DB.get(p, {}).get('type') in ['vowel', 'diphthong'])
                if missing_vowels == 0:
                    return {
                        'scenario': 'ending',
                        'category': 'ending',
                        'confidence': 0.7,
                        'feedback': f"You're missing the ending: '{''.join(missing_part)}'.",
                        'action': "complete_ending"
                    }
        
        if len(student_phonemes) >= 1 and len(reference_phonemes) >= 1:
            final_stu = student_phonemes[-1]
            final_ref = reference_phonemes[-1]
            
            if not is_exact_phoneme_match(final_ref, final_stu):
                return {
                    'scenario': 'ending',
                    'category': 'ending',
                    'confidence': 0.7,
                    'feedback': f"Final sound should be '{final_ref}' not '{final_stu}'.",
                    'action': "correct_final_sound",
                    'target_phoneme': final_ref
                }
        
        return None
    
    @staticmethod
    def detect_vowel_issues(student_phonemes, reference_phonemes):
        if not student_phonemes or not reference_phonemes:
            return None
        
        vowel_errors = []
        min_len = min(len(student_phonemes), len(reference_phonemes))
        
        for i in range(min_len):
            ref = reference_phonemes[i]
            stu = student_phonemes[i]
            
            ref_info = UK_PHONEME_DB.get(ref, {})
            if ref_info.get('type') in ['vowel', 'diphthong']:
                if not is_exact_phoneme_match(ref, stu):
                    vowel_errors.append({
                        'position': i + 1,
                        'expected': ref,
                        'actual': stu,
                        'tip': f"Use {ref} sound",
                    })
        
        if vowel_errors:
            primary = vowel_errors[0]
            return {
                'scenario': 'vowel',
                'category': 'vowel',
                'confidence': 0.9,
                'feedback': f"Vowel issue: {primary['tip']}",
                'action': "adjust_vowel",
                'target_phoneme': primary['expected']
            }
        
        return None
    
    @staticmethod
    def detect_consonant_issues(student_phonemes, reference_phonemes):
        if not student_phonemes or not reference_phonemes:
            return None
        
        consonant_errors = []
        min_len = min(len(student_phonemes), len(reference_phonemes))
        
        for i in range(min_len):
            ref = reference_phonemes[i]
            stu = student_phonemes[i]
            
            ref_info = UK_PHONEME_DB.get(ref, {})
            if ref_info.get('type') == 'consonant':
                if not is_exact_phoneme_match(ref, stu):
                    consonant_errors.append({
                        'position': i + 1,
                        'expected': ref,
                        'actual': stu,
                        'tip': ref_info.get('tip', f'Articulate {ref} clearly'),
                    })
        
        if consonant_errors:
            primary = consonant_errors[0]
            return {
                'scenario': 'consonant',
                'category': 'consonant',
                'confidence': 0.8,
                'feedback': f"Consonant: {primary['tip']}",
                'action': "articulate_consonant",
                'target_phoneme': primary['expected']
            }
        
        return None
    
    @staticmethod
    def detect_stress_issues(student_phonemes, reference_phonemes, word):
        if not student_phonemes or not reference_phonemes:
            return None
        
        word_info = get_word_info(word)
        if word_info["syllables"] < 2:
            return None
        
        correct_count = 0
        min_len = min(len(student_phonemes), len(reference_phonemes))
        for i in range(min_len):
            if is_exact_phoneme_match(reference_phonemes[i], student_phonemes[i]):
                correct_count += 1
        
        accuracy = correct_count / len(reference_phonemes) if len(reference_phonemes) > 0 else 0
        if accuracy >= 0.8 and word_info["syllables"] >= 2:
            stress_pattern = {
                "first": "first syllable",
                "second": "second syllable", 
                "third": "third syllable"
            }.get(word_info["stress"], "correct syllable")
            
            return {
                'scenario': 'stress',
                'category': 'stress',
                'confidence': 0.6,
                'feedback': f"For '{word}', emphasize the {stress_pattern}.",
                'action': "practice_stress"
            }
        
        return None
    
    @staticmethod
    def detect_success(analysis_result, score):
        if not analysis_result:
            return None
        
        if score >= 95:
            return {
                'scenario': 'success',
                'category': 'success',
                'confidence': 1.0,
                'feedback': "Excellent pronunciation! Perfect! 🎉",
                'action': "continue_excellent_work"
            }
        elif score >= 85:
            return {
                'scenario': 'success',
                'category': 'success',
                'confidence': 0.9,
                'feedback': "Very good pronunciation!",
                'action': "refine_pronunciation"
            }
        elif score >= 75:
            return {
                'scenario': 'success',
                'category': 'success',
                'confidence': 0.8,
                'feedback': "Good pronunciation! Keep practicing.",
                'action': "practice_more"
            }
        
        return None
    
    @classmethod
    def detect_scenarios(cls, student_phonemes, reference_phonemes, word, analysis_result, audio_error=None):
        score = analysis_result.get('score', 0) if analysis_result else 0
        
        detectors = [
            ('silence', lambda: cls.detect_silence(student_phonemes, audio_error)),
            ('multiple_words', lambda: cls.detect_multiple_words(student_phonemes, reference_phonemes)),
            ('wrong_word', lambda: cls.detect_wrong_word(student_phonemes, reference_phonemes, word)),
            ('syllable', lambda: cls.detect_syllable_issues(student_phonemes, reference_phonemes, word)),
            ('vowel', lambda: cls.detect_vowel_issues(student_phonemes, reference_phonemes)),
            ('consonant', lambda: cls.detect_consonant_issues(student_phonemes, reference_phonemes)),
            ('ending', lambda: cls.detect_ending_issues(student_phonemes, reference_phonemes)),
            ('stress', lambda: cls.detect_stress_issues(student_phonemes, reference_phonemes, word)),
            ('success', lambda: cls.detect_success(analysis_result, score)),
        ]
        
        for scenario_name, detector_func in detectors:
            result = detector_func()
            if result:
                if scenario_name == 'success' and score < 75:
                    continue
                return result
        
        return {
            'scenario': 'needs_improvement',
            'category': 'general',
            'confidence': 0.5,
            'feedback': "Pronunciation needs improvement.",
            'action': "practice_sounds"
        }

# ==================================================
# 6. VIDEO RAG BUILDER
# ==================================================
def build_feedback_video(category, feedback_message, target_phoneme=None):
    print(f"\n=== Building video for: {category} ===")
    print(f"Target phoneme: {target_phoneme}")

    if not target_phoneme:
        m = re.search(r"'([^']+)'", feedback_message)
        target_phoneme = m.group(1) if m else None

    selected_metadatas = []

    try:
        gen_results = collection.get(where={"category": category})
        if not gen_results or not gen_results.get("metadatas"):
            print(f"No clips found for category: {category}")
            return ""

        metadatas = gen_results["metadatas"]
        documents = gen_results.get("documents", [])

        items = []
        for idx, meta in enumerate(metadatas):
            text = documents[idx] if idx < len(documents) else ""
            items.append({"meta": meta, "text": text})

        # Split into:
        # - specific clips = has phoneme in metadata
        # - generic clips = no phoneme in metadata
        generic_clips = []
        specific_clips = []
        for it in items:
            meta = it["meta"]
            text = it["text"] or ""
            clip_phoneme = meta.get("phoneme")
            if clip_phoneme:
                specific_clips.append({"meta": meta, "phoneme": clip_phoneme, "text": text})
            else:
                meta_copy = dict(meta)
                meta_copy["_text"] = text
                generic_clips.append(meta_copy)

        print(f"Found {len(generic_clips)} generic clips, {len(specific_clips)} specific clips")

        def _seg_key(m):
            return f"{m.get('start')}_{m.get('end')}"

        def pick_generic(exclude_keys=None):
            exclude_keys = exclude_keys or set()
            pool = [m for m in generic_clips if _seg_key(m) not in exclude_keys]
            if pool:
                return random.choice(pool)
            return None

        def pick_specific_for_phoneme(target, related_map=None, exclude_keys=None):
            exclude_keys = exclude_keys or set()
            related_map = related_map or {}

            # 1) exact match
            if target:
                for it in specific_clips:
                    if it["phoneme"] == target and _seg_key(it["meta"]) not in exclude_keys:
                        return it["meta"]

            # 2) related (mainly for vowels)
            if target and target in related_map:
                for rel in related_map[target]:
                    for it in specific_clips:
                        if it["phoneme"] == rel and _seg_key(it["meta"]) not in exclude_keys:
                            return it["meta"]

            # 3) fallback any specific
            pool = [it["meta"] for it in specific_clips if _seg_key(it["meta"]) not in exclude_keys]
            if pool:
                return random.choice(pool)

            return None

        # -------------------------
        # REQUIRED CHANGE:
        # For vowel/consonant:
        # Always try to return TWO clips in this order:
        # 1) specific phoneme clip (target phoneme)
        # 2) general clip (generic feedback of vowel/consonant)
        # -------------------------
        if category in ["vowel", "consonant"]:
            exclude = set()

            vowel_groups = {
                "ɪ": ["iː", "i"], "iː": ["ɪ", "i"],
                "æ": ["a", "ɑː"], "ɑː": ["æ", "a"],
                "ʊ": ["uː", "u"], "uː": ["ʊ", "u"],
                "ɒ": ["ɔ", "ɔː"], "ɔː": ["ɒ", "ɔ"],
            }

            related_map = vowel_groups if category == "vowel" else {}

            # 1) Pick SPECIFIC (phoneme)
            specific_meta = pick_specific_for_phoneme(target_phoneme, related_map=related_map, exclude_keys=exclude)
            if specific_meta:
                selected_metadatas.append(specific_meta)
                exclude.add(_seg_key(specific_meta))
                print(f"✓ Selected specific {category} clip for phoneme: {target_phoneme}")

            # 2) Pick GENERAL (generic)
            generic_meta = pick_generic(exclude_keys=exclude)
            if generic_meta:
                selected_metadatas.append(generic_meta)
                exclude.add(_seg_key(generic_meta))
                print("✓ Selected general (generic) clip")

            # If still not 2 clips, try to fill with another different clip (best effort)
            if len(selected_metadatas) < 2:
                # try another generic first
                extra_generic = pick_generic(exclude_keys=exclude)
                if extra_generic:
                    selected_metadatas.append(extra_generic)
                    exclude.add(_seg_key(extra_generic))
                    print("✓ Filled missing slot with another generic clip")

            if len(selected_metadatas) < 2:
                # try another specific as last fallback
                extra_specific = pick_specific_for_phoneme(None, related_map=None, exclude_keys=exclude)
                if extra_specific:
                    selected_metadatas.append(extra_specific)
                    exclude.add(_seg_key(extra_specific))
                    print("✓ Filled missing slot with another specific clip")

            # If we still cannot make 2 clips, we proceed with whatever we have.
            # (Because the DB may not have enough clips.)
            if not selected_metadatas:
                print("✗ No clips selected for vowel/consonant.")
                return ""

        # -------------------------
        # Existing logic for other categories (unchanged)
        # -------------------------
        else:
            if category == "success":
                praise_keywords = ["good", "great", "perfect", "excellent", "well done", "nice", "clear"]
                next_keywords = ["next", "move"]

                praise_pool = [m for m in generic_clips if any(k in m.get("_text", "").lower() for k in praise_keywords)]
                next_pool = [m for m in generic_clips if any(k in m.get("_text", "").lower() for k in next_keywords)]

                print(f"Success classification: praise={len(praise_pool)} next={len(next_pool)}")

                first_clip = random.choice(praise_pool) if praise_pool else (random.choice(generic_clips) if generic_clips else None)

                if next_pool:
                    next_candidates = [m for m in next_pool if _seg_key(m) != _seg_key(first_clip)] if first_clip else next_pool
                    second_clip = random.choice(next_candidates) if next_candidates else None
                else:
                    alt_candidates = [m for m in generic_clips if _seg_key(m) != _seg_key(first_clip)] if first_clip else generic_clips
                    second_clip = random.choice(alt_candidates) if len(alt_candidates) > 0 else None

                selected_metadatas.clear()
                if first_clip:
                    selected_metadatas.append(first_clip)
                if second_clip:
                    selected_metadatas.append(second_clip)

            else:
                selection_strategy = "balanced"
                if category in ["syllable", "ending", "stress"]:
                    selection_strategy = "general_focus"

                print(f"Using selection strategy: {selection_strategy}")

                if selection_strategy == "general_focus":
                    if generic_clips:
                        selected_generic = random.sample(generic_clips, min(2, len(generic_clips)))
                        selected_metadatas.extend(selected_generic)
                else:
                    if generic_clips:
                        selected_metadatas.append(random.choice(generic_clips))

                # ensure at least 2 clips when possible
                if len(selected_metadatas) < 2 and generic_clips:
                    remaining = [c for c in generic_clips if _seg_key(c) not in {_seg_key(x) for x in selected_metadatas}]
                    if remaining:
                        selected_metadatas.append(random.choice(remaining))

        # Deduplicate (safety)
        unique_metadatas = []
        seen = set()
        for meta in selected_metadatas:
            key = _seg_key(meta)
            if key not in seen:
                seen.add(key)
                unique_metadatas.append(meta)
        selected_metadatas = unique_metadatas

        if len(selected_metadatas) == 0:
            print("No clips selected after filtering.")
            return ""

        print(f"Selected {len(selected_metadatas)} video clips:")

        # --- FFmpeg Processing ---
        if not os.path.exists(VIDEO_PATH):
            print(f"Video file not found: {VIDEO_PATH}")
            return ""

        clips = []
        concat_file = None
        final_video_path = None

        try:
            for i, seg in enumerate(selected_metadatas):
                tmp_clip = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{i}.mp4")
                tmp_clip.close()

                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-ss", str(seg["start"]), "-to", str(seg["end"]),
                        "-i", VIDEO_PATH,
                        "-c:v", "libx264", "-preset", "ultrafast",
                        "-crf", "28", "-c:a", "aac",
                        tmp_clip.name
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                clips.append(tmp_clip.name)

            concat_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w")
            for clip_path in clips:
                concat_file.write(f"file '{os.path.abspath(clip_path)}'\n")
            concat_file.close()

            final_video_path = tempfile.NamedTemporaryFile(delete=False, suffix="_final.mp4")
            final_video_path.close()

            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0", "-i", concat_file.name,
                    "-c", "copy", final_video_path.name
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            with open(final_video_path.name, "rb") as f:
                v_data = base64.b64encode(f.read()).decode()

            print(f"✓ Successfully merged {len(clips)} video clips")
            return v_data

        except Exception as e:
            print(f"✗ Video concatenation error: {e}")
            return ""

        finally:
            if concat_file and os.path.exists(concat_file.name):
                os.remove(concat_file.name)

            if final_video_path and os.path.exists(final_video_path.name):
                os.remove(final_video_path.name)

            for c in clips:
                if os.path.exists(c):
                    os.remove(c)

    except Exception as e:
        print(f"✗ Video generation error: {e}")
        return ""

# ==================================================
# 7. AUDIO PROCESSING
# ==================================================
def process_audio_file(audio_path):
    try:
        wav_path = audio_path.replace('.webm', '.wav')
        
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le",
            wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        speech, sample_rate = sf.read(wav_path)
        
        if len(speech) == 0:
            return None, "empty_audio"
        
        rms = np.sqrt(np.mean(speech**2))
        peak = np.max(np.abs(speech))
        
        if rms < 0.001 or peak < 0.02:
            return None, f"silent_rms_{rms:.6f}_peak_{peak:.4f}"
        
        if peak < 0.5:
            boost_factor = 0.5 / peak if peak > 0 else 1.0
            speech = speech * min(boost_factor, 3.0)
        
        inputs = processor(speech, sampling_rate=sample_rate,
                          return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = model(inputs.input_values.to(DEVICE)).logits
        
        pred_ids = torch.argmax(logits, dim=-1)
        raw_transcription = processor.batch_decode(pred_ids)[0]
        
        phonemes = [p for p in raw_transcription.replace(" ", "") if p.strip()]
        
        print(f"Extracted phonemes: {phonemes}")
        return phonemes, None
        
    except Exception as e:
        print(f"Audio processing error: {str(e)}")
        return None, f"error: {str(e)}"

# ==================================================
# 8. MAIN ENDPOINT
# ==================================================
@pronunciation_bp.route("/score", methods=["POST"])
def train_pronunciation():
    try:
        from flask import request, jsonify
        
        word = request.form.get('word', '').strip().lower()
        if not word:
            return jsonify({
                "success": False,
                "error": "No word provided",
                "scenario": "input_error"
            }), 400
        
        if 'audio' not in request.files:
            return jsonify({
                "success": False,
                "error": "No audio file",
                "scenario": "input_error"
            }), 400
        
        audio_file = request.files['audio']
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            audio_file.save(tmp_file.name)
            temp_path = tmp_file.name
        
        print(f"\n=== Processing: '{word}' ===")
        
        try:
            student_phonemes, audio_error = process_audio_file(temp_path)
            reference_phonemes = get_uk_pronunciation(word)
            analysis = analyze_pronunciation_strict(student_phonemes, reference_phonemes)
            score = analysis["score"]
            
            scenario_info = ScenarioDetector.detect_scenarios(
                student_phonemes=student_phonemes,
                reference_phonemes=reference_phonemes,
                word=word,
                analysis_result=analysis,
                audio_error=audio_error
            )
            
            scenario = scenario_info['scenario']
            category = scenario_info.get('category', scenario)
            feedback = scenario_info['feedback']
            action = scenario_info.get('action', '')
            target_phoneme = scenario_info.get('target_phoneme')
            
            print(f"Generating video for category: {category}")
            video_blob = build_feedback_video(category, feedback, target_phoneme)
            
            response = {
                "success": True,
                "scenario": scenario,
                "score": score,
                "is_acceptable": score >= 75,
                "word": word,
                "student_phonemes": student_phonemes if student_phonemes else [],
                "reference_phonemes": reference_phonemes,
                "ipa_notation": "/" + "".join(reference_phonemes) + "/",
                "feedback": feedback,
                "action_suggestion": action,
                "videoBlobBase64": video_blob if video_blob else "",
                "video_clips_merged": True if video_blob else False,
                "analysis": {
                    "accuracy": f"{analysis.get('exact_correct', 0)}/{analysis.get('total_expected', 0)} exact matches",
                    "accuracy_percentage": analysis.get('accuracy_percentage', 0),
                }
            }
            
            return jsonify(response)
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                wav_path = temp_path.replace('.webm', '.wav')
                if os.path.exists(wav_path):
                    os.remove(wav_path)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "scenario": "system_error"
        }), 500


