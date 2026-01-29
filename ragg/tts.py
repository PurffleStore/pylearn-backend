# tts_utils.py
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union
import uuid
from TTS.api import TTS

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
        raise FileNotFoundError(
            f"No reference audio files found. Checked: "
            f"{reference_files or []} and/or {reference_dir}"
        )

    if not hasattr(xtts_speak_to_file, "_model") or xtts_speak_to_file._model is None:
        xtts_speak_to_file._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    tts = xtts_speak_to_file._model

    out_path = Path(out_file) if out_file else Path(f"xtts_{uuid.uuid4().hex}.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        tts.tts_to_file(
            text=text,
            speaker_wav=speakers,
            language=language,
            file_path=str(out_path),
        )
    except Exception as e:
        raise RuntimeError(f"XTTS synthesis failed: {e}") from e

    return out_path
