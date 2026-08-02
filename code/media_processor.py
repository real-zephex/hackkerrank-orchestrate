import os
import json
import logging
from typing import Dict, Any
from PIL import Image

try:
    from google import genai
except ImportError:
    genai = None

try:
    from groq import Groq
except ImportError:
    Groq = None

from data_loader import DataStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "media_cache.json")

def process_image(image_path: str, gemini_client: Any) -> Dict[str, Any]:
    if not os.path.exists(image_path):
        logging.error(f"Image file not found: {image_path}")
        return {
            "caption": "[unavailable]",
            "flags": {"is_payment_qr_otp": False, "is_promotional": False, "is_scam_screenshot": False, "is_ordinary": True}
        }

    prompt = (
        "Analyze this image and provide a JSON response with the following format:\n"
        "{\n"
        '  "caption": "A short factual description (1-2 sentences)",\n'
        '  "flags": {\n'
        '    "is_payment_qr_otp": true/false,\n'
        '    "is_promotional": true/false,\n'
        '    "is_scam_screenshot": true/false,\n'
        '    "is_ordinary": true/false\n'
        "  }\n"
        "}\n"
        "Do not include any commentary or markdown outside the raw JSON object."
    )

    try:
        img = Image.open(image_path)
        if gemini_client:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[img, prompt]
            )
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()
            
            data = json.loads(raw_text)
            caption = data.get("caption", "[unavailable]")
            flags = data.get("flags", {})
            return {
                "caption": caption,
                "flags": {
                    "is_payment_qr_otp": bool(flags.get("is_payment_qr_otp", False)),
                    "is_promotional": bool(flags.get("is_promotional", False)),
                    "is_scam_screenshot": bool(flags.get("is_scam_screenshot", False)),
                    "is_ordinary": bool(flags.get("is_ordinary", True))
                }
            }
    except Exception as e:
        logging.error(f"Failed to process image {image_path}: {e}")

    return {
        "caption": "[unavailable]",
        "flags": {"is_payment_qr_otp": False, "is_promotional": False, "is_scam_screenshot": False, "is_ordinary": True}
    }

def process_voice_note(voice_path: str, groq_client: Any) -> Dict[str, Any]:
    if not os.path.exists(voice_path):
        logging.error(f"Voice note file not found: {voice_path}")
        return {"transcript": "[unavailable]"}

    try:
        if groq_client:
            with open(voice_path, "rb") as f:
                res = groq_client.audio.transcriptions.create(
                    file=(os.path.basename(voice_path), f.read()),
                    model="whisper-large-v3-turbo",
                    response_format="text"
                )
            transcript = str(res).strip()
            return {"transcript": transcript}
    except Exception as e:
        logging.error(f"Failed to transcribe voice note {voice_path}: {e}")

    return {"transcript": "[unavailable]"}

def run_media_processor(ds: Optional[DataStore] = None) -> Dict[str, Any]:
    if ds is None:
        ds = DataStore()

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception as e:
            logging.warning(f"Could not load cache file: {e}")

    if "images" not in cache:
        cache["images"] = {}
    if "voice_notes" not in cache:
        cache["voice_notes"] = {}

    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")

    gemini_client = genai.Client(api_key=gemini_key) if (genai and gemini_key) else None
    groq_client = Groq(api_key=groq_key) if (Groq and groq_key) else None

    # Collect image IDs referenced in messages
    referenced_images = set(
        m["media_id"] for m in ds.messages if m["media_type"] == "image" and m["media_id"]
    )
    # Collect voice IDs referenced in messages
    referenced_voices = set(
        m["media_id"] for m in ds.messages if m["media_type"] == "voice" and m["media_id"]
    )

    logging.info(f"Processing {len(referenced_images)} images and {len(referenced_voices)} voice notes...")

    for img_id in sorted(referenced_images):
        if img_id in cache["images"] and cache["images"][img_id].get("caption") != "[unavailable]":
            logging.info(f"Image {img_id} already cached.")
            continue
        path = ds.image_path.get(img_id)
        if not path:
            logging.error(f"Image ID {img_id} not found in DataStore.")
            cache["images"][img_id] = {
                "caption": "[unavailable]",
                "flags": {"is_payment_qr_otp": False, "is_promotional": False, "is_scam_screenshot": False, "is_ordinary": True}
            }
            continue
        logging.info(f"Processing image {img_id} ({path})...")
        res = process_image(path, gemini_client)
        cache["images"][img_id] = res

    for voice_id in sorted(referenced_voices):
        if voice_id in cache["voice_notes"] and cache["voice_notes"][voice_id].get("transcript") != "[unavailable]":
            logging.info(f"Voice note {voice_id} already cached.")
            continue
        path = ds.voice_path.get(voice_id)
        if not path:
            logging.error(f"Voice note ID {voice_id} not found in DataStore.")
            cache["voice_notes"][voice_id] = {"transcript": "[unavailable]"}
            continue
        logging.info(f"Transcribing voice note {voice_id} ({path})...")
        res = process_voice_note(path, groq_client)
        cache["voice_notes"][voice_id] = res

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    logging.info(f"Saved media cache to {CACHE_FILE}")
    return cache

if __name__ == "__main__":
    run_media_processor()
