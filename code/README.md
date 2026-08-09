# Message Notification Router — Code Implementation

This directory contains the Python modules for processing WhatsApp messages and context for notification routing.

## Requirements & Environment

Ensure required dependencies are installed:
```bash
uv sync
# or using pip:
pip install pandas requests pillow groq google-genai python-dotenv
```

Set required environment variables in `.env`:
```env
LLM_API_KEY=
LLM_BASE_URL=
GROQ_API_KEY=
GEMINI_API_KEY=
```

---

## How to Run Steps 1–3

### Step 1: Test Data Store (`code/data_loader.py`)
Loads all 13 dataset CSV files once and builds indexed lookups:
```bash
python3 code/data_loader.py
```
Or in Python:
```python
from code.data_loader import DataStore
ds = DataStore()
print("Loaded users:", len(ds.users_by_id))
```

### Step 2: Media Processor (`code/media_processor.py`)
Pre-processes all image and voice note media using Vision and Audio LLMs and caches results in `code/cache/media_cache.json`:
```bash
python3 code/media_processor.py
```

### Step 3: Context Builder (`code/context_builder.py`)
Combines `DataStore` and `media_cache.json` to generate complete, precomputed message context objects for all 110 messages and saves them to `code/cache/contexts.json`:
```bash
python3 code/context_builder.py
```

### Step 4: Run the program (`code/main.py`)
Uses all of the content generated in the previous step and classifies. It also calculates the accuracy scores.
```bash
python3 code/main.py
```
---

## Output Verification

- `code/cache/media_cache.json`: Stores vision captions, flags, and audio transcripts.
- `code/cache/contexts.json`: Stores precomputed contexts for all 110 incoming messages.
