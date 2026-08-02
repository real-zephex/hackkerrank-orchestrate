import os
import csv
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

from data_loader import DataStore, parse_datetime
from media_processor import run_media_processor
from context_builder import build_all_contexts, build_context
from classifier import classify

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
PRED_CACHE_FILE = os.path.join(CACHE_DIR, "predictions_cache.json")

def run_pipeline(dataset_dir: str = "dataset") -> str:
    logging.info("Starting Message Notification Router pipeline...")
    
    # Step 1: Load DataStore
    ds = DataStore(data_dir=dataset_dir)
    logging.info(f"Loaded {len(ds.messages)} messages to predict.")

    # Step 2: Media pre-processing
    media_cache = run_media_processor(ds)

    # Step 3: Context building
    contexts = build_all_contexts(ds, media_cache_path=os.path.join(CACHE_DIR, "media_cache.json"))
    logging.info(f"Built contexts for {len(contexts)} messages.")

    # Step 4: Parallel classification with caching
    pred_cache = {}
    if os.path.exists(PRED_CACHE_FILE):
        try:
            with open(PRED_CACHE_FILE, "r", encoding="utf-8") as f:
                pred_cache = json.load(f)
        except Exception as e:
            logging.warning(f"Could not load prediction cache: {e}")

    missing_contexts = [
        ctx for ctx in contexts if ctx["message"]["message_id"] not in pred_cache
    ]

    if missing_contexts:
        logging.info(f"Classifying {len(missing_contexts)} missing messages in parallel (10 workers)...")
        def _task(ctx):
            msg_id = ctx["message"]["message_id"]
            res = classify(ctx)
            return msg_id, res

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_msg = {executor.submit(_task, ctx): ctx for ctx in missing_contexts}
            completed_count = 0
            for future in as_completed(future_to_msg):
                completed_count += 1
                try:
                    msg_id, res = future.result()
                    pred_cache[msg_id] = res
                    if completed_count % 10 == 0 or completed_count == len(missing_contexts):
                        logging.info(f"Progress: {completed_count}/{len(missing_contexts)} classified.")
                except Exception as e:
                    ctx = future_to_msg[future]
                    logging.error(f"Error classifying {ctx['message']['message_id']}: {e}")

        with open(PRED_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(pred_cache, f, indent=2)

    # Step 5: Write predictions to dataset/output.csv
    output_path = os.path.join(ds.data_dir, "output.csv")
    fieldnames = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]

    rows = []
    for msg in ds.messages:
        msg_id = msg["message_id"]
        pred = pred_cache.get(msg_id, {
            "action": "digest",
            "message_type": "unknown",
            "reason": "Default prediction.",
            "confidence": 0.50,
            "evidence_message_ids": "none"
        })
        rows.append({
            "message_id": msg_id,
            "action": pred["action"],
            "message_type": pred["message_type"],
            "reason": pred["reason"],
            "confidence": f"{float(pred['confidence']):.2f}",
            "evidence_message_ids": pred["evidence_message_ids"]
        })

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logging.info(f"Pipeline complete! Wrote {len(rows)} predictions to {output_path}")

    # Evaluate on sample_messages.csv if available
    evaluate_on_sample(ds, media_cache)

    return output_path

def evaluate_on_sample(ds: DataStore, media_cache: Dict[str, Any]):
    sample_path = os.path.join(ds.data_dir, "sample_messages.csv")
    if not os.path.exists(sample_path):
        return

    with open(sample_path, "r", encoding="utf-8") as f:
        samples = list(csv.DictReader(f))

    if not samples:
        return

    # Register sample messages into ds.messages_by_id temporarily for context building
    for s in samples:
        mid = s["message_id"]
        if mid not in ds.messages_by_id:
            ds.messages_by_id[mid] = {
                "message_id": mid,
                "user_id": s["user_id"].strip(),
                "conversation_type": s["conversation_type"].strip(),
                "group_id": s.get("group_id", "").strip() or None,
                "business_id": s.get("business_id", "").strip() or None,
                "sender_user_id": s.get("sender_user_id", "").strip() or None,
                "created_at": parse_datetime(s.get("created_at")),
                "message_text": s.get("message_text", ""),
                "media_type": s.get("media_type", ""),
                "media_id": s.get("media_id", "") or None,
                "forwarded_count": int(s.get("forwarded_count", 0) or 0)
            }

    sample_cache_file = os.path.join(CACHE_DIR, "sample_predictions_cache.json")
    sample_preds = {}
    if os.path.exists(sample_cache_file):
        try:
            with open(sample_cache_file, "r", encoding="utf-8") as f:
                sample_preds = json.load(f)
        except Exception:
            pass

    missing_samples = [s for s in samples if s["message_id"] not in sample_preds]
    if missing_samples:
        def _eval(s):
            ctx = build_context(s["message_id"], ds, media_cache)
            res = classify(ctx)
            return s["message_id"], res

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_eval, s) for s in missing_samples]
            for future in as_completed(futures):
                try:
                    mid, res = future.result()
                    sample_preds[mid] = res
                except Exception as e:
                    logging.error(f"Error evaluating sample message: {e}")

        with open(sample_cache_file, "w", encoding="utf-8") as f:
            json.dump(sample_preds, f, indent=2)

    action_correct = 0
    type_correct = 0
    evidence_correct = 0
    total = len(samples)

    for s in samples:
        msg_id = s["message_id"]
        pred = sample_preds.get(msg_id)
        if not pred:
            continue

        if pred["action"] == s["action"]:
            action_correct += 1
        if pred["message_type"] == s["message_type"]:
            type_correct += 1
        if pred["evidence_message_ids"] == s["evidence_message_ids"]:
            evidence_correct += 1

    print("\n" + "="*50)
    print(f"Evaluation on {total} sample messages:")
    print(f"Action Accuracy:         {action_correct}/{total} ({action_correct/total*100:.1f}%)")
    print(f"Message Type Accuracy:   {type_correct}/{total} ({type_correct/total*100:.1f}%)")
    print(f"Evidence Match Accuracy: {evidence_correct}/{total} ({evidence_correct/total*100:.1f}%)")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_pipeline()
