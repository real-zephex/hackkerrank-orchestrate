import os
import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from data_loader import DataStore

def check_dnd(window_str: Optional[str], created_at: Optional[datetime]) -> bool:
    if not window_str or not created_at:
        return False
    parts = window_str.strip().split("-")
    if len(parts) != 2:
        return False
    try:
        sh, sm = map(int, parts[0].split(":"))
        eh, em = map(int, parts[1].split(":"))
        start_min = sh * 60 + sm
        end_min = eh * 60 + em
        msg_min = created_at.hour * 60 + created_at.minute
        
        if start_min <= end_min:
            return start_min <= msg_min <= end_min
        else:
            return msg_min >= start_min or msg_min <= end_min
    except Exception:
        return False

def derive_outcome(event: Optional[Dict[str, Any]]) -> str:
    if not event:
        return "no_signal"
    if event.get("message_reported"):
        return "reported"
    if event.get("muted_after_message"):
        return "muted_after_message"
    if event.get("notification_dismissed"):
        return "dismissed"
    if event.get("message_replied"):
        return "replied"
    if event.get("message_opened"):
        return "opened"
    return "no_signal"

def serialize_datetime(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    return obj

def clean_dict(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if d is None:
        return None
    res = {}
    for k, v in d.items():
        res[k] = serialize_datetime(v)
    return res

def tokenize(text: str) -> Set[str]:
    if not text:
        return set()
    return set(re.findall(r"\w+", text.lower()))

def score_evidence_candidate(
    curr_text: str,
    curr_media_type: str,
    hist_msg: Dict[str, Any],
    event: Optional[Dict[str, Any]]
) -> float:
    outcome = derive_outcome(event)
    
    # Base outcome weights
    outcome_weights = {
        "reported": 10.0,
        "muted_after_message": 8.0,
        "dismissed": 6.0,
        "replied": 5.0,
        "opened": 3.0,
        "no_signal": 1.0
    }
    score = outcome_weights.get(outcome, 1.0)

    # Lexical similarity score (Jaccard similarity)
    t1 = tokenize(curr_text)
    t2 = tokenize(hist_msg.get("message_text", ""))
    if t1 and t2:
        intersection = len(t1.intersection(t2))
        union = len(t1.union(t2))
        jaccard = intersection / union if union > 0 else 0.0
        score += jaccard * 15.0  # Heavy weight on text similarity

    # Media type match bonus
    hist_media = hist_msg.get("media_type", "")
    if curr_media_type and curr_media_type == hist_media:
        score += 3.0

    return score

def build_context(message_id: str, ds: DataStore, media_cache: Dict[str, Any]) -> Dict[str, Any]:
    msg = ds.messages_by_id.get(message_id)
    if not msg:
        raise ValueError(f"Message ID {message_id} not found in DataStore.")

    user_id = msg["user_id"]
    conv_type = msg["conversation_type"]
    group_id = msg["group_id"]
    business_id = msg["business_id"]
    sender_user_id = msg["sender_user_id"]
    created_at = msg["created_at"]
    curr_text = msg.get("message_text", "")
    curr_media_type = msg.get("media_type", "")

    user = ds.get_user(user_id)

    # 1. Resolve conversation-type-specific context
    conv_context = {}
    if conv_type == "personal":
        sender_user = ds.get_user(sender_user_id) if sender_user_id else None
        conv_context["type"] = "personal"
        conv_context["sender_user"] = clean_dict(sender_user)
    elif conv_type == "group":
        group = ds.get_group(group_id)
        rx_member = ds.get_group_member(group_id, user_id)
        tx_member = ds.get_group_member(group_id, sender_user_id) if sender_user_id else None
        conv_context["type"] = "group"
        conv_context["group"] = clean_dict(group)
        conv_context["receiving_user_membership"] = clean_dict(rx_member)
        conv_context["sender_membership"] = clean_dict(tx_member)
    elif conv_type == "business":
        business = ds.get_business(business_id)
        ub = ds.get_user_business(user_id, business_id)
        conv_context["type"] = "business"
        conv_context["business"] = clean_dict(business)
        conv_context["user_business"] = clean_dict(ub)

    # 2. Precompute flags
    is_in_dnd = check_dnd(user.get("do_not_disturb_window") if user else None, created_at)
    
    is_verified_business = None
    domain_mismatch = None
    user_has_opted_out = False
    if conv_type == "business":
        biz = ds.get_business(business_id)
        if biz:
            is_verified_business = biz.get("verified", False)
            domain_mismatch = (biz.get("domain_used_by_sender") != biz.get("official_domain"))
        ub = ds.get_user_business(user_id, business_id)
        if ub and ub.get("promotions_opted_out_at") is not None:
            user_has_opted_out = True

    is_group_muted_by_user = None
    sender_is_group_admin = None
    if conv_type == "group":
        rx_mem = ds.get_group_member(group_id, user_id)
        if rx_mem:
            is_group_muted_by_user = rx_mem.get("group_muted_by_user", False)
        tx_mem = ds.get_group_member(group_id, sender_user_id) if sender_user_id else None
        if tx_mem:
            sender_is_group_admin = (tx_mem.get("role") == "admin")

    is_forwarded = int(msg.get("forwarded_count", 0)) > 0

    flags = {
        "is_in_dnd_window": is_in_dnd,
        "is_verified_business": is_verified_business,
        "domain_mismatch": domain_mismatch,
        "is_group_muted_by_user": is_group_muted_by_user,
        "sender_is_group_admin": sender_is_group_admin,
        "is_forwarded": is_forwarded,
        "user_has_opted_out": user_has_opted_out
    }

    # 3. Smart evidence retrieval
    user_history = ds.history_by_user.get(user_id, [])
    matching_history = []
    for h in user_history:
        if conv_type == "personal" and sender_user_id and h.get("sender_user_id") == sender_user_id:
            matching_history.append(h)
        elif conv_type == "group" and group_id and h.get("group_id") == group_id:
            matching_history.append(h)
        elif conv_type == "business" and business_id and h.get("business_id") == business_id:
            matching_history.append(h)

    # Score each historical candidate
    scored_candidates = []
    for h in matching_history:
        event = ds.events_by_message_id.get(h["message_id"])
        s_score = score_evidence_candidate(curr_text, curr_media_type, h, event)
        # Add tiny recency tie-breaker
        created_dt = h["created_at"] if isinstance(h["created_at"], datetime) else datetime.min
        recency_bonus = (created_dt - datetime.min).total_seconds() / 1e10
        total_score = s_score + recency_bonus
        scored_candidates.append((total_score, h, event))

    # Sort descending by total score
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    evidence_candidates = []
    for score, h, event in scored_candidates[:3]:
        evidence_candidates.append({
            "message_id": h["message_id"],
            "created_at": serialize_datetime(h["created_at"]),
            "text": h["message_text"],
            "media_type": h.get("media_type", ""),
            "forwarded_count": h.get("forwarded_count", 0),
            "outcome": derive_outcome(event)
        })

    # 4. Attach media context
    media_context = None
    media_type = msg.get("media_type")
    media_id = msg.get("media_id")
    if media_type == "image" and media_id:
        img_info = media_cache.get("images", {}).get(media_id)
        if img_info:
            media_context = img_info
        else:
            media_context = {"caption": "[unavailable]", "flags": {}}
    elif media_type == "voice" and media_id:
        vn_info = media_cache.get("voice_notes", {}).get(media_id)
        if vn_info:
            media_context = vn_info.get("transcript", "[unavailable]")
        else:
            media_context = "[unavailable]"

    return {
        "message": clean_dict(msg),
        "user": clean_dict(user),
        "conversation_context": conv_context,
        "flags": flags,
        "media_context": media_context,
        "evidence_candidates": evidence_candidates
    }

def build_all_contexts(ds: Optional[DataStore] = None, media_cache_path: str = "code/cache/media_cache.json") -> List[Dict[str, Any]]:
    if ds is None:
        ds = DataStore()

    if not os.path.isabs(media_cache_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        media_cache_path = os.path.join(repo_root, media_cache_path)

    media_cache = {}
    if os.path.exists(media_cache_path):
        with open(media_cache_path, "r", encoding="utf-8") as f:
            media_cache = json.load(f)

    contexts = []
    for msg in ds.messages:
        ctx = build_context(msg["message_id"], ds, media_cache)
        contexts.append(ctx)

    cache_dir = os.path.dirname(media_cache_path)
    os.makedirs(cache_dir, exist_ok=True)
    out_file = os.path.join(cache_dir, "contexts.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(contexts, f, indent=2)

    print(f"Built {len(contexts)} message contexts and saved to {out_file}")
    return contexts

if __name__ == "__main__":
    build_all_contexts()
