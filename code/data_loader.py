import os
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

def parse_datetime(val: Any) -> Optional[datetime]:
    if val is None:
        return None
    val_str = str(val).strip()
    if not val_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(val_str, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(val_str)
    except Exception:
        return None

def parse_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    val_str = str(val).strip()
    if not val_str:
        return default
    try:
        return int(float(val_str))
    except (ValueError, TypeError):
        return default

def parse_bool(val: Any) -> bool:
    if val is None:
        return False
    val_str = str(val).strip().lower()
    return val_str in ("1", "true", "t", "yes", "y")

def clean_id(val: Any) -> Optional[str]:
    if val is None:
        return None
    val_str = str(val).strip()
    return val_str if val_str else None

class DataStore:
    def __init__(self, data_dir: str = "dataset"):
        # Resolve data_dir relative to current working dir or repo root
        if not os.path.isabs(data_dir):
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            possible_path = os.path.join(repo_root, data_dir)
            if os.path.exists(possible_path):
                data_dir = possible_path
        self.data_dir = data_dir

        self.users_by_id: Dict[str, Dict[str, Any]] = {}
        self.groups_by_id: Dict[str, Dict[str, Any]] = {}
        self.group_member: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.business_by_id: Dict[str, Dict[str, Any]] = {}
        self.user_business: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.daily_summary: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.image_path: Dict[str, str] = {}
        self.voice_path: Dict[str, str] = {}
        self.history_by_user: Dict[str, List[Dict[str, Any]]] = {}
        self.events_by_message_id: Dict[str, Dict[str, Any]] = {}

        self.messages: List[Dict[str, Any]] = []
        self.messages_by_id: Dict[str, Dict[str, Any]] = {}

        self._load_all()

    def _get_path(self, filename: str) -> str:
        return os.path.join(self.data_dir, filename)

    def _load_users(self):
        path = self._get_path("users.csv")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                user_id = row["user_id"].strip()
                row["user_id"] = user_id
                row["messages_opened_30d"] = parse_int(row.get("messages_opened_30d"))
                row["messages_replied_30d"] = parse_int(row.get("messages_replied_30d"))
                row["notifications_dismissed_30d"] = parse_int(row.get("notifications_dismissed_30d"))
                row["messages_reported_30d"] = parse_int(row.get("messages_reported_30d"))
                self.users_by_id[user_id] = row

    def _load_groups(self):
        path = self._get_path("groups.csv")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                group_id = row["group_id"].strip()
                row["group_id"] = group_id
                row["member_count"] = parse_int(row.get("member_count"))
                row["admin_count"] = parse_int(row.get("admin_count"))
                row["messages_30d"] = parse_int(row.get("messages_30d"))
                row["created_at"] = parse_datetime(row.get("created_at"))
                self.groups_by_id[group_id] = row

    def _load_group_members(self):
        path = self._get_path("group_members.csv")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                group_id = row["group_id"].strip()
                user_id = row["user_id"].strip()
                row["group_id"] = group_id
                row["user_id"] = user_id
                row["joined_at"] = parse_datetime(row.get("joined_at"))
                row["messages_sent_30d"] = parse_int(row.get("messages_sent_30d"))
                row["messages_read_30d"] = parse_int(row.get("messages_read_30d"))
                row["replies_sent_30d"] = parse_int(row.get("replies_sent_30d"))
                row["notifications_dismissed_30d"] = parse_int(row.get("notifications_dismissed_30d"))
                row["group_muted_by_user"] = parse_bool(row.get("group_muted_by_user"))
                self.group_member[(group_id, user_id)] = row

    def _load_business(self):
        path = self._get_path("business_accounts.csv")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                business_id = row["business_id"].strip()
                row["business_id"] = business_id
                row["verified"] = parse_bool(row.get("verified"))
                row["account_age_days"] = parse_int(row.get("account_age_days"))
                row["messages_sent_30d"] = parse_int(row.get("messages_sent_30d"))
                row["user_reports_30d"] = parse_int(row.get("user_reports_30d"))
                row["domain_used_by_sender_age_days"] = parse_int(row.get("domain_used_by_sender_age_days"))
                self.business_by_id[business_id] = row

    def _load_user_business(self):
        path = self._get_path("user_business_history.csv")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                user_id = row["user_id"].strip()
                business_id = row["business_id"].strip()
                row["user_id"] = user_id
                row["business_id"] = business_id
                row["last_activity_at"] = parse_datetime(row.get("last_activity_at"))
                row["allows_promotions"] = parse_bool(row.get("allows_promotions"))
                row["promotions_opted_out_at"] = parse_datetime(row.get("promotions_opted_out_at"))
                row["activity_count_180d"] = parse_int(row.get("activity_count_180d"))
                row["messages_opened_30d"] = parse_int(row.get("messages_opened_30d"))
                row["messages_dismissed_30d"] = parse_int(row.get("messages_dismissed_30d"))
                row["messages_replied_30d"] = parse_int(row.get("messages_replied_30d"))
                row["last_reply_at"] = parse_datetime(row.get("last_reply_at"))
                self.user_business[(user_id, business_id)] = row

    def _load_daily_summary(self):
        path = self._get_path("daily_notification_summary.csv")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                user_id = row["user_id"].strip()
                date_str = row["date"].strip()
                row["user_id"] = user_id
                row["date"] = date_str
                row["notifications_sent"] = parse_int(row.get("notifications_sent"))
                row["notifications_dismissed"] = parse_int(row.get("notifications_dismissed"))
                self.daily_summary[(user_id, date_str)] = row

    def _load_images(self):
        path = self._get_path("images.csv")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                image_id = row["image_id"].strip()
                rel_path = row["file_path"].strip()
                self.image_path[image_id] = os.path.abspath(os.path.join(self.data_dir, rel_path))

    def _load_voice_notes(self):
        path = self._get_path("voice_notes.csv")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                voice_id = row["voice_note_id"].strip()
                rel_path = row["file_path"].strip()
                self.voice_path[voice_id] = os.path.abspath(os.path.join(self.data_dir, rel_path))

    def _load_message_history(self):
        path = self._get_path("message_history.csv")
        if not os.path.exists(path):
            return
        temp_history: Dict[str, List[Dict[str, Any]]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                user_id = row["user_id"].strip()
                row["message_id"] = row["message_id"].strip()
                row["user_id"] = user_id
                row["group_id"] = clean_id(row.get("group_id"))
                row["business_id"] = clean_id(row.get("business_id"))
                row["sender_user_id"] = clean_id(row.get("sender_user_id"))
                row["created_at"] = parse_datetime(row.get("created_at"))
                row["message_text"] = row.get("message_text", "") or ""
                row["forwarded_count"] = parse_int(row.get("forwarded_count"))
                row["media_id"] = clean_id(row.get("media_id"))

                if user_id not in temp_history:
                    temp_history[user_id] = []
                temp_history[user_id].append(row)

        # Sort history by created_at ascending per user
        for user_id, msgs in temp_history.items():
            msgs.sort(key=lambda m: m["created_at"] if m["created_at"] else datetime.min)
            self.history_by_user[user_id] = msgs

    def _load_events(self):
        path = self._get_path("message_events.csv")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                message_id = row["message_id"].strip()
                row["message_id"] = message_id
                row["user_id"] = row["user_id"].strip()
                row["message_opened"] = parse_bool(row.get("message_opened"))
                row["message_replied"] = parse_bool(row.get("message_replied"))
                row["notification_dismissed"] = parse_bool(row.get("notification_dismissed"))
                row["muted_after_message"] = parse_bool(row.get("muted_after_message"))
                row["message_reported"] = parse_bool(row.get("message_reported"))
                rt = row.get("reaction_time_minutes")
                row["reaction_time_minutes"] = parse_int(rt) if rt and rt.strip() else None
                self.events_by_message_id[message_id] = row

    def _load_messages(self):
        path = self._get_path("messages.csv")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                msg_id = row["message_id"].strip()
                row["message_id"] = msg_id
                row["user_id"] = row["user_id"].strip()
                row["group_id"] = clean_id(row.get("group_id"))
                row["business_id"] = clean_id(row.get("business_id"))
                row["sender_user_id"] = clean_id(row.get("sender_user_id"))
                row["created_at"] = parse_datetime(row.get("created_at"))
                row["message_text"] = row.get("message_text", "") or ""
                row["forwarded_count"] = parse_int(row.get("forwarded_count"))
                row["media_id"] = clean_id(row.get("media_id"))

                self.messages.append(row)
                self.messages_by_id[msg_id] = row

    def _load_all(self):
        self._load_users()
        self._load_groups()
        self._load_group_members()
        self._load_business()
        self._load_user_business()
        self._load_daily_summary()
        self._load_images()
        self._load_voice_notes()
        self._load_message_history()
        self._load_events()
        self._load_messages()

    def get_user(self, user_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not user_id:
            return None
        return self.users_by_id.get(user_id)

    def get_group(self, group_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not group_id:
            return None
        return self.groups_by_id.get(group_id)

    def get_group_member(self, group_id: Optional[str], user_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not group_id or not user_id:
            return None
        return self.group_member.get((group_id, user_id))

    def get_business(self, business_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not business_id:
            return None
        return self.business_by_id.get(business_id)

    def get_user_business(self, user_id: Optional[str], business_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not user_id or not business_id:
            return None
        return self.user_business.get((user_id, business_id))
