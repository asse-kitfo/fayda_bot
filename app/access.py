import json
import os
import time

USERS_FILE = "users.json"

ADMINS = {"sonof_virginmary"}


# =========================================================
# LOAD / SAVE
# =========================================================
def _load():
    if not os.path.exists(USERS_FILE):
        return {"users": {}, "usernames": {}}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"users": {}, "usernames": {}}


def _save(data):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =========================================================
# REGISTER — call every time a user interacts so we always
# have an up-to-date user_id ↔ username mapping
# =========================================================
def register(user_id: int, username: str | None):
    data = _load()
    uid = str(user_id)
    uname = (username or "").lower().strip("@")

    if uid not in data["users"]:
        data["users"][uid] = {
            "username": uname,
            "points": 0,
            "unlimited": False,
            "first_seen": time.time()
        }
    else:
        if uname:
            data["users"][uid]["username"] = uname

    if uname:
        data["usernames"][uname] = uid

    # auto-grant unlimited for admins
    if uname in ADMINS:
        data["users"][uid]["unlimited"] = True

    _save(data)


# =========================================================
# ADMIN CHECK
# =========================================================
def is_admin(user_id: int, username: str | None) -> bool:
    uname = (username or "").lower().strip("@")
    return uname in ADMINS


# =========================================================
# ACCESS CHECK
# =========================================================
def has_access(user_id: int) -> bool:
    data = _load()
    u = data["users"].get(str(user_id))
    if not u:
        return False
    return u.get("unlimited", False) or u.get("points", 0) > 0


def get_points(user_id: int) -> int | None:
    """Returns point count, or None if unlimited."""
    data = _load()
    u = data["users"].get(str(user_id))
    if not u:
        return 0
    if u.get("unlimited", False):
        return None
    return u.get("points", 0)


# =========================================================
# DEDUCT — called after a complete front+back generation
# =========================================================
def deduct_point(user_id: int):
    data = _load()
    u = data["users"].get(str(user_id))
    if not u or u.get("unlimited", False):
        return
    u["points"] = max(0, u.get("points", 0) - 1)
    _save(data)


# =========================================================
# ADMIN: GRANT POINTS OR UNLIMITED
# =========================================================
def grant(target_username: str, amount) -> str:
    """
    amount: int for points, or the string 'unlimited'.
    Returns a human-readable result message.
    """
    data = _load()
    uname = target_username.lower().strip("@")
    uid = data["usernames"].get(uname)

    if not uid:
        return f"❌ User @{uname} not found. They must send /start first."

    u = data["users"][uid]

    if amount == "unlimited":
        u["unlimited"] = True
        u["points"] = 0
        _save(data)
        return f"✅ @{uname} now has unlimited access."

    try:
        n = int(amount)
    except ValueError:
        return "❌ Amount must be a number or 'unlimited'."

    u["unlimited"] = False
    u["points"] = u.get("points", 0) + n
    _save(data)
    return f"✅ @{uname} now has {u['points']} point(s) (+{n})."


# =========================================================
# ADMIN: REVOKE ALL ACCESS
# =========================================================
def revoke(target_username: str) -> str:
    data = _load()
    uname = target_username.lower().strip("@")
    uid = data["usernames"].get(uname)

    if not uid:
        return f"❌ User @{uname} not found."

    u = data["users"][uid]
    u["unlimited"] = False
    u["points"] = 0
    _save(data)
    return f"✅ Access revoked for @{uname}."


# =========================================================
# ADMIN: LIST USERS
# =========================================================
def list_users() -> str:
    data = _load()
    users = data["users"]

    if not users:
        return "📋 No registered users yet."

    lines = ["📋 *Registered users:*\n"]
    for uid, u in sorted(users.items(), key=lambda x: x[1].get("first_seen", 0)):
        uname = u.get("username") or uid
        if u.get("unlimited"):
            status = "♾ unlimited"
        else:
            pts = u.get("points", 0)
            status = f"{pts} pt(s)"
        lines.append(f"• @{uname} — {status}")

    return "\n".join(lines)
