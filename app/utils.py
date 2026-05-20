import os
import shutil
import time


# =========================================================
# USER WORKSPACE
# =========================================================
def get_user_dir(user_id):

    path = os.path.join("temp", str(user_id))

    os.makedirs(path, exist_ok=True)

    return path


# =========================================================
# USER FILE
# =========================================================
def user_file(user_id, filename):

    return os.path.join(
        get_user_dir(user_id),
        filename
    )


# =========================================================
# CLEAN USER WORKSPACE
# =========================================================
def cleanup_user_dir(user_id):

    path = os.path.join("temp", str(user_id))

    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


# =========================================================
# CLEAN OLD SESSIONS
# =========================================================
def cleanup_old_dirs(max_age_minutes=30):

    base = "temp"

    if not os.path.exists(base):
        return

    now = time.time()

    for folder in os.listdir(base):

        full = os.path.join(base, folder)

        if not os.path.isdir(full):
            continue

        age = now - os.path.getmtime(full)

        if age > max_age_minutes * 60:

            shutil.rmtree(full, ignore_errors=True)

            print(f"🧹 Removed old temp folder: {folder}")