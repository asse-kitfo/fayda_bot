import os
import uuid

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)

from telegram.error import TimedOut, NetworkError

from app.ocr import process_ocr
from app.generator import generate_id

from app.back_ocr import process_back_ocr
from app.back_generator import generate_back
from app.utils import cleanup_old_dirs
from app import access
import time
import asyncio

if os.name == "nt":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )
    
load_dotenv()
user_sessions = {}

HELP_TEXT = (
    "📋 *How to use this bot:*\n\n"
    "*Step 1 —* Send a screenshot of the *FRONT* of the Fayda Digital ID card.\n"
    "The bot will extract your name, date of birth, gender, and ID number.\n\n"
    "*Step 2 —* Send a clear *FACE PHOTO* of the person.\n"
    "The background will be removed automatically.\n\n"
    "*Step 3 —* Send a screenshot of the *BACK* of the ID card.\n"
    "The bot will extract the address, phone number, and QR code.\n\n"
    "✅ The bot will then generate high-quality front and back ID images.\n\n"
    "📸 *Tips for best results:*\n"
    "• Make sure the full ID is visible\n"
    "• Text should be clear and not blurry\n"
    "• No dark overlay or cropping\n"
    "• Use a well-lit face photo\n\n"
    "Type /help anytime to see these instructions again."
)

# =========================================================
# START COMMAND
# =========================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.message.from_user
    access.register(u.id, u.username)
    await update.message.reply_text(
        "👋 Welcome to the *Fayda ID Card Generator Bot!*\n\n"
        + HELP_TEXT,
        parse_mode="Markdown"
    )

# =========================================================
# HELP COMMAND
# =========================================================
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.message.from_user
    access.register(u.id, u.username)
    await update.message.reply_text(
        HELP_TEXT,
        parse_mode="Markdown"
    )

# =========================================================
# RESET COMMAND
# =========================================================
async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    access.register(user_id, update.message.from_user.username)
    user_sessions.pop(user_id, None)
    processing_users.discard(user_id)
    await update.message.reply_text(
        "🔄 Session reset.\n\n"
        "📸 Send a new *FRONT* ID screenshot to start again.",
        parse_mode="Markdown"
    )

# =========================================================
# REQUEST COMMAND — user asks for access
# =========================================================
async def request_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.message.from_user
    access.register(u.id, u.username)
    uname = u.username or str(u.id)

    pts = access.get_points(u.id)

    if pts is None:
        await update.message.reply_text("✅ You already have unlimited access.")
        return

    if pts > 0:
        await update.message.reply_text(
            f"✅ You already have *{pts}* point(s).",
            parse_mode="Markdown"
        )
        return

    await update.message.reply_text(
        "📨 Access request sent to the admin.\n"
        "You will be notified once access is granted."
    )
    print(f"🔔 ACCESS REQUEST from @{uname} (id={u.id})")

    # Notify every known admin via DM
    admin_ids = access.get_admin_ids()
    for admin_id in admin_ids:
        try:
            await context.bot.send_message(
                chat_id=admin_id,
                text=(
                    f"🔔 *New access request*\n\n"
                    f"User: @{uname}\n"
                    f"ID: `{u.id}`\n\n"
                    f"Reply with:\n"
                    f"`/grant @{uname} 10`  — give 10 points\n"
                    f"`/grant @{uname} unlimited`  — unlimited access"
                ),
                parse_mode="Markdown"
            )
        except Exception as e:
            print(f"⚠️ Could not notify admin {admin_id}: {e}")

# =========================================================
# MYPOINTS COMMAND
# =========================================================
async def mypoints_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.message.from_user
    access.register(u.id, u.username)
    pts = access.get_points(u.id)

    if pts is None:
        await update.message.reply_text("♾ You have *unlimited* access.", parse_mode="Markdown")
    elif pts == 0:
        await update.message.reply_text(
            "❌ You have *0 points*.\n\nUse /request to ask for access.",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(
            f"🎯 You have *{pts}* point(s) remaining.\n"
            f"Each front+back generation uses 1 point.",
            parse_mode="Markdown"
        )

# =========================================================
# GRANT COMMAND (admin only)
# Usage: /grant @username 10   or   /grant @username unlimited
# =========================================================
async def grant_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.message.from_user
    access.register(u.id, u.username)

    if not access.is_admin(u.id, u.username):
        await update.message.reply_text("❌ Admin only.")
        return

    args = context.args
    if len(args) < 2:
        await update.message.reply_text(
            "Usage: `/grant @username 10`  or  `/grant @username unlimited`",
            parse_mode="Markdown"
        )
        return

    target = args[0].strip("@")
    amount = args[1]
    result = access.grant(target, amount)
    await update.message.reply_text(result)

# =========================================================
# REVOKE COMMAND (admin only)
# Usage: /revoke @username
# =========================================================
async def revoke_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.message.from_user
    access.register(u.id, u.username)

    if not access.is_admin(u.id, u.username):
        await update.message.reply_text("❌ Admin only.")
        return

    args = context.args
    if not args:
        await update.message.reply_text(
            "Usage: `/revoke @username`",
            parse_mode="Markdown"
        )
        return

    target = args[0].strip("@")
    result = access.revoke(target)
    await update.message.reply_text(result)

# =========================================================
# USERS COMMAND (admin only) — list all registered users
# =========================================================
async def users_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.message.from_user
    access.register(u.id, u.username)

    if not access.is_admin(u.id, u.username):
        await update.message.reply_text("❌ Admin only.")
        return

    await update.message.reply_text(
        access.list_users(),
        parse_mode="Markdown"
    )
processing_users = set()

# prevent duplicate processing
processed_files = {}

# session timeout (20 minutes)
SESSION_TIMEOUT = 1200
# =========================================================
# SAFE REPLY
# prevents crash if telegram fails while replying
# =========================================================
async def safe_reply(method, *args, **kwargs):

    try:
        return await method(*args, **kwargs)

    except Exception as e:
        print("⚠️ Reply failed:", e)
        return None

# =========================================================
# CLEAR EXPIRED SESSION
# =========================================================
def clear_expired_session(user_id):

    session = user_sessions.get(user_id)

    if not session:
        return

    created = session.get("created_at", 0)

    if time.time() - created > SESSION_TIMEOUT:

        print(f"🧹 Expired session cleared: {user_id}")

        user_sessions.pop(user_id, None)
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_id = None

    try:

        user_id = update.message.from_user.id
        u = update.message.from_user
        access.register(user_id, u.username)

        # clear old session
        clear_expired_session(user_id)

        # =====================================================
        # GLOBAL PROCESS LOCK
        # =====================================================
        if user_id in processing_users:

            await update.message.reply_text(
                "⏳ Previous request still processing..."
            )

            return

        processing_users.add(user_id)

        photo = update.message.photo[-1]

        file = await photo.get_file()

        os.makedirs("temp", exist_ok=True)

        user_temp = f"temp/{user_id}"
        os.makedirs(user_temp, exist_ok=True)

        # =====================================================
        # UNIQUE TELEGRAM FILE ID
        # prevents duplicate upload retries
        # =====================================================
        telegram_file_id = photo.file_unique_id
        
        # =====================================================
        # GLOBAL DUPLICATE DETECTION
        # prevents Telegram resend/retry duplication
        # =====================================================
        cache_key = f"{user_id}_{telegram_file_id}"

        if cache_key in processed_files:

            await safe_reply(
                update.message.reply_text,
                "⚠️ This image was already uploaded."
            )

            return

        session = user_sessions.get(user_id)

        # =====================================================
        # BLOCK SAME IMAGE RETRY
        # =====================================================
        if session:

            last_file = session.get("last_file_id")

            if last_file == telegram_file_id:

                await update.message.reply_text(
                    "⚠️ This image was already uploaded."
                )

                return

        job_id = str(uuid.uuid4())

        file_path = f"{user_temp}/{job_id}.jpg"

        await file.download_to_drive(file_path)

        # =====================================================
        # STEP 1 → FRONT OCR
        # =====================================================
        if not session:

            # -------------------------------------------------
            # ACCESS CHECK — only for new sessions
            # -------------------------------------------------
            if not access.has_access(user_id):
                await update.message.reply_text(
                    "🔒 You don't have access to this bot.\n\n"
                    "Use /request to ask for access from the admin."
                )
                return

            await update.message.reply_text(
                "🔍 Processing front ID..."
            )

            front_data = process_ocr(
                file_path,
                confirm=False,
                debug_dir=user_temp
            )

            if front_data.get("problems"):

                problems = front_data["problems"]
                issues = front_data.get("issues", [])

                msg = "❌ OCR validation failed.\n\n"

                for p in problems:
                    msg += f"• {p}\n"

                if issues:

                    msg += "\nDetected issues:\n"

                    for issue in issues:
                        msg += f" - {issue}\n"

                msg += (
                    "\n📸 Please upload a clearer FRONT screenshot.\n"
                    "Make sure:\n"
                    "• The full ID is visible\n"
                    "• Text is not blurry\n"
                    "• No dark overlay exists\n"
                    "• Screenshot is not cropped"
                )

                await safe_reply(
                    update.message.reply_text,
                    msg
                )

                return
            
            processed_files[cache_key] = time.time()

            # =================================================
            # SAVE SESSION
            # =================================================
            user_sessions[user_id] = {

                "step": "waiting_face",

                "front_data": front_data,

                "job_id": job_id,

                "last_file_id": telegram_file_id,

                "created_at": time.time(),

                # locks
                "front_processing": False,
                "back_processing": False
            }

            await update.message.reply_text(
                "✅ Front data extracted.\n\n"
                "📸 Now send face photo."
            )

            return

        # =====================================================
        # EXISTING SESSION
        # =====================================================
        session = user_sessions[user_id]

        # =====================================================
        # STEP 2 → FACE
        # =====================================================
        if session["step"] == "waiting_face":

            # already processing
            if session.get("front_processing"):

                await safe_reply(
                    update.message.reply_text,
                    "⏳ Face processing already running..."
                )

                return

            session["front_processing"] = True

            # atomic transition
            session["step"] = "generating_front"

            session["last_file_id"] = telegram_file_id

            await safe_reply(
                update.message.reply_text,
                "🧑 Processing face..."
            )

            front_data = session["front_data"]

            front_output = (
                f"temp/{user_id}/front_{session['job_id']}.tif"
            )


            
            front_path = generate_id(
                front_data,
                file_path,
                front_output,
                debug_dir=user_temp
            )

            # =================================================
            # FACE FAILED
            # =================================================
            if not front_path:

                session["front_processing"] = False
                session["step"] = "waiting_face"

                await safe_reply(
                    update.message.reply_text,
                    "❌ No human face detected.\n\n"
                    "📸 Please upload a clear face photo."
                )

                return
            
            processed_files[cache_key] = time.time()
            # =================================================
            # MARK FRONT AS COMPLETED
            # IMPORTANT FIX
            # =================================================
            session["front_generated"] = True
            session["front_processing"] = False

            # =================================================
            # SEND FRONT FILE
            # =================================================
            try:

                with open(front_path, "rb") as f:

                    await update.message.reply_document(
                        document=f
                    )

            except Exception as e:

                print("⚠️ Failed sending front:", e)

            # =================================================
            # MOVE TO NEXT STEP IMMEDIATELY
            # even if telegram/network fails
            # =================================================
            session["step"] = "waiting_back"

            await safe_reply(
                update.message.reply_text,
                "✅ Front generated.\n\n"
                "📸 Now send BACK screenshot."
            )

            return
        # =====================================================
        # ALREADY GENERATING FRONT
        # =====================================================
        if session["step"] == "generating_front":

            await update.message.reply_text(
                "⏳ Front generation already in progress..."
            )

            return
        
        # =====================================================
        # BACK ALREADY FINISHED
        # =====================================================
        if session.get("back_generated"):

            await safe_reply(
                update.message.reply_text,
                "⚠️ Process already completed.\n\n"
                "📸 Send a new FRONT screenshot."
            )

            return
        


        # =====================================================
        # STEP 3 → BACK
        # =====================================================
        if session["step"] == "waiting_back":

            if session.get("back_processing"):

                await safe_reply(
                    update.message.reply_text,
                    "⏳ Back side processing already running..."
                )

                return

            session["back_processing"] = True

            # atomic lock
            session["step"] = "generating_back"

            session["last_file_id"] = telegram_file_id

            await safe_reply(
                update.message.reply_text,
                "🔍 Processing back side..."
            )
            
            # =================================================
            # VALIDATE BACK SCREENSHOT
            # =================================================
            back_data, qr_crop = process_back_ocr(file_path)

            # =================================================
            # INVALID BACK SCREENSHOT
            # =================================================
            if (
                not back_data
                or qr_crop is None
                or back_data.get("problems")
            ):
                
                session["back_processing"] = False
                session["step"] = "waiting_back"

                msg = "❌ BACK validation failed.\n\n"

                problems = back_data.get("problems", [])

                for p in problems:
                    msg += f"• {p}\n"

                msg += (
                    "\n📸 Please upload a valid BACK screenshot.\n"
                    "Make sure:\n"
                    "• QR code is visible\n"
                    "• Screenshot is not cropped\n"
                    "• Text is readable\n"
                    "• No dark overlay exists"
                )
                

                await safe_reply(
                    update.message.reply_text,
                    msg
                )

                return
            

            front_data = session["front_data"]

            processed_files[cache_key] = time.time()
            
            person_name = front_data.get(
                "name_en",
                "unknown"
            )

            back_output = (
                f"temp/{user_id}/back_{session['job_id']}.tif"
            )

            back_path = generate_back(
                back_data,
                qr_crop,
                back_output,
                person_name
            )

            # =================================================
            # MARK BACK GENERATED + DEDUCT POINT
            # =================================================
            session["back_generated"] = True
            session["back_processing"] = False
            access.deduct_point(user_id)

            # =================================================
            # SEND BACK FILE
            # =================================================
            try:

                with open(back_path, "rb") as f:

                    await update.message.reply_document(
                        document=f
                    )

            except Exception as e:

                print("⚠️ Failed sending back:", e)

            # =================================================
            # FINISH SESSION
            # IMPORTANT:
            # clear session even if sending failed
            # =================================================
            user_sessions.pop(user_id, None)

            await safe_reply(
                update.message.reply_text,
                "✅ Back generated.\n\n"
                "Send another FRONT screenshot to start again."
            )

            return
        
        # =====================================================
        # FRONT ALREADY FINISHED
        # =====================================================
        if (
            session.get("front_generated")
            and session["step"] != "waiting_back"
        ):

            await safe_reply(
                update.message.reply_text,
                "⚠️ Front already generated.\n\n"
                "📸 Please send BACK screenshot."
            )

            return
    # =========================================================
    # ERRORS
    # =========================================================
    except TimedOut:

        print("❌ Timeout")

        await safe_reply(
            update.message.reply_text,
            "⚠️ Network timeout."
        )

    except NetworkError:

        print("❌ Network Error")

        await safe_reply(
            update.message.reply_text,
            "⚠️ Network error."
        )

    except Exception as e:

        print("❌ ERROR:", e)

        await safe_reply(
            update.message.reply_text,
            "⚠️ Something went wrong."
        )

    finally:

        if user_id is not None:

            # cleanup old duplicate cache
            now = time.time()

            processed_files_copy = processed_files.copy()

            for key, ts in processed_files_copy.items():

                if now - ts > 120:

                    processed_files.pop(key, None)
            processing_users.discard(user_id)

# =========================================================
# MAIN
# =========================================================
def main():

    TOKEN = os.getenv("BOT_TOKEN")
    
    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .connect_timeout(60.0)
        .read_timeout(60.0)
        .write_timeout(60.0)
        .pool_timeout(60.0)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("request", request_command))
    app.add_handler(CommandHandler("mypoints", mypoints_command))
    app.add_handler(CommandHandler("grant", grant_command))
    app.add_handler(CommandHandler("revoke", revoke_command))
    app.add_handler(CommandHandler("users", users_command))
    app.add_handler(
        MessageHandler(
            filters.PHOTO,
            handle_image
        )
    )
    
    app.job_queue.run_repeating(
        lambda *_: cleanup_old_dirs(30),
        interval=600,
        first=600
    )
    
    print("🚀 Initializing OCR and segmentation models...")
    print("✅ Bot running...")
    
    app.run_polling()


if __name__ == "__main__":
    main()