import os
import uuid

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
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
_last_front_file_ids: dict = {}   # user_id → file_unique_id of last PAID front screenshot

ADMIN_USERNAME = next(iter(access.ADMINS), "admin")

WELCOME_TEXT = (
    "👋 Welcome to the <b>Fayda ID Card Generator Bot!</b>\n\n"
    "This bot generates high-quality, print-ready Fayda Digital ID card from your screenshots.\n\n"
    "📞 Need help or access?\n"
    f"Contact the admin: @{ADMIN_USERNAME}"
)

HELP_TEXT = (
    "📋 <b>How to use this bot:</b>\n\n"
    "<b>Step 1 —</b> Send a screenshot of the <b>FRONT</b> of the Fayda Digital ID card.\n"
    "The bot will extract your name, date of birth, gender, and ID number.\n\n"
    "<b>Step 2 —</b> Send a clear <b>FACE PHOTO</b> of the person.\n"
    "The background will be removed automatically.\n\n"
    "<b>Step 3 —</b> Send a screenshot of the <b>BACK</b> of the ID card.\n"
    "The bot will extract the address, phone number, and QR code.\n\n"
    "✅ The bot will then generate high-quality front and back ID images.\n\n"
    "📸 <b>Tips for best results:</b>\n"
    "• Make sure the full ID is visible\n"
    "• Text should be clear and not blurry\n"
    "• No dark overlay or cropping\n"
    "• Use a well-lit face photo"
)

# =========================================================
# KEYBOARD BUILDERS
# =========================================================
def main_menu_keyboard(has_access_flag: bool, pts) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton("📋 How to Use", callback_data="help")]
    ]
    if has_access_flag:
        if pts is None:
            points_label = "♾ Unlimited Access"
        else:
            points_label = f"🎯 My Points ({pts})"
        buttons.append([InlineKeyboardButton(points_label, callback_data="mypoints")])
        buttons.append([InlineKeyboardButton("🎨 Choose Template", callback_data="choose_template")])
    else:
        buttons.append([InlineKeyboardButton("📨 Request Access", callback_data="request_access")])
    buttons.append([InlineKeyboardButton("🔄 Reset Session", callback_data="reset")])
    return InlineKeyboardMarkup(buttons)


def no_access_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📨 Request Access", callback_data="request_access")],
        [InlineKeyboardButton("🎯 My Points", callback_data="mypoints")],
    ])


# =========================================================
# TEMPLATE HELPERS
# =========================================================
def get_available_templates() -> list:
    """Scan assets/templates/ and return list of available template IDs."""
    templates = []
    tdir = "assets/templates"
    if os.path.exists(f"{tdir}/front.tif"):
        templates.append("a")
    for letter in "bcdefghijklmnop":
        if os.path.exists(f"{tdir}/front_{letter}.tif"):
            templates.append(letter)
    return templates or ["a"]


def template_keyboard(current_tid: str = None) -> InlineKeyboardMarkup:
    templates = get_available_templates()
    row, buttons = [], []
    for tid in templates:
        label = f"✅ Template {tid.upper()}" if tid == current_tid else f"📄 Template {tid.upper()}"
        row.append(InlineKeyboardButton(label, callback_data=f"select_template_{tid}"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton("🔄 Reset Session", callback_data="reset")])
    return InlineKeyboardMarkup(buttons)


def request_picker_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("5 pts",   callback_data="request_pts_5"),
            InlineKeyboardButton("10 pts",  callback_data="request_pts_10"),
            InlineKeyboardButton("20 pts",  callback_data="request_pts_20"),
        ],
        [
            InlineKeyboardButton("50 pts",  callback_data="request_pts_50"),
            InlineKeyboardButton("100 pts", callback_data="request_pts_100"),
        ],
        [InlineKeyboardButton("⬅️ Back",   callback_data="back_to_menu")],
    ])


def done_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Generate Another", callback_data="start_again")],
        [InlineKeyboardButton("🎯 My Points", callback_data="mypoints")],
        [InlineKeyboardButton("📋 Menu", callback_data="back_to_menu")],
    ])


def reset_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Reset Session", callback_data="reset")],
        [InlineKeyboardButton("📋 Menu", callback_data="back_to_menu")],
    ])


def admin_grant_keyboard(user_id: int, username: str) -> InlineKeyboardMarkup:
    uname = username or str(user_id)
    uid   = user_id
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("5 pts",   callback_data=f"admin_grant_{uid}_5"),
            InlineKeyboardButton("10 pts",  callback_data=f"admin_grant_{uid}_10"),
            InlineKeyboardButton("20 pts",  callback_data=f"admin_grant_{uid}_20"),
        ],
        [
            InlineKeyboardButton("50 pts",  callback_data=f"admin_grant_{uid}_50"),
            InlineKeyboardButton("100 pts", callback_data=f"admin_grant_{uid}_100"),
            InlineKeyboardButton("♾ Unlimited", callback_data=f"admin_grant_{uid}_unlimited"),
        ],
        [InlineKeyboardButton("❌ Deny", callback_data=f"admin_deny_{uid}_{uname}")],
    ])


# =========================================================
# START COMMAND
# =========================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.message.from_user
    access.register(u.id, u.username)
    pts = access.get_points(u.id)
    has_acc = access.has_access(u.id)

    await update.message.reply_text(
        WELCOME_TEXT,
        parse_mode="HTML",
        reply_markup=main_menu_keyboard(has_acc, pts)
    )


# =========================================================
# HELP COMMAND
# =========================================================
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.message.from_user
    access.register(u.id, u.username)
    await update.message.reply_text(
        HELP_TEXT,
        parse_mode="HTML",
        reply_markup=reset_keyboard()
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
        "📸 Send a new <b>FRONT</b> ID screenshot to start again.",
        parse_mode="HTML"
    )


# =========================================================
# RETRY COMMAND — re-run last failed step without re-uploading
# =========================================================
async def retry_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_id = update.message.from_user.id
    access.register(user_id, update.message.from_user.username)

    session = user_sessions.get(user_id)

    if not session or not session.get("last_tg_dl_file_id"):
        await update.message.reply_text(
            "⚠️ Nothing to retry.\n\n"
            "📸 Please send a <b>FRONT</b> ID screenshot to start.",
            parse_mode="HTML"
        )
        return

    if session.get("front_processing") or session.get("back_processing"):
        await update.message.reply_text("⏳ Processing is already in progress...")
        return

    if user_id in processing_users:
        await update.message.reply_text("⏳ Processing is already in progress...")
        return

    last_attempted = session.get("last_attempted")
    dl_file_id     = session.get("last_tg_dl_file_id")

    if last_attempted == "front":
        await update.message.reply_text(
            "✅ Front data was already extracted.\n\n"
            "📸 Please send the <b>face photo</b> to continue.",
            parse_mode="HTML",
            reply_markup=reset_keyboard()
        )
        return

    if last_attempted not in ("face", "back"):
        await update.message.reply_text(
            "⚠️ Nothing to retry. Please continue with the next step.",
            reply_markup=reset_keyboard()
        )
        return

    os.makedirs("temp", exist_ok=True)
    user_temp = f"temp/{user_id}"
    os.makedirs(user_temp, exist_ok=True)

    retry_file_path = None

    try:
        tg_file = await context.bot.get_file(dl_file_id)
        retry_job_id    = str(uuid.uuid4())
        retry_file_path = f"{user_temp}/{retry_job_id}.jpg"
        await download_with_retry(tg_file, retry_file_path)
    except Exception as e:
        print("❌ Retry download failed:", e)
        await update.message.reply_text(
            "❌ Could not re-download the image from Telegram.\n\n"
            "Please upload the image again."
        )
        return

    processing_users.add(user_id)

    try:

        if last_attempted == "face":

            session["front_processing"] = True
            session["step"]             = "generating_front"

            await update.message.reply_text("🧑 Retrying face processing...")

            front_data   = session["front_data"]
            front_output = f"temp/{user_id}/front_{session['job_id']}.tif"

            async with _heavy_sem:
                front_path = await asyncio.to_thread(
                    generate_id,
                    front_data,
                    retry_file_path,
                    front_output,
                    user_temp,
                    session.get("template_id", "a")
                )

            try:
                os.remove(retry_file_path)
                retry_file_path = None
            except OSError:
                pass

            if not front_path:
                session["front_processing"] = False
                session["step"]             = "waiting_face"
                await safe_reply(
                    update.message.reply_text,
                    "❌ No human face detected.\n\n"
                    "📸 Please upload a clear face photo.",
                    reply_markup=reset_keyboard()
                )
                return

            session["front_generated"]  = True
            session["front_processing"] = False

            front_fuid = session.get("front_file_unique_id")
            if _last_front_file_ids.get(user_id) != front_fuid:
                access.deduct_point(user_id, 0.5)
                _last_front_file_ids[user_id] = front_fuid

            sent = await send_document_with_retry(
                update.message.reply_document, front_path
            )
            try:
                os.remove(front_path)
            except OSError:
                pass
            if not sent:
                await safe_reply(
                    update.message.reply_text,
                    "⚠️ Front ID was generated but couldn't be sent.\n\nUse /retry to try again.",
                    reply_markup=reset_keyboard()
                )
                return

            session["step"] = "waiting_back"

            pts_after = access.get_points(user_id)
            front_msg = "✅ Front ID generated.\n\n📸 Now send the <b>BACK</b> screenshot of the ID card."
            if pts_after is not None and 0 < pts_after <= 1:
                pts_fmt = int(pts_after) if pts_after == int(pts_after) else pts_after
                front_msg += f"\n\n⚠️ <b>Low points warning:</b> You only have <b>{pts_fmt}</b> point(s) left. Request more soon to avoid being blocked."

            await safe_reply(
                update.message.reply_text,
                front_msg,
                parse_mode="HTML",
                reply_markup=reset_keyboard()
            )

        elif last_attempted == "back":

            pts_now = access.get_points(user_id)
            if pts_now is not None and pts_now < 0.5:
                await safe_reply(
                    update.message.reply_text,
                    "⚠️ You don't have enough points to generate the Back ID.\n\n"
                    "You need at least <b>0.5 points</b>. Use the button below to request more.",
                    parse_mode="HTML",
                    reply_markup=no_access_keyboard()
                )
                return

            session["back_processing"] = True
            session["step"]            = "generating_back"

            await update.message.reply_text("🔍 Retrying back side processing...")

            async with _heavy_sem:
                back_data, qr_crop = await asyncio.to_thread(
                    process_back_ocr,
                    retry_file_path,
                    True,
                    user_temp
                )

            try:
                os.remove(retry_file_path)
                retry_file_path = None
            except OSError:
                pass

            if not back_data or qr_crop is None or back_data.get("problems"):
                session["back_processing"] = False
                session["step"]            = "waiting_back"

                msg = "❌ BACK validation failed.\n\n"
                problems = back_data.get("problems", []) if back_data else []
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
                    msg,
                    reply_markup=reset_keyboard()
                )
                return

            front_data  = session["front_data"]
            person_name = front_data.get("name_en", "unknown")
            back_output = f"temp/{user_id}/back_{session['job_id']}.tif"

            async with _heavy_sem:
                back_path = await asyncio.to_thread(
                    generate_back,
                    back_data,
                    qr_crop,
                    back_output,
                    person_name,
                    session.get("template_id", "a")
                )

            session["back_generated"]  = True
            session["back_processing"] = False
            access.deduct_point(user_id, 0.5)

            sent = await send_document_with_retry(
                update.message.reply_document, back_path
            )
            try:
                os.remove(back_path)
            except OSError:
                pass
            if not sent:
                await safe_reply(
                    update.message.reply_text,
                    "⚠️ Back ID was generated but couldn't be sent.\n\nUse /retry to try again.",
                    reply_markup=reset_keyboard()
                )
                return

            user_sessions.pop(user_id, None)

            pts = access.get_points(user_id)
            if pts is None:
                pts_line = "♾ Unlimited access remaining."
            elif pts == 0:
                pts_line = "⚠️ You have <b>0 points</b> left. Use the button below to request more."
            elif pts <= 1:
                pts_fmt = int(pts) if pts == int(pts) else pts
                pts_line = f"⚠️ <b>Low points:</b> Only <b>{pts_fmt}</b> point(s) left — enough for one more generation. Request more soon."
            else:
                pts_fmt = int(pts) if pts == int(pts) else pts
                pts_line = f"🎯 You have <b>{pts_fmt}</b> point(s) remaining."

            await safe_reply(
                update.message.reply_text,
                f"✅ Back ID generated. All done!\n\n{pts_line}",
                parse_mode="HTML",
                reply_markup=done_keyboard()
            )

    except TimedOut:
        await safe_reply(update.message.reply_text, "⚠️ Network timeout. Use /retry to try again.")

    except NetworkError:
        await safe_reply(update.message.reply_text, "⚠️ Network error. Use /retry to try again.")

    except Exception as e:
        print("❌ Retry error:", e)
        await safe_reply(
            update.message.reply_text,
            "⚠️ Retry failed. Please upload the image again.",
            reply_markup=reset_keyboard()
        )

    finally:
        s = user_sessions.get(user_id)
        if s:
            s["front_processing"] = False
            s["back_processing"]  = False
        processing_users.discard(user_id)
        if retry_file_path:
            try:
                os.remove(retry_file_path)
            except OSError:
                pass


# =========================================================
# REQUEST COMMAND — user asks for access
# =========================================================
async def request_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.message.from_user
    access.register(u.id, u.username)
    await _do_request_access(update.message.reply_text, context, u)


# =========================================================
# MYPOINTS COMMAND
# =========================================================
async def mypoints_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.message.from_user
    access.register(u.id, u.username)
    await _send_points(update.message.reply_text, u.id)


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
            parse_mode="HTML"
        )
        return

    target = args[0].strip("@")
    amount = args[1]
    result = access.grant(target, amount)
    await update.message.reply_text(result)


# =========================================================
# REVOKE COMMAND (admin only)
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
            parse_mode="HTML"
        )
        return

    target = args[0].strip("@")
    result = access.revoke(target)
    await update.message.reply_text(result)


# =========================================================
# USERS COMMAND (admin only)
# =========================================================
async def users_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.message.from_user
    access.register(u.id, u.username)

    if not access.is_admin(u.id, u.username):
        await update.message.reply_text("❌ Admin only.")
        return

    await update.message.reply_text(
        access.list_users(),
        parse_mode="HTML"
    )


# =========================================================
# BROADCAST COMMAND (admin only)
# Usage: /broadcast Your message here
# =========================================================
async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.message.from_user
    access.register(u.id, u.username)

    if not access.is_admin(u.id, u.username):
        await update.message.reply_text("❌ Admin only.")
        return

    msg = " ".join(context.args).strip()
    if not msg:
        await update.message.reply_text(
            "Usage: <code>/broadcast Your message here</code>",
            parse_mode="HTML"
        )
        return

    data = access._load()
    users = data["users"]
    sent = failed = 0

    status_msg = await update.message.reply_text(
        f"📢 Broadcasting to {len(users)} users..."
    )

    for uid in users:
        try:
            await context.bot.send_message(
                chat_id=int(uid),
                text=f"📢 <b>Announcement</b>\n\n{msg}",
                parse_mode="HTML"
            )
            sent += 1
        except Exception:
            failed += 1

    await status_msg.edit_text(
        f"📢 <b>Broadcast complete</b>\n\n"
        f"✅ Delivered: {sent}\n"
        f"❌ Failed: {failed}",
        parse_mode="HTML"
    )


# =========================================================
# STATUS COMMAND (admin only)
# =========================================================
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.message.from_user
    access.register(u.id, u.username)

    if not access.is_admin(u.id, u.username):
        await update.message.reply_text("❌ Admin only.")
        return

    await update.message.reply_text(
        access.get_status(),
        parse_mode="HTML"
    )


# =========================================================
# SETPREVIEW COMMAND (admin only)
# Send a photo with caption /setpreview  OR
# reply to a photo with /setpreview
# =========================================================
async def setpreview_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.message.from_user
    access.register(u.id, u.username)

    if not access.is_admin(u.id, u.username):
        await update.message.reply_text("❌ Admin only.")
        return

    # Determine side: /setpreview front  or  /setpreview back  (default = front)
    args = context.args
    side = (args[0].lower() if args else "front")
    if side not in ("front", "back"):
        await update.message.reply_text(
            "Usage: send a photo with caption <code>/setpreview front</code> or <code>/setpreview back</code>.",
            parse_mode="HTML"
        )
        return

    # Photo can be in the command message itself or the replied-to message
    photo = None
    if update.message.photo:
        photo = update.message.photo[-1]
    elif update.message.reply_to_message and update.message.reply_to_message.photo:
        photo = update.message.reply_to_message.photo[-1]

    if not photo:
        await update.message.reply_text(
            f"📸 Send a photo with caption <code>/setpreview {side}</code>, "
            f"or reply to a photo with <code>/setpreview {side}</code>.",
            parse_mode="HTML"
        )
        return

    await update.message.reply_text(f"⏳ Saving {side} preview image...")

    file = await photo.get_file()
    tmp_path  = f"assets/template_sample_{side}_tmp.jpg"
    final_path = f"assets/template_sample_{side}.jpg"

    await file.download_to_drive(tmp_path)

    from PIL import Image as _Img, ImageCms as _Cms
    import io as _io
    img = _Img.open(tmp_path)
    icc = img.info.get("icc_profile")
    if img.mode == "CMYK" and icc:
        src_profile = _Cms.ImageCmsProfile(_io.BytesIO(icc))
        dst_profile = _Cms.createProfile("sRGB")
        img = _Cms.profileToProfile(img, src_profile, dst_profile, outputMode="RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    img.save(final_path, "JPEG", quality=95)
    os.remove(tmp_path)

    await update.message.reply_photo(
        photo=open(final_path, "rb"),
        caption=(
            f"✅ <b>Template {side} preview updated!</b>\n\n"
            "Users will see this when they tap <b>Choose Template</b>."
        ),
        parse_mode="HTML"
    )


# =========================================================
# ADDTEMPLATE COMMAND (admin only)
# Usage: send a TIF document with caption:
#   /addtemplate b front   → saves as assets/templates/front_b.tif
#   /addtemplate b back    → saves as assets/templates/back_b.tif
# Or reply to a document message with the same caption.
# Template A is the default (front.tif / back.tif).
# All templates share the same coords.json / back_coords.json.
# =========================================================
async def addtemplate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.message.from_user
    access.register(u.id, u.username)

    if not access.is_admin(u.id, u.username):
        await update.message.reply_text("❌ Admin only.")
        return

    args = context.args
    if len(args) < 2:
        await update.message.reply_text(
            "📋 <b>Usage:</b>\n\n"
            "Send a <code>.tif</code> file with caption:\n"
            "<code>/addtemplate b front</code> — adds front template B\n"
            "<code>/addtemplate b back</code>  — adds back template B\n\n"
            "Or reply to a document with the same command.\n\n"
            "<b>Valid letters:</b> b, c, d, e, f, g, h, i, j, k, l, m, n, o, p\n"
            "<b>All templates share the same layout coords.</b>",
            parse_mode="HTML"
        )
        return

    letter = args[0].lower().strip()
    side   = args[1].lower().strip()

    if letter not in list("bcdefghijklmnop"):
        await update.message.reply_text("❌ Invalid letter. Use b–p (template A already exists).")
        return

    if side not in ("front", "back"):
        await update.message.reply_text("❌ Side must be <code>front</code> or <code>back</code>.", parse_mode="HTML")
        return

    # Get the document — from this message or a replied-to message
    doc = None
    if update.message.document:
        doc = update.message.document
    elif update.message.reply_to_message and update.message.reply_to_message.document:
        doc = update.message.reply_to_message.document

    if not doc:
        await update.message.reply_text(
            "📎 Please attach a <code>.tif</code> file to your message,\n"
            "or reply to a document with this command.",
            parse_mode="HTML"
        )
        return

    filename = (doc.file_name or "").lower()
    if not (filename.endswith(".tif") or filename.endswith(".tiff")):
        await update.message.reply_text("❌ File must be a <code>.tif</code> or <code>.tiff</code> file.", parse_mode="HTML")
        return

    await update.message.reply_text(f"⏳ Saving {side} template {letter.upper()}...")

    dest_name = f"front_{letter}.tif" if side == "front" else f"back_{letter}.tif"
    dest_path = os.path.join("assets", "templates", dest_name)

    file = await doc.get_file()
    await file.download_to_drive(dest_path)

    # Verify it's a valid image
    try:
        from PIL import Image as _Img
        with _Img.open(dest_path) as _test:
            w, h = _test.size
    except Exception as e:
        os.remove(dest_path)
        await update.message.reply_text(f"❌ Could not open the file as an image: {e}")
        return

    available = get_available_templates()
    await update.message.reply_text(
        f"✅ <b>Template {letter.upper()} {side}</b> saved ({w}×{h}px).\n\n"
        f"📋 Available templates now: {', '.join(t.upper() for t in available)}\n\n"
        f"Users can select it via <b>Choose Template</b> in the menu.",
        parse_mode="HTML"
    )


# =========================================================
# REMOVETEMPLATE COMMAND (admin only)
# Usage: /removetemplate b
# Deletes front_b.tif and back_b.tif, reassigns users to template A
# =========================================================
async def removetemplate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.message.from_user
    access.register(u.id, u.username)

    if not access.is_admin(u.id, u.username):
        await update.message.reply_text("❌ Admin only.")
        return

    args = context.args
    if not args:
        await update.message.reply_text(
            "Usage: <code>/removetemplate b</code>",
            parse_mode="HTML"
        )
        return

    letter = args[0].lower().strip()
    if letter == "a":
        await update.message.reply_text("❌ Cannot remove Template A — it is the default.")
        return
    if letter not in list("bcdefghijklmnop"):
        await update.message.reply_text("❌ Invalid letter. Use b–p.")
        return

    front_path = os.path.join("assets", "templates", f"front_{letter}.tif")
    back_path  = os.path.join("assets", "templates", f"back_{letter}.tif")

    removed = []
    if os.path.exists(front_path):
        os.remove(front_path)
        removed.append(f"front_{letter}.tif")
    if os.path.exists(back_path):
        os.remove(back_path)
        removed.append(f"back_{letter}.tif")

    if not removed:
        await update.message.reply_text(f"⚠️ Template {letter.upper()} files not found — nothing to remove.")
        return

    # Reassign any users on this template back to A
    affected = access.reassign_template(letter, "a")

    available = get_available_templates()
    await update.message.reply_text(
        f"🗑️ <b>Template {letter.upper()} removed.</b>\n\n"
        f"Deleted: {', '.join(removed)}\n"
        f"Users reassigned to Template A: <b>{affected}</b>\n\n"
        f"Remaining templates: {', '.join(t.upper() for t in available)}",
        parse_mode="HTML"
    )


# =========================================================
# SHARED HELPERS
# =========================================================
async def _send_points(reply_fn, user_id: int):
    pts = access.get_points(user_id)
    if pts is None:
        await reply_fn("♾ You have <b>unlimited</b> access.", parse_mode="HTML")
    elif pts == 0:
        await reply_fn(
            "❌ You have <b>0 points</b>.\n\nUse the button below to request access.",
            parse_mode="HTML",
            reply_markup=no_access_keyboard()
        )
    else:
        await reply_fn(
            f"🎯 You have <b>{pts}</b> point(s) remaining.\n"
            f"Each front+back generation uses 1 point.",
            parse_mode="HTML"
        )


async def _do_request_access(reply_fn, context, u):
    pts = access.get_points(u.id)

    if pts is None:
        await reply_fn("✅ You already have unlimited access.")
        return

    if pts > 0:
        await reply_fn(
            f"✅ You already have <b>{pts}</b> point(s).",
            parse_mode="HTML"
        )
        return

    # Show point picker — actual request sent when user picks an amount
    await reply_fn(
        "📨 <b>Request Access</b>\n\n"
        "How many points are you requesting?\n"
        "Tap an amount below:",
        parse_mode="HTML",
        reply_markup=request_picker_keyboard()
    )


async def _send_access_request(context, u, requested_pts: int):
    """Send the actual access request to all admins after user picked an amount."""
    uname = u.username or str(u.id)
    print(f"🔔 ACCESS REQUEST from @{uname} (id={u.id}) — {requested_pts} pts")

    admin_ids = access.get_admin_ids()
    for admin_id in admin_ids:
        try:
            await context.bot.send_message(
                chat_id=admin_id,
                text=(
                    f"🔔 <b>New access request</b>\n\n"
                    f"User: @{uname}\n"
                    f"ID: <code>{u.id}</code>\n"
                    f"Requested: <b>{requested_pts} points</b>"
                ),
                parse_mode="HTML",
                reply_markup=admin_grant_keyboard(u.id, uname)
            )
        except Exception as e:
            print(f"⚠️ Could not notify admin {admin_id}: {e}")


# =========================================================
# INLINE BUTTON CALLBACK HANDLER
# =========================================================
async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    u = query.from_user
    access.register(u.id, u.username)
    data = query.data

    # -------------------------------------------------------
    # HELP
    # -------------------------------------------------------
    if data == "help":
        await query.message.reply_text(
            HELP_TEXT,
            parse_mode="HTML",
            reply_markup=reset_keyboard()
        )

    # -------------------------------------------------------
    # MY POINTS
    # -------------------------------------------------------
    elif data == "mypoints":
        await _send_points(query.message.reply_text, u.id)

    # -------------------------------------------------------
    # REQUEST ACCESS — show point picker
    # -------------------------------------------------------
    elif data == "request_access":
        await _do_request_access(query.message.reply_text, context, u)

    # -------------------------------------------------------
    # REQUEST POINTS — user picked an amount
    # request_pts_{n}
    # -------------------------------------------------------
    elif data.startswith("request_pts_"):
        pts_check = access.get_points(u.id)
        if pts_check is None:
            await query.message.reply_text("✅ You already have unlimited access.")
            return
        if pts_check > 0:
            await query.message.reply_text(
                f"✅ You already have <b>{pts_check}</b> point(s).",
                parse_mode="HTML"
            )
            return

        requested = int(data.split("_")[2])
        await query.message.reply_text(
            f"📨 Your request for <b>{requested} points</b> has been sent to the admin.\n"
            "You will be notified once access is granted.",
            parse_mode="HTML"
        )
        await _send_access_request(context, u, requested)

    # -------------------------------------------------------
    # CHOOSE TEMPLATE — show preview + picker
    # -------------------------------------------------------
    elif data == "choose_template":
        if not access.has_access(u.id):
            await query.message.reply_text("🔒 You need access to choose a template.")
            return
        current_tid = access.get_template(u.id)
        front_path = "assets/template_sample_front.jpg"
        back_path  = "assets/template_sample_back.jpg"

        header = (
            "🎨 <b>Choose a Template</b>\n\n"
            f"Current: <b>Template {current_tid.upper()}</b>\n"
            "Tap a template below to set it as your default:"
        )

        media = []
        open_files = []
        if os.path.exists(front_path):
            fh = open(front_path, "rb")
            open_files.append(fh)
            media.append(InputMediaPhoto(media=fh, caption="🪪 Front side"))
        if os.path.exists(back_path):
            bh = open(back_path, "rb")
            open_files.append(bh)
            media.append(InputMediaPhoto(media=bh, caption="🪪 Back side"))

        try:
            if media:
                await query.message.reply_media_group(media=media)
            await query.message.reply_text(
                header,
                parse_mode="HTML",
                reply_markup=template_keyboard(current_tid)
            )
        finally:
            for fh in open_files:
                fh.close()

    # -------------------------------------------------------
    # TEMPLATE SELECTION
    # select_template_{tid}
    # -------------------------------------------------------
    elif data.startswith("select_template_"):
        tid     = data.split("_", 2)[2]
        session = user_sessions.get(u.id)

        # Save as user's default regardless of session state
        access.set_template(u.id, tid)

        # If mid-flow waiting for template, continue the flow
        if session and session.get("step") == "waiting_template":
            session["template_id"] = tid
            session["step"]        = "waiting_face"
            await query.message.reply_text(
                f"✅ <b>Template {tid.upper()}</b> selected.\n\n"
                "📸 Now send the <b>face photo</b> of the person.",
                parse_mode="HTML",
                reply_markup=reset_keyboard()
            )
        else:
            # Standalone selection from main menu
            await query.message.reply_text(
                f"✅ <b>Template {tid.upper()}</b> is now your default template.\n\n"
                "It will be used automatically for all future generations.",
                parse_mode="HTML"
            )

    # -------------------------------------------------------
    # BACK TO MENU
    # -------------------------------------------------------
    elif data == "back_to_menu":
        pts = access.get_points(u.id)
        has_acc = access.has_access(u.id)
        await query.message.reply_text(
            WELCOME_TEXT,
            parse_mode="HTML",
            reply_markup=main_menu_keyboard(has_acc, pts)
        )

    # -------------------------------------------------------
    # RESET SESSION
    # -------------------------------------------------------
    elif data == "reset" or data == "start_again":
        user_sessions.pop(u.id, None)
        processing_users.discard(u.id)
        await query.message.reply_text(
            "🔄 Session reset.\n\n"
            "📸 Send a <b>FRONT</b> ID screenshot to begin.",
            parse_mode="HTML"
        )

    # -------------------------------------------------------
    # ADMIN: GRANT
    # admin_grant_{user_id}_{amount}
    # -------------------------------------------------------
    elif data.startswith("admin_grant_"):
        if not access.is_admin(u.id, u.username):
            await query.message.reply_text("❌ Admin only.")
            return

        parts = data.split("_")
        # admin_grant_{user_id}_{amount}  → parts[2]=user_id, parts[3]=amount
        target_uid = parts[2]
        amount     = parts[3]

        target_data = access._load()
        target_user = target_data["users"].get(target_uid, {})
        target_uname = target_user.get("username") or target_uid

        result = access.grant(target_uname, amount)

        # edit the original request message to show it's been handled
        try:
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text(f"✅ Done — {result}")
        except Exception:
            await query.message.reply_text(f"✅ Done — {result}")

        # notify the user
        try:
            if amount == "unlimited":
                user_msg = "🎉 Your access has been <b>granted</b> — you now have <b>unlimited</b> access!\n\nSend a FRONT ID screenshot to begin."
            else:
                user_msg = f"🎉 Your access has been <b>granted</b> — you now have <b>{amount}</b> point(s)!\n\nSend a FRONT ID screenshot to begin."
            await context.bot.send_message(
                chat_id=int(target_uid),
                text=user_msg,
                parse_mode="HTML"
            )
        except Exception as e:
            print(f"⚠️ Could not notify user {target_uid}: {e}")

    # -------------------------------------------------------
    # ADMIN: DENY
    # admin_deny_{user_id}_{username}
    # -------------------------------------------------------
    elif data.startswith("admin_deny_"):
        if not access.is_admin(u.id, u.username):
            await query.message.reply_text("❌ Admin only.")
            return

        parts = data.split("_", 3)
        target_uid  = parts[2]
        target_uname = parts[3] if len(parts) > 3 else target_uid

        try:
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text(f"❌ Access denied for @{target_uname}.")
        except Exception:
            await query.message.reply_text(f"❌ Access denied for @{target_uname}.")

        try:
            await context.bot.send_message(
                chat_id=int(target_uid),
                text=(
                    "❌ Your access request was not approved at this time.\n\n"
                    "Contact the admin or try again later."
                )
            )
        except Exception as e:
            print(f"⚠️ Could not notify user {target_uid}: {e}")


# =========================================================
# STATE TRACKING
# =========================================================
processing_users = set()
processed_files  = {}
SESSION_TIMEOUT  = 1200

# Limit concurrent heavy CPU jobs (OCR + image generation)
# so the event loop stays responsive under multi-user load.
_heavy_sem = asyncio.Semaphore(3)


# =========================================================
# SAFE REPLY
# =========================================================
async def safe_reply(method, *args, **kwargs):
    try:
        return await method(*args, **kwargs)
    except Exception as e:
        print("⚠️ Reply failed:", e)
        return None


# =========================================================
# DOWNLOAD WITH RETRY
# Retries the Telegram file download up to 3 times with
# exponential backoff so a transient network blip doesn't
# fail the whole step.
# =========================================================
async def download_with_retry(tg_file, dest_path, attempts=3):
    for i in range(attempts):
        try:
            await tg_file.download_to_drive(dest_path)
            return
        except (TimedOut, NetworkError) as e:
            print(f"⚠️ Download attempt {i+1} failed: {e}")
            if i < attempts - 1:
                await asyncio.sleep(2 ** i)
    raise RuntimeError("Download failed after retries")


# =========================================================
# SEND DOCUMENT WITH RETRY
# Retries sending the output TIFF up to 3 times. The file
# is only deleted after a confirmed successful send, or
# after all retries are exhausted (caller handles fallback).
# Returns True on success, False on failure.
# =========================================================
async def send_document_with_retry(reply_fn, file_path, attempts=3):
    for i in range(attempts):
        try:
            with open(file_path, "rb") as f:
                await reply_fn(document=f)
            return True
        except (TimedOut, NetworkError) as e:
            print(f"⚠️ Send attempt {i+1} failed: {e}")
            if i < attempts - 1:
                await asyncio.sleep(2 ** i)
        except Exception as e:
            print(f"⚠️ Send failed: {e}")
            break
    return False


# =========================================================
# CLEAR EXPIRED SESSION
# =========================================================
def clear_expired_session(user_id):
    session = user_sessions.get(user_id)
    if not session:
        return
    if time.time() - session.get("created_at", 0) > SESSION_TIMEOUT:
        print(f"🧹 Expired session cleared: {user_id}")
        user_sessions.pop(user_id, None)


# =========================================================
# IMAGE HANDLER
# =========================================================
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_id = None

    try:

        user_id = update.message.from_user.id
        u = update.message.from_user
        access.register(user_id, u.username)

        clear_expired_session(user_id)

        # -----------------------------------------------
        # GLOBAL PROCESS LOCK
        # -----------------------------------------------
        if user_id in processing_users:
            await update.message.reply_text(
                "⏳ Previous request still processing..."
            )
            return

        processing_users.add(user_id)

        photo = update.message.photo[-1]
        file  = await photo.get_file()

        os.makedirs("temp", exist_ok=True)
        user_temp = f"temp/{user_id}"
        os.makedirs(user_temp, exist_ok=True)

        telegram_file_id    = photo.file_unique_id
        telegram_dl_file_id = photo.file_id
        cache_key = f"{user_id}_{telegram_file_id}"

        if cache_key in processed_files:
            await safe_reply(
                update.message.reply_text,
                "⚠️ This image was already uploaded."
            )
            return

        session = user_sessions.get(user_id)

        if session:
            if session.get("last_file_id") == telegram_file_id:
                await update.message.reply_text("⚠️ This image was already uploaded.")
                return

        job_id    = str(uuid.uuid4())
        file_path = f"{user_temp}/{job_id}.jpg"
        await download_with_retry(file, file_path)

        # =====================================================
        # STEP 1 → FRONT OCR
        # =====================================================
        if not session:

            if not access.has_access(user_id):
                await update.message.reply_text(
                    "🔒 You don't have access to this bot.\n\n"
                    "Tap the button below to request access from the admin.",
                    reply_markup=no_access_keyboard()
                )
                return

            await update.message.reply_text("🔍 Processing front ID...")

            async with _heavy_sem:
                front_data = await asyncio.to_thread(
                    process_ocr,
                    file_path,
                    False,
                    user_temp
                )

            if front_data.get("problems"):

                problems = front_data["problems"]
                issues   = front_data.get("issues", [])

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
                    msg,
                    reply_markup=reset_keyboard()
                )
                return

            try:
                os.remove(file_path)
            except OSError:
                pass

            processed_files[cache_key] = time.time()

            # Use user's saved default template (set via Choose Template button)
            saved_template = access.get_template(user_id)

            user_sessions[user_id] = {
                "step": "waiting_face",
                "front_data": front_data,
                "job_id": job_id,
                "last_file_id": telegram_file_id,
                "last_tg_dl_file_id": telegram_dl_file_id,
                "last_attempted": "front",
                "created_at": time.time(),
                "front_processing": False,
                "back_processing": False,
                "template_id": saved_template,
                "front_file_unique_id": telegram_file_id,
            }

            await update.message.reply_text(
                f"✅ Front data extracted.\n\n"
                f"📸 Now send the <b>face photo</b> of the person.\n"
                f"<i>Template {saved_template.upper()} will be used. Change it anytime from the menu.</i>",
                parse_mode="HTML",
                reply_markup=reset_keyboard()
            )
            return

        # =====================================================
        # EXISTING SESSION
        # =====================================================
        session = user_sessions[user_id]

        # =====================================================
        # WAITING FOR TEMPLATE SELECTION (legacy fallback)
        # =====================================================
        if session["step"] == "waiting_template":
            current_tid = access.get_template(user_id)
            await safe_reply(
                update.message.reply_text,
                "📋 Please choose a template first:",
                reply_markup=template_keyboard(current_tid)
            )
            return

        # =====================================================
        # STEP 2 → FACE
        # =====================================================
        if session["step"] == "waiting_face":

            if session.get("front_processing"):
                await safe_reply(
                    update.message.reply_text,
                    "⏳ Face processing already running..."
                )
                return

            session["front_processing"] = True
            session["step"] = "generating_front"
            session["last_file_id"] = telegram_file_id
            session["last_tg_dl_file_id"] = telegram_dl_file_id
            session["last_attempted"] = "face"

            await safe_reply(update.message.reply_text, "🧑 Processing face...")

            front_data   = session["front_data"]
            front_output = f"temp/{user_id}/front_{session['job_id']}.tif"

            async with _heavy_sem:
                front_path = await asyncio.to_thread(
                    generate_id,
                    front_data,
                    file_path,
                    front_output,
                    user_temp,
                    session.get("template_id", "a")
                )

            try:
                os.remove(file_path)
            except OSError:
                pass

            if not front_path:
                session["front_processing"] = False
                session["step"] = "waiting_face"
                await safe_reply(
                    update.message.reply_text,
                    "❌ No human face detected.\n\n"
                    "📸 Please upload a clear face photo.",
                    reply_markup=reset_keyboard()
                )
                return

            processed_files[cache_key] = time.time()
            session["front_generated"]  = True
            session["front_processing"] = False

            front_fuid = session.get("front_file_unique_id")
            if _last_front_file_ids.get(user_id) != front_fuid:
                access.deduct_point(user_id, 0.5)
                _last_front_file_ids[user_id] = front_fuid

            sent = await send_document_with_retry(
                update.message.reply_document, front_path
            )
            try:
                os.remove(front_path)
            except OSError:
                pass
            if not sent:
                await safe_reply(
                    update.message.reply_text,
                    "⚠️ Front ID was generated but couldn't be sent due to a network error.\n\n"
                    "Use /retry to resend it.",
                    reply_markup=reset_keyboard()
                )
                return

            session["step"] = "waiting_back"

            pts_after = access.get_points(user_id)
            front_msg = "✅ Front ID generated.\n\n📸 Now send the <b>BACK</b> screenshot of the ID card."
            if pts_after is not None and 0 < pts_after <= 1:
                pts_fmt = int(pts_after) if pts_after == int(pts_after) else pts_after
                front_msg += f"\n\n⚠️ <b>Low points warning:</b> You only have <b>{pts_fmt}</b> point(s) left. Request more soon to avoid being blocked."

            await safe_reply(
                update.message.reply_text,
                front_msg,
                parse_mode="HTML",
                reply_markup=reset_keyboard()
            )
            return

        # =====================================================
        # ALREADY GENERATING FRONT
        # =====================================================
        if session["step"] == "generating_front":
            await update.message.reply_text("⏳ Front generation already in progress...")
            return

        # =====================================================
        # BACK ALREADY FINISHED
        # =====================================================
        if session.get("back_generated"):
            await safe_reply(
                update.message.reply_text,
                "⚠️ Process already completed.\n\n"
                "Tap below to start a new generation.",
                reply_markup=done_keyboard()
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

            pts_now = access.get_points(user_id)
            if pts_now is not None and pts_now < 0.5:
                await safe_reply(
                    update.message.reply_text,
                    "⚠️ You don't have enough points to generate the Back ID.\n\n"
                    "You need at least <b>0.5 points</b>. Use the button below to request more.",
                    parse_mode="HTML",
                    reply_markup=no_access_keyboard()
                )
                return

            session["back_processing"] = True
            session["step"] = "generating_back"
            session["last_file_id"] = telegram_file_id
            session["last_tg_dl_file_id"] = telegram_dl_file_id
            session["last_attempted"] = "back"

            await safe_reply(update.message.reply_text, "🔍 Processing back side...")

            async with _heavy_sem:
                back_data, qr_crop = await asyncio.to_thread(
                    process_back_ocr,
                    file_path,
                    True,
                    user_temp
                )

            if (
                not back_data
                or qr_crop is None
                or back_data.get("problems")
            ):
                session["back_processing"] = False
                session["step"] = "waiting_back"

                msg = "❌ BACK validation failed.\n\n"
                problems = back_data.get("problems", []) if back_data else []
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
                    msg,
                    reply_markup=reset_keyboard()
                )
                return

            try:
                os.remove(file_path)
            except OSError:
                pass

            front_data  = session["front_data"]
            processed_files[cache_key] = time.time()

            person_name = front_data.get("name_en", "unknown")
            back_output = f"temp/{user_id}/back_{session['job_id']}.tif"

            async with _heavy_sem:
                back_path = await asyncio.to_thread(
                    generate_back,
                    back_data,
                    qr_crop,
                    back_output,
                    person_name,
                    session.get("template_id", "a")
                )

            session["back_generated"]  = True
            session["back_processing"] = False
            access.deduct_point(user_id, 0.5)

            sent = await send_document_with_retry(
                update.message.reply_document, back_path
            )
            try:
                os.remove(back_path)
            except OSError:
                pass
            if not sent:
                await safe_reply(
                    update.message.reply_text,
                    "⚠️ Back ID was generated but couldn't be sent due to a network error.\n\n"
                    "Use /retry to resend it.",
                    reply_markup=reset_keyboard()
                )
                return

            user_sessions.pop(user_id, None)

            pts = access.get_points(user_id)
            if pts is None:
                pts_line = "♾ Unlimited access remaining."
            elif pts == 0:
                pts_line = "⚠️ You have <b>0 points</b> left. Use the button below to request more."
            elif pts <= 1:
                pts_fmt = int(pts) if pts == int(pts) else pts
                pts_line = f"⚠️ <b>Low points:</b> Only <b>{pts_fmt}</b> point(s) left — enough for one more generation. Request more soon."
            else:
                pts_fmt = int(pts) if pts == int(pts) else pts
                pts_line = f"🎯 You have <b>{pts_fmt}</b> point(s) remaining."

            await safe_reply(
                update.message.reply_text,
                f"✅ Back ID generated. All done!\n\n{pts_line}",
                parse_mode="HTML",
                reply_markup=done_keyboard()
            )
            return

        # =====================================================
        # FRONT ALREADY FINISHED
        # =====================================================
        if session.get("front_generated") and session["step"] != "waiting_back":
            await safe_reply(
                update.message.reply_text,
                "⚠️ Front already generated.\n\n"
                "📸 Please send the <b>BACK</b> ID screenshot.",
                parse_mode="HTML",
                reply_markup=reset_keyboard()
            )
            return

    # =========================================================
    # ERRORS
    # =========================================================
    except TimedOut:
        print("❌ Timeout")
        await safe_reply(
            update.message.reply_text,
            "⚠️ Network timeout. Use /retry to try the last step again without re-uploading."
        )

    except NetworkError:
        print("❌ Network Error")
        await safe_reply(
            update.message.reply_text,
            "⚠️ Network error. Use /retry to try the last step again without re-uploading."
        )

    except Exception as e:
        print("❌ ERROR:", e)
        await safe_reply(
            update.message.reply_text,
            "⚠️ Something went wrong. Use /retry to try again, or tap below to reset.",
            reply_markup=reset_keyboard()
        )

    finally:
        if user_id is not None:
            session = user_sessions.get(user_id)
            if session:
                session["front_processing"] = False
                session["back_processing"] = False
            now = time.time()
            for key, ts in list(processed_files.items()):
                if now - ts > 120:
                    processed_files.pop(key, None)
            processing_users.discard(user_id)


# =========================================================
# MAIN
# =========================================================
def main():

    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .connect_timeout(60.0)
        .read_timeout(60.0)
        .write_timeout(60.0)
        .pool_timeout(60.0)
        .build()
    )

    app.add_handler(CommandHandler("start",    start))
    app.add_handler(CommandHandler("help",     help_command))
    app.add_handler(CommandHandler("reset",    reset_command))
    app.add_handler(CommandHandler("retry",    retry_command))
    app.add_handler(CommandHandler("request",  request_command))
    app.add_handler(CommandHandler("mypoints", mypoints_command))
    app.add_handler(CommandHandler("grant",    grant_command))
    app.add_handler(CommandHandler("revoke",   revoke_command))
    app.add_handler(CommandHandler("users",      users_command))
    app.add_handler(CommandHandler("status",     status_command))
    app.add_handler(CommandHandler("broadcast",  broadcast_command))
    app.add_handler(CommandHandler("setpreview",     setpreview_command))
    app.add_handler(CommandHandler("addtemplate",   addtemplate_command))
    app.add_handler(CommandHandler("removetemplate", removetemplate_command))

    app.add_handler(CallbackQueryHandler(handle_callback))

    app.add_handler(
        MessageHandler(filters.PHOTO, handle_image)
    )

    async def _cleanup_job(context):
        cleanup_old_dirs(30)

    async def _sweep_sessions(context):
        now = time.time()
        expired = [
            uid for uid, s in list(user_sessions.items())
            if now - s.get("created_at", 0) > SESSION_TIMEOUT
        ]
        for uid in expired:
            user_sessions.pop(uid, None)
            processing_users.discard(uid)
        if expired:
            print(f"🧹 Swept {len(expired)} expired session(s)")

    async def _sweep_processed_files(context):
        now = time.time()
        stale = [k for k, ts in list(processed_files.items()) if now - ts > 120]
        for k in stale:
            processed_files.pop(k, None)
        if stale:
            print(f"🧹 Swept {len(stale)} stale processed_files entries")

    app.job_queue.run_repeating(
        _cleanup_job,
        interval=600,
        first=600
    )

    app.job_queue.run_repeating(
        _sweep_sessions,
        interval=SESSION_TIMEOUT,
        first=SESSION_TIMEOUT
    )

    app.job_queue.run_repeating(
        _sweep_processed_files,
        interval=120,
        first=120
    )

    print("🚀 Initializing OCR and segmentation models...")
    print("✅ Bot running...")

    app.run_polling()


if __name__ == "__main__":
    main()
