from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
import json
import re
from rembg import remove, new_session
from datetime import datetime, timedelta
from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    ImageStat,
    ImageEnhance,
    ImageFilter,
    ImageOps
)

# =========================================================
# GPU SESSION (ONNXRUNTIME GPU)
# =========================================================
session = new_session(
    "u2net_human_seg",
    providers=["CPUExecutionProvider"]
)

# =========================================================
# MODEL WARMUP
# prevents first-request freeze / timeout
# =========================================================
try:

    print("🔥 Warming up U2NET model...")

    warmup_img = Image.new("RGB", (512, 512), "white")

    remove(
        warmup_img,
        session=session
    )

    print("✅ U2NET warmup completed")

except Exception as e:

    print("⚠️ Warmup failed:", e)
    
# =========================================================
# PATH HELPERS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def path(*args):
    return os.path.join(BASE_DIR, "..", *args)


# =========================================================
# DATE HELPER
# =========================================================
def calculate_issue_date(date_str):

    if not date_str:
        return ""

    try:
        if any(c.isalpha() for c in date_str):
            # Format: YYYY/Mon/DD  (Gregorian with alpha month)
            parts = date_str.split("/")
            if len(parts) != 3:
                return date_str

            year, mon, day = parts
            MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"]
            mon_cap = mon.capitalize()
            if mon_cap not in MONTH_NAMES:
                return date_str
            month_num = MONTH_NAMES.index(mon_cap) + 1
            dt = datetime(int(year), month_num, int(day))
            dt = dt.replace(year=dt.year - 8) + timedelta(days=2)
            abbrev = MONTH_NAMES[dt.month - 1]
            return f"{dt.year}/{abbrev}/{dt.day:02d}"

        elif re.match(r"^\d{2}/\d{2}/\d{4}$", date_str):
            # Format: DD/MM/YYYY  (Ethiopian calendar)
            dd, mm, yyyy = date_str.split("/")
            new_day  = int(dd) + 2
            new_year = int(yyyy) - 8
            # Ethiopian months have 30 days (month 13 has 5 or 6)
            max_day = 5 if int(mm) == 13 else 30
            if new_day > max_day:
                new_day -= max_day
                new_month = int(mm) + 1
                if new_month > 13:
                    new_month = 1
                    new_year += 1
            else:
                new_month = int(mm)
            return f"{new_day:02d}/{new_month:02d}/{new_year}"

        else:
            dt = datetime.strptime(date_str, "%Y/%m/%d")

            try:
                dt = dt.replace(year=dt.year - 8)
            except ValueError:
                dt = dt.replace(month=2, day=28, year=dt.year - 8)

            dt += timedelta(days=2)

            return dt.strftime("%Y/%m/%d")

    except Exception as e:
        print("⚠️ Date error:", e)
        return date_str


# =========================================================
# VERTICAL TEXT
# =========================================================
def draw_vertical_text(base_img, text, cfg, font_path):

    if not text:
        return

    font_size = cfg.get("size", 28)

    x = int(cfg.get("x", 20))
    y = int(cfg.get("y", 50))

    anchor = cfg.get("anchor", "top")

    font = ImageFont.truetype(
        font_path,
        font_size
    )

    bold = float(cfg.get("bold", 1))

    offsets = [(0, 0)]

    if bold >= 1.1:
        offsets.append((0.3, 0))

    if bold >= 1.2:
        offsets.append((0.6, 0))

    if bold >= 1.3:
        offsets.append((0.9, 0))

    if bold >= 1.5:
        offsets.append((1, 0))

    if bold >= 1.8:
        offsets.append((0, 1))

    if bold >= 2:
        offsets.append((1, 1))

    dummy = Image.new("RGBA", (1, 1))

    d = ImageDraw.Draw(dummy)

    bbox = d.textbbox(
        (0, 0),
        text,
        font=font
    )

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    pad = 20

    txt_img = Image.new(
        "RGBA",
        (w + pad * 2, h + pad * 2),
        (0, 0, 0, 0)
    )

    d = ImageDraw.Draw(txt_img)

    for dx, dy in offsets:

        d.text(
            (pad + dx, pad + dy),
            text,
            font=font,
            fill=(0, 0, 0)
        )

    rotated = txt_img.rotate(
        90,
        expand=True
    )

    rw, rh = rotated.size

    if anchor == "top":
        final_y = y

    elif anchor == "center":
        final_y = y - rh // 2

    elif anchor == "bottom":
        final_y = y - rh

    else:
        final_y = y

    base_img.paste(
        rotated,
        (x, int(final_y)),
        rotated
    )

# =========================================================
# FACE EXTRACTION + VALIDATION
# =========================================================
def _run_detector(detector, gray_img, scale=1.1, neighbors=5, min_sz=80):
    faces = detector.detectMultiScale(
        gray_img,
        scaleFactor=scale,
        minNeighbors=neighbors,
        minSize=(min_sz, min_sz)
    )
    return faces if len(faces) > 0 else []


def extract_face(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print("❌ Image not loaded")
        return None

    h, w = img.shape[:2]

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    # =====================================================
    # ATTEMPT 1 — fixed upper-centre region, strict params
    # =====================================================
    x1 = int(w * 0.20)
    y1 = int(h * 0.08)
    x2 = int(w * 0.80)
    y2 = int(h * 0.55)
    region = img[y1:y2, x1:x2]
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    faces = _run_detector(detector, gray_region, scale=1.1, neighbors=5, min_sz=60)
    source_img, offset_x, offset_y = region, x1, y1

    # =====================================================
    # ATTEMPT 2 — full image, strict params
    # =====================================================
    if len(faces) == 0:
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = _run_detector(detector, gray_full, scale=1.1, neighbors=5, min_sz=60)
        source_img, offset_x, offset_y = img, 0, 0

    # =====================================================
    # ATTEMPT 3 — full image, lenient params
    # =====================================================
    if len(faces) == 0:
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = _run_detector(detector, gray_full, scale=1.05, neighbors=3, min_sz=30)
        source_img, offset_x, offset_y = img, 0, 0

    # =====================================================
    # ATTEMPT 4 — centre crop fallback
    # user confirmed it's a face photo; use centre portion
    # =====================================================
    if len(faces) == 0:
        print("⚠️ Haar cascade failed — using centre crop fallback")
        cx1 = int(w * 0.15)
        cy1 = int(h * 0.05)
        cx2 = int(w * 0.85)
        cy2 = int(h * 0.75)
        fallback = img[cy1:cy2, cx1:cx2]
        final_face = cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(final_face).convert("RGBA")
        est = int(min(fallback.shape[0], fallback.shape[1]) * 0.5)
        return (pil, est, est)

    # =====================================================
    # USE LARGEST DETECTED FACE
    # =====================================================
    largest = max(faces, key=lambda f: f[2] * f[3])
    fx, fy, fw, fh = largest

    pad_left  = int(fw * 0.45)
    pad_right = int(fw * 0.45)
    pad_top   = int(fh * 0.60)
    # Bottom: face-relative cap — at most 1.1× face-height below the chin.
    # This scales with face size so it captures neck + chest without ever
    # reaching the white gap or QR code that appear below the face photo
    # on the ID card screenshot.
    pad_bottom = int(fh * 0.75)

    # Always translate coordinates back to the FULL image before clamping.
    full_fx = fx + offset_x
    full_fy = fy + offset_y

    sx1 = max(0, full_fx - pad_left)
    sy1 = max(0, full_fy - pad_top)
    sx2 = min(img.shape[1], full_fx + fw + pad_right)
    # Hard cap at 46 % of image height — keeps crop above white gap and QR code
    qr_safe_limit = int(img.shape[0] * 0.46)
    sy2 = min(qr_safe_limit, full_fy + fh + pad_bottom)

    final_face = img[sy1:sy2, sx1:sx2]
    final_face = cv2.cvtColor(final_face, cv2.COLOR_BGR2RGB)

    return (
        Image.fromarray(final_face).convert("RGBA"),
        fw,
        fh
    )

# =========================================================
# BG REMOVAL (GPU ACCELERATED)
# =========================================================
def remove_background(image):

    # Original RGB — correct colours everywhere (used for GrabCut and output)
    orig_np = np.array(image)[:, :, :3]

    # =====================================================
    # STEP 1 — u2net initial mask
    # =====================================================
    output   = remove(image, session=session).convert("RGBA")
    alpha_np = np.array(output)[:, :, 3]

    # =====================================================
    # STEP 2 — GrabCut refinement
    #
    # u2net mis-labels warm-coloured clothing (e.g. yellow
    # hijab) against white backgrounds as background.
    # GrabCut builds an explicit GMM colour model of
    # foreground vs background using the u2net mask as a
    # hint, then re-segments by colour — correctly pulling
    # those missed clothing pixels back into the foreground.
    #
    # GrabCut label map seeded from u2net alpha:
    #   alpha > 200  → GC_FGD     (definite foreground)
    #   alpha >  20  → GC_PR_FGD  (probable foreground)
    #   alpha <=  5  → GC_BGD     (definite background)
    #   otherwise    → GC_PR_BGD  (probable background)
    # =====================================================
    img_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)

    gc_mask = np.full(alpha_np.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[alpha_np <= 5]   = cv2.GC_BGD
    gc_mask[alpha_np > 20]   = cv2.GC_PR_FGD
    gc_mask[alpha_np > 200]  = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            img_bgr, gc_mask, None,
            bgd_model, fgd_model,
            5, cv2.GC_INIT_WITH_MASK
        )
        refined = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            np.uint8(255), np.uint8(0)
        )
    except Exception:
        # Fall back to plain u2net binarised mask if GrabCut fails
        _, refined = cv2.threshold(alpha_np, 20, 255, cv2.THRESH_BINARY)

    # =====================================================
    # STEP 3 — ALPHA CLEANUP  (border fix)
    #
    #   1  Erode inward   — pull edge inside the person
    #   2  Gaussian blur  — feather edge back outward
    #
    # Smaller kernel (5×5 instead of 11×11) preserves fine
    # edge detail (hair, ears) for a higher-quality cutout.
    # =====================================================
    k_erode      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha_eroded = cv2.erode(
        refined, k_erode,
        borderType=cv2.BORDER_REPLICATE
    )
    alpha_feathered = cv2.GaussianBlur(
        alpha_eroded, (5, 5), 1,
        borderType=cv2.BORDER_REPLICATE
    )

    result_np = np.dstack([orig_np, alpha_feathered.astype(np.uint8)])
    return Image.fromarray(result_np, "RGBA")

# =========================================================
# TEXT HELPERS
# =========================================================
def draw_text_box(
    draw,
    text,
    box,
    font_path,
    size,
    spacing
):

    if not text:
        return

    font = ImageFont.truetype(font_path, size)

    x, y, w, h = (
        box["x"],
        box["y"],
        box["w"],
        box["h"]
    )

    bold = float(box.get("bold", 1))

    bbox = draw.textbbox(
        (0, 0),
        text,
        font=font
    )

    tx = x + 5
    ty = y + (h - (bbox[3] - bbox[1])) // 2

    offsets = [(0, 0)]

    if bold >= 1.1:
        offsets.append((0.3, 0))

    if bold >= 1.2:
        offsets.append((0.6, 0))

    if bold >= 1.3:
        offsets.append((0.9, 0))

    if bold >= 1.5:
        offsets.append((1, 0))

    if bold >= 1.8:
        offsets.append((0, 1))

    if bold >= 2:
        offsets.append((1, 1))

    for dx, dy in offsets:

        draw.text(
            (tx + dx, ty + dy),
            text,
            font=font,
            fill=(0, 0, 0)
        )


def draw_mixed_text(
    draw,
    box,
    left_text,
    right_text,
    font_left_path,
    font_right_path,
    size_left=28,
    size_right=28
):

    if not left_text and not right_text:
        return

    x, y, w, h = (
        box["x"],
        box["y"],
        box["w"],
        box["h"]
    )

    font_left = ImageFont.truetype(
        font_left_path,
        size_left
    )

    font_right = ImageFont.truetype(
        font_right_path,
        size_right
    )

    sep = " | "

    left_w = (
        draw.textbbox(
            (0, 0),
            left_text,
            font=font_left
        )[2]
        if left_text else 0
    )

    sep_w = draw.textbbox(
        (0, 0),
        sep,
        font=font_right
    )[2]

    start_x = x + 5
    center_y = y + h // 2 - 12

    bold = float(box.get("bold", 1))

    offsets = [(0, 0)]

    if bold >= 1.1:
        offsets.append((0.3, 0))

    if bold >= 1.2:
        offsets.append((0.6, 0))

    if bold >= 1.3:
        offsets.append((0.9, 0))

    if bold >= 1.5:
        offsets.append((1, 0))

    if bold >= 1.8:
        offsets.append((0, 1))

    if bold >= 2:
        offsets.append((1, 1))

    for dx, dy in offsets:

        if left_text:
            draw.text(
                (start_x + dx, center_y + dy),
                left_text,
                font=font_left,
                fill=(0, 0, 0)
            )

        draw.text(
            (start_x + left_w + dx, center_y + dy),
            sep,
            font=font_right,
            fill=(0, 0, 0)
        )

        if right_text:
            draw.text(
                (
                    start_x + left_w + sep_w + dx,
                    center_y + dy
                ),
                right_text,
                font=font_right,
                fill=(0, 0, 0)
            )


# =========================================================
# PHOTO COPY (CONSISTENT)
# =========================================================
def place_photo_copy(template, face, layout):

    cfg = layout.get("photo_copy")

    if not cfg or face is None:
        return

    canvas_w = cfg["width"]
    canvas_h = cfg["height"]

    margin_r = cfg.get("margin_right", 0)
    margin_b = cfg.get("margin_bottom", 0)

    offset_x = cfg.get("offset_x", 0)
    offset_y = cfg.get("offset_y", 0)

    # =====================================================
    # REMOVE EMPTY TRANSPARENT SPACE
    # =====================================================

    alpha = face.getchannel("A")

    bbox = alpha.getbbox()

    if bbox:
        face = face.crop(bbox)

    fw, fh = face.size

    # =====================================================
    # CONSISTENT SCALE
    # =====================================================

    target_visible_height = int(canvas_h * 0.85)

    scale = target_visible_height / fh

    new_w = int(fw * scale)
    new_h = int(fh * scale)

    face = face.resize(
        (new_w, new_h),
        Image.LANCZOS
    )

    # =====================================================
    # CREATE FIXED CANVAS
    # =====================================================

    photo_canvas = Image.new(
        "RGBA",
        (canvas_w, canvas_h),
        (0, 0, 0, 0)
    )

    # =====================================================
    # CENTER PERSON
    # =====================================================

    paste_x = (canvas_w - new_w) // 2
    paste_y = canvas_h - new_h

    photo_canvas.paste(
        face,
        (paste_x, paste_y),
        face
    )

    # =====================================================
    # PLACE ON TEMPLATE
    # =====================================================

    tx = (
        template.width
        - canvas_w
        - margin_r
        + offset_x
    )

    ty = (
        template.height
        - canvas_h
        - margin_b
        + offset_y
    )

    template.paste(
        photo_canvas,
        (int(tx), int(ty)),
        photo_canvas
    )

# =========================================================
# MAIN
# =========================================================
def generate_id(data, image_path, output_path, debug_dir="temp", template_id="a"):

    # Template A → front.tif / Template B+ → front_b.tif etc.
    # All templates share the same coords.json
    tpl_file    = "front.tif" if template_id == "a" else f"front_{template_id}.tif"
    coords_file = "coords.json"

    template = Image.open(path("assets", "templates", tpl_file))
    template_dpi = template.info.get("dpi", (300, 300))
    draw = ImageDraw.Draw(template)

    with open(path("config", coords_file), "r", encoding="utf-8") as f:
        layout = json.load(f)

    ocr = layout["ocr"]

    font_en = path("assets", "fonts", "english.ttf")
    font_am = path("assets", "fonts", "amharic.ttf")

    dob = f"{data.get('dob_eth','')} | {data.get('dob_greg','')}"
    exp = f"{data.get('exp_eth','')} | {data.get('exp_greg','')}"

    issue_greg = calculate_issue_date(data.get("exp_greg",""))
    issue_eth  = calculate_issue_date(data.get("exp_eth",""))

    draw_text_box(draw, data.get("name_en",""), ocr["name_en"], font_en, 33, 2)
    draw_text_box(draw, data.get("name_am",""), ocr["name_am"], font_am, 35, 1)
    draw_text_box(draw, dob, ocr["dob_greg"], font_en, 30, 0)
    draw_text_box(draw, exp, ocr["exp_greg"], font_en, 30, 0)

    draw_mixed_text(
        draw,
        ocr["gender_am"],
        data.get("gender_am",""),
        data.get("gender_en",""),
        font_am,
        font_en,
        36,
        34
    )

    issue_cfg = layout.get("issue_date", {})

    draw_vertical_text(template, issue_eth, issue_cfg.get("greg", {}), font_en)
    draw_vertical_text(template, issue_greg, issue_cfg.get("eth", {}), font_en)

    # =====================================================
    # FAN IMAGE
    # =====================================================
    fan_img_path = os.path.join(
        debug_dir,
        "debug_fan_crop.jpg"
    )

    if os.path.exists(fan_img_path):

        try:
            crop_img = Image.open(fan_img_path).convert("RGBA")

            fan_cfg = layout.get("fan", {})

            x = fan_cfg.get("x", 460)
            y = fan_cfg.get("y", 500)

            offset_x = fan_cfg.get("offset_x", 0)
            offset_y = fan_cfg.get("offset_y", 0)

            scale = fan_cfg.get("scale", 1.0)

            base_w = fan_cfg.get("w", crop_img.width)
            base_h = fan_cfg.get("h", crop_img.height)

            w = int(base_w * scale)
            h = int(base_h * scale)

            max_w = fan_cfg.get("max_width")
            max_h = fan_cfg.get("max_height")

            if max_w and w > max_w:
                ratio = max_w / w
                w = int(w * ratio)
                h = int(h * ratio)

            if max_h and h > max_h:
                ratio = max_h / h
                w = int(w * ratio)
                h = int(h * ratio)

            crop_img = crop_img.resize((w, h), Image.LANCZOS)

            final_x = int(x + offset_x)
            final_y = int(y + offset_y)

            template.paste(crop_img, (final_x, final_y), crop_img)

        except Exception as e:
            print("⚠️ Failed to place cropped image:", e)

    # =====================================================
    # FACE
    # =====================================================
    result = extract_face(image_path)

    # =====================================================
    # FACE VALIDATION
    # =====================================================
    if not result:

        print("❌ FACE NOT FOUND")

        return None

    face, detected_fw, detected_fh = result

    face = remove_background(face)

    r, g, b, a = face.split()

    gray = Image.merge("RGB", (r, g, b)).convert("L")

    stat = ImageStat.Stat(gray, mask=a)

    avg = stat.mean[0]

    target = 165

    factor = target / max(avg, 1)

    factor = max(0.7, min(1.6, factor))

    gray = ImageEnhance.Brightness(gray).enhance(factor)

    face = Image.merge("RGBA", (gray, gray, gray, a))

    f = layout["face"]

    scale = f.get("scale", 1.0)

    # =====================================================
    # TRUE CONSISTENT SIZE SYSTEM
    # =====================================================

    fw, fh = face.size

    alpha = face.getchannel("A")

    bbox = alpha.getbbox()

    if bbox:

        bx1, by1, bx2, by2 = bbox

        visible_w = bx2 - bx1
        visible_h = by2 - by1

        # ==========================================
        # TARGET VISIBLE PERSON HEIGHT
        # ==========================================
        target_visible_height = f.get(
            "target_visible_height",
            300
        )

        scale_factor = (
            target_visible_height / visible_h
        )

        scale_factor *= f.get("scale", 1.0)

        new_w = int(fw * scale_factor)
        new_h = int(fh * scale_factor)

    else:

        fit_scale = min(
            f["width"] / fw,
            f["height"] / fh
        )

        scale_factor = fit_scale

        new_w = int(fw * scale_factor)
        new_h = int(fh * scale_factor)

    # =====================================================
    # RESIZE
    # =====================================================

    face = face.resize(
        (new_w, new_h),
        Image.LANCZOS
    )

    # =====================================================
    # SHARPEN + CONTRAST BOOST (post-resize quality)
    #
    # UnsharpMask recovers detail softened by LANCZOS
    # resizing. A mild contrast lift makes the face crisp.
    # =====================================================
    r, g, b_ch, a_ch = face.split()
    rgb = Image.merge("RGB", (r, g, b_ch))
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.0, percent=110, threshold=2))
    rgb = ImageEnhance.Contrast(rgb).enhance(1.12)
    r, g, b_ch = rgb.split()
    face = Image.merge("RGBA", (r, g, b_ch, a_ch))

    # =====================================================
    # CENTER INSIDE FACE BOX
    # =====================================================

    x_offset = int(
        f["x"] + (f["width"] - new_w) / 2
    )

    y_offset = int(
        f["y"] + (f["height"] - new_h) / 2
    )

    # =====================================================
    # CLAMP TO TEMPLATE BOUNDS — prevent PIL from silently
    # clipping the face when it overflows the template edge.
    # Instead, crop the face image to the visible region.
    # =====================================================
    tmpl_w, tmpl_h = template.size

    # How much of the face sticks outside each edge
    clip_left   = max(0, -x_offset)
    clip_top    = max(0, -y_offset)
    clip_right  = max(0, (x_offset + new_w) - tmpl_w)
    clip_bottom = max(0, (y_offset + new_h) - tmpl_h)

    if clip_left or clip_top or clip_right or clip_bottom:
        face = face.crop((
            clip_left,
            clip_top,
            new_w - clip_right,
            new_h - clip_bottom
        ))
        x_offset = max(0, x_offset)
        y_offset = max(0, y_offset)

    template.paste(face, (x_offset, y_offset), face)

    place_photo_copy(template, face, layout)

    # =====================================================
    # SAVE
    # =====================================================
    name = data.get("name_en", "").strip()

    if not name:
        name = "unknown"

    safe_name = "".join(
        c for c in name
        if c.isalnum() or c in (" ", "_")
    ).strip()

    filename = f"{safe_name} front.tif"

    output_dir = os.path.dirname(output_path)

    os.makedirs(output_dir, exist_ok=True)

    final_path = os.path.join(output_dir, filename)

    template.save(
        final_path,
        format="TIFF",
        compression="raw",
        dpi=template_dpi
    )

    print("✅ ID Generated:", final_path)

    return final_path