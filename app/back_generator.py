import re
from PIL import Image, ImageDraw, ImageFont
import os
import json
import cv2

# =========================================================
# PATH
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def path(*args):
    return os.path.join(BASE_DIR, "..", *args)


# =========================================================
# TEXT
# =========================================================
def draw_text(draw, text, cfg, font_path):

    if not text:
        return

    size = cfg.get("size", 28)

    font = ImageFont.truetype(font_path, size)

    x = cfg["x"]
    y = cfg["y"]

    # thickness
    bold = cfg.get("bold", 1)

    # NORMAL
    if bold <= 1:

        draw.text(
            (x, y),
            text,
            font=font,
            fill=(0, 0, 0)
        )

    # LIGHT BOLD (1.5)
    elif bold <= 1.5:

        offsets = [
            (0, 0),
            (0.4, 0)
        ]

        for dx, dy in offsets:

            draw.text(
                (x + dx, y + dy),
                text,
                font=font,
                fill=(0, 0, 0)
            )

    # MEDIUM BOLD (2)
    elif bold <= 2:

        offsets = [
            (0, 0),
            (1, 0),
            (0, 1)
        ]

        for dx, dy in offsets:

            draw.text(
                (x + dx, y + dy),
                text,
                font=font,
                fill=(0, 0, 0)
            )

    # STRONG BOLD (3)
    else:

        offsets = [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1)
        ]

        for dx, dy in offsets:

            draw.text(
                (x + dx, y + dy),
                text,
                font=font,
                fill=(0, 0, 0)
            )
   
# =========================================================
# FORMAT FIN
# =========================================================
def format_fin(fin):

    if not fin:
        return ""

    # keep digits only
    fin = re.sub(r"\D", "", fin)

    # split every 4 digits
    parts = [
        fin[i:i+4]
        for i in range(0, len(fin), 4)
    ]

    return "-".join(parts)    

# =========================================================
# CARD NUMBER COUNTER
# =========================================================
COUNTER_FILE = path("card_counter.json")

def generate_card_number():

    # create file if missing
    if not os.path.exists(COUNTER_FILE):

        with open(COUNTER_FILE, "w") as f:

            json.dump(
                {
                    "last_number": 20038464
                },
                f
            )

    # read current number
    with open(COUNTER_FILE, "r") as f:

        data = json.load(f)

    current = data.get(
        "last_number",
        20038464
    )

    # increment
    new_number = current + 1

    # save updated number
    with open(COUNTER_FILE, "w") as f:

        json.dump(
            {
                "last_number": new_number
            },
            f
        )

    # return 8 digits
    return str(new_number).zfill(8)

# =========================================================
# GENERATE BACK
# =========================================================
def generate_back(data, qr_crop, output_path, name="unknown"):

    template = Image.open(
        path("assets", "templates", "back.tif")
    )

    draw = ImageDraw.Draw(template)
    
    card_number = generate_card_number()
    # coords
    with open(
        path("config", "back_coords.json"),
        "r",
        encoding="utf-8"
    ) as f:

        layout = json.load(f)

    font_en = path("assets", "fonts", "english.ttf")
    font_amh = path("assets", "fonts", "amharic.ttf")
    
    # =====================================================
    # LOAD TEMPLATE (NO COLOR CHANGE)
    # =====================================================
    template = Image.open(
        path("assets", "templates", "back.tif")
    )

    draw = ImageDraw.Draw(template)

    # =====================================================
    # QR
    # =====================================================
    if qr_crop is not None:

        qr_img = Image.fromarray(
            cv2.cvtColor(qr_crop, cv2.COLOR_BGR2RGB)
        ).convert("RGBA")

        qr_cfg = layout["qr"]

        qr_img = qr_img.resize(
            (
                qr_cfg["w"],
                qr_cfg["h"]
            ),
            Image.LANCZOS
        )

        template.paste(
            qr_img,
            (
                qr_cfg["x"],
                qr_cfg["y"]
            ),
            qr_img
        )
    # =====================================================
    # TEXT
    # =====================================================
    draw_text(
        draw,
        data.get("phone", ""),
        layout["phone"],
        font_en,
    )

    formatted_fin = format_fin(
        data.get("fin", "")
    )

    draw_text(
        draw,
        formatted_fin,
        layout["fin"],
        font_en,
    )


    draw_text(
        draw,
        data.get("region", ""),
        layout["region"],
        font_en,
    )

    draw_text(
        draw,
        data.get("zone", ""),
        layout["zone"],
        font_en,
    )

    draw_text(
        draw,
        data.get("woreda", ""),
        layout["woreda"],
        font_en,
    )
    
    draw_text(
        draw,
        card_number,
        layout["card_number"],
        font_en,
    )
        
    # =====================================================
    # AMHARIC ADDRESS
    # =====================================================
    draw_text(
        draw,
        data.get("region_amh", ""),
        layout["region_amh"],
        font_amh,
    )

    draw_text(
        draw,
        data.get("zone_amh", ""),
        layout["zone_amh"],
        font_amh,
    )

    draw_text(
        draw,
        data.get("woreda_amh", ""),
        layout["woreda_amh"],
        font_amh,
    )

    # =====================================================
    # SAVE
    # =====================================================
    safe_name = "".join(
        c for c in name
        if c.isalnum() or c in (" ", "_")
    ).strip()

    filename = f"{safe_name} back.tif"

    output_dir = os.path.dirname(output_path)

    os.makedirs(output_dir, exist_ok=True)

    final_path = os.path.join(
        output_dir,
        filename
    )

    template.save(
        final_path,
        format="TIFF",
        compression="raw"
    )

    print("✅ Back generated:", final_path)

    return final_path