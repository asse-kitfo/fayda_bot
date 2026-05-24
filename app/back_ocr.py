import os
import cv2
import pytesseract
import re
import numpy as np

# use high-quality tessdata_best trained models
_TESSDATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tessdata")
os.environ["TESSDATA_PREFIX"] = _TESSDATA

# =========================================================
# DEBUG SAVE
# =========================================================
def save_debug(debug_dir, filename, image):

    if not debug_dir:
        return

    os.makedirs(debug_dir, exist_ok=True)

    cv2.imwrite(
        os.path.join(debug_dir, filename),
        image
    )


# =========================================================
# SHARED NATIONALITY ANCHOR
# =========================================================
def get_nationality_anchor(card_img, ocr_data):

    h, w = card_img.shape[:2]

    nat_x = None
    nat_y = None

    # =========================================
    # FIND "ETHIOPIAN"
    # =========================================
    for i, word in enumerate(ocr_data["text"]):

        word = word.strip().lower()

        if "ethiopian" in word:

            nat_x = ocr_data["left"][i]
            nat_y = ocr_data["top"][i]

            print(
                f"✅ NATIONALITY FOUND: "
                f"x={nat_x}, y={nat_y}"
            )

            break

    # =========================================
    # FALLBACK
    # =========================================
    if nat_x is None:

        print("⚠️ NATIONALITY NOT FOUND")

        nat_x = int(w * 0.15)
        nat_y = int(h * 0.30)

    return nat_x, nat_y


# =========================================================
# GENERIC OCR PREPROCESS
# =========================================================
def preprocess_ocr_crop(
    crop,
    scale=6,
    alpha=2.5,
    beta=20,
    blur=True
):

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    # enlarge
    gray = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    # contrast
    gray = cv2.convertScaleAbs(
        gray,
        alpha=alpha,
        beta=beta
    )

    # blur
    if blur:

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

    # threshold
    _, gray = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return gray


# =========================================================
# OCR RETRY ENGINE
# =========================================================
def run_ocr_variants(
    gray,
    config
):

    variants = []

    # original
    variants.append(gray)

    # inverted
    variants.append(255 - gray)

    # adaptive
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )

    variants.append(adaptive)
    # median blur
    median = cv2.medianBlur(gray, 3)
    variants.append(median)

    # morphology close
    kernel = np.ones((2, 2), np.uint8)

    morph = cv2.morphologyEx(
        gray,
        cv2.MORPH_CLOSE,
        kernel
    )

    variants.append(morph)

    # stronger threshold
    _, strong = cv2.threshold(
        gray,
        180,
        255,
        cv2.THRESH_BINARY
    )

    variants.append(strong)

    results = []

    for idx, variant in enumerate(variants):

        text = pytesseract.image_to_string(
            variant,
            config=config
        )

        print(f"\n🔍 OCR VARIANT {idx}:\n")
        print(repr(text))

        results.append(text)

    # choose longest cleaned text
    best = max(
        results,
        key=lambda x: len(
            re.sub(r"\W", "", x)
        )
    )

    return best
# =========================================================
# AUTO CROP CARD FROM SCREENSHOT
# =========================================================
def crop_card(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # edge detect
    edges = cv2.Canny(blur, 50, 150)

    # contours
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    biggest = None
    biggest_area = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 100000:
            continue

        peri = cv2.arcLength(cnt, True)

        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # rectangle only
        if len(approx) == 4:

            if area > biggest_area:
                biggest = approx
                biggest_area = area

    # fallback
    if biggest is None:
        return img

    x, y, w, h = cv2.boundingRect(biggest)

    crop = img[y:y+h, x:x+w]

    cv2.imwrite("debug_back_crop.jpg", crop)

    print("✅ Card cropped")

    return crop

# =========================================================
# OCR BOX CACHE
# =========================================================
def get_ocr_data(card_img):

    data = pytesseract.image_to_data(
        card_img,
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT
    )

    return data

# =========================================================
# QR EXTRACTOR (NATIONALITY ANCHOR)
# =========================================================
def extract_qr(card_img, ocr_data, debug_dir=None):
    
    h, w = card_img.shape[:2]

    nat_x, nat_y = get_nationality_anchor(
        card_img,
        ocr_data
    )
    # =========================================
    # RELATIVE QR CROP
    # =========================================

    # QR is BELOW nationality
    shift_down = 0.027 * h

    # QR starts slightly left
    shift_left = 0.195 * w

    qr_w = int(w * 0.69)
    qr_h = int(h * 0.31)

    x1 = int(nat_x - shift_left)
    y1 = int(nat_y + shift_down)

    x2 = x1 + qr_w
    y2 = y1 + qr_h

    # =========================================
    # CLAMP
    # =========================================
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))

    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))

    # =========================================
    # HIGH RESOLUTION
    # =========================================
    scale = 3

    big = cv2.resize(
        card_img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    qr_crop = big[
        y1 * scale:y2 * scale,
        x1 * scale:x2 * scale
    ]

    # =========================================
    # SHARPEN
    # =========================================
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])

    qr_crop = cv2.filter2D(
        qr_crop,
        -1,
        kernel
    )

    save_debug(
        debug_dir,
        "debug_qr.jpg",
        qr_crop
    )

    print(
        f"📦 QR CROP: "
        f"x1={x1}, y1={y1}, "
        f"x2={x2}, y2={y2}"
    )

    return qr_crop

# =========================================================
# OCR ONLY LOWER AREA
# =========================================================
def extract_text_area(card_img):

    h, w = card_img.shape[:2]

    # bottom area only
    bottom = card_img[int(h * 0.45):h, :]

    cv2.imwrite("debug_bottom.jpg", bottom)

    gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)

    gray = cv2.convertScaleAbs(
        gray,
        alpha=2.2,
        beta=35
    )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, gray = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    cv2.imwrite("debug_bottom_processed.jpg", gray)

    text = pytesseract.image_to_string(
        gray,
        config="--oem 3 --psm 6 -l amh+eng"
    )

    print("\n🔍 BACK OCR:\n")
    print(text)

    return text


# =========================================================
# FIN EXTRACTOR (FAN-STYLE RELATIVE CROP)
# =========================================================
def extract_fin(card_img, ocr_data, debug_dir=None):
        
    h, w = card_img.shape[:2]

    nat_x, nat_y = get_nationality_anchor(
        card_img,
        ocr_data
    )
    # =========================================
    # EXACT RELATIVE FIN CROP
    # =========================================

    # FIN is BELOW nationality
    shift_down = 0.36 * h

    # stable height
    box_height = 0.028 * h

    center_y = int(nat_y + shift_down)

    # balanced crop like FAN logic
    y1 = int(center_y - box_height * 0.4)
    y2 = int(center_y + box_height * 0.6)

    # right side
    x1 = int(w * 0.61)
    x2 = int(w * 0.85)

    # =========================================
    # CLAMP
    # =========================================
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))

    crop = card_img[y1:y2, x1:x2]

    save_debug(
        debug_dir,
        "debug_fin_crop.jpg",
        crop
    )

    print(
        f"📦 FIN CROP: "
        f"x1={x1}, y1={y1}, "
        f"x2={x2}, y2={y2}"
    )

    # =========================================
    # OCR PREPROCESS
    # =========================================
    gray = preprocess_ocr_crop(
        crop,
        scale=8,
        alpha=2.5,
        beta=20
    )

    # =========================================
    # OCR DIGITS ONLY
    # =========================================
    text = run_ocr_variants(
        gray,
        (
            "--oem 3 "
            "--psm 7 "
            "-c tessedit_char_whitelist=0123456789"
        )
    )

    print("\n🔍 FIN RAW:\n")
    print(repr(text))

    # keep digits only
    text = re.sub(r"\D", "", text)

    print("\n🔍 FIN CLEANED:\n")
    print(text)

    # =========================================
    # FIND FIN
    # =========================================
    matches = re.findall(r"\d{12}", text)

    if matches:

        best = max(matches, key=len)

        print("✅ FIN FOUND:", best)

        return best

    # fallback
    partial = re.findall(r"\d+", text)

    if partial:

        best = max(partial, key=len)

        if len(best) >= 8:

            print("⚠️ PARTIAL FIN:", best)

            return best

    print("❌ FIN NOT FOUND")

    return ""

# =========================================================
# PHONE EXTRACTOR
# =========================================================
def extract_phone(card_img, ocr_data):

    h, w = card_img.shape[:2]

    nat_x, nat_y = get_nationality_anchor(
        card_img,
        ocr_data
    )

    # =========================================
    # PHONE CROP
    # =========================================

    # =========================================
    # PHONE POSITION RELATIVE TO NATIONALITY
    # =========================================

    # phone is BELOW nationality
    shift_down = 0.372 * h

    # phone is LEFT of nationality
    shift_left = 0.25 * w

    # stable line height
    box_height = 0.032 * h

    center_y = int(nat_y + shift_down)

    # balanced vertical crop
    y1 = int(center_y - box_height * 0.45)
    y2 = int(center_y + box_height * 0.55)

    # phone starts left from nationality
    x1 = int(nat_x - shift_left)

    # phone width
    phone_width = int(w * 0.31)

    x2 = x1 + phone_width

    # =========================================
    # CLAMP
    # =========================================
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))

    crop = card_img[y1:y2, x1:x2]
    
    # remove noisy borders
    ch, cw = crop.shape[:2]

    crop = crop[
        int(ch * 0.12):int(ch * 0.88),
        int(cw * 0.04):int(cw * 0.96)
    ]

    cv2.imwrite("debug_phone_crop.jpg", crop)

    print(
        f"📦 PHONE CROP: "
        f"x1={x1}, y1={y1}, "
        f"x2={x2}, y2={y2}"
    )

    # =========================================
    # OCR PREPROCESS
    # =========================================
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # enlarge
    gray = cv2.resize(
        gray,
        None,
        fx=8,
        fy=8,
        interpolation=cv2.INTER_CUBIC
    )

    # contrast
    gray = cv2.convertScaleAbs(
        gray,
        alpha=2.5,
        beta=20
    )

    # blur
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # threshold
    _, gray = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    cv2.imwrite(
        "debug_phone_processed.jpg",
        gray
    )

    # =========================================
    # OCR DIGITS ONLY
    # =========================================
    text = run_ocr_variants(
        gray,
        (
            "--oem 3 "
            "--psm 7 "
            "-c tessedit_char_whitelist=0123456789"
        )
    )

    print("\n🔍 PHONE RAW:\n")
    print(repr(text))

    # digits only
    text = re.sub(r"\D", "", text)

    print("\n🔍 PHONE CLEANED:\n")
    print(text)

    # =========================================
    # FIND PHONE
    # =========================================
    matches = re.findall(r"09\d{8}", text)

    if matches:

        phone = matches[0]

        print("✅ PHONE FOUND:", phone)

        return phone

    # fallback
    partial = re.findall(r"\d+", text)

    if partial:

        best = max(partial, key=len)

        if len(best) >= 8:

            print("⚠️ PARTIAL PHONE:", best)

            return best

    print("❌ PHONE NOT FOUND")

    return ""

# =========================================================
# WOREDA EXTRACTOR
# =========================================================
def extract_woreda(card_img, ocr_data):

    h, w = card_img.shape[:2]

    nat_x, nat_y = get_nationality_anchor(
        card_img,
        ocr_data
    )
    # =========================================
    # WOREDA POSITION
    # =========================================

    # slightly BELOW phone
    shift_down = 0.54 * h

    # same left alignment as phone
    shift_left = 0.19 * w

    # woreda text area
    box_height = 0.032 * h

    center_y = int(nat_y + shift_down)

    y1 = int(center_y - box_height * 0.45)
    y2 = int(center_y + box_height * 0.55)

    # left aligned
    x1 = int(nat_x - shift_left)

    # wider because woreda is text
    woreda_width = int(w * 0.50)

    x2 = x1 + woreda_width

    # =========================================
    # CLAMP
    # =========================================
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))

    crop = card_img[y1:y2, x1:x2]

    cv2.imwrite("debug_woreda_crop.jpg", crop)

    print(
        f"📦 WOREDA CROP: "
        f"x1={x1}, y1={y1}, "
        f"x2={x2}, y2={y2}"
    )

    # =========================================
    # OCR PREPROCESS
    # =========================================
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=6,
        fy=6,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.convertScaleAbs(
        gray,
        alpha=2.3,
        beta=20
    )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, gray = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = np.ones((2, 2), np.uint8)

    gray = cv2.morphologyEx(
        gray,
        cv2.MORPH_CLOSE,
        kernel
    )

    cv2.imwrite(
        "debug_woreda_processed.jpg",
        gray
    )

    # =========================================
    # OCR TEXT
    # =========================================
    text = run_ocr_variants(
        gray,
        (
            "--oem 3 "
            "--psm 6 "
            "-l amh+eng"
        )
    )

    print("\n🔍 WOREDA RAW:\n")
    print(repr(text))

    # cleanup
    text = text.strip()

    text = re.sub(r"[^A-Za-z0-9\u1200-\u137F\s]", "", text)
    
    text = re.sub(r"\s+", " ", text)

    print("\n🔍 WOREDA CLEANED:\n")
    print(text)

    return text

# =========================================================
# HELPERS
# =========================================================
def clean_line(x):

    x = x.strip()

    x = re.sub(r"\s+", " ", x)

    return x

# =========================================================
# PARSE BACK DATA
# =========================================================
def parse_back(text):

    lines = [
        clean_line(l)
        for l in text.split("\n")
        if clean_line(l)
    ]

    data = {
        "phone": "",
        "fin": "",
        "nationality": "",

        "region": "",
        "zone": "",
        "woreda": "",

        "region_amh": "",
        "zone_amh": "",
        "woreda_amh": ""
    }

    # =====================================================
    # PHONE
    # =====================================================
    phone_index = -1

    for i, line in enumerate(lines):

        phone_match = re.search(r"09\d{8}", line)

        if phone_match:

            data["phone"] = phone_match.group()
            phone_index = i
            break

    # =====================================================
    # FIN
    # =====================================================
    fin_match = re.search(r"(\d{12,13})", text)

    if fin_match:
        data["fin"] = fin_match.group(1)

    # =====================================================
    # NATIONALITY
    # =====================================================
    for line in lines:

        if "ethiopian" in line.lower():

            data["nationality"] = "Ethiopian"
            break

    # =====================================================
    # ADDRESS BLOCK
    # =====================================================
    if phone_index != -1:
        addr = lines[phone_index + 1:]
    else:
        addr = lines

    print("\nADDRESS BLOCK:")

    for x in addr:
        print(x)

    # =====================================================
    # CLEAN GARBAGE
    # =====================================================
    cleaned = []

    for line in addr:

        if len(line) < 2:
            continue

        weird = len(
            re.findall(
                r"[^A-Za-z0-9\u1200-\u137F\s]",
                line
            )
        )

        if weird > 5:
            continue

        cleaned.append(line)

    addr = cleaned


    # =====================================================
    # ENGLISH EXTRACTION
    # =====================================================
    for line in addr:

        low = line.lower().strip()

        # =========================================
        # REGION
        # =========================================
        if low == "amhara":

            data["region"] = "Amhara"

        elif low == "oromia":

            data["region"] = "Oromia"

        elif low == "addis ababa":

            data["region"] = "Addis Ababa"

        elif "central ethiopia" in low:

            data["region"] = "Central Ethiopia Region"

        # =========================================
        # ZONE
        # =========================================
        elif "zone" in low:

            data["zone"] = line.strip()

        # Addis Ababa subcities
        elif low in [
            "kolfe keranio",
            "bole",
            "yeka",
            "arada",
            "lideta",
            "kirkos",
            "akaky kaliti",
            "gulele",
            "nifas silk lafto",
            "addis ketema",
            "lemi kura"
        ]:

            data["zone"] = line.strip()

        # =========================================
        # WOREDA
        # =========================================
        elif re.fullmatch(r"[A-Za-z0-9 ]+", line):

            if (
                "ethiopian" not in low
                and "zone" not in low
                and "region" not in low
                and low not in [
                    "amhara",
                    "oromia",
                    "addis ababa"
                ]
            ):

                # prioritize actual woreda lines
                if "woreda" in low:
                    data["woreda"] = line.strip()

                # fallback only if woreda still empty
                elif not data["woreda"]:
                    data["woreda"] = line.strip()
                    
   # =====================================================
    # FIX KNOWN VALUES
    # =====================================================
    zone_map = {
        "gurage zone": 
            "Gurage Zone",
        "bole": 
            "ቦሌ",
        "yeka": 
            "የካ",
        "arada": 
            "አራዳ",
        "lideta": 
            "ልደታ",
        "kirkos": 
            "ቂርቆስ",
        "akaki kaliti": 
            "አቃቂ ቃሊቲ",
        "gulele": "ጉለሌ",
        "nifas silk lafto": 
            "ንፋስ ስልክ ላፍቶ",
        "addis ketema": 
            "አዲስ ከተማ",
        "lemi kura": 
            "ለሚ ኩራ",
        "nifas silk lafto": 
            "ንፋስ ስልክ ላፍቶ",
        
    }

    woreda_map = {
        "enor": "Enor"
    }

    woreda_amh_map = {
        "enor":
            "እኖር",
        "እኖር":
            "enor",

        "gunchere city administration":
            "ጉንችሬ ከተማ አስተዳደር",
        "ጉንችሬ ከተማ አስተዳደር":
            "gunchere city administration",

        "wolkite town administration":
            "ወልቂጤ ከተማ አስተዳደር",
        "ወልቂጤ ከተማ አስተዳደር":
            "wolkite town administration",

        "welkite city administration":
            "ወልቂጤ ከተማ አስተዳደር",

        "abeshge":
            "አበሽጌ",
        "አበሽጌ":
            "abeshge",

        "kombolcha city administration":
            "ኮምቦልቻ ከተማ አስተዳደር",
        "ኮምቦልቻ ከተማ አስተዳደር":
            "kombolcha city administration",
        
        "geta": "ጌታ",
        "goro": "ጎሮ",
        "ጎሮ": "goro",
        
        "emdibir city administration": 
            "እምድብር ከተማ አስተዳደር",
            
        "እምድብር ከተማ አስተዳደር":
            "emdibir city administration", 
            
        "arekit city administration": 
            "አረቅጥ ከተማ አስተዳደር",
        
        "አረቅጥ ከተማ አስተዳደር": 
            "arekit city administration",
        
        "agena city administration": 
            "አገና ከተማ አስተዳደር",
        "አገና ከተማ አስተዳደር": 
            "agena city administration",
            
        "cheha": "ቸሀ",
        "ቸሀ": "cheha",
        
        "mohrna aklil": "ሞህርና አክሊል",
        "ሞህርና አክሊል": "mohrna aklil",
       
        "ezja": "ኧዣ",
        "ኧዣ": "ezja",
        
        "geta": "ጌታ",
        "ጌታ": "geta",
        
        "gumer": "ጉመር",
        "ጉመር": "gumer",
        
        "endegagn": "እንደጋኝ",
        "እንደጋኝ": "endegagn",
        
        "ener meger": "ኢነር መገር",
        "ኢነር መገር": "ener meger"
        
        
        
        
    }

    z = data["zone"].lower()
    w = data["woreda"].lower()

    if z in zone_map:
        data["zone"] = zone_map[z]

    if w in woreda_map:
        data["woreda"] = woreda_map[w]
            
    # =====================================================
    # DYNAMIC WOREDA NUMBER
    # example:
    # "Woreda 05" -> "ወረዳ 05"
    # =====================================================
    match = re.fullmatch(
        r"woreda\s+(\d+)",
        w,
        re.IGNORECASE
    )

    if match:
        number = match.group(1)
        data["woreda_amh"] = f"ወረዳ {number}"
        
    # =====================================================
    # ENGLISH -> AMHARIC MAPPING
    # =====================================================
    region_amh_map = {
        "central ethiopia region":
            "ማዕከላዊ ኢትዮጵያ ክልል",

        "amhara":
            "አማራ",

        "addis ababa":
            "አዲስ አበባ",
            
        "ማዕከላዊ ኢትዮጵያ ክልል":
            "central ethiopia region",

        "አማራ":
            "amhara",

        "አዲስ አበባ":
            "addis ababa",
    }
   
    zone_amh_map = {
        "gurage zone":
            "ጉራጌ ዞን",

        "south wollo zone":
            "ደቡብ ወሎ ዞን",

        "kolfe keranio":
            "ኮልፌ ቀራንዮ",
            
        "south west shawa":
            "ደቡብ ምዕራብ ሸዋ",
            
        "bole":
            "ቦሌ",

        "yeka":
            "የካ",

        "arada":
            "አራዳ",

        "lideta":
            "ልደታ",

        "kirkos":
            "ቂርቆስ",

        "akaki kaliti":
            "አቃቂ ቃሊቲ",

        "gulele":
            "ጉለሌ",

        "nifas silk lafto":
            "ንፋስ ስልክ ላፍቶ",

        "addis ketema":
            "አዲስ ከተማ",

        "lemi kura":
            "ለሚ ኩራ",
                "bole": 
            "ቦሌ",
        "yeka": 
            "የካ",
        "arada": 
            "አራዳ",
        "lideta": 
            "ልደታ",
        "kirkos": 
            "ቂርቆስ",
        "akaki kaliti": 
            "አቃቂ ቃሊቲ",
        "gulele": "ጉለሌ",
        "nifas silk lafto": 
            "ንፋስ ስልክ ላፍቶ",
        "addis ketema": 
            "አዲስ ከተማ",
        "lemi kura": 
            "ለሚ ኩራ",
        "nifas silk lafto": 
            "ንፋስ ስልክ ላፍቶ",
        
        }


    # region
    r = data["region"].lower()

    if r in region_amh_map:
        data["region_amh"] = region_amh_map[r]

    # zone
    z = data["zone"].lower()

    if z in zone_amh_map:
        data["zone_amh"] = zone_amh_map[z]

    # woreda
    w = data["woreda"].lower()

    if w in woreda_amh_map:
        data["woreda_amh"] = woreda_amh_map[w]

    # =====================================================
    # DEBUG
    # =====================================================
    print("\nFINAL ADDRESS DATA:\n")

    for k in [
        "region",
        "zone",
        "woreda",
        "region_amh",
        "zone_amh",
        "woreda_amh"
    ]:
        print(k, ":", data[k])

    return data

# =========================================================
# VALIDATE BACK DATA
# =========================================================
def validate_back_data(data):

    problems = []

    # =========================================
    # FIN VALIDATION
    # =========================================
    fin = re.sub(r"\D", "", data.get("fin", ""))

    if len(fin) != 12:

        problems.append(
            f"FIN must be exactly 12 digits "
            f"(detected: {len(fin)})"
        )

    # =========================================
    # PHONE VALIDATION
    # =========================================
    phone = re.sub(r"\D", "", data.get("phone", ""))

    if len(phone) != 10:

        problems.append(
            f"Phone number must be exactly 10 digits "
            f"(detected: {len(phone)})"
        )

    # =========================================
    # ADDRESS VALIDATION
    # =========================================
    def invalid_text(value):

        if not value:
            return True

        allowed = r"^[A-Za-z0-9\u1200-\u137F\s]+$"

        return re.fullmatch(
            allowed,
            value.strip()
        ) is None

    address_fields = [
        "region",
        "zone",
        "woreda",
        "region_amh",
        "zone_amh",
        "woreda_amh"
    ]

    for field in address_fields:

        value = data.get(field, "")

        if invalid_text(value):

            problems.append(
                f"Invalid or empty {field}"
            )

    return problems

# =========================================================
# MAIN
# =========================================================
def process_back_ocr(image_path, confirm=True):

    # crop card
    card = crop_card(image_path)
    
    # shared OCR box data
    ocr_data = get_ocr_data(card)

    # qr crop
    qr = extract_qr(card, ocr_data)

    # text
    text = extract_text_area(card)

    # fin
    fin = extract_fin(card, ocr_data)
    
    # phone
    phone = extract_phone(card, ocr_data)    

    # parse
    data = parse_back(text)

    # dedicated woreda extractor
    woreda = extract_woreda(card, ocr_data)

    if woreda:
        data["woreda"] = woreda
        
    # update amh woreda after dedicated OCR overwrite
    w = data["woreda"].lower()

    woreda_amh_map = {
        "enor": "እኖር",
        "gunchere city administration": "ጉንችሬ ከተማ አስተዳደር",
        "wolkite town administration": "ወልቂጤ ከተማ አስተዳደር",
        "abeshge": "አበሽጌ",
        "welkite city administration": "ወልቂጤ ከተማ አስተዳደር",
        "kombolcha city administration":
        "ኮምቦልቻ ከተማ አስተዳደር",
        "geta": "ጌታ",
        "goro": "ጎሮ",
        "emdibir city administration": "እምድብር ከተማ አስተዳደር",
        "arekit city administration": "አረቅጥ ከተማ አስተዳደር",
        "agena city administration": "አገና ከተማ አስተዳደር",
        "cheha": "ቸሀ",
        "mohrna aklil": "ሞህርና አክሊል",
        "ezja": "ኧዣ",
        "geta": "ጌታ",
        "gumer": "ጉመር",
        "endegagn": "እንደጋኝ",
        "ener meger": "ኢነር መገር"
    }

    if w in woreda_amh_map:
        data["woreda_amh"] = woreda_amh_map[w]
            
    # =====================================================
    # DYNAMIC WOREDA NUMBER
    # example:
    # "Woreda 05" -> "ወረዳ 05"
    # =====================================================
    match = re.fullmatch(
        r"woreda\s+(\d+)",
        w,
        re.IGNORECASE
    )

    if match:
        number = match.group(1)
        data["woreda_amh"] = f"ወረዳ {number}"

    data["phone"] = phone

    # overwrite fin with dedicated extractor
    data["fin"] = fin
    
    

    # =========================================
    # VALIDATE
    # =========================================
    problems = validate_back_data(data)

    if problems:

        data["problems"] = problems

        print("\n❌ BACK VALIDATION FAILED:\n")

        for p in problems:
            print("-", p)

    else:

        print("\n✅ BACK VALIDATION PASSED")

    print("\n✅ BACK DATA:\n")

    for k, v in data.items():
        print(f"{k}: {v}")

    return data, qr