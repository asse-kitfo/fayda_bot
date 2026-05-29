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
    return


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
# OCR VARIANT COLLECTOR — returns ALL variant texts as list
# =========================================================
def run_ocr_all_results(gray, config):
    """
    Same image variants as run_ocr_variants but returns every
    result as a list instead of picking the longest.
    Used for digit-majority voting.
    """
    kernel2 = np.ones((2, 2), np.uint8)

    variants = [
        gray,
        255 - gray,
        cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 2
        ),
        cv2.medianBlur(gray, 3),
        cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel2),
        cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1],
        cv2.dilate(gray, kernel2, iterations=1),
    ]

    results = []
    for variant in variants:
        text = pytesseract.image_to_string(variant, config=config)
        results.append(text)

    return results


# =========================================================
# DIGIT MAJORITY VOTE
# =========================================================
def _digit_majority_vote(candidates, expected_len):
    """
    Given a list of digit-only strings all of exactly expected_len,
    return the per-position plurality-voted consensus string.
    Returns None if candidates is empty.
    """
    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    result = []
    for i in range(expected_len):
        digits_at_pos = [c[i] for c in candidates]
        best = max(set(digits_at_pos), key=digits_at_pos.count)
        result.append(best)

    return "".join(result)
# =========================================================
# AUTO CROP CARD FROM SCREENSHOT
# =========================================================
def crop_card(image_path, debug_dir=None):

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

    save_debug(debug_dir, "debug_back_crop.jpg", crop)

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
def extract_text_area(card_img, debug_dir=None):

    h, w = card_img.shape[:2]

    # bottom area only
    bottom = card_img[int(h * 0.45):h, :]

    save_debug(debug_dir, "debug_bottom.jpg", bottom)

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

    save_debug(debug_dir, "debug_bottom_processed.jpg", gray)

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
    # OCR — 3 PSM modes × 7 image variants = 21 readings
    # Majority-vote per digit position for max accuracy.
    # =========================================
    _DIGIT_CFG = "--oem 3 -c tessedit_char_whitelist=0123456789"
    all_raw = []
    for psm in (7, 8, 13):
        all_raw.extend(
            run_ocr_all_results(gray, f"{_DIGIT_CFG} --psm {psm}")
        )

    # collect all 12-digit sequences from every reading
    candidates_12 = []
    for r in all_raw:
        d = re.sub(r"\D", "", r)
        for m in re.finditer(r"\d{12}", d):
            candidates_12.append(m.group())

    print(f"\n🔍 FIN candidates ({len(candidates_12)}): {candidates_12}")

    voted = _digit_majority_vote(candidates_12, 12)
    if voted:
        print("✅ FIN VOTED:", voted)
        return voted

    # fallback: longest partial ≥ 8 digits across all readings
    all_digits = sorted(
        [re.sub(r"\D", "", r) for r in all_raw],
        key=len, reverse=True
    )
    for d in all_digits:
        if len(d) >= 8:
            print("⚠️ PARTIAL FIN:", d)
            return d

    print("❌ FIN NOT FOUND")
    return ""

# =========================================================
# PHONE EXTRACTOR
# =========================================================
def extract_phone(card_img, ocr_data, debug_dir=None):

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

    save_debug(debug_dir, "debug_phone_crop.jpg", crop)

    print(
        f"📦 PHONE CROP: "
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

    save_debug(debug_dir, "debug_phone_processed.jpg", gray)

    # =========================================
    # OCR — 3 PSM modes × 7 image variants = 21 readings
    # Majority-vote per digit position for max accuracy.
    # =========================================
    _DIGIT_CFG = "--oem 3 -c tessedit_char_whitelist=0123456789"
    all_raw = []
    for psm in (7, 8, 13):
        all_raw.extend(
            run_ocr_all_results(gray, f"{_DIGIT_CFG} --psm {psm}")
        )

    # collect all 09XXXXXXXX sequences from every reading
    candidates_09 = []
    for r in all_raw:
        d = re.sub(r"\D", "", r)
        for m in re.finditer(r"09\d{8}", d):
            candidates_09.append(m.group())

    print(f"\n🔍 PHONE 09-candidates ({len(candidates_09)}): {candidates_09}")

    voted = _digit_majority_vote(candidates_09, 10)
    if voted:
        print("✅ PHONE VOTED:", voted)
        return voted

    # fallback: any 10-digit sequence (in case 09 prefix was garbled)
    candidates_10 = []
    for r in all_raw:
        d = re.sub(r"\D", "", r)
        for m in re.finditer(r"\d{10}", d):
            candidates_10.append(m.group())

    voted10 = _digit_majority_vote(candidates_10, 10)
    if voted10:
        print("⚠️ PHONE (no 09 prefix) VOTED:", voted10)
        return voted10

    # fallback: longest partial ≥ 8 digits across all readings
    all_digits = sorted(
        [re.sub(r"\D", "", r) for r in all_raw],
        key=len, reverse=True
    )
    for d in all_digits:
        if len(d) >= 8:
            print("⚠️ PARTIAL PHONE:", d)
            return d

    print("❌ PHONE NOT FOUND")
    return ""

# =========================================================
# WOREDA EXTRACTOR
# =========================================================
def extract_woreda(card_img, ocr_data, debug_dir=None):

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

    save_debug(debug_dir, "debug_woreda_crop.jpg", crop)

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

    save_debug(debug_dir, "debug_woreda_processed.jpg", gray)

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
# WOREDA TABLE + FUZZY MATCHER  (module-level so both
# parse_back and process_back_ocr can share it)
# =========================================================
_WOREDA_TABLE = [
    # Gurage Zone woredas
    (["enor"],               "Enor",                         "እኖር"),
    (["cheha"],              "Cheha",                        "ቸሀ"),
    (["abeshge"],            "Abeshge",                      "አበሽጌ"),
    (["gunchere"],           "Gunchere City Administration", "ጉንችሬ ከተማ አስተዳደር"),
    (["wolkite", "welkite"], "Wolkite City Administration",  "ወልቂጤ ከተማ አስተዳደር"),
    (["emdibir"],            "Emdibir City Administration",  "እምድብር ከተማ አስተዳደር"),
    (["arekit"],             "Arekit City Administration",   "አረቅጥ ከተማ አስተዳደር"),
    (["agena"],              "Agena City Administration",    "አገና ከተማ አስተዳደር"),
    (["geta"],               "Geta",                         "ጌታ"),
    (["goro"],               "Goro",                         "ጎሮ"),
    (["gumer"],              "Gumer",                        "ጉመር"),
    (["endegagn"],           "Endegagn",                     "እንደጋኝ"),
    (["ener meger"],         "Ener Meger",                   "ኢነር መገር"),
    (["mohrna", "mohrena"],  "Mohrna Aklil",                 "ሞህርና አክሊል"),
    (["ezja"],               "Ezja",                         "ኧዣ"),
    (["kombolcha"],          "Kombolcha City Administration","ኮምቦልቻ ከተማ አስተዳደር"),
]


def detect_woreda(raw_woreda):
    """Fuzzy-match a raw OCR woreda string to a canonical (en, amh) pair."""
    if not raw_woreda:
        return "", ""
    low = raw_woreda.lower().strip()

    # Dynamic "Woreda NN" pattern (Addis Ababa numbered woredas)
    m = re.fullmatch(r"woreda\s+(\d+)", low)
    if m:
        num = m.group(1)
        return f"Woreda {num}", f"ወረዳ {num}"

    for variants, canon_en, canon_amh in _WOREDA_TABLE:
        for v in variants:
            if v in low:
                return canon_en, canon_amh

    # No match — return cleaned raw value, amh stays empty
    return raw_woreda.strip(), ""


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
    # REGION DETECTION
    # Scans all address lines at once.
    # Amharic keywords checked FIRST — they survive OCR
    # garbling better than English words do.
    # Each region has unique Amharic + fuzzy English signals.
    # =====================================================

    # (canonical_en, canonical_amh, amh_signals, eng_signals, eng_excludes)
    # eng_signals  : any substring that must appear in a lowercased line
    # eng_excludes : words that must NOT appear (to separate similar names)
    REGION_TABLE = [
        (
            "Addis Ababa", "አዲስ አበባ",
            ["አዲስ አበባ"],
            ["addis ababa"],
            []
        ),
        (
            "Southwest Ethiopia Peoples Region",
            "ደቡብ ምዕራብ ኢትዮጵያ ህዝቦች ክልላዊ መንግስት",
            ["ደቡብ ምዕራብ"],
            ["southwest ethiopia", "south west ethiopia"],
            []
        ),
        (
            "South Ethiopia Region", "ደቡብ ኢትዮጵያ ክልል",
            ["ደቡብ ኢትዮጵያ"],
            ["south ethiopia"],
            ["west", "southwest", "peoples"]   # exclude Southwest
        ),
        (
            "Central Ethiopia Region", "ማዕከላዊ ኢትዮጵያ ክልል",
            ["ማዕከላዊ"],
            ["central ethiopia", "centr"],
            []
        ),
        (
            "Amhara", "አማራ",
            ["አማራ"],
            ["amhara"],
            []
        ),
        (
            "Oromia", "ኦሮሚያ",
            ["ኦሮሚያ"],
            ["oromia"],
            []
        ),
        (
            "Tigray", "ትግራይ",
            ["ትግራይ"],
            ["tigray", "tigrai"],
            []
        ),
        (
            "Afar", "አፋር",
            ["አፋር"],
            ["afar"],
            []
        ),
        (
            "Somali", "ሶማሌ",
            ["ሶማሌ"],
            ["somali"],
            []
        ),
        (
            "Sidama", "ሲዳማ",
            ["ሲዳማ"],
            ["sidama"],
            []
        ),
        (
            "Gambella", "ጋምቤላ",
            ["ጋምቤላ"],
            ["gambella"],
            []
        ),
        (
            "Benishangul-Gumuz", "ቤኒሻንጉል ጉሙዝ",
            ["ቤኒሻንጉል"],
            ["benishangul"],
            []
        ),
        (
            "Harari", "ሀረሪ",
            ["ሀረሪ", "ሐረሪ"],
            ["harari"],
            []
        ),
        (
            "Dire Dawa", "ድሬ ዳዋ",
            ["ድሬ ዳዋ", "ድሬዳዋ"],
            ["dire dawa"],
            []
        ),
    ]

    def detect_region(lines):
        full_text     = " ".join(lines)
        full_text_low = full_text.lower()

        for canon_en, canon_amh, amh_sigs, eng_sigs, eng_excl in REGION_TABLE:

            # --- Amharic pass (highest priority) ---
            for sig in amh_sigs:
                if sig in full_text:
                    print(f"✅ REGION via Amharic '{sig}': {canon_en}")
                    return canon_en, canon_amh

            # --- English pass ---
            for sig in eng_sigs:
                if sig in full_text_low:
                    excluded = any(ex in full_text_low for ex in eng_excl)
                    if not excluded:
                        print(f"✅ REGION via English '{sig}': {canon_en}")
                        return canon_en, canon_amh

        return "", ""

    region_en, region_amh = detect_region(addr)
    if region_en:
        data["region"]     = region_en
        data["region_amh"] = region_amh

    # =====================================================
    # ZONE DETECTION
    # Same signal-table approach as detect_region():
    # scan all lines, Amharic first, fuzzy English fallback.
    # Covers Gurage Zone and all Addis Ababa subcities.
    # =====================================================

    # (canonical_en, canonical_amh, amh_signals, eng_signals)
    ZONE_TABLE = [
        (
            "Gurage Zone", "ጉራጌ ዞን",
            # Amharic: "ጉራጌ" is unique. Also catch garbled
            # versions like "ጉሬዞን" / "ጉሬቴዞንነ" via "ጉ"+"ዞን"
            ["ጉራጌ"],
            ["gurage"],
        ),
        (
            "Kolfe Keranio", "ኮልፌ ቀራንዮ",
            ["ኮልፌ"],
            ["kolfe"],
        ),
        (
            "Bole", "ቦሌ",
            ["ቦሌ"],
            ["bole"],
        ),
        (
            "Yeka", "የካ",
            ["የካ"],
            ["yeka"],
        ),
        (
            "Arada", "አራዳ",
            ["አራዳ"],
            ["arada"],
        ),
        (
            "Lideta", "ልደታ",
            ["ልደታ"],
            ["lideta"],
        ),
        (
            "Kirkos", "ቂርቆስ",
            ["ቂርቆ"],
            ["kirkos"],
        ),
        (
            "Akaky Kaliti", "አቃቂ ቃሊቲ",
            ["አቃቂ"],
            ["akaki", "akaky"],
        ),
        (
            "Gulele", "ጉለሌ",
            ["ጉለሌ"],
            ["gulele"],
        ),
        (
            "Nifas Silk Lafto", "ንፋስ ስልክ ላፍቶ",
            ["ንፋስ"],
            ["nifas"],
        ),
        (
            "Addis Ketema", "አዲስ ከተማ",
            ["አዲስ ከተማ"],
            ["addis ketema"],
        ),
        (
            "Lemi Kura", "ለሚ ኩራ",
            ["ለሚ"],
            ["lemi"],
        ),
    ]

    def detect_zone(lines):
        full_text     = " ".join(lines)
        full_text_low = full_text.lower()

        for canon_en, canon_amh, amh_sigs, eng_sigs in ZONE_TABLE:

            # Amharic pass
            for sig in amh_sigs:
                if sig in full_text:
                    print(f"✅ ZONE via Amharic '{sig}': {canon_en}")
                    return canon_en, canon_amh

            # English pass — also catch garbled "gurage zon", "gurage zone", etc.
            for sig in eng_sigs:
                if sig in full_text_low:
                    print(f"✅ ZONE via English '{sig}': {canon_en}")
                    return canon_en, canon_amh

        # Fallback: pick any line that contains "zone" (raw capture)
        for line in lines:
            if "zone" in line.lower():
                raw = line.strip()
                print(f"⚠️  ZONE raw fallback: {raw}")
                return raw, ""

        return "", ""

    zone_en, zone_amh = detect_zone(addr)
    if zone_en:
        data["zone"]     = zone_en
        data["zone_amh"] = zone_amh

    # =====================================================
    # WOREDA EXTRACTION
    # detect_woreda() is defined at module level above.
    for line in addr:

        low = line.lower().strip()

        # =========================================
        # WOREDA (English-only lines)
        # =========================================
        if re.fullmatch(r"[A-Za-z0-9 ]+", line):

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

                # prioritize explicit "woreda" keyword
                if "woreda" in low:
                    data["woreda"] = line.strip()

                # fallback only if woreda still empty
                elif not data["woreda"]:
                    data["woreda"] = line.strip()
                    
    # zone + zone_amh already set by detect_zone() above.
    # Woreda: run detect_woreda() on whatever the address loop captured.
    canon_w, canon_w_amh = detect_woreda(data["woreda"])
    if canon_w:
        data["woreda"] = canon_w
    if canon_w_amh:
        data["woreda_amh"] = canon_w_amh

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
def process_back_ocr(image_path, confirm=True, debug_dir=None):

    # crop card
    card = crop_card(image_path, debug_dir=debug_dir)
    
    # shared OCR box data
    ocr_data = get_ocr_data(card)

    # qr crop
    qr = extract_qr(card, ocr_data, debug_dir=debug_dir)

    # text
    text = extract_text_area(card, debug_dir=debug_dir)

    # fin
    fin = extract_fin(card, ocr_data, debug_dir=debug_dir)
    
    # phone
    phone = extract_phone(card, ocr_data, debug_dir=debug_dir)

    # parse
    data = parse_back(text)

    # dedicated woreda extractor
    woreda = extract_woreda(card, ocr_data, debug_dir=debug_dir)

    if woreda:
        data["woreda"] = woreda
        
    # Re-run detect_woreda() after dedicated extractor overwrites data["woreda"]
    canon_w, canon_w_amh = detect_woreda(data["woreda"])
    if canon_w:
        data["woreda"] = canon_w
    if canon_w_amh:
        data["woreda_amh"] = canon_w_amh

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