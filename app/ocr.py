import cv2
import pytesseract
import re
import os

# use high-quality tessdata_best trained models
_TESSDATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tessdata")
os.environ["TESSDATA_PREFIX"] = _TESSDATA

# =========================================================
# PREPROCESS (UNCHANGED)
# =========================================================
def preprocess(image_path):

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray


# =========================================================
# OCR (UNCHANGED)
# =========================================================
def get_text(img):

    config = "--oem 3 --psm 6 -l amh+eng"
    text = pytesseract.image_to_string(img, config=config)

    print("\n🔍 FULL OCR TEXT:\n")
    print(text)

    return text


# =========================================================
# CLEANERS (UNCHANGED)
# =========================================================
def clean_english(text):
    if not text:
        return ""

    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^A-Za-z ]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_amharic(text):
    if not text:
        return ""

    text = "".join(
        ch for ch in text
        if ("\u1200" <= ch <= "\u137F") or ch == " "
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# =========================================================
# RAW CHARACTER FILTERS
# reject names containing numbers/special chars
# BEFORE cleaning
# =========================================================
def has_invalid_english_chars(text):

    if not text:
        return True

    # allow only letters + spaces
    return bool(
        re.search(r"[^A-Za-z ]", text)
    )


def has_invalid_amharic_chars(text):

    if not text:
        return True

    # allow only Amharic chars + spaces
    return bool(
        re.search(r"[^\u1200-\u137F ]", text)
    )

# =========================================================
# VALIDATORS (UNCHANGED)
# =========================================================
def is_valid_english_name(text):

    if not text:
        return False

    # =========================================
    # REJECT SPECIAL CHARACTERS / NUMBERS
    # only English letters + spaces allowed
    # =========================================
    if re.search(r"[^A-Za-z ]", text):
        return False

    text = clean_english(text)

    if not text:
        return False

    bad_words = [
        "FAYDA",
        "DIGITAL",
        "COPY",
        "CARD",
        "ETHIOPIAN",
        "DETAILS",
        "ID",
        "MALE",
        "FEMALE"
    ]

    words_upper = text.upper().split()

    # reject forbidden words
    if any(b == w for w in words_upper for b in bad_words):
        return False

    words = text.split()

    # =========================================
    # MUST BE EXACTLY 3 WORDS
    # =========================================
    if len(words) != 3:
        return False

    # each word must be at least 2 chars
    if not all(len(w) >= 2 for w in words):
        return False

    return True


def is_valid_amharic_name(text):

    if not text:
        return False

    # =========================================
    # REJECT SPECIAL CHARACTERS / NUMBERS
    # allow only Amharic fidels + spaces
    # =========================================
    if re.search(r"[^\u1200-\u137F ]", text):
        return False

    text = clean_amharic(text)

    if not text:
        return False

    words = text.split()

    # =========================================
    # MUST BE EXACTLY 3 WORDS
    # =========================================
    if len(words) != 3:
        return False

    # each word must be at least 2 chars
    if not all(len(w) >= 2 for w in words):
        return False

    return True


def is_amharic(text):
    return bool(re.search(r"[\u1200-\u137F]", text or ""))


# =========================================================
# HELPERS (UNCHANGED)
# =========================================================
def normalize(x):
    if not x:
        return ""

    x = x.strip()
    x = re.sub(r"\s+", "", x)
    x = re.sub(r"(?<=\d)O(?=\d)", "0", x)

    # OCR fix (49 → 19)
    x = re.sub(r"\b49/", "19/", x)

    return x


def split_lr(line):
    if "|" in line:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2:
            return parts[0], parts[1]
    return None, None


# =========================================================
# DATE VALIDATION (UNCHANGED)
# =========================================================
def is_date(x):
    x = normalize(x)

    match = (
        re.match(r"(\d{2})/(\d{2})/(\d{4})", x) or
        re.match(r"(\d{4})/(\d{2})/(\d{2})", x) or
        re.match(r"(\d{2})/([A-Za-z]{3})/(\d{4})", x, re.IGNORECASE) or
        re.match(r"(\d{4})/([A-Za-z]{3})/(\d{2})", x, re.IGNORECASE)
    )

    if not match:
        return False

    parts = match.groups()

    try:
        day = int(parts[0]) if len(parts[0]) == 2 else int(parts[2])
    except:
        return False

    if day < 1 or day > 31:
        return False

    return True

def is_gregorian_date(x):

    x = normalize(x)

    # Gregorian usually contains English month
    return bool(
        re.search(r"[A-Za-z]{3}", x)
    )


def is_ethiopian_date(x):

    x = normalize(x)

    # Ethiopian usually numeric only
    return bool(
        re.fullmatch(r"\d{4}/\d{2}/\d{2}", x)
    )
# =========================================================
# FIND DOB (UNCHANGED)
# =========================================================
def find_dob_anchor(lines):

    for i, line in enumerate(lines):

        if "|" not in line:
            continue

        left, right = split_lr(line)
        if not left or not right:
            continue

        if is_date(left) or is_date(right):
            return i

    return -1


# =========================================================
# NAME EXTRACTION (UNCHANGED)
# =========================================================
def extract_names(lines, anchor_i):

    name_en = ""
    name_am = ""

    search_area = lines[:anchor_i]

    for i in range(len(search_area) - 1, -1, -1):

        line = search_area[i]

        if is_amharic(line):
            continue

        # reject raw OCR line if contains symbols/numbers
        if has_invalid_english_chars(line):
            continue

        cleaned_en = clean_english(line)

        if is_valid_english_name(cleaned_en) and not cleaned_en.isupper():
            name_en = cleaned_en

            for j in range(i - 1, -1, -1):

                am_line = search_area[j]

                if not is_amharic(am_line):
                    continue

                # reject raw OCR line if contains symbols/numbers
                if has_invalid_amharic_chars(am_line):
                    continue

                cleaned_am = clean_amharic(am_line)

                if is_valid_amharic_name(cleaned_am):
                    name_am = cleaned_am
                    break

            break

    return name_en, name_am


# =========================================================
# 🚀 FINAL FAN EXTRACTOR (UPDATED FLEXIBLE CROP)
# =========================================================
def extract_fan(text, image_path=None, debug_dir="temp"):

    if not text or not image_path:
        return "", None

    def is_valid_fan(x):
        return (
            len(x) == 16 and
            x.isdigit() and
            not x.startswith("20") and
            len(set(x)) >= 6
        )

    img = cv2.imread(image_path)
    if img is None:
        return "", None

    h, w = img.shape[:2]

    # =========================================
    # 🔥 USE OCR BOX DATA (REAL POSITIONS)
    # =========================================
    data = pytesseract.image_to_data(
        img,
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT
    )

    exp_y = None

    for i, word in enumerate(data["text"]):
        if re.search(r"\d{4}/\d{2}/\d{2}|\d{4}/[A-Za-z]{3}/\d{2}", word):
            exp_y = data["top"][i]
            break

    # fallback if not found
    if exp_y is None:
        exp_y = int(0.45 * h)

    # =========================================
    # 🔥 STABLE RELATIVE CROP (TUNABLE)
    # =========================================
    # THESE are now reliable to tweak 👇

    shift = 0.11 * h      # 🔥 move DOWN (use negative for UP)
    box_height = 0.054 * h

    center_y = int(exp_y + shift)

    y1 = int(center_y - box_height * 0.3)
    y2 = int(center_y + box_height * 0.7)

    x1 = int(w * 0.28)
    x2 = int(w * 0.68)

    # clamp
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    if y2 <= y1:
        return "", None

    crop = img[y1:y2, x1:x2]

    # DEBUG
    fan_debug_path = os.path.join(debug_dir, "debug_fan_crop.jpg")
    cv2.imwrite(fan_debug_path, crop)

    print(f"\n📦 CROP: y1={y1}, y2={y2}, exp_y={exp_y}")

    # =========================================
    # OCR ON CROPPED REGION
    # =========================================
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.convertScaleAbs(g, alpha=3.0, beta=40)

    _, t = cv2.threshold(
        g, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    zoom = cv2.resize(t, None, fx=3, fy=3)

    ocr = pytesseract.image_to_string(
        zoom,
        config="--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"
    )

    matches = re.findall(r"\d{16}", ocr)

    for m in matches:
        if is_valid_fan(m):
            print("✅ FAN FOUND")
            return m, crop

    return "", crop


# =========================================================
# SIMPLE NAME CROP (RELATIVE TO DOB/EXP)
# =========================================================
def extract_names_from_crop(image_path, debug_dir="temp"):

    img = cv2.imread(image_path)

    if img is None:
        return "", "", None

    h, w = img.shape[:2]

    # =========================================
    # FIND DATE POSITION
    # =========================================
    data = pytesseract.image_to_data(
        img,
        config="--oem 3 --psm 6 -l amh+eng",
        output_type=pytesseract.Output.DICT
    )

    anchor_y = None

    for i, word in enumerate(data["text"]):

        word = normalize(word)

        if re.search(
            r"\d{4}/\d{2}/\d{2}|\d{4}/[A-Za-z]{3}/\d{2}",
            word
        ):
            anchor_y = data["top"][i]
            break

    # fallback
    if anchor_y is None:
        anchor_y = int(h * 0.42)

    # =========================================
    # NAME CROP
    # =========================================

    # move ABOVE dates
    shift_up = int(h * 0.02)

    # crop height
    crop_height = int(h * 0.10)

    center_y = anchor_y - shift_up

    y1 = max(0, center_y - crop_height)
    y2 = max(0, center_y)

    x1 = int(w * 0.12)
    x2 = int(w * 0.83)

    crop = img[y1:y2, x1:x2]

    cv2.imwrite(
        os.path.join(debug_dir, "debug_name_crop.jpg"),
        crop
    )

    print(f"\n📦 NAME CROP: y1={y1}, y2={y2}")

    # =========================================
    # PREPROCESS (SMOOTH + CLEAN)
    # =========================================

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # upscale FIRST using Lanczos (best for text)
    zoom = cv2.resize(
        gray,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_LANCZOS4
    )

    # smooth edges slightly
    zoom = cv2.GaussianBlur(zoom, (3, 3), 0)

    # increase contrast
    zoom = cv2.convertScaleAbs(
        zoom,
        alpha=1.8,
        beta=15
    )

    # adaptive threshold keeps smooth curves
    zoom = cv2.adaptiveThreshold(
        zoom,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        8
    )

    # tiny median cleanup
    zoom = cv2.medianBlur(zoom, 3)

    cv2.imwrite(
        os.path.join(debug_dir, "debug_name_processed.jpg"),
        zoom
    )

    # =========================================
    # OCR ENGLISH
    # =========================================
    text_en = pytesseract.image_to_string(
        zoom,
        config="--oem 1 --psm 6 -l eng"
    )

    print("\n🔤 NAME OCR ENGLISH:\n")
    print(text_en)

    name_en = ""

    # =========================================
    # PICK BEST ENGLISH NAME
    # =========================================
    candidates = []

    for line in text_en.split("\n"):

        # reject raw OCR line first
        if has_invalid_english_chars(line):
            continue

        cleaned = clean_english(line)

        print("EN TEST:", cleaned)

        if not is_valid_english_name(cleaned):
            continue

        words = cleaned.split()

        score = 0

        # prefer proper capitalization
        proper_case = sum(
            1 for w in words
            if w[0].isupper() and w[1:].islower()
        )

        score += proper_case * 10

        # prefer longer words
        score += sum(len(w) for w in words)

        # penalize weird OCR-looking words
        weird = sum(
            1 for w in words
            if (
                len(set(w.lower())) <= 2 or
                w.isupper()
            )
        )

        score -= weird * 15

        candidates.append((score, cleaned))

    # choose highest score
    name_en = ""

    if candidates:
        candidates.sort(reverse=True)
        name_en = candidates[0][1]


    # =========================================
    # OCR AMHARIC
    # =========================================
    text_am = pytesseract.image_to_string(
        zoom,
        config="--oem 3 --psm 6 -l amh"
    )

    print("\n🔠 NAME OCR AMHARIC:\n")
    print(text_am)

    name_am = ""

    for line in text_am.split("\n"):

        # reject raw OCR line first
        if has_invalid_amharic_chars(line):
            continue

        cleaned = clean_amharic(line)

        print("AM TEST:", cleaned)

        if is_valid_amharic_name(cleaned):
            name_am = cleaned
            break

    return name_en, name_am, crop
# =========================================================
# PARSER (UNCHANGED)
# =========================================================
def parse_text(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    name_en, name_am = "", ""
    gender_en, gender_am = "", ""
    dob_greg, dob_eth = "", ""
    exp_greg, exp_eth = "", ""

    anchor_i = find_dob_anchor(lines)

    if anchor_i == -1:
        return (name_en, gender_en, dob_greg, exp_greg,
                name_am, gender_am, dob_eth, exp_eth)

    name_en, name_am = extract_names(lines, anchor_i)

    for i in range(anchor_i, len(lines)):
        line = lines[i].lower()

        if "female" in line:
            gender_en = "Female"
            break
        elif "male" in line:
            gender_en = "Male"
            break

    gender_am = "ወንድ" if gender_en == "Male" else "ሴት" if gender_en == "Female" else ""

    date_pairs = []

    for i in range(anchor_i, len(lines)):
        if "|" not in lines[i]:
            continue

        left, right = split_lr(lines[i])

        if is_date(left) or is_date(right):
            date_pairs.append((normalize(left), normalize(right)))

    if len(date_pairs) >= 1:

        a, b = date_pairs[0]

        # Gregorian contains month abbreviation
        if is_gregorian_date(a):
            dob_greg, dob_eth = a, b
        elif is_gregorian_date(b):
            dob_greg, dob_eth = b, a
        else:
            # fallback
            dob_greg, dob_eth = a, b


    if len(date_pairs) >= 2:

        eth, greg = date_pairs[1]

        exp_eth = eth
        exp_greg = greg

    return (
        name_en, gender_en, dob_greg, exp_greg,
        name_am, gender_am, dob_eth, exp_eth
    )

# =========================================================
# NAME VALIDATION FOR BOT
# =========================================================
def confirm_names(data):

    name_en = clean_english(data.get("name_en", ""))
    name_am = clean_amharic(data.get("name_am", ""))

    issues = []

    # =====================================================
    # ENGLISH NAME CHECKS
    # =====================================================
    if not is_valid_english_name(name_en):

        if not name_en:
            issues.append("English name is empty")

        elif len(name_en.split()) != 3:
            issues.append("English name must contain exactly 3 names")

        else:
            issues.append("English name is invalid")

    # =====================================================
    # AMHARIC NAME CHECKS
    # =====================================================
    if not is_valid_amharic_name(name_am):

        if not name_am:
            issues.append("Amharic name is empty")

        elif len(name_am.split()) != 3:
            issues.append("Amharic name must contain exactly 3 names")

        else:
            issues.append("Amharic name is invalid")

    # =====================================================
    # RESULT
    # =====================================================
    if issues:

        return {
            "valid": False,
            "message": "incorrect name",
            "issues": issues
        }

    return {
        "valid": True,
        "message": "",
        "issues": []
    }


# =========================================================
# DATE VALIDATION FOR BOT
# =========================================================
def confirm_dates(data):

    issues = []

    # =====================================================
    # DATE CHECK FUNCTION
    # =====================================================
    def suspicious(x):

        if not x:
            return True

        cleaned = re.sub(r"[0-9A-Za-z/]", "", x)

        return len(cleaned.strip()) > 0

    # =====================================================
    # DOB GREG
    # =====================================================
    dob_greg = data.get("dob_greg", "")

    if suspicious(dob_greg) or not is_date(dob_greg):
        issues.append("DOB Gregorian is invalid")

    # =====================================================
    # DOB ETH
    # =====================================================
    dob_eth = data.get("dob_eth", "")

    if suspicious(dob_eth) or not is_date(dob_eth):
        issues.append("DOB Ethiopian is invalid")

    # =====================================================
    # EXP GREG
    # =====================================================
    exp_greg = data.get("exp_greg", "")

    if suspicious(exp_greg) or not is_date(exp_greg):
        issues.append("EXP Gregorian is invalid")

    # =====================================================
    # EXP ETH
    # =====================================================
    exp_eth = data.get("exp_eth", "")

    if suspicious(exp_eth) or not is_date(exp_eth):
        issues.append("EXP Ethiopian is invalid")

    # =====================================================
    # RESULT
    # =====================================================
    if issues:

        return {
            "valid": False,
            "message": "incorrect date",
            "issues": issues
        }

    return {
        "valid": True,
        "message": "",
        "issues": []
    }
# =========================================================
# AMHARIC OCR FIXES
# =========================================================
def fix_amharic_from_english(name_en, name_am):

    if not name_en or not name_am:
        return name_am

    # split by name position
    en_parts = clean_english(name_en).lower().split()
    am_parts = clean_amharic(name_am).split()

    if not en_parts or not am_parts:
        return name_am

    # =====================================================
    # APPLY RULES TO MATCHING NAME SEGMENT
    # =====================================================
    def apply_rules(en_word, am_word):

        if not en_word or not am_word:
            return am_word

        # =================================================
        # RULE 1:
        # if English contains "go"
        # and Amharic doesn't contain "ጎ"
        # replace ALL possible OCR mistakes
        # =================================================
        if "go" in en_word and "ጎ" not in am_word:

            am_word = am_word.replace("ነ", "ጎ")

        # =================================================
        # RULE 2:
        # if English contains "ch"
        # and Amharic doesn't contain "ች"
        # =================================================
        if "ch" in en_word and "ች" not in am_word:

            am_word = am_word.replace("ት", "ች")

        # =================================================
        # RULE 3:
        # if English contains "ra"
        # and Amharic doesn't contain "ራ"
        # =================================================
        if "ra" in en_word and "ራ" not in am_word:

            am_word = am_word.replace("ሪ", "ራ")

        return am_word

    # =====================================================
    # POSITIONAL ALIGNMENT
    # first ↔ first
    # father ↔ father
    # last ↔ last
    # =====================================================
    fixed_parts = []

    max_len = max(len(en_parts), len(am_parts))

    for i in range(max_len):

        en_word = en_parts[i] if i < len(en_parts) else ""
        am_word = am_parts[i] if i < len(am_parts) else ""

        fixed_word = apply_rules(en_word, am_word)

        fixed_parts.append(fixed_word)

    return " ".join(fixed_parts)

# =========================================================
# MAIN (UNCHANGED)
# =========================================================
def process_ocr(image_path, confirm=True, debug_dir="temp"):

    img = preprocess(image_path)
    text = get_text(img)

    (
        name_en, gender_en, dob_greg, exp_greg,
        name_am, gender_am, dob_eth, exp_eth
    ) = parse_text(text)

    # =========================================
    # NAME CROP OCR
    # =========================================
    crop_name_en, crop_name_am, name_crop = (
        extract_names_from_crop(
            image_path,
            debug_dir
        )
    )

    if crop_name_en:
        name_en = crop_name_en

    if crop_name_am:
        name_am = crop_name_am
    # =========================================
    # FIX AMHARIC OCR USING ENGLISH
    # =========================================
    name_am = fix_amharic_from_english(name_en, name_am)
    # =========================================
    # FAN OCR
    # =========================================
    fan, fan_crop = extract_fan(
        text,
        image_path,
        debug_dir
    )

    data = {
        "name_en": name_en,
        "gender_en": gender_en,
        "dob_greg": dob_greg,
        "exp_greg": exp_greg,
        "name_am": name_am,
        "gender_am": gender_am,
        "dob_eth": dob_eth,
        "exp_eth": exp_eth,
        "fan": fan
    }

    # =========================================
    # VALIDATION
    # =========================================
    problems = []

    name_check = confirm_names(data)

    if not name_check["valid"]:
        problems.append(name_check["message"])

    date_check = confirm_dates(data)

    if not date_check["valid"]:
        problems.append(date_check["message"])

    # attach issues
    data["issues"] = (
        name_check["issues"] +
        date_check["issues"]
    )

    data["problems"] = problems

    print("\n✅ FINAL EXTRACTED DATA:")
    for k, v in data.items():
        print(f"{k}: {v}")

    return data