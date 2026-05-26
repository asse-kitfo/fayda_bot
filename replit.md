# Telegram OCR Bot

A Telegram bot that extracts data from Ethiopian ID card images using OCR and generates new ID card TIFF images.

## How It Works

1. User sends a **Front ID** screenshot → bot extracts name, DOB, gender, etc. via Tesseract OCR
2. User sends a **Face Photo** → bot removes background and generates a Front ID TIFF
3. User sends a **Back ID** screenshot → bot extracts FIN, QR code, and address, then generates a Back ID TIFF

## Setup

- **Runtime**: Python 3.12
- **Entry point**: `python main.py`
- **Required secret**: `BOT_TOKEN` — your Telegram bot token from @BotFather

## Project Structure

- `app/bot.py` — Main bot logic and Telegram handler
- `app/ocr.py` — Front-side OCR (Tesseract)
- `app/back_ocr.py` — Back-side OCR and QR extraction
- `app/generator.py` — Front ID image generation
- `app/back_generator.py` — Back ID image generation
- `assets/fonts/` — TTF fonts for Amharic and English text rendering
- `assets/templates/` — TIFF templates for front and back ID
- `config/coords.json` — Drawing coordinates for front side
- `config/back_coords.json` — Drawing coordinates for back side

## User Preferences

- No frontend; this is a pure backend Telegram bot worker
