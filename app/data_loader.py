import os
import re
from tqdm import tqdm
from pathlib import Path
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEANED_DIR = BASE_DIR / "data" / "cleaned"

CLEANED_DIR.mkdir(parents=True, exist_ok=True)

def is_scanned_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            if page.extract_text():
                return False
        return True
    except:
        return True

def extract_text_from_pdf(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

def extract_text_from_scanned(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    text = ""
    for img in images:
        gray = img.convert("L")  # escala de grises
        text += pytesseract.image_to_string(gray, lang="spa+eng") + "\n"
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    return text.strip()

def process_all_pdfs():
    pdf_files = list(RAW_DIR.glob("*.pdf"))
    for pdf_path in tqdm(pdf_files, desc="Procesando PDFs"):
        try:
            if is_scanned_pdf(pdf_path):
                print(pdf_path)
                print("Procesando pdf escaneado")
                raw_text = extract_text_from_scanned(str(pdf_path))
            else:
                print(pdf_path)
                print("Procesando pdf no escaneado")
                raw_text = extract_text_from_pdf(str(pdf_path))

            cleaned = clean_text(raw_text)
            output_file = CLEANED_DIR / (pdf_path.stem + ".txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(cleaned)
        except Exception as e:
            print(f"Error procesando {pdf_path.name}: {e}")

# if __name__ == "__main__":
#     print("Procesando documentos")
#     process_all_pdfs()
