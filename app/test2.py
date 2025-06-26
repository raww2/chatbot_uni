import os
import easyocr
from pdf2image import convert_from_path
from pathlib import Path
from PIL import Image
import numpy as np

# Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEANED_DIR = BASE_DIR / "data" / "cleaned2"
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

# Inicializar EasyOCR
reader = easyocr.Reader(['es', 'en'])  # español e inglés

# Procesar cada archivo PDF
for filename in os.listdir(RAW_DIR):
    if filename.lower().endswith('.pdf'):
        pdf_path = os.path.join(RAW_DIR, filename)
        print(f"[INFO] Procesando: {filename}")

        try:
            # Convertir PDF a imágenes
            images = convert_from_path(pdf_path, dpi=300)

            full_text = ""
            for page_num, img in enumerate(images):
                print(f"    - Página {page_num + 1}")
                img_array = np.array(img)
                results = reader.readtext(img_array, detail=0, paragraph=True)
                page_text = "\n".join(results)
                full_text += page_text + "\n\n"

            # Guardar texto extraído
            clean_name = os.path.splitext(filename)[0]
            txt_path = os.path.join(CLEANED_DIR, f"{clean_name}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            print(f"[OK] Guardado en: {txt_path}")

        except Exception as e:
            print(f"[ERROR] Falló en {filename}: {e}")
