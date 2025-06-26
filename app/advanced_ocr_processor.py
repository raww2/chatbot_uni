import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
import time
import warnings
import sys

# Configurar variables de entorno para CPU-only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Deshabilitar CUDA
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())

# Suprimir advertencias específicas
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*GPU.*")
warnings.filterwarnings("ignore", message=".*CUDA.*")

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suprimir logs de OCR libraries
for lib in ['ppocr', 'paddle', 'easyocr', 'tesseract']:
    logging.getLogger(lib).setLevel(logging.ERROR)

@dataclass
class OCRConfig:
    """Configuración para el procesamiento OCR"""
    dpi: int = 300  # DPI optimizado para CPU
    cpu_cores: int = max(1, multiprocessing.cpu_count() - 2)  # Dejar 2 cores libres
    chunk_size: int = 2
    languages: List[str] = None
    ocr_engine: str = "paddle"  # paddle, easyocr, tesseract
    confidence_threshold: float = 0.6
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ['es', 'en']

class AdvancedImagePreprocessor:
    """Preprocesador avanzado de imágenes optimizado para CPU"""
    
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """Aplica técnicas de mejora optimizadas para CPU"""
        try:
            # Asegurar que la imagen esté en formato correcto
            if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB
                pass  # Ya está en formato correcto
            elif len(image.shape) == 2:  # Escala de grises
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # 1. Redimensionar si es muy grande (optimización para CPU)
            height, width = image.shape[:2]
            if height > 2000 or width > 2000:
                scale_factor = min(2000/height, 2000/width)
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # 2. Convertir a PIL para mejoras
            pil_image = Image.fromarray(image)
            
            # 3. Mejorar contraste y brillo
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.3)
            
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(1.1)
            
            # 4. Aplicar nitidez
            pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
            
            # Convertir de vuelta a numpy
            enhanced = np.array(pil_image)
            
            # 5. Corrección geométrica
            enhanced = AdvancedImagePreprocessor.correct_skew(enhanced)
            
            # 6. Filtrado de ruido optimizado
            enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
            
            # 7. Conversión a escala de grises optimizada
            if len(enhanced.shape) == 3:
                gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
            else:
                gray = enhanced
            
            # 8. Binarización adaptativa mejorada
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
            )
            
            # 9. Operaciones morfológicas para limpiar
            kernel = np.ones((2,2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return binary
            
        except Exception as e:
            logger.warning(f"Error en preprocesamiento: {e}")
            # Fallback: convertir a escala de grises simple
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return image
    
    @staticmethod
    def correct_skew(image: np.ndarray) -> np.ndarray:
        """Corrección de inclinación optimizada"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Detección de bordes más eficiente
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Transformada de Hough optimizada
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=80)
            
            if lines is not None and len(lines) > 5:
                angles = []
                for rho, theta in lines[:20, 0]:  # Limitar número de líneas
                    angle = theta * 180 / np.pi
                    if 85 <= angle <= 95 or -5 <= angle <= 5:
                        if angle > 45:
                            angle = angle - 90
                        angles.append(angle)
                
                if len(angles) > 3:  # Necesitar suficientes líneas
                    median_angle = np.median(angles)
                    
                    if abs(median_angle) > 0.5:
                        rows, cols = image.shape[:2]
                        center = (cols/2, rows/2)
                        M = cv2.getRotationMatrix2D(center, median_angle, 1)
                        image = cv2.warpAffine(image, M, (cols, rows), 
                                             flags=cv2.INTER_CUBIC, 
                                             borderMode=cv2.BORDER_REPLICATE)
            
            return image
        except Exception as e:
            logger.warning(f"Error en corrección de inclinación: {e}")
            return image

class MultiEngineOCRProcessor:
    """Procesador OCR con múltiples engines como fallback"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.preprocessor = AdvancedImagePreprocessor()
        self.engines = {}
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Inicializa los engines OCR disponibles"""
        
        # 1. Intentar PaddleOCR
        if self.config.ocr_engine == "paddle" or "paddle" not in self.engines:
            try:
                from paddleocr import PaddleOCR
                
                # Configuración CPU-only compatible con versiones nuevas
                paddle_config = {
                    'lang': 'es',
                    'show_log': False,
                    'use_angle_cls': False,  # Desactivar para mejor compatibilidad
                }
                
                # Intentar parámetros adicionales solo si están disponibles
                try:
                    # Parámetros para versiones más nuevas
                    paddle_config.update({
                        'det_limit_side_len': 960,
                        'det_limit_type': 'min',
                    })
                except:
                    pass
                
                self.engines['paddle'] = PaddleOCR(**paddle_config)
                logger.info("✓ PaddleOCR inicializado correctamente (CPU)")
                
            except Exception as e:
                logger.warning(f"PaddleOCR no disponible: {e}")
        
        # 2. Intentar EasyOCR como fallback
        try:
            import easyocr
            self.engines['easyocr'] = easyocr.Reader(['es', 'en'], gpu=False, verbose=False)
            logger.info("✓ EasyOCR inicializado como fallback")
        except Exception as e:
            logger.warning(f"EasyOCR no disponible: {e}")
        
        # 3. Tesseract como último recurso
        try:
            import pytesseract
            # Verificar que tesseract esté instalado
            pytesseract.get_tesseract_version()
            self.engines['tesseract'] = pytesseract
            logger.info("✓ Tesseract disponible como último recurso")
        except Exception as e:
            logger.warning(f"Tesseract no disponible: {e}")
        
        if not self.engines:
            raise RuntimeError("No hay engines OCR disponibles. Instalar al menos uno: pip install paddlepaddle paddleocr easyocr pytesseract")
    
    def process_image(self, image: np.ndarray) -> str:
        """Procesa imagen con el engine principal y fallbacks"""
        
        # Preprocesar imagen
        try:
            enhanced_image = self.preprocessor.enhance_image(image)
        except Exception as e:
            logger.warning(f"Error en preprocesamiento, usando imagen original: {e}")
            enhanced_image = image
        
        # Intentar engines en orden de preferencia
        engines_to_try = []
        
        # Priorizar engine configurado
        if self.config.ocr_engine in self.engines:
            engines_to_try.append(self.config.ocr_engine)
        
        # Agregar otros engines como fallback
        for engine in ['paddle', 'easyocr', 'tesseract']:
            if engine in self.engines and engine not in engines_to_try:
                engines_to_try.append(engine)
        
        # Intentar cada engine
        for engine_name in engines_to_try:
            try:
                text = self._process_with_engine(enhanced_image, engine_name)
                if text and len(text.strip()) > 0:
                    return text
            except Exception as e:
                logger.warning(f"Engine {engine_name} falló: {e}")
                continue
        
        logger.warning("Todos los engines OCR fallaron")
        return ""
    
    def _process_with_engine(self, image: np.ndarray, engine_name: str) -> str:
        """Procesa imagen con un engine específico"""
        
        if engine_name == 'paddle':
            return self._process_paddle(image)
        elif engine_name == 'easyocr':
            return self._process_easyocr(image)
        elif engine_name == 'tesseract':
            return self._process_tesseract(image)
        else:
            raise ValueError(f"Engine no soportado: {engine_name}")
    
    def _process_paddle(self, image: np.ndarray) -> str:
        """Procesa con PaddleOCR"""
        results = self.engines['paddle'].ocr(image, cls=False)
        
        text_parts = []
        if results and results[0]:
            for line_result in results[0]:
                if len(line_result) >= 2:
                    text = line_result[1][0]
                    confidence = line_result[1][1]
                    
                    if confidence > self.config.confidence_threshold:
                        text_parts.append(text)
        
        return '\n'.join(text_parts)
    
    def _process_easyocr(self, image: np.ndarray) -> str:
        """Procesa con EasyOCR"""
        results = self.engines['easyocr'].readtext(image, paragraph=True)
        
        text_parts = []
        for (bbox, text, confidence) in results:
            if confidence > self.config.confidence_threshold:
                text_parts.append(text)
        
        return '\n'.join(text_parts)
    
    def _process_tesseract(self, image: np.ndarray) -> str:
        """Procesa con Tesseract"""
        import pytesseract
        
        # Configuración para español
        config = '--oem 3 --psm 6 -l spa+eng'
        
        # Convertir imagen si es necesario
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        text = pytesseract.image_to_string(image, config=config)
        return text

class AdvancedTextCleaner:
    """Limpiador de texto mejorado"""
    
    def __init__(self):
        # Correcciones específicas de OCR en español
        self.ocr_corrections = {
            # Números y letras confundidas
            r'\b0\b': 'o',
            r'\bl\b': 'I',
            r'\bI(?=\w)': 'l',
            r'\b1\b': 'l',
            r'rn\b': 'm',
            r'\bvv': 'w',
            r'([a-záéíóúñü])0([a-záéíóúñü])': r'\1o\2',
            r'([a-záéíóúñü])5([a-záéíóúñü])': r'\1s\2',
            r'([a-záéíóúñü])6([a-záéíóúñü])': r'\1b\2',
            r'([a-záéíóúñü])8([a-záéíóúñü])': r'\1b\2',
            
            # Caracteres especiales mal interpretados
            r'[''`´]': "'",
            r'["""]': '"',
            r'—': '-',
            r'…': '...',
            
            # Espacios problemáticos
            r'\s+': ' ',
            r'^\s+|\s+$': '',
        }
        
        # Patrones de limpieza
        self.cleanup_patterns = [
            (r'\n\s*\n\s*\n+', '\n\n'),  # Múltiples saltos de línea
            (r'[^\w\s.,;:()\-¿?¡!áéíóúñüÁÉÍÓÚÑÜ]', ''),  # Caracteres extraños
            (r'(.)\1{4,}', r'\1\1'),  # Caracteres repetidos excesivamente
        ]
    
    def clean_text(self, text: str) -> str:
        """Limpia el texto extraído"""
        if not text or not text.strip():
            return ""
        
        try:
            # 1. Normalización Unicode
            text = unicodedata.normalize('NFKC', text)
            
            # 2. Correcciones de OCR
            for pattern, replacement in self.ocr_corrections.items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
            # 3. Limpieza general
            for pattern, replacement in self.cleanup_patterns:
                text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
            
            # 4. Corrección de palabras partidas
            text = self._fix_broken_words(text)
            
            # 5. Mejorar estructura
            text = self._improve_structure(text)
            
            # 6. Limpieza final
            text = self._final_cleanup(text)
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"Error en limpieza de texto: {e}")
            return text.strip()
    
    def _fix_broken_words(self, text: str) -> str:
        """Corrige palabras partidas"""
        # Palabras con guión al final de línea
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Palabras partidas sin guión (conservador)
        lines = text.split('\n')
        result = []
        
        i = 0
        while i < len(lines):
            current = lines[i].strip()
            
            if i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                
                # Condiciones para unir líneas
                if (current and next_line and 
                    current[-1].islower() and next_line[0].islower() and
                    len(current.split()[-1]) < 4 and len(next_line.split()[0]) < 6):
                    
                    result.append(current + next_line)
                    i += 2  # Saltar la siguiente línea
                    continue
            
            result.append(current)
            i += 1
        
        return '\n'.join(result)
    
    def _improve_structure(self, text: str) -> str:
        """Mejora la estructura del documento"""
        # Detectar títulos y secciones
        text = re.sub(r'\n([A-ZÁÉÍÓÚÑÜ][A-ZÁÉÍÓÚÑÜ\s]{10,})\n', r'\n\n\1\n\n', text)
        
        # Detectar párrafos
        text = re.sub(r'\n([A-ZÁÉÍÓÚÑÜ])', r'\n\n\1', text)
        
        # Limpiar múltiples espacios
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def _final_cleanup(self, text: str) -> str:
        """Limpieza final"""
        lines = [line.strip() for line in text.split('\n')]
        
        # Eliminar líneas muy cortas que probablemente son artefactos
        cleaned_lines = []
        for line in lines:
            if not line:
                cleaned_lines.append('')
            elif len(line) > 2 or any(c.isalnum() for c in line):
                cleaned_lines.append(line)
        
        # Eliminar líneas vacías al inicio y final
        while cleaned_lines and not cleaned_lines[0]:
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)

class DocumentProcessor:
    """Procesador principal optimizado para CPU"""
    
    def __init__(self, config: OCRConfig = None):
        self.config = config or OCRConfig()
        self.ocr_processor = MultiEngineOCRProcessor(self.config)
        self.text_cleaner = AdvancedTextCleaner()
        
        # Configurar directorios
        self.base_dir = Path(__file__).resolve().parent.parent
        self.raw_dir = self.base_dir / "data" / "raw"
        self.cleaned_dir = self.base_dir / "data" / "cleaned_advanced"
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        
        # Estadísticas
        self.reset_stats()
    
    def reset_stats(self):
        """Reinicia estadísticas"""
        self.stats = {
            'processed_files': 0,
            'total_pages': 0,
            'processing_time': 0,
            'errors': [],
            'successful_files': [],
            'failed_files': []
        }
    
    def process_pdf_file(self, pdf_path: Path) -> Dict:
        """Procesa un archivo PDF individual"""
        start_time = time.time()
        file_stats = {
            'filename': pdf_path.name,
            'pages': 0,
            'processing_time': 0,
            'success': True,
            'error': None,
            'text_length': 0,
            'word_count': 0
        }
        
        try:
            logger.info(f"[INICIO] Procesando: {pdf_path.name}")
            
            # Convertir PDF a imágenes con configuración optimizada
            try:
                images = convert_from_path(
                    str(pdf_path), 
                    dpi=self.config.dpi,
                    thread_count=min(4, self.config.cpu_cores),  # Limitar threads
                    fmt='RGB'  # Formato específico
                )
            except Exception as e:
                logger.error(f"Error convirtiendo PDF {pdf_path.name}: {e}")
                raise
            
            file_stats['pages'] = len(images)
            logger.info(f"  - Convertidas {len(images)} páginas")
            
            # Procesar páginas
            if len(images) <= 4:  # Pocos archivos: procesamiento secuencial
                full_text = self._process_pages_sequential(images)
            else:  # Muchas páginas: procesamiento paralelo controlado
                full_text = self._process_pages_parallel(images)
            
            # Limpieza final
            final_text = self.text_cleaner.clean_text(full_text)
            
            # Estadísticas del texto
            file_stats['text_length'] = len(final_text)
            file_stats['word_count'] = len(final_text.split())
            
            # Guardar resultado
            output_path = self.cleaned_dir / f"{pdf_path.stem}.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_text)
            
            processing_time = time.time() - start_time
            file_stats['processing_time'] = processing_time
            
            logger.info(f"[ÉXITO] {pdf_path.name} -> {output_path.name}")
            logger.info(f"  - Tiempo: {processing_time:.2f}s, Palabras: {file_stats['word_count']}")
            
            self.stats['successful_files'].append(pdf_path.name)
            
        except Exception as e:
            file_stats['success'] = False
            file_stats['error'] = str(e)
            logger.error(f"[ERROR] {pdf_path.name}: {e}")
            self.stats['errors'].append(f"{pdf_path.name}: {e}")
            self.stats['failed_files'].append(pdf_path.name)
        
        return file_stats
    
    def _process_pages_sequential(self, images: List[Image.Image]) -> str:
        """Procesamiento secuencial para pocos archivos"""
        page_texts = []
        
        for i, img in enumerate(images):
            try:
                img_array = np.array(img)
                text = self.ocr_processor.process_image(img_array)
                page_texts.append(f"--- PÁGINA {i+1} ---\n{text}\n")
                logger.info(f"    ✓ Página {i+1} procesada")
            except Exception as e:
                logger.warning(f"    ✗ Error página {i+1}: {e}")
                page_texts.append(f"--- PÁGINA {i+1} (ERROR) ---\n")
        
        return "\n".join(page_texts)
    
    def _process_pages_parallel(self, images: List[Image.Image]) -> str:
        """Procesamiento paralelo controlado"""
        # Convertir a arrays numpy
        image_arrays = [(i, np.array(img)) for i, img in enumerate(images)]
        
        # Usar menos workers para evitar sobrecarga
        max_workers = min(self.config.cpu_cores // 2, len(images), 4)
        
        page_texts = [''] * len(images)  # Preservar orden
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Crear tareas
            futures = {
                executor.submit(self._process_single_page, i, img_array): i 
                for i, img_array in image_arrays
            }
            
            # Recoger resultados
            for future in futures:
                page_idx = futures[future]
                try:
                    text = future.result(timeout=180)  # 3 minutos por página
                    page_texts[page_idx] = f"--- PÁGINA {page_idx+1} ---\n{text}\n"
                    logger.info(f"    ✓ Página {page_idx+1} procesada")
                except Exception as e:
                    logger.warning(f"    ✗ Error página {page_idx+1}: {e}")
                    page_texts[page_idx] = f"--- PÁGINA {page_idx+1} (ERROR) ---\n"
        
        return "\n".join(page_texts)
    
    def _process_single_page(self, page_num: int, image_array: np.ndarray) -> str:
        """Procesa una página individual"""
        return self.ocr_processor.process_image(image_array)
    
    def process_all_documents(self):
        """Procesa todos los documentos"""
        pdf_files = list(self.raw_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No se encontraron PDFs en {self.raw_dir}")
            return
        
        logger.info(f"[INICIO] Procesando {len(pdf_files)} documentos")
        logger.info(f"Configuración: DPI={self.config.dpi}, CPU={self.config.cpu_cores}")
        
        start_time = time.time()
        
        for pdf_file in pdf_files:
            file_stats = self.process_pdf_file(pdf_file)
            self.stats['processed_files'] += 1
            self.stats['total_pages'] += file_stats['pages']
        
        self.stats['processing_time'] = time.time() - start_time
        self._print_final_stats()
    
    def _print_final_stats(self):
        """Muestra estadísticas finales"""
        stats = self.stats
        
        print("\n" + "="*70)
        print("ESTADÍSTICAS FINALES - PROCESAMIENTO OCR")
        print("="*70)
        print(f"Archivos procesados: {stats['processed_files']}")
        print(f"Archivos exitosos: {len(stats['successful_files'])}")
        print(f"Archivos fallidos: {len(stats['failed_files'])}")
        print(f"Total páginas: {stats['total_pages']}")
        print(f"Tiempo total: {stats['processing_time']:.2f} segundos")
        
        if stats['processed_files'] > 0:
            print(f"Tiempo promedio por archivo: {stats['processing_time']/stats['processed_files']:.2f}s")
        if stats['total_pages'] > 0:
            print(f"Tiempo promedio por página: {stats['processing_time']/stats['total_pages']:.2f}s")
        
        if stats['successful_files']:
            print(f"\nArchivos exitosos:")
            for filename in stats['successful_files'][:5]:
                print(f"  ✓ {filename}")
            if len(stats['successful_files']) > 5:
                print(f"  ... y {len(stats['successful_files']) - 5} más")
        
        if stats['errors']:
            print(f"\nErrores encontrados:")
            for error in stats['errors'][:3]:
                print(f"  ✗ {error}")
            if len(stats['errors']) > 3:
                print(f"  ... y {len(stats['errors']) - 3} más")
        
        print(f"\nResultados guardados en: {self.cleaned_dir}")
        print("="*70)

def main():
    """Función principal"""
    # Configuración optimizada para CPU
    config = OCRConfig(
        dpi=300,  # Balanceado entre calidad y velocidad
        cpu_cores=max(2, multiprocessing.cpu_count() - 2),  # Dejar cores libres
        ocr_engine="paddle",  # Engine principal
        confidence_threshold=0.6,
        languages=['es', 'en']
    )
    
    logger.info("Iniciando procesador OCR avanzado (CPU-only)")
    logger.info(f"Configuración: {config}")
    
    try:
        processor = DocumentProcessor(config)
        processor.process_all_documents()
    except Exception as e:
        logger.error(f"Error crítico: {e}")
        sys.exit(1)

# if __name__ == "__main__":
#     main()