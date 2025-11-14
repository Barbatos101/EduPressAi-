import os
import cv2
import fitz
import numpy as np
import pytesseract
import json
import re
import logging
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import torch
from transformers import pipeline
import threading
from datetime import datetime, timezone
from config import *

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_language(text: str) -> str:
    """Simple language detection for English/Hindi"""
    if not text:
        return "en"
    
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    total_chars = hindi_chars + english_chars
    
    if total_chars == 0:
        return "en"
    
    hindi_ratio = hindi_chars / total_chars
    return "hi" if hindi_ratio > 0.3 else "en"

def crop_and_deskew_minimal(img):
    """Ultra-minimal image preprocessing for speed"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        # Skip deskewing for maximum speed on HF Spaces
        return gray
    except Exception:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

def clean_ocr_text_bilingual(text: str, language: str = "en") -> str:
    """Optimized bilingual text cleaning"""
    if not text:
        return ""
    
    try:
        # Remove common OCR artifacts
        text = re.sub(r'[|\\/=↔_•¤©®™]+', '', text)
        
        if language == "hi":
            # Hindi-specific cleaning
            text = re.sub(r'[^\u0900-\u097F\w\s.,!?:;\'\"()\-\n]', ' ', text)
        else:
            # English cleaning
            text = re.sub(r'[^\w\s.,!?:;\'\"()\-\n]', ' ', text)
        
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'-\s*\n\s*', '', text)
        
        # Filter lines
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 8 and (re.search(r'[a-zA-Z\u0900-\u097F]', line)):
                cleaned_lines.append(line)
        
        text = ' '.join(cleaned_lines)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    except Exception:
        return str(text).strip() if text else ""

class BilingualEducationFilter:
    """Optimized bilingual education filtering"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.education_keywords = EDUCATION_KEYWORDS
        self.context_exclusions = CONTEXT_EXCLUSIONS
        self.core_keywords = CORE_EDUCATION_KEYWORDS
        self.logger.info("Initialized bilingual education filter")
    
    def quick_education_filter(self, text: str) -> bool:
        """Ultra-fast keyword check"""
        if len(text.strip()) < 20:
            return False
        
        text_lower = text.lower()
        
        # Check exclusions first
        for exclusion in self.context_exclusions:
            if exclusion in text_lower:
                return False
        
        # Quick keyword count (bilingual)
        keyword_count = 0
        for keyword in self.core_keywords:
            if keyword.lower() in text_lower:
                keyword_count += 1
                if keyword_count >= 1:  # Early exit
                    return True
        
        return False
    
    def is_education_article(self, text: str, min_keywords: int = 1) -> Tuple[bool, List[str], Dict]:
        """Optimized bilingual education analysis"""
        if not text or len(text.strip()) < 20:
            return False, [], {}
        
        text_lower = text.lower()
        
        # Check exclusions
        for exclusion in self.context_exclusions:
            if exclusion in text_lower:
                return False, [], {"exclusion_found": exclusion}
        
        # Find keywords
        found_keywords = []
        for keyword in self.education_keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        # Check core keywords
        has_core_keyword = any(core.lower() in text_lower for core in self.core_keywords)
        
        analysis_details = {
            "keyword_count": len(found_keywords),
            "has_core_keyword": has_core_keyword,
            "language": detect_language(text),
            "bilingual_processing": True
        }
        
        is_education = len(found_keywords) >= min_keywords and has_core_keyword
        analysis_details["is_education"] = is_education
        
        return is_education, found_keywords, analysis_details

class BilingualNewspaperExtractor:
    def __init__(
        self,
        min_keyword_matches: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        num_workers: Optional[int] = None,
        save_crops: bool = None,
    ):
        """Initialize bilingual extractor"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Configure OpenCV for minimal resource usage
        cv2.setNumThreads(1)
        try:
            cv2.ocl.setUseOpenCL(False)
        except:
            pass
        
        # Settings
        self.keyword_min_match = min_keyword_matches or KEYWORD_MIN_MATCH
        self.confidence_threshold = confidence_threshold or CONFIDENCE_THRESHOLD
        self.num_workers = num_workers or NUM_WORKERS
        self.save_crops = save_crops if save_crops is not None else SAVE_CROPS_DEFAULT
        
        # Threading locks
        self._ocr_lock = threading.Lock()
        self._summ_lock = threading.Lock()
        
        # Load bilingual YOLO model
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Bilingual model not found: {MODEL_PATH}")
        
        self.yolo_model = YOLO(str(MODEL_PATH))
        self.education_filter = BilingualEducationFilter()
        
        self.logger.info("Loaded bilingual YOLO model and education filter")
        
        # Initialize lightweight summarization
        self.logger.info("Loading ultra-lightweight summarization...")
        try:
            self.summarizer_en = pipeline(
                "summarization",
                model=SUMMARIZATION_MODEL_EN,
                device=-1,
                torch_dtype=torch.float16,
                model_kwargs={"low_cpu_mem_usage": True}
            )
            self.logger.info(f"English summarizer loaded: {SUMMARIZATION_MODEL_EN}")
        except Exception as e:
            self.logger.warning(f"English summarization failed: {e}")
            self.summarizer_en = None
        
        # Hindi uses extractive summarization (no model needed)
        self.summarizer_hi = None
        self.logger.info("Hindi extractive summarization ready")
        
        self.logger.info("Bilingual extractor initialized successfully")
    
    def process_newspaper(self, pdf_path: str) -> Dict:
        """Updated bilingual processing pipeline - ALL PAGES, IMPROVED DPI"""
        self.logger.info(f"Processing bilingual newspaper with improved settings: {pdf_path}")
        
        # Convert PDF with improved DPI (180) and NO page limit
        image_paths = self.pdf_to_images(pdf_path)
        if not image_paths:
            return {"error": "Failed to convert PDF to images"}
        
        # Process ALL pages (no limit)
        self.logger.info(f"Processing ALL {len(image_paths)} pages at {REDUCED_DPI} DPI")
        
        education_articles = []
        stats = {
            'total_pages': len(image_paths),
            'total_articles_detected': 0,
            'education_articles_found': 0,
            'english_articles': 0,
            'hindi_articles': 0,
            'bilingual_model': True,
            'optimized_for_spaces': True,
            'dpi_quality': REDUCED_DPI,
            'pages_processed': len(image_paths)  # Track actual pages processed
        }
        
        # Process each page sequentially for stability
        for page_num, image_path in enumerate(image_paths, 1):
            self.logger.info(f"Processing page {page_num}/{len(image_paths)} at {REDUCED_DPI} DPI")
            
            articles = self.detect_articles(image_path, page_num)
            stats['total_articles_detected'] += len(articles)
            
            for article in articles:
                try:
                    result = self._process_single_article(article, page_num)
                    if result:
                        education_articles.append(result)
                        stats['education_articles_found'] += 1
                        
                        # Track language stats
                        lang = result.get('language', 'en')
                        if lang == 'hi':
                            stats['hindi_articles'] += 1
                        else:
                            stats['english_articles'] += 1
                            
                except Exception as e:
                    self.logger.warning(f"Article processing error: {e}")
            
            # Log progress for long documents
            if page_num % 5 == 0:
                self.logger.info(f"Progress: {page_num}/{len(image_paths)} pages completed, {stats['education_articles_found']} education articles found")
        
        # Compile results
        results = {
            'pdf_path': pdf_path,
            'processing_stats': stats,
            'education_articles': education_articles,
            'analysis_summary': {
                'bilingual_support': True,
                'languages_detected': ['English', 'Hindi'],
                'total_articles_analyzed': stats['total_articles_detected'],
                'dpi_quality': REDUCED_DPI,
                'all_pages_processed': True
            },
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'optimized_processing': True
        }
        
        # Save results
        try:
            pdf_name = Path(pdf_path).stem
            results_file = OUTPUT_DIR / "results" / f"{pdf_name}_bilingual_education_improved.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Processing complete! Found {len(education_articles)} education articles from {len(image_paths)} pages")
            self.logger.info(f"Results saved to: {results_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
        
        return results
    
    def pdf_to_images(self, pdf_path: str, dpi: int = None) -> List[str]:
        """PDF conversion with improved 180 DPI"""
        dpi = dpi or REDUCED_DPI  # Now 180 DPI
        self.logger.info(f"Converting PDF at improved {dpi} DPI (bilingual optimized)")
        
        try:
            pdf_document = fitz.open(pdf_path)
            image_paths = []
            pdf_name = Path(pdf_path).stem
            
            # Process ALL pages (no limit)
            total_pages = pdf_document.page_count
            self.logger.info(f"Converting all {total_pages} pages at {dpi} DPI")
            
            for page_num in range(total_pages):
                page = pdf_document[page_num]
                mat = fitz.Matrix(dpi/72, dpi/72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                image_filename = f"{pdf_name}_page_{page_num + 1}.jpg"
                image_path = OUTPUT_DIR / "images" / image_filename
                pix.save(str(image_path), "JPEG", jpg_quality=85)  # Slightly higher quality
                image_paths.append(str(image_path))
                
                # Progress logging for large documents
                if (page_num + 1) % 10 == 0:
                    self.logger.info(f"Converted {page_num + 1}/{total_pages} pages")
            
            pdf_document.close()
            self.logger.info(f"Successfully converted all {len(image_paths)} pages")
            return image_paths
        except Exception as e:
            self.logger.error(f"PDF conversion error: {e}")
            return []
    
    def detect_articles(self, image_path: str, page_num: int) -> List[Dict]:
        """Optimized bilingual article detection"""
        try:
            # Use moderate image size for good balance of speed and quality
            results = self.yolo_model.predict(
                source=image_path,
                conf=self.confidence_threshold,
                imgsz=416,  # Moderate size for better quality with 180 DPI
                verbose=False,
                save=False
            )
            
            articles = []
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for i, (box, conf) in enumerate(zip(boxes, confidences)):
                        x1, y1, x2, y2 = map(int, box)
                        area = (x2 - x1) * (y2 - y1)
                        
                        # Adjusted area threshold for 180 DPI
                        min_area = 15000  # Increased for higher DPI
                        if area > min_area:
                            articles.append({
                                'article_id': i + 1,
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(conf),
                                'area': area,
                                'image_path': image_path,
                                'page': page_num
                            })
            
            return articles
        except Exception as e:
            self.logger.error(f"Article detection error: {e}")
            return []
    
    def _process_single_article(self, article: Dict, page_num: int) -> Optional[Dict]:
        """Optimized bilingual article processing"""
        try:
            # Extract text with bilingual OCR
            crop_path, text, language = self.extract_article_crop_and_text(article)
            
            if len(text.strip()) < 30:
                return None
            
            # Quick pre-filtering
            if not self.education_filter.quick_education_filter(text):
                return None
            
            # Education filtering
            is_education, found_keywords, analysis_details = self.education_filter.is_education_article(
                text, self.keyword_min_match
            )
            
            if not is_education:
                return None
            
            # Generate summary based on language
            summary = self.summarize_text(text, language)
            
            return {
                'page': page_num,
                'article_id': article['article_id'],
                'confidence': article['confidence'],
                'bbox': article['bbox'],
                'language': language,
                'keywords_found': found_keywords,
                'full_text': text,
                'summary': summary,
                'crop_path': crop_path,
                'text_length': len(text),
                'analysis_details': analysis_details,
                'bilingual_processing': True
            }
            
        except Exception as e:
            self.logger.error(f"Article processing error: {e}")
            return None
    
    def extract_article_crop_and_text(self, article_data: Dict) -> Tuple[str, str, str]:
        """Optimized bilingual OCR with improved quality"""
        try:
            img = cv2.imread(article_data['image_path'])
            if img is None:
                return "", "", "en"
            
            x1, y1, x2, y2 = article_data['bbox']
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0:
                return "", "", "en"
            
            crop_path_str = ""
            if self.save_crops:
                try:
                    image_name = Path(article_data['image_path']).stem
                    crop_filename = f"{image_name}_article_{article_data['article_id']}.jpg"
                    crop_path = OUTPUT_DIR / "crops" / crop_filename
                    cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    crop_path_str = str(crop_path)
                except Exception:
                    pass
            
            # Bilingual OCR with minimal preprocessing
            with self._ocr_lock:
                gray = crop_and_deskew_minimal(crop)
                
                # Try bilingual OCR first
                ocr_config = f'--oem 1 --psm {OCR_PSM_PRIMARY} -l {OCR_LANG_BILINGUAL}'
                text = pytesseract.image_to_string(gray, config=ocr_config)
                
                # Detect language
                language = detect_language(text)
                
                # Clean text based on detected language
                text = clean_ocr_text_bilingual(text, language)
                
                return crop_path_str, text, language
                
        except Exception as e:
            self.logger.error(f"OCR error: {e}")
            return "", "", "en"
    
    def summarize_text(self, text: str, language: str) -> str:
        """Bilingual text summarization"""
        try:
            if not text or len(text.strip()) < 40:
                return text if text else ""
            
            cleaned_text = clean_ocr_text_bilingual(text, language)
            if len(cleaned_text.strip()) < 40:
                return cleaned_text
            
            if language == "hi":
                # Extractive summarization for Hindi
                sentences = re.split(r'[।|.|!|?]\s*', cleaned_text)
                sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
                
                if len(sentences) >= 2:
                    return sentences[0] + "। " + sentences[1] + "।"
                elif len(sentences) == 1:
                    return sentences[0] + "।"
                else:
                    return cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text
            
            else:
                # English summarization with ultra-light model
                if self.summarizer_en:
                    try:
                        with self._summ_lock:
                            input_text = cleaned_text[:MAX_INPUT_CHARS_FOR_SUMMARY]
                            summary = self.summarizer_en(
                                input_text,
                                max_length=MAX_SUMMARY_LENGTH,
                                min_length=20,
                                do_sample=False
                            )
                            return summary[0]['summary_text']
                    except Exception:
                        pass
                
                # Fallback extractive summarization for English
                sentences = re.split(r'[.!?]\s+', cleaned_text)
                sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
                
                if len(sentences) >= 2:
                    return sentences[0] + ". " + sentences[1] + "."
                elif len(sentences) == 1:
                    return sentences[0] + "."
                else:
                    return cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text
        
        except Exception as e:
            self.logger.error(f"Summarization error: {e}")
            return text[:200] + "..." if text and len(text) > 200 else (text if text else "")
