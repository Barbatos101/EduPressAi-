import os
from pathlib import Path
import requests
import logging
from huggingface_hub import login

# Optional HF token for better reliability
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("✅ HuggingFace authentication successful")
    except Exception as e:
        print(f"⚠️ HF authentication failed: {e}")

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Detect if running on Hugging Face Spaces
IS_SPACES = os.getenv("SPACE_ID") is not None

# Use writable directories based on environment
if IS_SPACES:
    BASE_DIR = Path.home()
    MODEL_PATH = BASE_DIR / ".cache" / "models" / "BilingualModel.pt"
    OUTPUT_DIR = BASE_DIR / ".cache" / "output"
else:
    BASE_DIR = Path(__file__).parent
    MODEL_PATH = BASE_DIR / "models" / "BilingualModel.pt"
    OUTPUT_DIR = BASE_DIR / "output"

# Create required directories
for d in [MODEL_PATH.parent, OUTPUT_DIR / "images", OUTPUT_DIR / "crops", OUTPUT_DIR / "results"]:
    d.mkdir(parents=True, exist_ok=True)

def ensure_model_downloaded():
    """Download bilingual YOLO model if not present"""
    if not MODEL_PATH.exists():
        print(f"Downloading Bilingual YOLO model to {MODEL_PATH}...")
        # Updated URL for bilingual model
        url = "https://github.com/Barbatos101/newspaper-education-extractor/releases/download/v2.0/BilingualModel.pt"
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with open(MODEL_PATH, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and downloaded % (1024 * 1024) == 0:
                            percent = (downloaded / total_size) * 100
                            print(f"Download progress: {percent:.1f}%")
            print(f"✅ Bilingual model downloaded successfully to {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"❌ Error downloading bilingual model: {e}")
            print("⚠️ Continuing without model - some features may not work")
            return False
    return True

# Download model on import
model_available = ensure_model_downloaded()

# Updated settings with improved DPI and no page limits
CONFIDENCE_THRESHOLD = 0.82  # Higher confidence for bilingual model
KEYWORD_MIN_MATCH = 1  # Reduced for bilingual support
NUM_WORKERS = 1  # Single worker for stability
REDUCED_DPI = 180  # IMPROVED DPI from 120 to 180
MAX_PAGES_BATCH = 2  # Process 2 pages at a time for better memory management
MAX_INPUT_CHARS_FOR_SUMMARY = 400  # Slightly increased for better summaries
MAX_SUMMARY_LENGTH = 50  # Slightly longer summaries
SEMANTIC_ANALYSIS_ENABLED = False  # Disabled for performance
SAVE_CROPS_DEFAULT = False
MAX_PAGES_TO_PROCESS = None  # REMOVED PAGE LIMIT - Process all pages
ENABLE_QUICK_FILTER = True

print(f"🔧 Using {NUM_WORKERS} workers for processing")
print(f"🌐 Bilingual mode (English + Hindi)")
print(f"📄 DPI Quality: {REDUCED_DPI}")
print(f"📑 Page Processing: All pages (no limit)")
print(f"🎯 HF Spaces mode: {IS_SPACES}")

# Bilingual education keywords
EDUCATION_KEYWORDS_EN = [
    'school', 'schools', 'education', 'educational', 'student', 'students',
    'teacher', 'teachers', 'university', 'college', 'academic', 'classroom',
    'curriculum', 'exam', 'exams', 'graduation', 'scholarship', 'principal',
    'kindergarten', 'elementary', 'secondary', 'admission', 'enrollment',
    'learning', 'study', 'studies', 'degree', 'diploma', 'certificate'
]

EDUCATION_KEYWORDS_HI = [
    'स्कूल', 'विद्यालय', 'शिक्षा', 'शैक्षिक', 'छात्र', 'छात्रा', 'विद्यार्थी',
    'शिक्षक', 'अध्यापक', 'विश्वविद्यालय', 'महाविद्यालय', 'कॉलेज', 'पाठशाला',
    'पाठ्यक्रम', 'परीक्षा', 'प्रवेश', 'दाखिला', 'छात्रवृत्ति', 'प्रधानाचार्य',
    'प्राथमिक', 'माध्यमिक', 'उच्च', 'अध्ययन', 'पढ़ाई', 'डिग्री', 'डिप्लोमा',
    'प्रमाणपत्र', 'कक्षा', 'ज्ञान', 'बोर्ड', 'सीबीएसई', 'आईसीएसई'
]

# Combined keywords for bilingual search
EDUCATION_KEYWORDS = EDUCATION_KEYWORDS_EN + EDUCATION_KEYWORDS_HI
CORE_EDUCATION_KEYWORDS = ['school', 'education', 'student', 'teacher', 'स्कूल', 'शिक्षा', 'छात्र', 'विद्यालय']

# OCR configuration for bilingual support
OCR_LANG_EN = "eng"
OCR_LANG_HI = "hin"
OCR_LANG_BILINGUAL = "eng+hin"
OCR_PSM_PRIMARY = 6
OCR_PSM_FALLBACK = 4

# Ultra-lightweight models for HF Spaces
SUMMARIZATION_MODEL_EN = "sshleifer/distilbart-cnn-12-6"  # Smallest BART variant
SUMMARIZATION_MODEL_HI = None  # Use extractive summarization for Hindi
SEMANTIC_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"  # Smallest semantic model

SEMANTIC_THRESHOLD = 0.4

# Context exclusions (bilingual)
CONTEXT_EXCLUSIONS = [
    'weather', 'temperature', 'celsius', 'fahrenheit', 'मौसम', 'तापमान',
    'clinical study', 'medical study', 'चिकित्सा', 'स्वास्थ्य',
    'stock market', 'financial report', 'शेयर', 'बाजार',
    'sports score', 'match result', 'खेल', 'मैच'
]

# Processing timeouts - increased for all-page processing
PDF_PROCESSING_TIMEOUT = 300  # Increased for more pages
MODEL_INFERENCE_TIMEOUT = 15

# File size limits - slightly increased for better quality processing
MAX_FILE_SIZE_MB = 15  # Increased from 10MB to handle better DPI

# Language detection patterns
HINDI_PATTERNS = [
    r'[\u0900-\u097F]',  # Devanagari script
    r'[\u0980-\u09FF]',  # Bengali script (sometimes used)
]

ENGLISH_PATTERNS = [
    r'[a-zA-Z]+'
]
