#!/usr/bin/env python3
"""
EduPress AI - Educational Content Extractor
"""

import os
import sys
import argparse
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from extractor import BilingualNewspaperExtractor
from config import IS_SPACES

def main():
    parser = argparse.ArgumentParser(
        description='Extract education articles from newspaper PDFs (English + Hindi)'
    )
    parser.add_argument('pdf_path', help='Path to the newspaper PDF file')
    parser.add_argument('--min-keywords', type=int, default=None, help='Minimum education keywords required')
    parser.add_argument('--conf-threshold', type=float, default=None, help='YOLO confidence threshold')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker threads')
    parser.add_argument('--save-crops', action='store_true', help='Save cropped article images')

    args = parser.parse_args()

    # Validate PDF path
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    try:
        print("Initializing Bilingual Newspaper Education Extractor...")
        print("Languages supported: English + Hindi | समर्थित भाषाएं: अंग्रेजी + हिंदी")
        
        if IS_SPACES:
            print("Mode: HuggingFace Spaces optimized")
        else:
            print("Mode: Full feature processing")

        extractor = BilingualNewspaperExtractor(
            min_keyword_matches=args.min_keywords,
            confidence_threshold=args.conf_threshold,
            num_workers=args.workers,
            save_crops=args.save_crops,
        )

        print(f"Processing: {pdf_path}")
        results = extractor.process_newspaper(str(pdf_path))

        # Display results
        stats = results.get('processing_stats', {})
        print(f"\n📊 Processing Results:")
        print(f"📄 Pages processed: {stats.get('total_pages', 0)}")
        print(f"🔍 Articles detected: {stats.get('total_articles_detected', 0)}")
        print(f"🎓 Education articles found: {stats.get('education_articles_found', 0)}")
        print(f"🇺🇸 English articles: {stats.get('english_articles', 0)}")
        print(f"🇮🇳 Hindi articles: {stats.get('hindi_articles', 0)}")
        
        print(f"\n📁 Detailed results saved to: output/results/")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
