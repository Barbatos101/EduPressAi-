import tempfile
from pathlib import Path
import json
import os
import streamlit as st

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from extractor import BilingualNewspaperExtractor
from config import CONFIDENCE_THRESHOLD, KEYWORD_MIN_MATCH, NUM_WORKERS, IS_SPACES, MAX_FILE_SIZE_MB

st.set_page_config(
    page_title="EduPressAi - Educational Content Extractor",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌐 EduPressAi - Educational Content Extractor")
st.caption("Upload a newspaper PDF to detect education-related articles in English & Hindi | अंग्रेजी और हिंदी में शिक्षा संबंधी लेख खोजने के लिए PDF अपलोड करें")

# Health check
try:
    if st.query_params.get("health") == "check":
        st.write("OK")
        st.stop()
except AttributeError:
    try:
        params = st.experimental_get_query_params()
        if params.get("health", [None])[0] == "check":
            st.write("OK")
            st.stop()
    except:
        pass

def display_image_compatible(image_path, caption, width=400):
    """Display image with compatibility"""
    try:
        st.image(str(image_path), caption=caption, use_container_width=True)
    except TypeError:
        st.image(str(image_path), caption=caption, width=width)

def main():
    # Initialize session state
    if "results" not in st.session_state:
        st.session_state.results = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False

    with st.sidebar:
        st.header("⚙️ Settings | सेटिंग्स")
        
        st.info("🌐 **Bilingual Support** | **द्विभाषी समर्थन**")
        st.write("• English articles | अंग्रेजी लेख")
        st.write("• Hindi articles | हिंदी लेख")
        st.write("• Fast processing | तेज़ प्रसंस्करण")
        
        conf_threshold = st.slider(
            "YOLO Confidence | YOLO विश्वसनीयता",
            0.5, 0.95,
            value=0.82,
            step=0.01,
            help="Higher values = more precise detection | उच्च मान = अधिक सटीक पहचान"
        )

        min_keywords = st.slider(
            "Min Keywords | न्यूनतम कीवर्ड",
            1, 3,
            value=1,
            step=1,
            help="Minimum education keywords required | आवश्यक न्यूनतम शिक्षा कीवर्ड"
        )

        save_crops = st.checkbox(
            "Save article crops | लेख क्रॉप सेव करें",
            value=False,
            help="Save cropped images of detected articles | पहचाने गए लेखों की क्रॉप की गई छवियां सहेजें"
        )

        st.markdown("---")
        st.markdown("🚀 **Powered by | द्वारा संचालित:**")
        st.markdown("• Bilingual YOLO v8 | द्विभाषी YOLO v8")
        st.markdown("• Tesseract OCR (En+Hi)")
        st.markdown("• DistilBART + Extractive Summarization")

        st.info("💡 **Performance Tips | प्रदर्शन सुझाव:**")
        st.write("• Use clear PDFs under 15MB | 15MB से कम के स्पष्ट PDF का उपयोग करें")
        st.write("• Both languages supported | दोनों भाषाएं समर्थित हैं")

    # File uploader
    uploaded_pdf = st.file_uploader(
        f"Upload newspaper PDF (max {MAX_FILE_SIZE_MB}MB) | अख़बार PDF अपलोड करें",
        type=["pdf"],
        help=f"Select a clear newspaper PDF file | स्पष्ट अखबार PDF फ़ाइल का चयन करें"
    )

    # File validation
    if uploaded_pdf is not None:
        file_size_mb = uploaded_pdf.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"📄 File too large: {file_size_mb:.1f}MB | फ़ाइल बहुत बड़ी है")
            st.error(f"🚫 Maximum allowed: {MAX_FILE_SIZE_MB}MB | अधिकतम अनुमतित")
            return
        elif file_size_mb > MAX_FILE_SIZE_MB * 0.8:
            st.warning(f"⚠️ Large file ({file_size_mb:.1f}MB) - processing may take longer | बड़ी फ़ाइल - प्रसंस्करण में अधिक समय लग सकता है")
        else:
            st.success(f"✅ File ready: {file_size_mb:.1f}MB | फ़ाइल तैयार है")

        st.session_state.uploaded_file_name = uploaded_pdf.name

    # Main extraction button
    extract_button = st.button(
        "🚀 Extract Bilingual Education Articles | द्विभाषी शिक्षा लेख निकालें",
        type="primary",
        disabled=uploaded_pdf is None
    )

    # Processing
    if extract_button and uploaded_pdf is not None:
        st.session_state.results = None
        st.session_state.processing_complete = False

        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            tmp_path = tmp.name

        # Initialize extractor
        with st.spinner("🔧 Loading bilingual AI models... | द्विभाषी AI मॉडल लोड हो रहे हैं..."):
            try:
                extractor = BilingualNewspaperExtractor(
                    min_keyword_matches=min_keywords,
                    confidence_threshold=conf_threshold,
                    num_workers=NUM_WORKERS,
                    save_crops=save_crops,
                )
            except Exception as e:
                st.error(f"❌ Failed to load models: {str(e)} | मॉडल लोड करने में विफल")
                return

        # Processing with progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Replace the processing section with this updated version:
        try:
            status_text.text("📄 Converting PDF pages at 180 DPI... | 180 DPI पर PDF पृष्ठ परिवर्तित कर रहे हैं...")
            progress_bar.progress(20)
        
            status_text.text("🎯 Detecting articles with bilingual YOLO... | द्विभाषी YOLO से लेख खोज रहे हैं...")
            progress_bar.progress(40)
        
            status_text.text("📝 Extracting bilingual text from all pages... | सभी पृष्ठों से द्विभाषी टेक्स्ट निकाल रहे हैं...")
            progress_bar.progress(60)
        
            status_text.text("🧠 Analyzing education content across all pages... | सभी पृष्ठों में शिक्षा सामग्री का विश्लेषण...")
            progress_bar.progress(80)
        
            # Process the PDF
            results = extractor.process_newspaper(tmp_path)
            
            progress_bar.progress(100)
            status_text.text("✅ All pages processed successfully! | सभी पृष्ठ सफलतापूर्वक प्रसंस्करित!")


            # Store results
            st.session_state.results = results
            st.session_state.processing_complete = True

            # Cleanup
            try:
                os.unlink(tmp_path)
            except:
                pass

            progress_bar.empty()
            status_text.empty()

        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)} | प्रसंस्करण विफल")
            st.info("This might be due to: | यह हो सकता है:")
            st.info("• File complexity | फ़ाइल जटिलता")
            st.info("• Memory limitations | मेमोरी सीमा")
            st.info("• Model loading issues | मॉडल लोडिंग समस्याएं")
            progress_bar.empty()
            status_text.empty()
            try:
                os.unlink(tmp_path)
            except:
                pass
            return

    # Display results
    if st.session_state.results is not None and st.session_state.processing_complete:
        results = st.session_state.results
        stats = results.get("processing_stats", {})

        # Summary metrics
        st.subheader("📊 Results Summary | परिणाम सारांश")
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("📄 Pages | पृष्ठ", stats.get("total_pages", 0))
        col2.metric("🔍 Detected | पहचाने गए", stats.get("total_articles_detected", 0))
        col3.metric("🎓 Education | शिक्षा", stats.get("education_articles_found", 0))
        col4.metric("🌐 Languages | भाषाएं", f"EN: {stats.get('english_articles', 0)} | HI: {stats.get('hindi_articles', 0)}")

        st.info("⚡ **Bilingual Processing** | द्विभाषी प्रसंस्करण: Optimized for HuggingFace Spaces")

        # Education articles display
        articles = results.get("education_articles", [])
        if articles:
            st.subheader(f"🎓 Education Articles Found ({len(articles)}) | पाए गए शिक्षा लेख")

            # Language filter
            col1, col2 = st.columns(2)
            with col1:
                language_filter = st.selectbox(
                    "🌐 Filter by language | भाषा के अनुसार फ़िल्टर करें:",
                    ["All | सभी", "English | अंग्रेजी", "Hindi | हिंदी"],
                    index=0
                )

            with col2:
                min_confidence = st.slider("📊 Minimum confidence | न्यूनतम विश्वसनीयता", 0.0, 1.0, 0.0, 0.05)

            # Apply filters
            filtered_articles = articles
            if language_filter == "English | अंग्रेजी":
                filtered_articles = [a for a in articles if a.get('language', 'en') == 'en']
            elif language_filter == "Hindi | हिंदी":
                filtered_articles = [a for a in articles if a.get('language', 'en') == 'hi']

            if min_confidence > 0:
                filtered_articles = [a for a in articles if a.get('confidence', 0) >= min_confidence]

            if not filtered_articles:
                st.info("🔍 No articles match your filter criteria | कोई लेख आपके फ़िल्टर मानदंड से मेल नहीं खाता")

            # Article display
            for i, article in enumerate(filtered_articles, 1):
                confidence = article.get('confidence', 0)
                language = article.get('language', 'en')
                lang_emoji = "🇮🇳" if language == 'hi' else "🇺🇸"

                # Confidence indicator
                if confidence > 0.8:
                    conf_emoji = "🟢"
                elif confidence > 0.6:
                    conf_emoji = "🟡"
                else:
                    conf_emoji = "🔴"

                with st.expander(f"{conf_emoji} {lang_emoji} Article {i} - Page {article['page']} (conf: {confidence:.2f})"):
                    # Metadata
                    meta_cols = st.columns(4)
                    keywords = article.get('keywords_found', [])[:5]
                    meta_cols[0].write(f"**🏷️ Keywords:** {', '.join(keywords)}")
                    meta_cols[1].write(f"**📝 Length:** {article.get('text_length', 0)} chars")
                    meta_cols[2].write(f"**🌐 Language | भाषा:** {'Hindi | हिंदी' if language == 'hi' else 'English | अंग्रेजी'}")
                    meta_cols[3].write(f"**📍 Position:** Page {article['page']}")

                    # Show crop image
                    if article.get("crop_path") and Path(article["crop_path"]).exists():
                        display_image_compatible(article["crop_path"], "🖼️ Article Crop", width=600)

                    # AI Summary
                    st.markdown("**🤖 AI Summary | AI सारांश:**")
                    summary = article.get("summary", "No summary available | कोई सारांश उपलब्ध नहीं")
                    st.write(summary)

                    # Full text
                    with st.expander("📄 View full extracted text | पूर्ण निकाला गया टेक्स्ट देखें"):
                        full_text = article.get("full_text", "No text extracted")
                        if full_text and len(full_text) > 20:
                            st.text_area(
                                label=f"Article {i} Text",
                                value=full_text,
                                height=150,
                                key=f"text_{article['page']}_{i}",
                                label_visibility="collapsed"
                            )
                        else:
                            st.info("📝 No readable text could be extracted | कोई पठनीय टेक्स्ट निकाला नहीं जा सका")

        else:
            st.info("🔍 No education articles found | कोई शिक्षा लेख नहीं मिला")
            st.info("Try adjusting the confidence threshold | विश्वसनीयता सीमा समायोजित करने का प्रयास करें")

        # Download results
        st.subheader("💾 Download Results | परिणाम डाउनलोड करें")
        json_data = json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8")
        filename = f"bilingual_education_articles_{st.session_state.uploaded_file_name or 'results'}.json"

        st.download_button(
            "📥 Download JSON Results | JSON परिणाम डाउनलोड करें",
            data=json_data,
            file_name=filename,
            mime="application/json"
        )

        # Reset button
        if st.button("🔄 Process Another PDF | अन्य PDF प्रसंस्करण करें"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]

    # Footer
    st.markdown("---")
    st.markdown("⚡ Bilingual performance optimized for HuggingFace Spaces | HuggingFace स्पेसेस के लिए द्विभाषी प्रदर्शन अनुकूलित")

if __name__ == "__main__":
    main()
