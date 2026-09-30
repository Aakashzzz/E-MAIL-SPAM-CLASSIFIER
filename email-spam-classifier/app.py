import os
import sys
import streamlit as st
from PIL import Image

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predict import predict

# Set Page Config
st.set_page_config(
    page_title="SentriSpam - Spam Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS injection for vibrant glassmorphic UI)
st.markdown("""
<style>
    /* Main Background & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    
    /* Title styling */
    .main-title {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-align: center;
        letter-spacing: -1px;
    }
    
    .main-subtitle {
        color: #8A99AD;
        font-size: 1.15rem;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 300;
    }
    
    /* Cards / Containers */
    .css-1r6g72x, .stTextArea {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
    }
    
    /* Custom Result Cards */
    .result-card-spam {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(239, 68, 68, 0.3);
        animation: pulse 2s infinite alternate;
    }
    
    .result-card-ham {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.3);
    }
    
    .result-label {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: 1px;
    }
    
    .result-desc {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 400;
    }
    
    /* Custom Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        100% { transform: scale(1.02); }
    }
</style>
""", unsafe_allow_html=True)

# Layout: Main Columns
col1, col2 = st.columns([7, 5])

with col1:
    st.markdown('<div class="main-title">🛡️ SentriSpam AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Production-Grade Email & SMS Spam Classifier</div>', unsafe_allow_html=True)
    
    # Check if models are trained
    project_dir = os.path.dirname(os.path.abspath(__file__))
    vectorizer_path = os.path.join(project_dir, 'models', 'vectorizer.pkl')
    best_model_path = os.path.join(project_dir, 'models', 'best_model.pkl')
    
    if not os.path.exists(vectorizer_path) or not os.path.exists(best_model_path):
        st.warning("⚠️ Machine Learning Models are not trained yet! Please run the training script from the console: `python src/train.py` before classifying messages.")
        st.info("You can also run training right now using the button below (make sure virtual environment is active).")
        if st.button("Train Models Now"):
            with st.spinner("Training models in background... This takes about 15 seconds."):
                import subprocess
                try:
                    result = subprocess.run(
                        [sys.executable, os.path.join(project_dir, "src", "train.py")],
                        capture_output=True, text=True, check=True
                    )
                    st.success("✅ Models trained successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to run training script: {e}")
                    if 'result' in locals():
                        st.code(result.stderr)
    else:
        # Message input area
        message_input = st.text_area(
            "Enter message text below:",
            height=220,
            placeholder="Paste your email, newsletter, or SMS here to check if it's spam..."
        )
        
        classify_btn = st.button("Analyze Message")
        
        if classify_btn or message_input.strip() != "":
            if not message_input.strip():
                st.warning("Please enter some text before analyzing.")
            else:
                with st.spinner("Analyzing message semantics..."):
                    label, confidence, model_name = predict(message_input)
                
                # Display Results
                if label == 'spam':
                    st.markdown(f"""
                    <div class="result-card-spam">
                        <div class="result-label" style="color: #ef4444;">🚨 SPAM</div>
                        <div class="result-desc">
                            We are <strong>{confidence:.1f}%</strong> confident that this message is spam.
                        </div>
                        <div style="font-size: 0.85rem; margin-top: 1rem; opacity: 0.7;">
                            Classified by {model_name}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card-ham">
                        <div class="result-label" style="color: #10b981;">✅ HAM (LEGIT)</div>
                        <div class="result-desc">
                            We are <strong>{confidence:.1f}%</strong> confident that this message is legitimate.
                        </div>
                        <div style="font-size: 0.85rem; margin-top: 1rem; opacity: 0.7;">
                            Classified by {model_name}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

with col2:
    st.markdown("<h3 style='margin-bottom:1.5rem; font-weight:600;'>📊 Model Architecture & Insights</h3>", unsafe_allow_html=True)
    
    # Model info and visualizations
    outputs_dir = os.path.join(project_dir, 'outputs')
    models_dir = os.path.join(project_dir, 'models')
    
    metadata_path = os.path.join(models_dir, 'best_model_metadata.txt')
    if os.path.exists(metadata_path):
        st.markdown("#### Selected Model Performance")
        with open(metadata_path, 'r') as f:
            metadata_lines = f.readlines()
            for line in metadata_lines:
                st.write(f"- **{line.strip()}**")
    
    # Show Visualizations
    tab1, tab2 = st.tabs(["Spam Keywords", "Confusion Matrices"])
    
    with tab1:
        kw_img_path = os.path.join(outputs_dir, 'top_spam_keywords.png')
        if os.path.exists(kw_img_path):
            st.image(Image.open(kw_img_path), caption="Top features driving spam predictions")
        else:
            st.info("Keyword importance plot will appear here once models are trained.")
            
    with tab2:
        cm_img_path = os.path.join(outputs_dir, 'confusion_matrices.png')
        if os.path.exists(cm_img_path):
            st.image(Image.open(cm_img_path), caption="Comparative Confusion Matrices")
        else:
            st.info("Confusion matrix plots will appear here once models are trained.")