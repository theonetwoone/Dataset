import streamlit as st
from PIL import Image, UnidentifiedImageError
import os
import zipfile
from pathlib import Path
import pandas as pd
import base64
import requests
import openai
import google.generativeai as genai
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import hashlib
import shutil
import time
import re
import json

# Try to import imagehash for better duplicate detection
try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

# Configuration
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]
MAX_DIMENSION = 1024  # Keep longest side at 1024px while preserving aspect ratio

def resize_image_preserve_aspect(image, max_dimension=1024):
    """Resize image to fit within max_dimension while preserving aspect ratio"""
    width, height = image.size
    
    # If image is already smaller than max_dimension, don't upscale
    if max(width, height) <= max_dimension:
        return image
    
    # Calculate scaling factor
    if width > height:
        # Landscape or square
        new_width = max_dimension
        new_height = int((height * max_dimension) / width)
    else:
        # Portrait
        new_height = max_dimension
        new_width = int((width * max_dimension) / height)
    
    return image.resize((new_width, new_height), Image.LANCZOS)

def extract_retry_delay(error_message):
    """Extract retry delay from API error message"""
    try:
        # Look for retry_delay in the error message
        match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', str(error_message))
        if match:
            return int(match.group(1))
    except:
        pass
    return None

def retry_api_call(api_function, max_retries=3, base_delay=5, status_placeholder=None):
    """Retry API calls with intelligent backoff and live countdown"""
    for attempt in range(max_retries + 1):
        try:
            return api_function()
        except Exception as e:
            error_str = str(e)
            
            # Check if it's a rate limit error
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                if attempt < max_retries:
                    # Try to extract suggested delay from error message
                    suggested_delay = extract_retry_delay(error_str)
                    delay = suggested_delay if suggested_delay else min(base_delay * (2 ** attempt), 60)
                    
                    # Live countdown with progress bar
                    if status_placeholder:
                        countdown_container = status_placeholder.container()
                        
                        # Show initial rate limit message
                        countdown_container.error(f"🚫 **Rate Limit Hit** - API quota exceeded (attempt {attempt + 1}/{max_retries + 1})")
                        
                        progress_bar = countdown_container.progress(0)
                        status_text = countdown_container.empty()
                        
                        for remaining in range(delay, 0, -1):
                            progress = (delay - remaining) / delay
                            progress_bar.progress(progress)
                            
                            minutes, seconds = divmod(remaining, 60)
                            if minutes > 0:
                                time_str = f"{minutes}m {seconds}s"
                            else:
                                time_str = f"{seconds}s"
                            
                            status_text.warning(f"⏳ **Waiting to retry** - {time_str} remaining...")
                            time.sleep(1)
                        
                        # Clear countdown display and show retry message
                        progress_bar.empty()
                        status_text.info("🔄 **Retrying API call now...**")
                        time.sleep(0.5)  # Brief pause to show retry message
                    else:
                        # Fallback without live countdown
                        st.warning(f"⏳ Rate limit hit. Waiting {delay} seconds before retry (attempt {attempt + 1}/{max_retries + 1})...")
                        time.sleep(delay)
                    
                    continue
                else:
                    # Final attempt failed
                    raise ValueError(f"API rate limit exceeded. Please wait before processing more images. Original error: {error_str}")
            else:
                # Non-rate-limit error, don't retry
                raise e
    
    # This shouldn't be reached, but just in case
    raise ValueError("Max retries exceeded")

# Initialize session state for unique directories
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(int(time.time() * 1000))

def get_output_dir():
    """Get unique output directory for this session"""
    return Path(f"/tmp/flux_lora_dataset_{st.session_state.session_id}")

def reset_dataset():
    """Reset the dataset and create new session"""
    # Remove old directory if it exists
    old_dir = get_output_dir()
    if old_dir.exists():
        shutil.rmtree(old_dir)
    
    # Create new session ID
    st.session_state.session_id = str(int(time.time() * 1000))
    
    # Clear any cached results
    if 'last_results' in st.session_state:
        del st.session_state.last_results

def create_system_prompt(concept_info=""):
    """Create system prompt with optional concept information"""
    base_prompt = """You are an AI assistant helping to generate training captions for a machine learning model. Given an image, describe what is visually present in a detailed and neutral way. Do not infer emotions, context, or artistic intent. Be literal and descriptive, naming objects, styles, colors, and physical arrangement.

CRITICALLY IMPORTANT: Always identify and describe the artistic medium/style of the image. Specify if it is:
- A photograph (specify if digital photo, film photo, etc.)
- Digital artwork/digital painting
- Traditional painting (oil painting, watercolor, acrylic, etc.)
- Drawing/sketch (pencil, charcoal, ink, etc.)
- 3D rendering/CGI
- Vector illustration
- Comic/cartoon style
- Mixed media
- Any other specific artistic medium

Include this style description naturally in your caption as it's essential for the AI model to understand the difference between photographic and artistic content."""
    
    if concept_info.strip():
        return f"{base_prompt}\n\nIt is very important that you honor the following instructions about the concept/subject: {concept_info.strip()}. Only provide the caption text, no additional commentary."
    else:
        return f"{base_prompt}\n\nOnly provide the caption text, no additional commentary."
st.set_page_config(page_title="Flux LoRA Dataset Builder", page_icon="🎨", layout="wide")

# Streamlit app
st.title("🎨 Flux LoRA Dataset Builder with Image Variety Analysis")
st.markdown("""
Build high-quality datasets for Flux LoRA training with automated captioning and comprehensive image analysis.

**🔬 Research-Backed Approach** • Based on extensive analysis of successful Flux LoRA training methodologies from community experts, official documentation, and academic sources.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    concept_token = st.text_input("Concept Token", value="CSKULL", help="Unique token for your concept (e.g., CSKULL, GXSTYLE)")
    mode = st.selectbox("Training Mode", ["Subject (Concept)", "Style"], help="Subject for objects/characters, Style for artistic looks")
    
    st.subheader("📝 Caption Guidance")
    concept_info = st.text_area(
        "Concept Description (Optional)", 
        placeholder="e.g., 'a cybernetic skull with glowing blue circuits and mechanical jaw components, focus on biomechanical details' or 'brutalist concrete textures with geometric greeble details, emphasize industrial surface patterns'",
        help="Provide additional context about your concept to help the AI generate more accurate captions. The AI will automatically identify the artistic medium (photo, digital art, painting, etc.)",
        height=100
    )
    
    st.subheader("🤖 Vision API")
    vision_model = st.radio("Choose Vision Model", [
        "Gemini Flash 1.5",
        "Gemini Pro Vision", 
        "OpenAI GPT-4 Vision",
        "HuggingFace Multi-Model"
    ])
    
    # Show API key link based on selected model
    if "Gemini" in vision_model:
        st.markdown("🔑 **Get API Key**: [Google AI Studio](https://aistudio.google.com/apikey)")
        st.caption("Free tier available with rate limits")
    elif "OpenAI" in vision_model:
        st.markdown("🔑 **Get API Key**: [OpenAI Platform](https://platform.openai.com/api-keys)")
        st.caption("Requires OpenAI account with billing setup")
    elif "HuggingFace" in vision_model:
        st.markdown("🔑 **Get API Key**: [HuggingFace Tokens](https://huggingface.co/settings/tokens)")
        st.caption("Free tier available with rate limits")
        with st.expander("📋 HuggingFace Models Used", expanded=False):
            st.write("The app will automatically try these models in order:")
            st.write("1. **microsoft/git-base** - Microsoft's GenerativeImage2Text")
            st.write("2. **Salesforce/blip-image-captioning-large** - Large BLIP model")
            st.write("3. **nlpconnect/vit-gpt2-image-captioning** - ViT + GPT2")
            st.write("4. **Salesforce/blip-image-captioning-base** - Original BLIP")
    
    api_key = st.text_input("API Key", type="password", help="Your API key for the selected vision model")
    
    # Rate limiting options
    st.subheader("⏱️ Rate Limiting")
    enable_rate_limiting = st.checkbox("Enable Rate Limiting", value=True, help="Add delays between API calls to avoid quota limits")
    
    if enable_rate_limiting:
        delay_between_calls = st.slider(
            "Delay Between Calls (seconds)", 
            min_value=0.5, 
            max_value=5.0, 
            value=1.0, 
            step=0.5,
            help="Time to wait between each API call"
        )
        max_retries = st.selectbox(
            "Max Retries on Rate Limit", 
            options=[1, 2, 3, 4, 5], 
            index=2,
            help="How many times to retry when hitting rate limits"
        )
    else:
        delay_between_calls = 0
        max_retries = 3
    
    st.subheader("📊 Analysis Options")
    run_analysis = st.checkbox("Perform Image Variety Analysis", value=True, help="Analyze dataset quality and diversity")
    show_detailed_plots = st.checkbox("Show Detailed Plots", value=False, help="Display additional analysis charts")
    
    st.subheader("🔄 Dataset Management")
    if st.button("🗑️ Reset Dataset", help="Clear all previous files and start fresh", type="secondary"):
        reset_dataset()
        st.success("✅ Dataset reset! Upload new images to start fresh.")
        st.rerun()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📁 Upload Images")
    uploaded_files = st.file_uploader(
        "Select your training images", 
        accept_multiple_files=True, 
        type=[ext[1:] for ext in ALLOWED_EXTENSIONS],
        help="Upload multiple images for your LoRA training dataset"
    )

with col2:
    if uploaded_files:
        st.metric("Images Uploaded", len(uploaded_files))
        file_sizes = [len(f.read()) for f in uploaded_files]
        for f in uploaded_files: f.seek(0)  # Reset file pointers
        st.metric("Total Size", f"{sum(file_sizes) / (1024*1024):.1f} MB")

# Image Variety Analysis Functions
def calculate_image_hash(image):
    """Calculate perceptual hash for duplicate detection"""
    if HAS_IMAGEHASH:
        return str(imagehash.phash(image))
    else:
        # Fallback: use histogram-based hash
        img_array = np.array(image.resize((8, 8)))  # Resize to small size
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # Create a simple hash from the image histogram
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
        hist_str = ''.join([str(int(x)) for x in hist.flatten()])
        return hashlib.md5(hist_str.encode()).hexdigest()[:16]

def analyze_color_distribution(image):
    """Analyze color distribution and dominant colors"""
    img_array = np.array(image)
    
    # Calculate color histograms
    hist_r = cv2.calcHist([img_array], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img_array], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([img_array], [2], None, [256], [0, 256])
    
    # Calculate dominant colors
    pixels = img_array.reshape(-1, 3)
    
    # Use k-means to find dominant colors
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    
    # Calculate color diversity (entropy)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    return {
        'dominant_colors': colors.tolist(),
        'color_entropy': entropy,
        'brightness': np.mean(gray),
        'contrast': np.std(gray)
    }

def analyze_composition(image):
    """Analyze image composition and features"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Edge detection for detail analysis
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Texture analysis - use simple gradient-based method as fallback
    try:
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray, 24, 8, method='uniform')
        texture_variance = np.var(lbp)
    except ImportError:
        # Fallback texture analysis using gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        texture_variance = np.var(gradient_magnitude)
    
    # Brightness distribution
    brightness_std = np.std(gray)
    
    return {
        'edge_density': edge_density,
        'texture_variance': texture_variance,
        'brightness_std': brightness_std,
        'sharpness': cv2.Laplacian(gray, cv2.CV_64F).var()
    }

def perform_dataset_analysis(uploaded_files):
    """Perform comprehensive dataset analysis"""
    if not uploaded_files:
        return None
    
    analysis_data = []
    image_hashes = []
    original_sizes = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            progress = (idx + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing image {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            # Load and analyze image
            img = Image.open(uploaded_file).convert("RGB")
            original_sizes.append(img.size)
            
            # Calculate hash for duplicate detection
            img_hash = calculate_image_hash(img)
            image_hashes.append(img_hash)
            
            # Analyze colors and composition
            color_analysis = analyze_color_distribution(img)
            composition_analysis = analyze_composition(img)
            
            # File info
            file_size = len(uploaded_file.read())
            uploaded_file.seek(0)  # Reset file pointer
            
            analysis_data.append({
                'filename': uploaded_file.name,
                'original_width': img.size[0],
                'original_height': img.size[1],
                'aspect_ratio': img.size[0] / img.size[1],
                'file_size_mb': file_size / (1024 * 1024),
                'hash': img_hash,
                **color_analysis,
                **composition_analysis
            })
            
        except Exception as e:
            st.warning(f"Could not analyze {uploaded_file.name}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    
    # Detect duplicates
    hash_counts = Counter(image_hashes)
    duplicates = [hash_val for hash_val, count in hash_counts.items() if count > 1]
    
    # Calculate dataset statistics
    df = pd.DataFrame(analysis_data)
    
    stats = {
        'total_images': len(df),
        'duplicates_found': len(duplicates),
        'unique_images': len(df) - len(duplicates),
        'avg_brightness': df['brightness'].mean(),
        'brightness_std': df['brightness'].std(),
        'avg_contrast': df['contrast'].mean(),
        'contrast_std': df['contrast'].std(),
        'avg_color_entropy': df['color_entropy'].mean(),
        'avg_edge_density': df['edge_density'].mean(),
        'avg_sharpness': df['sharpness'].mean(),
        'resolution_variety': df['aspect_ratio'].std(),
        'size_variety_mb': df['file_size_mb'].std()
    }
    
    return df, stats, duplicates

# Vision API Functions
def query_gemini(api_key, image_bytes, system_prompt, model="gemini-1.5-flash"):
    """Query Gemini API with proper authentication"""
    try:
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)
        
        # Convert bytes to PIL Image for Gemini
        image = Image.open(io.BytesIO(image_bytes))
        
        response = model_instance.generate_content([system_prompt, image])
        return response.text.strip()
    except Exception as e:
        raise ValueError(f"Gemini API error: {str(e)}")

def query_openai(api_key, image_bytes, system_prompt):
    """Query OpenAI GPT-4 Vision API"""
    try:
        client = openai.OpenAI(api_key=api_key)
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise ValueError(f"OpenAI API error: {str(e)}")

def query_huggingface(api_key, image_bytes, system_prompt=None):
    """Query HuggingFace vision model (system_prompt ignored - these models don't support it)"""
    try:
        # Try multiple working models with proper API format
        models_to_try = [
            "Salesforce/blip-image-captioning-base",
            "Salesforce/blip-image-captioning-large", 
            "nlpconnect/vit-gpt2-image-captioning",
            "microsoft/git-base-coco"
        ]
        
        last_error = None
        
        for model in models_to_try:
            try:
                # Use direct binary data format (most reliable)
                headers = {"Authorization": f"Bearer {api_key}"}
                
                response = requests.post(
                    f"https://api-inference.huggingface.co/models/{model}",
                    headers=headers,
                    data=image_bytes,
                    timeout=30
                )
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        
                        # Debug: Print the actual response structure (comment out in production)
                        # print(f"Model {model} response: {result}")
                        
                        # Handle different response formats
                        if isinstance(result, list) and len(result) > 0:
                            item = result[0]
                            if isinstance(item, dict):
                                # Look for different possible keys
                                if 'generated_text' in item and item['generated_text'].strip():
                                    return item['generated_text'].strip()
                                elif 'caption' in item and item['caption'].strip():
                                    return item['caption'].strip()
                                elif 'label' in item and item['label'].strip():
                                    return item['label'].strip()
                            elif isinstance(item, str) and item.strip():
                                return item.strip()
                        
                        elif isinstance(result, dict):
                            if 'generated_text' in result and result['generated_text'].strip():
                                return result['generated_text'].strip()
                            elif 'caption' in result and result['caption'].strip():
                                return result['caption'].strip()
                        
                        elif isinstance(result, str) and result.strip():
                            return result.strip()
                        
                        # If we got a 200 response but couldn't extract caption, try next model
                        last_error = f"Model {model}: Got response but couldn't extract caption: {result}"
                        continue
                        
                    except json.JSONDecodeError:
                        # Sometimes the response isn't JSON
                        response_text = response.text.strip()
                        if response_text and len(response_text) > 5:  # Basic validity check
                            return response_text
                        last_error = f"Model {model}: Invalid JSON response"
                        continue
                        
                elif response.status_code == 503:
                    # Model is loading, wait and retry once
                    time.sleep(20)  # Longer wait for model loading
                    response = requests.post(
                        f"https://api-inference.huggingface.co/models/{model}",
                        headers=headers,
                        data=image_bytes,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            if isinstance(result, list) and len(result) > 0:
                                item = result[0]
                                if isinstance(item, dict):
                                    if 'generated_text' in item and item['generated_text'].strip():
                                        return item['generated_text'].strip()
                                    elif 'caption' in item and item['caption'].strip():
                                        return item['caption'].strip()
                                elif isinstance(item, str) and item.strip():
                                    return item.strip()
                            elif isinstance(result, dict):
                                if 'generated_text' in result and result['generated_text'].strip():
                                    return result['generated_text'].strip()
                        except json.JSONDecodeError:
                            response_text = response.text.strip()
                            if response_text and len(response_text) > 5:
                                return response_text
                    
                    last_error = f"Model {model}: Still loading after retry"
                    continue
                    
                else:
                    # Store error details for debugging
                    try:
                        error_details = response.json()
                        last_error = f"Model {model}: {response.status_code} - {error_details}"
                    except:
                        last_error = f"Model {model}: {response.status_code} - {response.text[:200]}"
                    continue
                    
            except requests.exceptions.RequestException as e:
                last_error = f"Model {model}: Network error - {str(e)}"
                continue
            except Exception as e:
                last_error = f"Model {model}: Unexpected error - {str(e)}"
                continue
        
        # If all models failed, raise an error instead of returning generic text
        raise ValueError(f"All HuggingFace models failed. Last error: {last_error}")
        
    except Exception as e:
        if "All HuggingFace models failed" in str(e):
            raise e  # Re-raise our specific error
        else:
            raise ValueError(f"HuggingFace API error: {str(e)}")

# Main processing
if uploaded_files:
    
    # Show image variety analysis
    if run_analysis:
        st.header("📊 Dataset Quality Analysis")
        
        # Show which duplicate detection method is being used
        if not HAS_IMAGEHASH:
            st.info("ℹ️ Using histogram-based duplicate detection (install `imagehash` for improved accuracy)")
        
        with st.spinner("Analyzing image variety and quality..."):
            analysis_df, stats, duplicates = perform_dataset_analysis(uploaded_files)
        
        if analysis_df is not None:
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Images", stats['total_images'])
                st.metric("Unique Images", stats['unique_images'])
            
            with col2:
                st.metric("Duplicates Found", stats['duplicates_found'], 
                         delta=f"-{stats['duplicates_found']}" if stats['duplicates_found'] > 0 else None)
                st.metric("Avg Brightness", f"{stats['avg_brightness']:.1f}")
            
            with col3:
                st.metric("Avg Contrast", f"{stats['avg_contrast']:.1f}")
                st.metric("Color Variety", f"{stats['avg_color_entropy']:.2f}")
            
            with col4:
                st.metric("Detail Level", f"{stats['avg_edge_density']:.3f}")
                st.metric("Sharpness", f"{stats['avg_sharpness']:.1f}")
            
            # Quality assessment and recommendations
            st.subheader("🎯 Dataset Quality Assessment")
            
            quality_score = 0
            recommendations = []
            
            # Check image count
            if stats['total_images'] >= 20:
                quality_score += 25
                st.success(f"✅ Good dataset size ({stats['total_images']} images)")
            elif stats['total_images'] >= 10:
                quality_score += 15
                st.warning(f"⚠️ Adequate dataset size ({stats['total_images']} images)")
                recommendations.append("Consider adding more images (20+ recommended for robust training)")
            else:
                st.error(f"❌ Small dataset size ({stats['total_images']} images)")
                recommendations.append("Add more images! At least 10-15 images recommended")
            
            # Check for duplicates
            if stats['duplicates_found'] == 0:
                quality_score += 20
                st.success("✅ No duplicate images found")
            else:
                st.warning(f"⚠️ {stats['duplicates_found']} duplicate images detected")
                recommendations.append("Remove duplicate images to improve training variety")
            
            # Check brightness variety
            if stats['brightness_std'] > 20:
                quality_score += 20
                st.success(f"✅ Good brightness variety (std: {stats['brightness_std']:.1f})")
            else:
                st.warning(f"⚠️ Limited brightness variety (std: {stats['brightness_std']:.1f})")
                recommendations.append("Add images with different lighting conditions")
            
            # Check contrast variety
            if stats['contrast_std'] > 10:
                quality_score += 15
                st.success(f"✅ Good contrast variety (std: {stats['contrast_std']:.1f})")
            else:
                recommendations.append("Consider adding images with varied contrast levels")
            
            # Check color variety
            if stats['avg_color_entropy'] > 6:
                quality_score += 20
                st.success(f"✅ Good color variety (entropy: {stats['avg_color_entropy']:.2f})")
            else:
                recommendations.append("Add images with more color variety")
            
            # Overall score
            st.subheader(f"Overall Quality Score: {quality_score}/100")
            
            if quality_score >= 80:
                st.success("🎉 Excellent dataset quality! Ready for training.")
            elif quality_score >= 60:
                st.warning("⚠️ Good dataset with room for improvement.")
            else:
                st.error("❌ Dataset needs improvement before training.")
            
            # Show recommendations
            if recommendations:
                st.subheader("💡 Recommendations")
                for rec in recommendations:
                    st.write(f"• {rec}")
            
            # Detailed plots
            if show_detailed_plots:
                st.subheader("📈 Detailed Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Brightness distribution
                    fig = px.histogram(analysis_df, x='brightness', title='Brightness Distribution',
                                     nbins=20, color_discrete_sequence=['#1f77b4'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Aspect ratio distribution
                    fig = px.histogram(analysis_df, x='aspect_ratio', title='Aspect Ratio Distribution',
                                     nbins=20, color_discrete_sequence=['#ff7f0e'])
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Contrast vs Brightness scatter
                    fig = px.scatter(analysis_df, x='brightness', y='contrast', 
                                   title='Brightness vs Contrast', 
                                   hover_data=['filename'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # File size distribution
                    fig = px.histogram(analysis_df, x='file_size_mb', title='File Size Distribution (MB)',
                                     nbins=15, color_discrete_sequence=['#2ca02c'])
                    st.plotly_chart(fig, use_container_width=True)

    # Process images if API key is provided and not already processed
    if api_key and uploaded_files:
        # Check if we already have results for this exact configuration
        current_config_key = f"{concept_token}_{vision_model}_{len(uploaded_files)}_{hash(tuple(f.name for f in uploaded_files))}"
        
        # Only process if we don't have results or configuration changed
        should_process = (
            'last_results' not in st.session_state or 
            st.session_state.get('last_config_key') != current_config_key or
            st.button("🔄 Reprocess Images", help="Click to regenerate captions with current settings")
        )
        
        if should_process:
            st.header("🤖 Caption Generation & Dataset Creation")
            
            # Show concept info status
            if concept_info.strip():
                with st.expander("📝 Using Concept Guidance", expanded=False):
                    st.write("**Concept Description:**")
                    st.write(concept_info)
                    st.info("💡 This guidance will help the AI generate more accurate captions for your specific concept.")
            else:
                st.info("💡 Tip: Add a concept description in the sidebar to help the AI generate more accurate captions!")
            
            # Style identification notice
            st.info("🎨 **Artistic Medium Detection**: The AI will automatically identify whether each image is a photograph, digital artwork, painting, sketch, 3D render, etc. This is crucial for Flux training since the base model was trained primarily on photographs.")
            
            # Get unique output directory for this session
            output_dir = get_output_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Show session info and rate limiting status
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📁 **Session ID**: {st.session_state.session_id}")
            with col2:
                if enable_rate_limiting:
                    st.info(f"⏱️ **Rate Limiting**: {delay_between_calls}s delay, {max_retries} retries")
                else:
                    st.warning("⚠️ **Rate Limiting**: Disabled (may hit quota limits)")
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            retry_status = st.empty()  # For retry countdown display
            
            # Calculate estimated time
            total_files = len(uploaded_files)
            estimated_time = total_files * delay_between_calls if enable_rate_limiting else total_files * 2
            
            if enable_rate_limiting and delay_between_calls > 0:
                st.info(f"⏱️ Estimated processing time: ~{estimated_time:.0f} seconds ({delay_between_calls}s per image)")
            
            start_time = time.time()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            file_ext = Path(uploaded_file.name).suffix.lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                continue
            
            # Update progress
            progress = (idx + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            
            elapsed_time = time.time() - start_time
            remaining_files = len(uploaded_files) - (idx + 1)
            
            if enable_rate_limiting and idx > 0:
                avg_time_per_file = elapsed_time / (idx + 1)
                eta = remaining_files * avg_time_per_file
                status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name} (ETA: {eta:.0f}s)")
            else:
                status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            new_basename = f"{concept_token.lower()}_{idx:04d}"
            image_path = output_dir / f"{new_basename}.jpg"
            text_path = output_dir / f"{new_basename}.txt"
            
            try:
                # Process image
                img = Image.open(uploaded_file).convert("RGB")
                img = resize_image_preserve_aspect(img, MAX_DIMENSION)
                img.save(image_path, format="JPEG", quality=95)
                
                # Generate caption
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                
                # Create dynamic system prompt with concept info
                current_system_prompt = create_system_prompt(concept_info)
                
                # Add rate limiting delay (except for first image)
                if idx > 0 and enable_rate_limiting and delay_between_calls > 0:
                    time.sleep(delay_between_calls)
                
                # API call with retry logic
                def make_api_call():
                    if vision_model == "Gemini Flash 1.5":
                        return query_gemini(api_key, image_bytes, current_system_prompt, "gemini-1.5-flash")
                    elif vision_model == "Gemini Pro Vision":
                        return query_gemini(api_key, image_bytes, current_system_prompt, "gemini-pro-vision")
                    elif vision_model == "OpenAI GPT-4 Vision":
                        return query_openai(api_key, image_bytes, current_system_prompt)
                    elif vision_model == "HuggingFace Multi-Model":
                        return query_huggingface(api_key, image_bytes, current_system_prompt)
                    else:
                        raise ValueError("Invalid model selected")
                
                description = retry_api_call(make_api_call, max_retries=max_retries, status_placeholder=retry_status)
                
                # Clear retry status after successful call
                retry_status.empty()
                
                # Format caption based on mode
                if mode == "Subject (Concept)":
                    caption = f"{concept_token}, {description}"
                else:  # Style mode
                    caption = f"{concept_token} style, {description}"
                
                # Save caption
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(caption)
                
                results.append({
                    "Image": image_path.name,
                    "Caption": caption,
                    "Original": uploaded_file.name
                })
                
            except UnidentifiedImageError:
                retry_status.empty()  # Clear retry status on error
                st.warning(f"Skipped unreadable file: {uploaded_file.name}")
            except Exception as e:
                retry_status.empty()  # Clear retry status on error
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()
        retry_status.empty()  # Clear retry status when done
        
        if results:
            # Store results in session state with configuration tracking
            st.session_state.last_results = results
            st.session_state.last_output_dir = str(output_dir)
            st.session_state.last_concept_token = concept_token
            st.session_state.last_config_key = current_config_key
            
            # Show results
            df = pd.DataFrame(results)
            st.success(f"Successfully processed {len(results)} images!")
            
            # Display results table
            st.subheader("Generated Captions")
            st.dataframe(df, use_container_width=True)

    elif 'last_results' in st.session_state:
        # Show existing results without reprocessing
        st.header("🤖 Caption Generation & Dataset Creation") 
        st.info("✅ Using previously processed results. Click '🔄 Reprocess Images' above to regenerate.")
        df = pd.DataFrame(st.session_state.last_results)
        st.subheader("Generated Captions")
        st.dataframe(df, use_container_width=True)

# Show download section if we have results (either from current run or previous)
if 'last_results' in st.session_state and st.session_state.last_results:
    st.header("📦 Download Dataset")
    
    results = st.session_state.last_results
    output_dir = Path(st.session_state.last_output_dir)
    concept_token = st.session_state.last_concept_token
    
    # Verify files still exist
    if output_dir.exists() and any(output_dir.rglob("*.jpg")):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.success(f"✅ Dataset ready: {len(results)} processed images")
            
            # Show file list
            with st.expander("📁 View Dataset Files", expanded=False):
                image_files = list(output_dir.glob("*.jpg"))
                text_files = list(output_dir.glob("*.txt"))
                st.write(f"**Image files:** {len(image_files)}")
                st.write(f"**Caption files:** {len(text_files)}")
                for img_file in sorted(image_files)[:5]:  # Show first 5
                    st.write(f"📸 {img_file.name}")
                if len(image_files) > 5:
                    st.write(f"... and {len(image_files) - 5} more")
        
        with col2:
            # Create download package with unique name
            zip_filename = f"{concept_token.lower()}_flux_lora_dataset_{st.session_state.session_id}.zip"
            zip_path = output_dir.parent / zip_filename
            
            # Always check and recreate ZIP to ensure it has current files
            with st.spinner("📦 Creating download package..."):
                # Get all image and text files from output directory
                image_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
                text_files = list(output_dir.glob("*.txt"))
                
                st.write(f"Found {len(image_files)} images and {len(text_files)} captions to package")
                
                if image_files or text_files:
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        # Add all files with proper structure
                        for file_path in image_files + text_files:
                            if file_path.is_file():
                                arcname = file_path.name  # Just the filename, not full path
                                zipf.write(file_path, arcname=arcname)
                                st.write(f"Added: {arcname}")
                
                # Verify ZIP contents
                if zip_path.exists():
                    with zipfile.ZipFile(zip_path, 'r') as zipf:
                        zip_contents = zipf.namelist()
                        st.write(f"ZIP contains {len(zip_contents)} files: {zip_contents[:5]}...")
            
            # Show file size and download
            if zip_path.exists() and zip_path.stat().st_size > 0:
                zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
                st.metric("ZIP Size", f"{zip_size_mb:.1f} MB")
                
                # Download button
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="📦 Download Dataset ZIP",
                        data=f,
                        file_name=zip_filename,
                        mime="application/zip",
                        help="Download your processed dataset ready for Flux LoRA training",
                        use_container_width=True,
                        key=f"download_{st.session_state.session_id}"
                    )
            else:
                st.error("❌ Failed to create ZIP file or ZIP is empty")
            
    else:
        st.warning("⚠️ Dataset files not found. Please process images again or reset the dataset.")
        if st.button("🔄 Reset and Start Over"):
            reset_dataset()
            st.rerun()

# Show info messages when no processing is happening
if uploaded_files and api_key and 'last_results' not in st.session_state:
    st.info("💡 Click above to start processing your images!")
elif uploaded_files and not api_key:
    st.info("💡 Enter your API key to generate captions and create the dataset.")
elif not uploaded_files:
    st.info("👆 Upload some images to get started!")

# Footer with tips
st.markdown("---")

# Research-based guidance section
st.markdown("""
### 📚 Research-Based Best Practices

This app implements findings from extensive research on **Flux LoRA training best practices** based on community experience, official guidelines, and expert recommendations:
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 🎯 **Subject/Concept LoRAs**
    - **Dataset Size**: 15-30 high-quality images minimum
    - **Resolution**: Max 1024px on longest side (aspect ratio preserved)
    - **Variety**: Different angles, lighting, backgrounds
    - **Trigger Words**: Use unique tokens (e.g., `CSKULL`)
    - **Training Steps**: ~100 steps per image (2000-2500 total)
    - **Network Rank**: 32 recommended for detail vs. generalization
    - **Captioning**: Detailed descriptions with trigger word
    
    #### 🏛️ **Key Research Sources**
    - **FineTuners AI** - Dataset quality guidelines
    - **TheFluxTrain Blog** - Rank and parameter optimization  
    - **Reddit r/StableDiffusion** - Community findings
    - **Fal.ai Documentation** - Platform-specific best practices
    """)

with col2:
    st.markdown("""
    #### 🎨 **Style LoRAs**
    - **Dataset Size**: 30-50+ images for robust style learning
    - **Content Variety**: Apply style to different subjects
    - **Patch Augmentation**: Use "Is Style" toggle on fal.ai
    - **Captioning Strategy**: Minimal captions or style token only
    - **Training Steps**: ~3000-5000 steps for complex styles
    - **Learning Rate**: 1e-4 to 5e-5 depending on steps
    - **Flexibility**: Train separately from subjects for modularity
    
    #### 🔬 **Research Methodology**
    - **Overfitting Prevention**: Analyzed through varied datasets
    - **Quality Metrics**: Based on community success rates
    - **Parameter Tuning**: Derived from comparative studies
    - **Photorealism**: Optimized for Flux.1 Dev capabilities
    
    #### 🤖 **Vision API Comparison**
    - **Gemini Flash 1.5**: Fast, free tier, good for most concepts
    - **Gemini Pro Vision**: Higher quality, better for complex images
    - **OpenAI GPT-4 Vision**: Most accurate, requires paid account
    - **HuggingFace Multi-Model**: Multiple models tried automatically, free
    
    #### ⏱️ **Rate Limit Management**
    - **Free Tiers**: Limited requests per minute/day
    - **Gemini**: 15 requests/minute (free), respects retry delays
    - **OpenAI**: Pay-per-use, higher limits with billing
    - **Best Practice**: Enable rate limiting, use 1-2s delays
    """)

st.markdown("""
#### 🎓 **Academic & Community Research Foundation**

This tool synthesizes findings from multiple sources including:
- **John Shi's Medium** - Flux training challenges and solutions
- **Dev Rajput's Analysis** - Fal.ai optimization techniques  
- **Cadmium9094's Experiments** - Parameter testing with 50+ image datasets
- **Multiple Reddit Case Studies** - Real-world success and failure analysis

**🎯 Originally researched for**: Training two specific LoRAs - a **Cybernetic Skull subject** and a **Graphic Greeble/Brutalist style** - this methodology has been generalized for any concept or style training.

The image variety analysis implemented here directly addresses common training failures identified in the research, including:
- **Overfitting detection** through duplicate analysis and variety scoring
- **Dataset quality assessment** based on successful training patterns
- **Automated recommendations** derived from expert best practices

#### 🛠️ **Research → Implementation**
| **Research Finding** | **App Implementation** |
|---------------------|----------------------|
| "100 steps per image" rule | Quality score considers 10+ image minimum |
| Flux works well with 1024px dimension | Aspect ratio preserved, max 1024px side |
| Unique trigger words essential | Concept token validation and guidance |
| Style LoRAs need content variety | Analysis checks subject diversity |
| Duplicates cause overfitting | Perceptual hash duplicate detection |
| Brightness variety prevents bias | Brightness std deviation scoring |
| Captioning improves results | Multi-API vision model integration |
| Flux trained primarily on photos | Automatic artistic medium identification |
| Style recognition crucial for non-photo | Enhanced prompts specify art medium |

---

### 💡 Quick Training Tips:
- **Subject Training**: Use 15-30 varied images of your concept with different angles, lighting, and backgrounds
- **Style Training**: Use 30-50+ images showcasing the style applied to different subjects  
- **Quality over Quantity**: Ensure images are clear, well-lit, and represent what you want to learn
- **Avoid Duplicates**: Remove similar or identical images to prevent overfitting
- **Test Variety**: Use the analysis above to ensure good diversity in your dataset
- **Concept Guidance**: Add a detailed description of your concept in the sidebar to help AI generate more accurate captions
- **Be Specific**: Describe key visual features, colors, materials, and distinctive characteristics of your concept
- **Artistic Medium Matters**: Captions will automatically identify if images are photos, digital art, paintings, etc. - crucial since Flux was trained primarily on photographs
- **Style Consistency**: For style LoRAs, ensure your artistic medium is consistent across training images (all digital art, all photos, etc.)
- **API Selection**: Start with **Gemini Flash 1.5** (free) for testing, **HuggingFace** for reliable free option, upgrade to **GPT-4 Vision** for best quality
- **Aspect Ratio**: Images are resized to max 1024px on the longest side while preserving original proportions
- **Rate Limiting**: Use 1-2 second delays between calls to avoid quota limits, especially with free API tiers
- **Large Datasets**: For 50+ images, consider processing in smaller batches or using paid API tiers

📖 **For comprehensive guidance**, refer to the complete research documentation: `LORA.md`
""")