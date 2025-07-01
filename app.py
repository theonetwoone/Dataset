import streamlit as st
from PIL import Image, UnidentifiedImageError
import os
import shutil
import zipfile
from pathlib import Path
import pandas as pd
import random
import uuid
import base64
import requests

# Configuration
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]
TARGET_SIZE = (1024, 1024)
SYSTEM_PROMPT = "You are an AI assistant helping to generate training captions for a machine learning model. Given an image, describe what is visually present in a detailed and neutral way. Do not infer emotions, context, or artistic intent. Be literal and descriptive, naming objects, styles, colors, and physical arrangement."

# Streamlit app
st.title("Flux LoRA Dataset Builder with Gemini Vision API")

concept_token = st.text_input("Concept Token (e.g. CSKULL or GXSTYLE)", "CSKULL")
mode = st.selectbox("Mode", ["Subject (Concept)", "Style"])
uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=ALLOWED_EXTENSIONS)
gemini_api_key = st.text_input("Gemini API Key", type="password")


def query_gemini(api_key, image_bytes, system_prompt):
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "contents": [
            {
                "parts": [
                    {"text": system_prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_image
                        }
                    }
                ]
            }
        ]
    }
    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent",
        headers=headers,
        json=data
    )
    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        raise ValueError(f"Gemini API error: {response.status_code}, {response.text}")

if uploaded_files and gemini_api_key:
    output_dir = Path("/tmp/flux_lora_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for idx, uploaded_file in enumerate(uploaded_files):
        file_ext = Path(uploaded_file.name).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            continue

        new_basename = f"{concept_token.lower()}_{idx:04d}"
        image_path = output_dir / f"{new_basename}.jpg"
        text_path = output_dir / f"{new_basename}.txt"

        try:
            img = Image.open(uploaded_file).convert("RGB")
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
            img.save(image_path, format="JPEG")

            # Generate caption with Gemini
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            description = query_gemini(gemini_api_key, image_bytes, SYSTEM_PROMPT).strip()

            caption = f"{concept_token}, {description}" if mode == "Subject (Concept)" else f"{concept_token} style, {description}"

            with open(text_path, "w") as f:
                f.write(caption)

            results.append({"Image": image_path.name, "Caption": caption})

        except UnidentifiedImageError:
            st.warning(f"Skipped unreadable file: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

    # Display dataset summary
    df = pd.DataFrame(results)
    st.success(f"Processed {len(results)} images.")
    st.dataframe(df)

    # Zip the dataset
    zip_path = Path("/tmp/flux_lora_dataset.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in output_dir.rglob("*"):
            zipf.write(file, arcname=file.relative_to(output_dir))

    with open(zip_path, "rb") as f:
        st.download_button("Download ZIP", f, file_name="flux_lora_dataset.zip", mime="application/zip")
