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
import openai

# Configuration
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]
TARGET_SIZE = (1024, 1024)
SYSTEM_PROMPT = "You are an AI assistant helping to generate training captions for a machine learning model. Given an image, describe what is visually present in a detailed and neutral way. Do not infer emotions, context, or artistic intent. Be literal and descriptive, naming objects, styles, colors, and physical arrangement."

# Streamlit app
st.title("Flux LoRA Dataset Builder with Vision API Selection")

concept_token = st.text_input("Concept Token (e.g. CSKULL or GXSTYLE)", "CSKULL")
mode = st.selectbox("Mode", ["Subject (Concept)", "Style"])
vision_model = st.radio("Choose Vision Model API", ["Gemini Flash (v2.5)", "Gemini Pro Vision", "OpenAI GPT-4 Vision", "HuggingFace Inference", "Gemini Pro Vision", "OpenAI GPT-4 Vision", "HuggingFace Inference"])
uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=ALLOWED_EXTENSIONS)
api_key = st.text_input("API Key (or Bearer Token)", type="password")


def query_gemini(api_key, image_bytes, system_prompt, model="gemini-pro-vision"):
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
    endpoint = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
    response = requests.post(endpoint, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        raise ValueError(f"Gemini API error: {response.status_code}, {response.text}")

def query_openai(api_key, image_bytes, system_prompt):
    openai.api_key = api_key
    b64_image = base64.b64encode(image_bytes).decode('utf-8')
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]
            }
        ],
        max_tokens=120
    )
    return response.choices[0].message['content']

def query_huggingface(api_key, image_bytes, system_prompt):
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.post(
        "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base",
        headers=headers,
        files={"file": image_bytes}
    )
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        raise ValueError(f"HuggingFace API error: {response.status_code}, {response.text}")

if uploaded_files and api_key:
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

            with open(image_path, "rb") as f:
                image_bytes = f.read()

            if vision_model == "Gemini Flash (v2.5)":
                description = query_gemini(api_key, image_bytes, SYSTEM_PROMPT, model="gemini-1.5-flash-latest").strip()
            elif vision_model == "Gemini Pro Vision":
                description = query_gemini(api_key, image_bytes, SYSTEM_PROMPT, model="gemini-pro-vision").strip()
            elif vision_model == "OpenAI GPT-4 Vision":
                description = query_openai(api_key, image_bytes, SYSTEM_PROMPT).strip()
            elif vision_model == "HuggingFace Inference":
                description = query_huggingface(api_key, image_bytes, SYSTEM_PROMPT).strip()
            else:
                raise ValueError("Invalid model selected.")

            caption = f"{concept_token}, {description}" if mode == "Subject (Concept)" else f"{concept_token} style, {description}"

            with open(text_path, "w") as f:
                f.write(caption)

            results.append({"Image": image_path.name, "Caption": caption})

        except UnidentifiedImageError:
            st.warning(f"Skipped unreadable file: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

    df = pd.DataFrame(results)
    st.success(f"Processed {len(results)} images.")
    st.dataframe(df)

    zip_path = Path("/tmp/flux_lora_dataset.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in output_dir.rglob("*"):
            zipf.write(file, arcname=file.relative_to(output_dir))

    with open(zip_path, "rb") as f:
        st.download_button("Download ZIP", f, file_name="flux_lora_dataset.zip", mime="application/zip")