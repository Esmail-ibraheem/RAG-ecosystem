# app.py

import os
import base64
import requests
import streamlit as st
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
from typing import Union

# Load environment variables from .env file
load_dotenv()

def is_remote_file(file_path: Union[str, Image.Image]) -> bool:
    """
    Check if the provided file path is a remote URL.
    
    Args:
        file_path (str or Image.Image): The file path or Image object.
    
    Returns:
        bool: True if it's a remote URL, False otherwise.
    """
    if isinstance(file_path, str):
        return file_path.startswith("http://") or file_path.startswith("https://")
    return False

def encode_image(image: Image.Image) -> str:
    """
    Encode a PIL Image to a base64 string.
    
    Args:
        image (Image.Image): PIL Image object.
    
    Returns:
        str: Base64 encoded string of the image.
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return encoded_string

def encode_image_from_path(image_path: str) -> str:
    """
    Encode a local image file to a base64 string.
    
    Args:
        image_path (str): Path to the local image file.
    
    Returns:
        str: Base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def perform_ocr(file_input: Union[str, Image.Image], api_key: str, model: str) -> str:
    """
    Perform OCR on the provided image file and convert it to Markdown.
    
    Args:
        file_input (str or Image.Image): Path/URL to the image file or PIL Image object.
        api_key (str): Together AI API key.
        model (str): Vision model to use.
    
    Returns:
        str: OCR result in Markdown format.
    
    Raises:
        ValueError: If the API key is not provided.
        Exception: For any other errors during the OCR process.
    """
    # Determine the vision model
    if model == "free":
        vision_llm = "meta-llama/Llama-Vision-Free"
    else:
        vision_llm = f"meta-llama/{model}-Instruct-Turbo"
    
    # Prepare the system prompt
    system_prompt = """
Convert the provided image into Markdown format. Ensure that all content from the page is included, such as headers, footers, subtexts, images (with alt text if possible), tables, and any other elements.

Requirements:

- Output Only Markdown: Return solely the Markdown content without any additional explanations or comments.
- No Delimiters: Do not use code fences or delimiters like ```markdown.
- Complete Content: Do not omit any part of the page, including headers, footers, and subtext.
""".strip()
    
    # Determine if the file is remote or local
    if isinstance(file_input, Image.Image):
        # Encode the PIL Image to base64
        final_image_url = f"data:image/jpeg;base64,{encode_image(file_input)}"
    elif isinstance(file_input, str):
        if is_remote_file(file_input):
            final_image_url = file_input
        else:
            # Encode the local image to base64
            encoded_image = encode_image_from_path(file_input)
            final_image_url = f"data:image/jpeg;base64,{encoded_image}"
    else:
        raise ValueError("file_input must be a string (path or URL) or a PIL Image object.")
    
    # Prepare the payload for the API request
    payload = {
        "model": vision_llm,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": final_image_url
                        }
                    }
                ]
            }
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    api_endpoint = "https://api.together.ai/chat/completions"  # Replace with the actual Together AI API endpoint
    
    try:
        response = requests.post(api_endpoint, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        
        # Extract the Markdown content
        markdown_content = data["choices"][0]["message"]["content"]
        return markdown_content
    
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err} - {response.text}")
        return ""
    except Exception as err:
        st.error(f"An error occurred: {err}")
        return ""

def perform_image_question(file_input: Union[str, Image.Image], api_key: str, model: str, question: str) -> str:
    """
    Perform a question-answering task on the provided image.
    
    Args:
        file_input (str or Image.Image): Path/URL to the image file or PIL Image object.
        api_key (str): Together AI API key.
        model (str): Vision model to use.
        question (str): The question to ask about the image.
    
    Returns:
        str: Answer to the question.
    
    Raises:
        ValueError: If the API key or question is not provided.
        Exception: For any other errors during the process.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    
    # Determine the vision model
    if model == "free":
        vision_llm = "meta-llama/Llama-Vision-Free"
    else:
        vision_llm = f"meta-llama/{model}-Instruct-Turbo"
    
    # Prepare the system prompt for question answering
    system_prompt = f"""
You are an assistant that can answer questions about the content of an image.

Question: {question}

Please provide a clear and concise answer based solely on the content of the provided image.
""".strip()
    
    # Determine if the file is remote or local
    if isinstance(file_input, Image.Image):
        # Encode the PIL Image to base64
        final_image_url = f"data:image/jpeg;base64,{encode_image(file_input)}"
    elif isinstance(file_input, str):
        if is_remote_file(file_input):
            final_image_url = file_input
        else:
            # Encode the local image to base64
            encoded_image = encode_image_from_path(file_input)
            final_image_url = f"data:image/jpeg;base64,{encoded_image}"
    else:
        raise ValueError("file_input must be a string (path or URL) or a PIL Image object.")
    
    # Prepare the payload for the API request
    payload = {
        "model": vision_llm,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": final_image_url
                        }
                    }
                ]
            }
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    api_endpoint = "https://api.together.ai/chat/completions"  # Replace with the actual Together AI API endpoint
    
    try:
        response = requests.post(api_endpoint, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        
        # Extract the answer
        answer = data["choices"][0]["message"]["content"]
        return answer
    
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err} - {response.text}")
        return ""
    except Exception as err:
        st.error(f"An error occurred: {err}")
        return ""

def run():
    # st.set_page_config(page_title="OCR and Image QA Converter", layout="wide")
    st.title("📄 OCR to Markdown & 🧠 Image Question Answering")
    st.write("Convert your images into Markdown format and ask questions about them using Together AI's capabilities.")
    
    # Sidebar for API Key and Model Selection
    st.sidebar.header("Configuration")
    
    # API Key Input
    api_key = st.sidebar.text_input("Together AI API Key", type="password", placeholder="Enter your API key here")
    
    if not api_key:
        # Try to load from environment variable
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            st.sidebar.warning("Please enter your Together AI API Key or set it in the .env file.")
    
    # Model Selection
    model = st.sidebar.selectbox(
        "Select Vision Model",
        ["Llama-3.2-90B-Vision", "Llama-3.2-11B-Vision", "free"],
        index=0
    )
    
    # File Upload
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image file (JPEG, PNG)", type=["jpg", "jpeg", "png"])
    
    # Alternatively, provide a URL
    st.subheader("Or Provide an Image URL")
    image_url = st.text_input("Enter Image URL", placeholder="https://example.com/image.jpg")
    
    file_input = None
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            file_input = image  # Pass the PIL Image object
        except Exception as e:
            st.error(f"Error loading image: {e}")
    elif image_url:
        # Validate the URL
        if image_url.startswith(("http://", "https://")):
            try:
                response = requests.get(image_url)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                st.image(image, caption='Image from URL', use_column_width=True)
                file_input = image_url
            except Exception as e:
                st.error(f"Failed to load image from URL: {e}")
                file_input = None
        else:
            st.error("Please enter a valid image URL starting with http:// or https://")
            file_input = None
    else:
        st.info("Please upload an image or provide an image URL to proceed.")
    
    # Tabs for OCR and Q&A
    if file_input:
        tab1, tab2 = st.tabs(["🔍 Convert to Markdown", "❓ Ask About the Image"])
        
        with tab1:
            st.subheader("Convert Image to Markdown")
            if st.button("Convert to Markdown"):
                if not api_key:
                    st.error("API Key is required to perform OCR.")
                else:
                    with st.spinner("Performing OCR..."):
                        markdown_result = perform_ocr(file_input, api_key, model)
                    
                    if markdown_result:
                        st.success("OCR Completed Successfully!")
                        st.markdown("### **Markdown Output:**")
                        st.code(markdown_result, language='markdown')
                        
                        # Option to download the Markdown
                        st.download_button(
                            label="Download Markdown",
                            data=markdown_result,
                            file_name='output.md',
                            mime='text/markdown'
                        )
        
        with tab2:
            st.subheader("Ask a Question About the Image")
            question = st.text_input("Enter your question here", placeholder="e.g., What is the title on the banner?")
            if st.button("Get Answer"):
                if not api_key:
                    st.error("API Key is required to ask questions.")
                elif not question.strip():
                    st.error("Please enter a valid question.")
                else:
                    with st.spinner("Processing your question..."):
                        answer = perform_image_question(file_input, api_key, model, question)
                    
                    if answer:
                        st.success("Answer Retrieved Successfully!")
                        st.markdown("### **Answer:**")
                        st.write(answer)
    else:
        st.info("Please upload an image or provide an image URL to access OCR and Q&A features.")

if __name__ == "__main__":
    run()
