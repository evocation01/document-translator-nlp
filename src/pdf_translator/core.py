import logging
import os
import re
import time
from io import BytesIO

import fitz  # PyMuPDF
import pytesseract
import torch
from fpdf import FPDF
from PIL import Image
from transformers import MarianMTModel, MarianTokenizer

from pdf_translator.config import config, get_project_root

# Get project root to resolve absolute paths
project_root = get_project_root()

# Set up logging
log_file_path = project_root / config["paths"]["log_file"]
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
)


def get_device():
    """Detects and returns the appropriate device for PyTorch."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    return device

def load_model_and_tokenizer(model_name, device):
    """Loads the translation model and tokenizer."""
    logging.info(f"Loading model: {model_name}")
    try:
        model = MarianMTModel.from_pretrained(model_name).to(device)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load model or tokenizer: {e}")
        raise

def extract_text_with_ocr(page):
    """Extracts text from a PDF page using OCR as a fallback."""
    logging.info("Using OCR for text extraction")
    try:
        pix = page.get_pixmap()
        img = Image.open(BytesIO(pix.tobytes()))
        ocr_text = pytesseract.image_to_string(img)
        return ocr_text
    except Exception as e:
        logging.error(f"OCR failed: {e}")
        return ""

def extract_text_from_pdf_per_page(pdf_path):
    """Extracts text from each page of a PDF."""
    logging.info(f"Extracting text from {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        page_texts = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if not text.strip():
                text = extract_text_with_ocr(page)
            page_texts.append(text)
        return page_texts
    except Exception as e:
        logging.error(f"Failed to extract text from {pdf_path}: {e}")
        return []

def split_text_into_sentences(text):
    """Splits text into sentences using regex."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]

def translate_text(text, model, tokenizer, device, initial_batch_size):
    """Translates text in batches."""
    sentences = split_text_into_sentences(text)
    if not sentences:
        return ""
    
    translated_texts = []
    logging.info(f"Total sentences to translate: {len(sentences)}")
    batch_size = initial_batch_size

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        try:
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            translated_tokens = model.generate(**inputs)
            translated_batch = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            translated_texts.extend(translated_batch)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.warning("Out of memory error. Reducing batch size.")
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                # Retry with the smaller batch size
                i -= batch_size 
            else:
                logging.error(f"Unexpected error during translation: {e}")
                # Skip batch
        except Exception as e:
            logging.error(f"Translation failed for a batch: {e}")
            # Skip batch
            
    return "\n".join(translated_texts)

def recreate_pdf(input_pdf_path, translated_pages, output_pdf_path, font_path):
    """Recreates a PDF with original and translated text."""
    logging.info(f"Recreating PDF for {input_pdf_path}")
    try:
        doc = fitz.open(input_pdf_path)
        pdf = FPDF()
        
        pdf.add_font("CustomFont", "", font_path, uni=True)
        pdf.set_font("CustomFont", size=12)

        for page_num, translated_page_text in enumerate(translated_pages):
            # Add original page as an image
            pdf.add_page()
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.open(BytesIO(pix.tobytes()))
            with BytesIO() as img_buffer:
                img.save(img_buffer, format="PNG")
                pdf.image(img_buffer, x=10, y=10, w=190)

            # Add translated page
            pdf.add_page()
            pdf.set_xy(10, 10)
            pdf.multi_cell(0, 10, translated_page_text)
            
        pdf.output(output_pdf_path)
        logging.info(f"PDF saved as {output_pdf_path}")
    except Exception as e:
        logging.error(f"Failed to recreate PDF {output_pdf_path}: {e}")

def process_pdf(pdf_file: str, model, tokenizer, device) -> None:

    """Main processing logic for a single PDF file."""

    input_folder = project_root / config["paths"]["input_folder"]

    output_folder = project_root / config["paths"]["output_folder"]

    font_path = project_root / config["paths"]["font_path"]



    input_pdf_path = input_folder / pdf_file

    output_pdf_path = output_folder / f"TR_{pdf_file}"



    if output_pdf_path.exists():

        logging.info(f"Skipping {pdf_file}, already translated.")

        return



    logging.info(f"Processing {pdf_file}")

    start_time = time.time()

    

    try:

        page_texts = extract_text_from_pdf_per_page(input_pdf_path)

        if not page_texts:

            return



        translated_pages = [

            translate_text(page, model, tokenizer, device, config["translation"]["initial_batch_size"]) 

            for page in page_texts

        ]



        if not any(translated_pages):

            logging.error("Translation resulted in empty text. Skipping PDF creation.")

            return

            

        recreate_pdf(input_pdf_path, translated_pages, output_pdf_path, font_path)

        end_time = time.time()

        logging.info(f"Completed {pdf_file} in {end_time - start_time:.2f} seconds")

        

    except Exception as e:

        logging.error(f"An error occurred while processing {pdf_file}: {e}")



def main() -> None:

    """Main function to run the PDF translation process."""

    output_folder = project_root / config["paths"]["output_folder"]

    output_folder.mkdir(exist_ok=True)

    

    device = get_device()

    model, tokenizer = load_model_and_tokenizer(config["translation"]["model_name"], device)

    

    input_folder = project_root / config["paths"]["input_folder"]

    pdf_files = [f.name for f in input_folder.glob("*.pdf")]

    

    for pdf_file in pdf_files:

        process_pdf(pdf_file, model, tokenizer, device)
