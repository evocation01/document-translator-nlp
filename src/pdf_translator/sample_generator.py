import random
from pathlib import Path
from fpdf import FPDF
from pdf_translator.config import config, get_project_root

project_root = get_project_root()

def load_excerpts(excerpts_file: Path) -> list[str]:
    """Loads text excerpts from a file."""
    with open(excerpts_file, "r") as f:
        excerpts = [line.strip() for line in f if line.strip()]
    return excerpts

def generate_random_pdf(output_path: Path, num_pages: int, excerpts_per_page: int, excerpts: list[str]) -> None:
    """
    Generates a PDF with random excerpts.

    Args:
        output_path (Path): The path to save the generated PDF.
        num_pages (int): The number of pages to generate.
        excerpts_per_page (int): The number of random excerpts per page.
        excerpts (list[str]): A list of text excerpts to use.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    for page_num in range(1, num_pages + 1):
        pdf.add_page()
        pdf.set_xy(10, 10)
        pdf.set_font("Arial", style="B", size=16)
        pdf.cell(0, 10, f"Page {page_num}", ln=True, align="C")
        pdf.set_font("Arial", size=12)
        
        for _ in range(excerpts_per_page):
            pdf.multi_cell(0, 10, random.choice(excerpts), align="L")
            pdf.ln(2)
            
    try:
        pdf.output(output_path)
        print(f"Successfully generated sample PDF: {output_path}")
    except Exception as e:
        print(f"Error saving PDF: {e}")

def main() -> None:
    """
    Main function to generate multiple sample PDFs.
    """
    sample_generator_config = config["sample_generator"]
    paths_config = config["paths"]
    
    output_folder = project_root / paths_config["input_folder"]
    output_folder.mkdir(exist_ok=True)
    
    excerpts_file = project_root / paths_config["excerpts_file"]
    excerpts = load_excerpts(excerpts_file)
    
    num_pdfs = sample_generator_config["num_pdfs"]
    num_pages = sample_generator_config["num_pages"]
    excerpts_per_page = sample_generator_config["excerpts_per_page"]
    
    for i in range(1, num_pdfs + 1):
        output_path = output_folder / f"sample_{i}.pdf"
        generate_random_pdf(output_path, num_pages, excerpts_per_page, excerpts)
