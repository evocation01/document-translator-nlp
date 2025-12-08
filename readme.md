# PDF Translation Tool

![PDF Translation Tool Banner](https://example.com/banner.png) <!-- Replace with a real banner image if you have one -->

## Overview

This powerful and flexible tool automates the translation of PDF documents. It extracts text from PDF files, leverages the state-of-the-art MarianMT model from Hugging Face for translation, and recreates the PDF with a side-by-side comparison of the original and translated text. To ensure comprehensive text extraction, the tool seamlessly integrates OCR as a fallback for images with embedded text.

This project is ideal for anyone who needs to translate documents while preserving the original layout and context. It's built with efficiency in mind, skipping already-translated files and providing detailed logging for easy monitoring.

## Key Features

-   **Accurate Text Extraction**: Extracts text from both text-based and image-based PDFs using an OCR fallback.
-   **High-Quality Translation**: Utilizes the pre-trained MarianMT model from Hugging Face for state-of-the-art translation.
-   **Side-by-Side Comparison**: Recreates the PDF with the original and translated text on the same page for easy comparison.
-   **Efficient Processing**: Skips files that have already been translated to save time and resources.
-   **Custom Fonts**: Supports custom fonts for the translated text, allowing for a personalized look.
-   **Detailed Logging**: Logs all processing steps and errors to a `translation_log.txt` file for easy debugging.
-   **Sample Generation**: Includes a script to generate random sample PDFs for testing and demonstration.

## Requirements

### Python Packages

-   `fitz` (PyMuPDF)
-   `pytesseract`
-   `torch`
-   `fpdf`
-   `Pillow`
-   `transformers`

You can install these packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Additional Requirements

#### Tesseract OCR

This tool requires Tesseract OCR for extracting text from images. Follow the instructions for your operating system:

**Windows:**

1.  Download and install Tesseract from the [official Tesseract repository](https://github.com/tesseract-ocr/tesseract).
2.  Add the Tesseract installation directory to your system's `PATH` environment variable.

**Linux:**

```bash
sudo apt update
sudo apt install tesseract-ocr
```

**macOS:**

```bash
brew install tesseract
```

To verify the installation, run `tesseract --version` in your terminal.

## Folder Structure

```
document-translator-nlp/
|— src/
|   |— translate_pdf.py
|   |— generate_samples.py
|— pdfs_original/
|— pdfs_translated/
|— fonts/
|— .gitignore
|— LICENSE
|— README.md
|— requirements.txt
|— translation_log.txt
```

## Getting Started

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/your-username/document-translator-nlp.git
    cd document-translator-nlp
    ```

2.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Add PDFs**

    Place the PDF files you want to translate into the `pdfs_original/` directory.

4.  **Run the Translation Script**

    ```bash
    python src/translate_pdf.py
    ```

    The translated PDFs will be saved in the `pdfs_translated/` directory.

### Generating Sample PDFs

To generate random sample PDFs for testing, run the following command:

```bash
python src/generate_samples.py
```

## Configuration

### Translation Model

The script uses the MarianMT model for translation. You can change the model by updating the `model_name` variable in `src/translate_pdf.py`:

```python
model_name = "Helsinki-NLP/opus-mt-tc-big-en-tr"
```

### Device

The script automatically uses a GPU (CUDA) if available. If not, it defaults to the CPU. You can change this behavior by modifying the `device` variable in `src/translate_pdf.py`:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### Custom Font

You can specify a custom font for the translated text by updating the `font_path` in `src/translate_pdf.py`:

```python
font_path = os.path.join(base_dir, "fonts", "alegreya-sans-sc", "AlegreyaSansSC-Regular.ttf")
```

## Contributing

Contributions are welcome! If you have any suggestions or find any bugs, please open an issue or submit a pull request.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature`).
6.  Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

-   This project was created to help a friend and has been adapted with public domain documents to avoid copyright issues.
-   Thanks to the developers of the open-source libraries used in this project.
