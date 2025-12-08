from setuptools import setup, find_packages

def get_long_description():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

setup(
    name="pdf-translator",
    version="1.0.0",
    description="A tool to translate PDF documents using NLP models.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/your-username/document-translator-nlp",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        "pdf_translator": ["fonts/alegreya-sans-sc/*", "data/excerpts.txt"],
    },
    install_requires=[
        "PyMuPDF",
        "pytesseract",
        "torch",
        "fpdf",
        "Pillow",
        "transformers",
        "PyYAML",
        "numpy<2",
    ],
    entry_points={
        "console_scripts": [
            "pdf-translator = pdf_translator.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
