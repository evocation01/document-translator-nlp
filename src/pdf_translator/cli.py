import argparse
from pdf_translator import core, sample_generator

def main():
    """Main function for the command-line interface."""
    parser = argparse.ArgumentParser(description="PDF Translation Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Translate command
    parser_translate = subparsers.add_parser("translate", help="Translate PDFs")
    parser_translate.set_defaults(func=core.main)

    # Generate samples command
    parser_generate = subparsers.add_parser("generate-samples", help="Generate sample PDFs")
    parser_generate.set_defaults(func=sample_generator.main)

    args = parser.parse_args()
    args.func()

if __name__ == "__main__":
    main()
