from pathlib import Path
from nltk.stem import PorterStemmer


def remove_stopwords(tokens, stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stop_words = set(f.read().splitlines())
    return [token for token in tokens if token not in stop_words]


def case_folding(tokens):
    return [token.lower() for token in tokens]


def perform_stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def process_file(input_file: Path, stopwords_file: Path, output_dir: Path):
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            tokens = f.read().splitlines()

        # Apply all linguistic operations in sequence
        tokens = remove_stopwords(tokens, stopwords_file)  # Remove stopwords
        tokens = case_folding(tokens)  # Convert to lowercase
        tokens = perform_stemming(tokens)  # Apply stemming

        # Create output file
        output_file = output_dir / f"{input_file.stem}_processed.txt"
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write("\n".join(tokens))

        print(f"Successfully processed: {input_file.name}")

    except Exception as e:
        print(f"Error processing {input_file.name}: {str(e)}")


def process_documents(input_dir: Path, stopwords_file: Path):
    # Verify input paths exist
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not stopwords_file.exists():
        raise FileNotFoundError(f"Stopwords file does not exist: {stopwords_file}")

    # Create output directory
    output_dir = input_dir / "processed_files"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing files from: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Get list of input files
    input_files = list(input_dir.glob("*.txt"))

    if not input_files:
        print(f"No .txt files found in {input_dir}")
        return

    # Process each file
    print("\nStarting processing...")
    for file in input_files:
        if "_processed" not in file.stem:  # Skip already processed files
            process_file(file, stopwords_file, output_dir)

    print(f"\nProcessing completed. Processed files are saved in: {output_dir}")


if __name__ == "__main__":
    # Define directories and files using raw strings
    input_dir = Path(
        r"C:\Users\yonat\OneDrive\Documents\Information Systems\Year 3\Semester A\Information Retrieval\Documents txt\tokenized")
    stopwords_file = Path(
        r"C:\Users\yonat\OneDrive\Documents\Information Systems\Year 3\Semester A\Information Retrieval\Documents txt\StopWords.txt")

    try:
        process_documents(input_dir, stopwords_file)
    except Exception as e:
        print(f"Error: {str(e)}")