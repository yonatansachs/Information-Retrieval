from pathlib import Path
from nltk.stem import PorterStemmer


def remove_stopwords(tokens, stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stop_words = set(f.read().splitlines())
    return [token for token in tokens if token not in stop_words]


def safe_write_file(output_file: Path, content: str):
    """Safely write content to file, ensuring directory exists."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write(content)


def process_stopwords(input_dir: Path, stopwords_file: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    counter=0
    input_files = list(input_dir.glob("*.txt"))

    if not input_files:
        print(f"Warning: No .txt files found in {input_dir}")
        return

    for file in input_files:
        try:
            if "_no_stopwords" not in file.stem:
                #print(f"Processing file: {file.name}")
                with open(file, 'r', encoding='utf-8') as f:
                    tokens = f.read().splitlines()

                filtered_tokens = remove_stopwords(tokens, stopwords_file)
                output_file = output_dir / f"{file.stem}_no_stopwords.txt"
                safe_write_file(output_file, "\n".join(filtered_tokens))

        except Exception as e:
            counter+=1
            print(f"Error processing {file.name}: {str(e)}")
    if counter == 0:
        print(f"Successfully Removed Stop Words from all files")

def case_folding(tokens):
    return [token.lower() for token in tokens]


def process_case_folding(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    counter=0
    input_files = list(input_dir.glob("*.txt"))

    if not input_files:
        print(f"Warning: No .txt files found in {input_dir}")
        return

    for file in input_files:
        try:
            if "_case_folded" not in file.stem:
                #print(f"Processing file: {file.name}")
                with open(file, 'r', encoding='utf-8') as f:
                    tokens = f.read().splitlines()

                folded_tokens = case_folding(tokens)
                output_file = output_dir / f"{file.stem}_case_folded.txt"
                safe_write_file(output_file, "\n".join(folded_tokens))

        except Exception as e:
            counter+=1
            print(f"Error processing {file.name}: {str(e)}")
    if counter == 0:
        print(f"Successfully Case Folded all files")

def perform_stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def process_stemming(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    counter=0
    input_files = list(input_dir.glob("*.txt"))

    if not input_files:
        print(f"Warning: No .txt files found in {input_dir}")
        return

    for file in input_files:
        try:
            if "_stemmed" not in file.stem:
                #print(f"Processing file: {file.name}")
                with open(file, 'r', encoding='utf-8') as f:
                    tokens = f.read().splitlines()

                stemmed_tokens = perform_stemming(tokens)
                output_file = output_dir / f"{file.stem}_stemmed.txt"
                safe_write_file(output_file, "\n".join(stemmed_tokens))

        except Exception as e:
            counter+=1
            print(f"Error processing {file.name}: {str(e)}")
    if counter == 0:
        print(f"Successfully Stemmed all files")


def process_documents(input_dir: Path, stopwords_file: Path):
    # Verify input paths exist
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not stopwords_file.exists():
        raise FileNotFoundError(f"Stopwords file does not exist: {stopwords_file}")

    # Create main output directory
    output_base_dir = input_dir  / "linguistic_processed"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for each processing step
    stopwords_dir = output_base_dir / "1_no_stopwords"
    case_folded_dir = output_base_dir / "2_case_folded"
    stemmed_dir = output_base_dir / "3_stemmed"

    #print(f"Processing files from: {input_dir}")
    #print(f"Output directory: {output_base_dir}")

    # Process files through each step
    print("\nStep 1: Removing stopwords...")
    process_stopwords(input_dir, stopwords_file, stopwords_dir)

    print("\nStep 2: Performing case folding...")
    process_case_folding(stopwords_dir, case_folded_dir)

    print("\nStep 3: Performing stemming...")
    process_stemming(case_folded_dir, stemmed_dir)

    print(f"\nLinguistic processing completed. Results are saved in: {output_base_dir}")


"""if __name__ == "__main__":
    # Define directories and files using raw strings
    input_dir = Path("OurDocuments/tokenized")
    stopwords_file =Path("OurDocuments/StopWords.txt")

    try:
        process_documents(input_dir, stopwords_file)
    except Exception as e:
        print(f"Error: {str(e)}")"""