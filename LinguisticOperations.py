from pathlib import Path
from nltk.stem import PorterStemmer

from LanguageModel import create_advanced_language_model


def remove_stopwords(tokens, stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stop_words = set(f.read().splitlines())
    return [token for token in tokens if token not in stop_words]


def case_folding(tokens):
    return [token.lower() for token in tokens]


def perform_stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def safe_write_file(file_path: Path, content: str):
    """Safely write content to file."""
    with open(file_path, 'w', encoding='utf-8') as out_f:
        out_f.write(content)


def read_tokens(file_path: Path):
    """Read tokens from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().splitlines()


def process_documents(input_dir: Path, stopwords_file: Path):
    # Verify input paths exist
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not stopwords_file.exists():
        raise FileNotFoundError(f"Stopwords file does not exist: {stopwords_file}")

    input_files = list(input_dir.glob("*.txt"))
    if not input_files:
        print(f"Warning: No .txt files found in {input_dir}")
        return

    error_count = 0

    # Initial state language model
    print("\nCreating initial language model...")
    create_advanced_language_model(step_identifier="0_Initial_State")

    # Step 1: Remove stopwords for all files
    print("\nStep 1: Removing stopwords...")
    for file in input_files:
        try:
            tokens = read_tokens(file)
            tokens = remove_stopwords(tokens, stopwords_file)
            safe_write_file(file, "\n".join(tokens))
        except Exception as e:
            error_count += 1
            print(f"Error processing {file.name}: {str(e)}")
    print("\nCreating language model after removing stopwords...")
    create_advanced_language_model(step_identifier="1_After_Stopwords_Removal")

    # Step 2: Case folding for all files
    print("\nStep 2: Performing case folding...")
    for file in input_files:
        try:
            tokens = read_tokens(file)
            tokens = case_folding(tokens)
            safe_write_file(file, "\n".join(tokens))
        except Exception as e:
            error_count += 1
            print(f"Error processing {file.name}: {str(e)}")
    print("\nCreating language model after case folding...")
    create_advanced_language_model(step_identifier="2_After_Case_Folding")

    # Step 3: Stemming for all files
    print("\nStep 3: Performing stemming...")
    for file in input_files:
        try:
            tokens = read_tokens(file)
            tokens = perform_stemming(tokens)
            safe_write_file(file, "\n".join(tokens))
        except Exception as e:
            error_count += 1
            print(f"Error processing {file.name}: {str(e)}")
    print("\nCreating language model after stemming...")
    create_advanced_language_model(step_identifier="3_After_Stemming")

    if error_count == 0:
        print("\nSuccessfully processed all files")
    else:
        print(f"\nCompleted with {error_count} errors")


"""if __name__ == "__main__":
    input_dir = Path("OurDocuments/tokenized")
    stopwords_file = Path("OurDocuments/StopWords.txt")

    try:
        process_documents(input_dir, stopwords_file)
    except Exception as e:
        print(f"Error: {str(e)}")"""