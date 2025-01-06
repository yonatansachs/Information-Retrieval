import re
import logging
from pathlib import Path
import json

from Clustering import main2
from LanguageModel import create_advanced_language_model
from MachineLearning import main
from tokenization_rules import TOKENIZATION_RULES
from LinguisticOperations import process_documents


class IRTokenizer:
    def __init__(self, rules):
        self.rules = rules
        self.compile_patterns()

    def compile_patterns(self):
        """Compile regex patterns based on rules"""
        patterns = []

        # Special cases (must be checked first)
        if self.rules['special_cases']:
            special_cases = '|'.join(map(re.escape, self.rules['special_cases']))
            patterns.append(f'(?:{special_cases})')

        # Emails
        if self.rules['keep_emails']:
            patterns.append(r'[\w\.-]+@[\w\.-]+\.\w+')

        # URLs
        if self.rules['keep_urls']:
            patterns.append(r'https?://\S+|www\.\S+')

        # Numbers (if keeping them)
        if self.rules['keep_numbers']:
            patterns.append(r'\b\d+(?:\.\d+)?\b')

        # Words with hyphens
        if self.rules['keep_hyphens']:
            patterns.append(r'\b\w+(?:-\w+)+\b')

        # Words with underscores
        if self.rules['keep_underscores']:
            patterns.append(r'\b\w+(?:_\w+)+\b')

        # Regular words
        patterns.append(r'\b\w+\b')

        # Combine all patterns
        self.pattern = re.compile('|'.join(patterns), re.IGNORECASE)

    def tokenize(self, text):
        """Tokenize text according to rules"""
        if not text:
            return []

        # Convert to lowercase if specified
        if self.rules['lowercase']:
            text = text.lower()

        # Find all tokens
        tokens = self.pattern.findall(text)

        # Apply minimum length filter
        tokens = [t for t in tokens if len(t) >= self.rules['min_length']]

        return tokens


def process_directory(input_dir, output_dir=None, rules_file=None):
    """Process all text files in a directory"""
    counter=0
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir / 'Processed_Files'
    output_dir.mkdir(exist_ok=True)

    # Load custom rules if provided
    rules = TOKENIZATION_RULES
    if rules_file:
        with open(rules_file, 'r') as f:
            rules = json.load(f)

    # Initialize tokenizer
    tokenizer = IRTokenizer(rules)

    # Save tokenization rules for reference
    with open(output_dir / 'tokenization_rules.json', 'w') as f:
        json.dump(rules, f, indent=4)

    # Process each file
    for file_path in input_dir.glob('*.txt'):
        try:
            # Read file with UTF-8 encoding, falling back to latin1
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin1') as f:
                    text = f.read()

            # Tokenize
            tokens = tokenizer.tokenize(text)

            # Save tokens
            output_path = output_dir / f"processed_{file_path.name}"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(tokens))

            #print(f"Successfully processed {file_path}: {len(tokens)} tokens")

        except Exception as e:
            counter+=1
            print(f"Error processing {file_path}: {str(e)}")
    if counter==0:
        print(f"Successfully Tokenized all files to {output_dir}")



if __name__ == "__main__":
    # Directory containing your text files
    input_dir = Path("OurDocuments")
    # Process files
    print("Starting to Tokenize...")
    process_directory(input_dir)
    #Linguistic Operations
    print()
    print("Starting Linguistic Operations...")
    LinguisticInput_dir = Path("OurDocuments/Processed_Files")
    stopwords_file = Path("StopWords.txt")
    try:
        process_documents(LinguisticInput_dir, stopwords_file)
    except Exception as e:
        print(f"Error: {str(e)}")

    print("***************************************")
    print("***************************************")
    print("Moving on to Search engine bias")
    print("***************************************")
    print("***************************************")
    print()
    print("Labels documents can be found at search_engine_bias_lables.csv")
    print()
    print("Starting to Tokenize...")
    input_dir = Path("SearchEngineBias")
    process_directory(input_dir)
    # Linguistic Operations
    print()
    print("Starting Linguistic Operations...")
    NewInput_dir = Path("SearchEngineBias/Processed_Files")
    stopwords_file = Path("StopWords.txt")
    try:
        process_documents(NewInput_dir, stopwords_file)
    except Exception as e:
        print(f"Error: {str(e)}")

    print("***************************************")
    print("***************************************")
    print("Moving on to Classification")
    print("***************************************")
    print("***************************************")
    main()

    print("***************************************")
    print("***************************************")
    print("Moving on to Clusters")
    print("***************************************")
    print("***************************************")
    print("Preforming Operations on Enhancing the musuem documents")
    input_dir = Path("Enhancing_the_museum")
    # Process files
    print("Starting to Tokenize...")
    process_directory(input_dir)
    # Linguistic Operations
    print()
    print("Starting Linguistic Operations...")
    LinguisticInput_dir = Path("Enhancing_the_museum/Processed_Files")
    try:
        process_documents(LinguisticInput_dir, stopwords_file)
    except Exception as e:
        print(f"Error: {str(e)}")

    print("Preforming Operations on Indoor positioning documents")
    input_dir = Path("Indoor_positioning")
    # Process files
    print("Starting to Tokenize...")
    process_directory(input_dir)
    # Linguistic Operations
    print()
    print("Starting Linguistic Operations...")
    LinguisticInput_dir = Path("Indoor_positioning/Processed_Files")
    try:
        process_documents(LinguisticInput_dir, stopwords_file)
    except Exception as e:
        print(f"Error: {str(e)}")

    main2()