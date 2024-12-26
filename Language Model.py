from pathlib import Path
import json
from collections import Counter
import math


def create_language_model():
    # Directory containing tokenized files
    tokenized_dir = Path(
        r'C:\Users\yonat\OneDrive\Documents\Information Systems\Year 3\Semester A\Information Retrieval\Documents txt\tokenized')

    # Counter for all terms
    term_counts = Counter()
    total_terms = 0

    # Process each tokenized file
    for file_path in tokenized_dir.glob('tokenized_*.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            tokens = f.read().splitlines()  # Read tokens (one per line)
            term_counts.update(tokens)
            total_terms += len(tokens)

    # Calculate probabilities and create language model
    language_model = {
        'total_terms': total_terms,
        'vocabulary_size': len(term_counts),
        'term_probabilities': {},
        'collection_stats': {
            'total_documents': len(list(tokenized_dir.glob('tokenized_*.txt'))),
            'average_document_length': total_terms / len(list(tokenized_dir.glob('tokenized_*.txt')))
        }
    }

    # Calculate probability for each term
    for term, count in term_counts.items():
        language_model['term_probabilities'][term] = {
            'count': count,
            'probability': count / total_terms,
            'log_probability': math.log(count / total_terms)
        }

    # Save the language model
    output_file = tokenized_dir.parent / 'language_model.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(language_model, f, indent=4)

    print(f"Language model created with {len(term_counts)} unique terms")
    print(f"Total terms: {total_terms}")
    print(f"Average document length: {language_model['collection_stats']['average_document_length']:.2f} terms")

    # Print some example terms with highest probabilities
    print("\nMost common terms:")
    for term, stats in sorted(language_model['term_probabilities'].items(),
                              key=lambda x: x[1]['count'], reverse=True)[:10]:
        print(f"{term}: {stats['count']} occurrences, probability: {stats['probability']:.6f}")


if __name__ == "__main__":
    create_language_model()