from pathlib import Path
import json
from collections import Counter, defaultdict
import math


class AdvancedLanguageModel:
    def __init__(self, smoothing_method='laplace', laplace_alpha=1.0):
        self.smoothing_method = smoothing_method
        self.laplace_alpha = laplace_alpha

        # Storage for different n-gram models
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()

        # Document-specific statistics
        self.doc_term_counts = defaultdict(Counter)
        self.doc_lengths = defaultdict(int)

        # Collection statistics
        self.total_terms = 0
        self.vocabulary = set()
        self.docs_containing_term = defaultdict(set)

    def _get_ngrams(self, tokens, n):
        """Generate n-grams from a list of tokens"""
        if n == 1:
            return tokens
        padded = ['<s>'] * (n - 1) + tokens + ['</s>'] * (n - 1)
        ngrams = []
        for i in range(len(padded) - n + 1):
            ngram = tuple(padded[i:i + n])
            ngrams.append(ngram)
        return ngrams

    def _ngram_to_string(self, ngram):
        """Convert ngram tuple to string representation"""
        return ' '.join(ngram)

    def train(self, tokenized_dir: Path):
        """Train the language model on all documents in the directory"""
        doc_paths = list(tokenized_dir.glob('tokenized_*.txt'))

        print("Processing documents...")
        for doc_path in doc_paths:
            doc_id = doc_path.stem
            with open(doc_path, 'r', encoding='utf-8') as f:
                tokens = f.read().splitlines()

            # Update document-specific counts
            self.doc_term_counts[doc_id].update(tokens)
            self.doc_lengths[doc_id] = len(tokens)

            # Update vocabulary and document frequency
            for token in set(tokens):
                self.vocabulary.add(token)
                self.docs_containing_term[token].add(doc_id)

            # Update n-gram counts
            self.unigram_counts.update(tokens)

            # Convert n-grams to strings and update counts
            bigrams = [self._ngram_to_string(ng) for ng in self._get_ngrams(tokens, 2)]
            trigrams = [self._ngram_to_string(ng) for ng in self._get_ngrams(tokens, 3)]

            self.bigram_counts.update(bigrams)
            self.trigram_counts.update(trigrams)

            self.total_terms += len(tokens)

        # Calculate IDF scores
        self.idf_scores = {}
        num_docs = len(doc_paths)
        for term in self.vocabulary:
            self.idf_scores[term] = math.log(num_docs / len(self.docs_containing_term[term]))

    def _smooth_probability(self, count: int, total: int, vocab_size: int) -> float:
        """Apply smoothing to probability calculation"""
        if self.smoothing_method == 'laplace':
            return (count + self.laplace_alpha) / (total + self.laplace_alpha * vocab_size)
        elif self.smoothing_method == 'good_turing':
            if count == 0:
                return 1 / (total + len(self.vocabulary))
            return count / total
        return count / total if total > 0 else 0

    def save_model(self, output_path: Path):
        """Save the language model to disk"""
        model_data = {
            'collection_stats': {
                'total_terms': self.total_terms,
                'vocabulary_size': len(self.vocabulary),
                'total_documents': len(self.doc_lengths),
                'average_document_length': self.total_terms / len(self.doc_lengths)
            },
            'unigram_probabilities': {
                term: {
                    'count': count,
                    'probability': self._smooth_probability(count, self.total_terms, len(self.vocabulary)),
                    'idf': self.idf_scores[term]
                }
                for term, count in self.unigram_counts.most_common()
            },
            'top_bigrams': {
                bigram: {
                    'count': count,
                    'probability': self._smooth_probability(count, self.total_terms, len(self.vocabulary))
                }
                for bigram, count in self.bigram_counts.most_common(1000)
            },
            'top_trigrams': {
                trigram: {
                    'count': count,
                    'probability': self._smooth_probability(count, self.total_terms, len(self.vocabulary))
                }
                for trigram, count in self.trigram_counts.most_common(1000)
            },
            'model_params': {
                'smoothing_method': self.smoothing_method,
                'laplace_alpha': self.laplace_alpha
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=4)

        print(f"\nModel saved to {output_path}")


def create_advanced_language_model():
    # Directory containing tokenized files
    directories = [
        Path(
            r'C:\Users\yonat\OneDrive\Documents\Information Systems\Year 3\Semester A\Information Retrieval\Documents txt\tokenized'),
        Path(
            r'C:\Users\yonat\OneDrive\Documents\Information Systems\Year 3\Semester A\Information Retrieval\Documents txt\linguistic_processed\1_no_stopwords'),
        Path(
            r'C:\Users\yonat\OneDrive\Documents\Information Systems\Year 3\Semester A\Information Retrieval\Documents txt\linguistic_processed\2_case_folded'),
        Path(
            r'C:\Users\yonat\OneDrive\Documents\Information Systems\Year 3\Semester A\Information Retrieval\Documents txt\linguistic_processed\3_stemmed')
    ]
    #tokenized_dir = Path(
       # r'C:\Users\yonat\OneDrive\Documents\Information Systems\Year 3\Semester A\Information Retrieval\Documents txt\tokenized')
    for directory in directories:

        # Initialize and train model
        model = AdvancedLanguageModel(smoothing_method='laplace', laplace_alpha=1.0)
        model.train(directory)

        # Save model
        model.save_model(directory.parent / 'language_model.json')

        # Print some statistics
        print("\nLanguage Model Statistics:")
        print(f"Total terms: {model.total_terms}")
        print(f"Vocabulary size: {len(model.vocabulary)}")
        print(f"Total documents: {len(model.doc_lengths)}")
        print(f"Average document length: {model.total_terms / len(model.doc_lengths):.2f}")

        print("\nMost common terms and their probabilities:")
        for term, count in model.unigram_counts.most_common(10):
            prob = model._smooth_probability(count, model.total_terms, len(model.vocabulary))
            idf = model.idf_scores[term]
            print(f"{term}: count={count}, prob={prob:.6f}, idf={idf:.2f}")


if __name__ == "__main__":
    create_advanced_language_model()