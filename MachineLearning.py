import warnings
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# scikit-learn imports
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Optional: If you have a stopwords removal function
# from LinguisticOperations import remove_stopwords


class TextClassifier:
    # Suppress certain scikit-learn warnings
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    def __init__(self, processed_dir, labels_file):
        self.processed_dir = Path(processed_dir)
        self.labels_df = pd.read_csv(labels_file, encoding='latin1')  # or 'ISO-8859-1'
        # Convert the "Label" to a binary label
        self.labels_df['Binary_Label'] = (self.labels_df['Label'] == 'Belongs').astype(int)

        # Four scikit-learn classifiers
        self.classifiers = {
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(kernel='linear'),
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
        }

    def load_documents(self):
        """Load processed documents from the specified directory."""
        documents = []
        filenames = []
        labels = []

        for idx, row in self.labels_df.iterrows():
            filename = f"processed_{row['File Name']}.txt"
            file_path = self.processed_dir / filename

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents.append(content)
                filenames.append(row['File Name'])
                labels.append(row['Binary_Label'])
            except FileNotFoundError:
                print(f"Warning: File not found - {filename}")
                continue
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

        return documents, filenames, labels

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, filenames_test):
        """
        Train and evaluate each classifier on the train/test split.
        X_train, y_train -> training data (90%)
        X_test, y_test -> testing data (10%)
        filenames_test -> the filenames corresponding to X_test documents
        """
        for name, clf in self.classifiers.items():
            print(f"\n=== {name} ===")

            # Train classifier
            clf.fit(X_train, y_train)

            # Predict on test set
            y_pred = clf.predict(X_test)

            # Print metrics
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
            plt.close()

            # Misclassification analysis
            misclassified_indices = np.where(y_pred != y_test)[0]
            print(f"\nMisclassifications for {name}:")
            if len(misclassified_indices) == 0:
                print("  No misclassifications!")
            else:
                # Print details for a few misclassified documents
                for idx in misclassified_indices[:5]:  # limit to first 5 for brevity
                    true_label_str = "Belongs" if y_test[idx] == 1 else "Does not belong"
                    pred_label_str = "Belongs" if y_pred[idx] == 1 else "Does not belong"
                    print(f"  - Document: {filenames_test[idx]}")
                    print(f"    True label: {true_label_str}, Predicted label: {pred_label_str}")

    def run(self):
        # 1) Load documents
        print("Loading documents...")
        documents, filenames, labels = self.load_documents()

        # 2) Create TF-IDF features
        print("\nCreating TF-IDF features...")
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(documents)  # shape: (num_docs, num_features)

        # 3) Split into train (90%) and test (10%)
        # Stratify by labels so the class distribution is maintained
        X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
            X, labels, filenames, test_size=0.1, random_state=42, stratify=labels
        )

        # 4) Train and evaluate
        print("\nTraining classifiers on 90% and testing on 10% of the data...")
        self.train_and_evaluate(X_train, y_train, X_test, y_test, filenames_test)


def main():
    processed_dir = Path("SearchEngineBias/Processed_Files")
    labels_file = Path("search_engine_bias_labels.csv")

    classifier = TextClassifier(processed_dir, labels_file)
    classifier.run()


if __name__ == "__main__":
    main()
