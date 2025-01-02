import warnings
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# scikit-learn imports
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class TextClassifier:
    # Suppress certain scikit-learn warnings
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    def __init__(self, search_enging_processed_dir, our_processed_dir, our_labels_file, search_labels_file):
        """
        :param search_enging_processed_dir: Directory containing processed .txt files
                                            for 'Does not belong' documents
        :param our_processed_dir: Directory containing processed .txt files
                                  for 'Belongs' documents
        :param our_labels_file: CSV file containing docs labeled 'Belongs'
        :param search_labels_file: CSV file containing docs labeled 'Does not belong'
        """
        self.search_enging_processed_dir = Path(search_enging_processed_dir)
        self.our_processed_dir = Path(our_processed_dir)

        # 1) Read the CSV that contains 'Belongs' documents
        df_belong = pd.read_csv(our_labels_file, encoding='latin1')
        df_belong['Label'] = 'Belongs'  # Set label explicitly

        # 2) Read the CSV that contains 'Does not belong' documents
        df_not_belong = pd.read_csv(search_labels_file, encoding='latin1')
        df_not_belong['Label'] = 'Does not belong'  # Set label explicitly

        # 3) Combine the two dataframes
        self.labels_df = pd.concat([df_belong, df_not_belong], ignore_index=True)

        # 4) Convert "Label" column to a binary label: 1 -> Belongs, 0 -> Does not belong
        self.labels_df['Binary_Label'] = (self.labels_df['Label'] == 'Belongs').astype(int)

        # Define classifiers
        self.classifiers = {
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(kernel='linear'),
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
        }

    def load_documents(self):
        """Load processed documents from the two separate directories."""
        documents = []
        filenames = []
        labels = []

        for idx, row in self.labels_df.iterrows():
            # Decide which directory to read from based on the label
            if row['Label'] == 'Belongs':
                file_dir = self.our_processed_dir
            else:
                file_dir = self.search_enging_processed_dir

            # Construct the filename (processed_<File Name>.txt)
            filename = f"processed_{row['File Name']}.txt"
            file_path = file_dir / filename

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

        return np.array(documents), np.array(filenames), np.array(labels)

    def cross_validate(self, documents, filenames, labels, n_splits=10):
        """
        10-fold cross validation:
        - בכל קיפול נבנה מודל, נאמן אותו על כ-90% מהמסמכים, ונבדוק על 10% הנותרים.
        - StratifiedKFold דואג לחלוקה פרופורציונלית של התוויות.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Dictionaries to accumulate scores for each classifier
        accuracy_dict = {name: [] for name in self.classifiers.keys()}
        precision_dict = {name: [] for name in self.classifiers.keys()}
        recall_dict = {name: [] for name in self.classifiers.keys()}
        f1_dict = {name: [] for name in self.classifiers.keys()}

        fold_index = 1
        for train_index, test_index in skf.split(documents, labels):
            print(f"\n=== Fold {fold_index}/{n_splits} ===")
            fold_index += 1

            # פיצול ל-Train/Test
            docs_train, docs_test = documents[train_index], documents[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            filenames_train, filenames_test = filenames[train_index], filenames[test_index]

            # בונים את ה-Vectorizer רק על קבוצת האימון
            vectorizer = TfidfVectorizer(max_features=1000)
            X_train = vectorizer.fit_transform(docs_train)
            X_test = vectorizer.transform(docs_test)

            # אימון ובדיקה של כל מסווג
            for name, clf in self.classifiers.items():
                # מאמנים
                clf.fit(X_train, y_train)
                # חוזים
                y_pred = clf.predict(X_test)

                # חישוב דיוק (Accuracy)
                acc = accuracy_score(y_test, y_pred)

                # Classification report as dict to get precision, recall, f1
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                # נשתמש ב-'macro avg' להדפסת הממוצע על פני שתי הקטגוריות (1 ו-0)
                avg_precision = report_dict['macro avg']['precision']
                avg_recall = report_dict['macro avg']['recall']
                avg_f1 = report_dict['macro avg']['f1-score']

                # צוברים את התוצאות בכל dict
                accuracy_dict[name].append(acc)
                precision_dict[name].append(avg_precision)
                recall_dict[name].append(avg_recall)
                f1_dict[name].append(avg_f1)

                # הדפסה עבור כל קיפול
                #print(f"\nClassifier: {name}")
                #print(f"Accuracy: {acc:.3f}")
                #print(f"Avg Precision (macro): {avg_precision:.3f}")
                #print(f"Avg Recall (macro): {avg_recall:.3f}")
                #print(f"Avg F1-score (macro): {avg_f1:.3f}")

                #print("\nClassification Report:")
                #print(classification_report(y_test, y_pred))

                # מטריצת בלבול
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {name} - Fold {fold_index-1}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                # נשמור כל מטריצה בקובץ נפרד (בתיקיה שתגדירו, למשל ConfustionMatrixes/)
                plt.savefig(
                    f'ConfustionMatrixes/conf_matrix_{name.lower().replace(" ", "_")}_fold_{fold_index-1}.png'
                )
                plt.close()

        # לאחר סיום כל הקיפולים, נדפיס ממוצעים וסטיות תקן
        print("\n=== Cross-Validation Summary ===")
        for name in self.classifiers.keys():
            avg_acc = np.mean(accuracy_dict[name])
            std_acc = np.std(accuracy_dict[name])
            avg_prec = np.mean(precision_dict[name])
            std_prec = np.std(precision_dict[name])
            avg_rec = np.mean(recall_dict[name])
            std_rec = np.std(recall_dict[name])
            avg_f1s = np.mean(f1_dict[name])
            std_f1s = np.std(f1_dict[name])

            print(f"\n{name}:")
            print(f"  Accuracy: {avg_acc:.3f} (± {std_acc:.3f})")
            print(f"  Precision (macro): {avg_prec:.3f} (± {std_prec:.3f})")
            print(f"  Recall (macro): {avg_rec:.3f} (± {std_rec:.3f})")
            print(f"  F1-score (macro): {avg_f1s:.3f} (± {std_f1s:.3f})")

    def run_cross_validation(self):
        """
        פונקציה מרכזית שרק טוענת מסמכים ואז מפעילה cross_validate.
        """
        print("Loading documents...")
        documents, filenames, labels = self.load_documents()
        print(f"Loaded {len(documents)} documents.")

        # נבצע 10-Fold Cross Validation
        print("\nStarting 10-Fold Cross Validation...")
        self.cross_validate(documents, filenames, labels, n_splits=10)


def main():
    # Update these paths as necessary
    search_enging_processed_dir = "SearchEngineBias/Processed_Files"
    our_processed_dir = "OurDocuments/Processed_Files"

    # The CSV with documents that 'Belong'
    our_labels_file = "OurDocument_Labels.csv"

    # The CSV with documents that 'Does not belong'
    search_labels_file = "search_engine_bias_labels.csv"

    classifier = TextClassifier(
        search_enging_processed_dir=search_enging_processed_dir,
        our_processed_dir=our_processed_dir,
        our_labels_file=our_labels_file,
        search_labels_file=search_labels_file
    )

    # מפעילים 10-Fold Cross Validation
    classifier.run_cross_validation()


if __name__ == "__main__":
    main()
