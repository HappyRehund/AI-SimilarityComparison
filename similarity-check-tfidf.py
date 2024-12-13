import os
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_lg")

# Preprocessing function
def preprocess_text(text):
    doc = nlp(text.lower()) 
    filtered_tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and not token.is_stop and token.pos_ in {"NOUN", "VERB", "ADJ"}
    ]
    return " ".join(filtered_tokens)

# baca doc dari folder abstracts dan preprocess
def load_and_preprocess_files(directory):
    txt_files = [file for file in os.listdir(directory) if file.endswith('.txt')]
    documents = {}
    for file in txt_files:
        file_path = os.path.join(directory, file)
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
            processed_content = preprocess_text(content)
            documents[file] = processed_content
    return documents

def calculate_similarity(doc1, doc2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([doc1, doc2])
    similarity_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return similarity_score

def compare_abstracts():
    abstracts_directory = "./abstracts" 
    if not os.path.exists(abstracts_directory):
        print(f"Directory '{abstracts_directory}' not found!")
        return

    documents = load_and_preprocess_files(abstracts_directory)
    file_names = list(documents.keys())
    if len(file_names) < 2:
        print("Need at least two documents in the directory to compare!")
        return

    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            file1, file2 = file_names[i], file_names[j]
            similarity_score = calculate_similarity(documents[file1], documents[file2])
            print(f"Similarity between {file1} and {file2}: {similarity_score:.4f}")

compare_abstracts()
