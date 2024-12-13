import os
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nlp = spacy.load("en_core_web_lg")

# Preprocessing function
def preprocess_text(text):
    doc = nlp(text.lower()) 
    filtered_tokens = [
        token for token in doc
        if token.is_alpha and not token.is_stop and token.pos_ in {"NOUN", "VERB", "ADJ"}
    ]
    return filtered_tokens

# Representasi dokumen dengan Word2Vec
def get_doc_vector(tokens):
    if not tokens: 
        return np.zeros(nlp.vocab.vectors.shape[1])
    return np.mean([token.vector for token in tokens], axis=0)
    

# baca dokumen dari folder abstracts dan representasikan sebagai vektor
def load_and_vectorize_files(directory):
    txt_files = [file for file in os.listdir(directory) if file.endswith('.txt')]
    documents = {}
    for file in txt_files:
        file_path = os.path.join(directory, file)
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
            tokens = preprocess_text(content)
            vector = get_doc_vector(tokens)
            documents[file] = vector
    return documents

def calculate_similarity_vector(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

def compare_abstracts_vector():
    abstracts_directory = "./abstracts" 
    if not os.path.exists(abstracts_directory):
        print(f"Directory '{abstracts_directory}' not found!")
        return

    documents = load_and_vectorize_files(abstracts_directory)
    file_names = list(documents.keys())
    if len(file_names) < 2:
        print("Need at least two documents in the directory to compare!")
        return

    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            file1, file2 = file_names[i], file_names[j]
            similarity_score = calculate_similarity_vector(documents[file1], documents[file2])
            print(f"Similarity between {file1} and {file2}: {similarity_score:.4f}")

compare_abstracts_vector()
