import csv
import string

import nltk
import pandas as pd
from gensim.downloader import load
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

from trash_final import Assistant

# Load pre-trained word vectors from Gensim (GloVe vectors in this case)
word_vectors = load('glove-wiki-gigaword-100')

# Load the pre-trained Sentence-BERT model
sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def download_nltk_resources():
    """Ensure that necessary NLTK resources are available."""
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading WordNet...")
        nltk.download('wordnet')
        nltk.download('omw-1.4')

def preprocess(text):
    """Preprocess text for WMD by tokenising, lowering the text, and removing punctuation."""
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return [word for word in tokens if word in word_vectors]

def calculate_wmd(response, ground_truth):
    """Calculate Word Mover's Distance (WMD) for response vs ground truth."""
    preprocessed_response = preprocess(response)
    preprocessed_ground_truth = preprocess(ground_truth)
    
    if not preprocessed_response or not preprocessed_ground_truth:
        return 2.0  # Assign a large distance for sentences that can't be compared
    
    try:
        distance = word_vectors.wmdistance(preprocessed_response, preprocessed_ground_truth)
        return distance if distance != float('inf') else 2.0  # Handle inf distance
    except KeyError:
        return 2.0

def calculate_metrics(expected, generated):
    """Calculate precision, recall, and F1 score."""
    expected_set = set(expected)
    generated_set = set(generated)

    true_positives = len(expected_set.intersection(generated_set))
    false_positives = len(generated_set - expected_set)
    false_negatives = len(expected_set - generated_set)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0

    return precision, recall, f1

def calculate_rouge_l_score(expected, generated):
    """Calculate ROUGE-L score for the expected and generated responses."""
    rouge = Rouge()
    scores = rouge.get_scores(generated, expected, avg=True)
    return scores['rouge-l']['f']  # Return only ROUGE-L score

def calculate_sentence_similarity(response, ground_truth):
    """Calculate cosine similarity between sentence embeddings."""
    response_embedding = sbert_model.encode([response])
    ground_truth_embedding = sbert_model.encode([ground_truth])
    similarity = cosine_similarity(response_embedding, ground_truth_embedding)[0][0]
    return similarity

def validate_chatbot(validation_file, chatbot):
    """Validate the chatbot responses using METEOR, BLEU, ROUGE-L, WMD, and Sentence-BERT scores."""
    results = []
    
    with open(validation_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',')
        for row_num, row in enumerate(reader, start=1):
            if len(row) < 2:
                print(f"Skipping invalid row at line {row_num}: {row}")
                continue

            question = row[0].strip()
            expected_response = row[1].strip()

            generated_response = chatbot.process_chat(question)

            if isinstance(generated_response, tuple):
                generated_response = generated_response[0]

            expected_tokens = expected_response.split()
            generated_tokens = generated_response.split()

            meteor_score_value = meteor_score([expected_tokens], generated_tokens)
            bleu_score_value = sentence_bleu([expected_tokens], generated_tokens)
            precision, recall, f1 = calculate_metrics(expected_tokens, generated_tokens)
            rouge_l = calculate_rouge_l_score(expected_response, generated_response)
            wmd_score = calculate_wmd(generated_response, expected_response)
            sbert_similarity = calculate_sentence_similarity(generated_response, expected_response)

            results.append({
                'question': question,
                'ground_truth': expected_response,
                'generated_response': generated_response,
                'meteor_score': round(meteor_score_value, 2),
                'bleu_score': round(bleu_score_value, 2),
                'rouge_l': round(rouge_l, 2),
                'precision': round(precision, 2),
                'recall': round(recall, 2),
                'f1_score': round(f1, 2),
                'wmd_score': round(wmd_score, 2),
                'sbert_similarity': round(sbert_similarity, 2)
            })

    return results

def main():
    """Main function to run the chatbot validation."""
    download_nltk_resources()

    validation_file = 'Val_vishu.csv'  # Path to your validation CSV file
    context = 'map navigation'  # Provide the context as required by the Assistant class

    chatbot = Assistant(context)  # Initialise the chatbot instance with the correct argument

    results = validate_chatbot(validation_file, chatbot)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV
    results_df.to_csv('val_results_vishu.csv', index=False)

    # Print average scores
    if not results_df.empty:
        avg_scores = {
            'avg_meteor': round(results_df['meteor_score'].mean(), 2),
            'avg_bleu': round(results_df['bleu_score'].mean(), 2),
            'avg_rouge_l': round(results_df['rouge_l'].mean(), 2),
            'avg_precision': round(results_df['precision'].mean(), 2),
            'avg_recall': round(results_df['recall'].mean(), 2),
            'avg_f1': round(results_df['f1_score'].mean(), 2),
            'avg_wmd': round(results_df['wmd_score'].mean(), 2),
            'avg_sbert_similarity': round(results_df['sbert_similarity'].mean(), 2)
        }
        
        print("Average Scores:")
        for metric, avg in avg_scores.items():
            print(f"{metric.replace('avg_', 'Average ').title()}: {avg:.2f}")

if __name__ == "__main__":
    main()
