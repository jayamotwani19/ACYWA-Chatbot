import os
import json
import openai
import numpy as np
import constants  # Import your constants module which contains the API key
from sklearn.metrics import classification_report

# Set the API key from constants
openai.api_key = constants.APIKEY

# Use the fine-tuned chat model ID
FT_MODEL = "gpt-4o-2024-08-06"
EXPERIMENT_NAME = "gpt-4o-2024-08-06"

def ensure_directory_exists(directory):
    """Ensure the directory exists, and if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def evaluate_model():
    """Evaluate the model by running it on the test dataset."""
    output_dir = ensure_directory_exists("output")  # Ensure the output directory exists
    test_data_path = os.path.join("prepared", "test.jsonl")  # Path to the test data
    
    inputs = []
    outputs_gold = []
    outputs_pred = []

    # Load and process test data from the JSONL file
    with open(test_data_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            user_input = data['messages'][1]['content']  # Extract user input
            gold_output = data['messages'][2]['content']  # Extract expected response
            inputs.append(user_input)
            outputs_gold.append(gold_output)

            # Generate prediction using the chat completion method
            response = openai.ChatCompletion.create(
                model=FT_MODEL,
                messages=data['messages'],  # Include system, user, and assistant messages
                max_tokens=50,
                temperature=0
            )
            predicted_output = response['choices'][0]['message']['content'].strip()
            outputs_pred.append(predicted_output)

    # Generate classification report
    report = classification_report(outputs_gold, outputs_pred, labels=np.unique(outputs_gold), zero_division=0)

    print("Classification Report:")
    print(report)

    # Save the report and predictions to files
    report_path = os.path.join(output_dir, f"evaluation_report-{EXPERIMENT_NAME}.txt")
    predictions_path = os.path.join(output_dir, f"model_predictions-{EXPERIMENT_NAME}.txt")

    with open(report_path, "w") as f:
        f.write(report)

    with open(predictions_path, "w") as f:
        for inp, pred in zip(inputs, outputs_pred):
            f.write(f"Input: {inp}\nPredicted: {pred}\n\n")

def main():
    evaluate_model()

if __name__ == "__main__":
    main()
