import os
import json
import openai
import time
import constants

# Setup the API key from constants.py
openai.api_key = constants.APIKEY

def finetune_model():
    """Finetune a model in ChatGPT."""
    
    current_dir = os.getcwd()
    prepared_dir = os.path.join(current_dir, "prepared")
    
    # Correctly upload the train file to OpenAI
    train_path = os.path.join(prepared_dir, "train.jsonl")
    with open(train_path, "rb") as f:  # Use "rb" to open the file in binary read mode
        f_train = openai.File.create(file=f, purpose="fine-tune")
    
    # Correctly upload the validation file to OpenAI
    val_path = os.path.join(prepared_dir, "val.jsonl")
    with open(val_path, "rb") as f:  # Use "rb" for binary read mode
        f_dev = openai.File.create(file=f, purpose="fine-tune")

    print("Waiting for file processing...", end="")
    while True:
        f_train_status = openai.File.retrieve(f_train.id)["status"]
        f_dev_status = openai.File.retrieve(f_dev.id)["status"]

        if f_train_status == "processed" and f_dev_status == "processed":
            break

        time.sleep(5)
        print(".", end="")
        sys.stdout.flush()

    print("\nFiles ready. Creating fine-tune job...")

    # Create the fine-tuning job using file IDs
    ftj = openai.FineTuningJob.create(
        training_file=f_train.id,
        validation_file=f_dev.id,
        model="gpt-4o-2024-08-06",
    )

    print(f"Created fine-tune job with id {ftj.id}. Writing to finetune-job.json...")
    with open("finetune-job.json", "w") as f:
        json.dump(ftj, f)

    print("Waiting for fine-tuning to complete...")
    while True:
        ftj_status = openai.FineTuningJob.retrieve(ftj.id)["status"]
        if ftj_status in ["succeeded", "failed"]:
            break

        time.sleep(5)

    print("Fine tuning complete. Final results:")
    ftj_result = openai.FineTuningJob.retrieve(ftj.id)
    print(ftj_result)

    with open("finetune-result.json", "w") as f:
        json.dump(ftj_result, f)

def main():
    finetune_model()

if __name__ == "__main__":
    main()
