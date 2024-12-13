import random
import os

# Step 1: Read the .txt file with Q&A pairs
with open('prepared_data_ver3.txt', 'r', encoding='utf-8') as f:
    qa_data = f.read().split("\n\n")  # Assuming Q&A pairs are separated by two newlines

# Step 2: Shuffle the Q&A pairs to ensure randomness
random.shuffle(qa_data)

# Step 3: Calculate the number of items for each set (60% train, 20% test, 20% validation)
total_size = len(qa_data)
train_size = int(0.6 * total_size)
test_size = int(0.2 * total_size)
val_size = total_size - train_size - test_size  # Ensuring all data is used

# Step 4: Split the data into train, test, and validation sets
train_data = qa_data[:train_size]
test_data = qa_data[train_size:train_size + test_size]
val_data = qa_data[train_size + test_size:]

# Prepare output directory inside the current directory
current_dir = os.getcwd()  # Get the current working directory
prepared_dir = os.path.join(current_dir, "prepared")
os.makedirs(prepared_dir, exist_ok=True)  # Ensure the directory exists

# Step 5: Write each set into separate .txt files in the 'prepared' directory
train_file_path = os.path.join(prepared_dir, 'train_data.txt')
test_file_path = os.path.join(prepared_dir, 'test_data.txt')
val_file_path = os.path.join(prepared_dir, 'val_data.txt')

# Writing train data
with open(train_file_path, 'w', encoding='utf-8') as train_file:
    train_file.write("\n\n".join(train_data))

# Writing test data
with open(test_file_path, 'w', encoding='utf-8') as test_file:
    test_file.write("\n\n".join(test_data))

# Writing validation data
with open(val_file_path, 'w', encoding='utf-8') as val_file:
    val_file.write("\n\n".join(val_data))

print(f"Data split complete. Train: {len(train_data)} items, Test: {len(test_data)} items, Val: {len(val_data)} items")
