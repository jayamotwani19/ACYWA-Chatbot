import random

# Step 1: Read the .txt file with Q&A pairs
with open('qa_data_maps.txt', 'r', encoding='utf-8') as f:
    qa_data = f.read().split("\n\n")  # Assuming Q&A pairs are separated by two newlines

# Step 2: Shuffle the Q&A pairs to ensure randomness
random.shuffle(qa_data)

# Step 3: Calculate the number of items for each set (60% train, 20% test, 20% validation)
total_size = len(qa_data)
train_size = int(0.6 * total_size)
test_size = int(0.2 * total_size)
validation_size = total_size - train_size - test_size  # Ensuring all data is used

# Step 4: Split the data into train, test, and validation sets
train_data = qa_data[:train_size]
test_data = qa_data[train_size:train_size + test_size]
validation_data = qa_data[train_size + test_size:]

# Step 5: Write each set into separate .txt files

# Writing train data
with open('train_data.txt', 'w', encoding='utf-8') as train_file:
    train_file.write("\n\n".join(train_data))

# Writing test data
with open('test_data.txt', 'w', encoding='utf-8') as test_file:
    test_file.write("\n\n".join(test_data))

# Writing validation data
with open('validation_data.txt', 'w', encoding='utf-8') as validation_file:
    validation_file.write("\n\n".join(validation_data))

print(f"Data split complete. Train: {len(train_data)} items, Test: {len(test_data)} items, Validation: {len(validation_data)} items")
