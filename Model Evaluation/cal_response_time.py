import csv

def calculate_average_response_time(filename):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        response_times = [float(row[2]) for row in reader if row]  # Collect all response times from the third column

    if response_times:
        average_time = sum(response_times) / len(response_times)
        print(f"Average Response Time: {average_time:.2f} seconds")
        return average_time
    else:
        print("No response times found in the file.")
        return None

if __name__ == "__main__":
    # Replace 'chat_logs_YYYY-MM-DD.csv' with your actual log file name
    calculate_average_response_time('chat_logs_2024-10-15.csv')
