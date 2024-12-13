import csv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import constants

# Set up the API key from constants.py
api_key = constants.APIKEY

# Load the input TXT file that contains the context, keywords, and response (colon-separated)
input_txt = "prepared_data_ver3.txt"
output_csv = "validation_data_paraphrased2.csv"

# Step 1: Load the data from input TXT
data = []
with open(input_txt, mode='r', encoding='utf-8') as file:
    for line in file:
        # Split each line by the colon character
        parts = line.strip().split(':')
        if len(parts) >= 3:
            context = parts[0].strip()
            keywords = parts[1].strip()
            response = parts[2].strip()
            data.append((context, keywords, response))

# Step 2: Set up the LLM and prompt for generating paraphrased questions
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=api_key)

# Define a prompt template to generate paraphrased questions
paraphrase_prompt_template = PromptTemplate(
    template=(
        "Context: {context}\n"
        "Keywords: {keywords}\n"
        "Generate 2 paraphrased very simple generic questions that could be asked by a first time user of this website, related to context or keywords and strictly about navigating the website. The website has many themes and subcategories related to keywords, so you have to generate simple broadly scoped questions trying to find information on this website such as what is where on this website.  Do not generate serial number for each question."
    ),
    input_variables=["context", "keywords"]
)

# Create a chain for generating paraphrased questions
paraphrase_chain = LLMChain(llm=llm, prompt=paraphrase_prompt_template)

# Step 3: Generate paraphrased questions and save them
with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Question', 'Answer'])  # Header for the output CSV

    for context, keywords, response in data:
        # Run the chain to generate paraphrased questions
        try:
            paraphrased_questions = paraphrase_chain.run({
                "context": context,
                "keywords": keywords
            })

            # Split the paraphrased questions by line breaks
            question_list = paraphrased_questions.split("\n")

            # Write each question with the associated response to the CSV
            for question in question_list:
                if question.strip():  # Make sure the question is not empty
                    writer.writerow([question.strip(), response])
        except Exception as e:
            print(f"An error occurred while generating questions for context '{context}': {e}")

print(f"Generated paraphrased Q&A pairs saved to {output_csv}")
