import os
import csv
import logging
from datetime import datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from difflib import SequenceMatcher  # Import SequenceMatcher for string similarity matching
import constants

# Load environment variables
load_dotenv()

# Neo4j Connection
def connect_to_neo4j():
    uri = os.getenv("NEO4J_URI", "neo4j+s://35dfdea4.databases.neo4j.io")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "KDYeM8NVhvtTEoIEut_Cvuc68oL778vzAvnlT028Xb0")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver

# Extract keywords from subtheme stored in Neo4j
def extract_keywords_from_subtheme(subtheme):
    return [keyword.strip().lower() for keyword in subtheme.split(",")]

# Enhanced matching to handle different phrasings
def match_user_input(user_input, stored_data):
    user_input = user_input.strip().lower()
    best_match = None
    highest_similarity = 0
    
    for data_item in stored_data:
        subtheme_keywords = extract_keywords_from_subtheme(data_item['subtheme'])
        
        # Check for an exact match first
        if user_input in subtheme_keywords:
            return data_item['response']  # Return immediately on an exact match
        
        # Use partial matching (sequence matcher) for looser matching
        for keyword in subtheme_keywords:
            similarity = SequenceMatcher(None, user_input, keyword).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = data_item['response']
    
    # Return if similarity is strong enough, otherwise fall back to text file
    return best_match if highest_similarity > 0.5 else None  # Lowered threshold for better flexibility

# Query Neo4j for the response based on the subtheme
def get_response_for_subtheme(user_input):
    driver = connect_to_neo4j()
    stored_data = []
    query = """
    MATCH (s:Subtheme)-[:HAS_RESPONSE]->(r:Response)
    RETURN s.name AS subtheme, r.text AS response
    """
    
    with driver.session() as session:
        results = session.run(query)
        for record in results:
            stored_data.append({'subtheme': record['subtheme'], 'response': record['response']})
    
    response = match_user_input(user_input, stored_data)
    return response if response else None

# Create date-specific log filenames
log_dir = 'chat_logs'
os.makedirs(log_dir, exist_ok=True)  # Ensure the chat_logs directory exists

current_date = datetime.now().strftime("%Y-%m-%d")
log_filename_csv = os.path.join(log_dir, f"chat_logs_{current_date}.csv")
log_filename_txt = os.path.join(log_dir, f"chat_logs_{current_date}.txt")

# Set up logs for CSV file
if not os.path.exists(log_filename_csv):
    with open(log_filename_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Response', 'Response Time'])  # Write header row if file is new

# Set up logs for TXT file
logging.basicConfig(filename=log_filename_txt, level=logging.INFO, format='%(asctime)s - %(message)s')

# Assistant Class
class Assistant:
    def __init__(self, file_path, context):
        self.context = context
        self.docs = self.load_text(file_path)
        self.vectorStore = self.create_db(self.docs)
        self.chain = self.create_chain()
        self.chat_history = []

    # Load text from file
    def load_text(self, file_path):
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()

    # Create vector database
    def create_db(self, docs):
        embedding = OpenAIEmbeddings(openai_api_key=constants.APIKEY)
        return Chroma.from_documents(docs, embedding=embedding)

    # Define the conversation chain
    def create_chain(self):
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=constants.APIKEY
        )

        # Define conversation prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a friendly and helpful AI assistant chatbot guiding users on how to navigate the Atlas map."),
            ("system", "Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        chain = create_stuff_documents_chain(llm=model, prompt=prompt)

        retriever = self.vectorStore.as_retriever(search_kwargs={"k": 1})

        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", f"Given the above conversation about {self.context}, generate a search query.")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm=model,
            retriever=retriever,
            prompt=retriever_prompt
        )

        return create_retrieval_chain(history_aware_retriever, chain)

    # Process user input and generate a response
    def process_chat(self, question):
        start_time = datetime.now()  # Timestamp when question is received
        neo4j_response = get_response_for_subtheme(question)
        if neo4j_response:
            response = neo4j_response
            logging.info("Response source: Neo4j")
        else:
            response = self.chain.invoke({
                "input": question,
                "chat_history": self.chat_history,
                "context": self.context
            })["answer"]
            logging.info("Response source: Text File / LangChain")

        end_time = datetime.now()  # Timestamp after response is generated
        response_time = (end_time - start_time).total_seconds()

        self.log_to_csv(question, response, response_time)
        self.log_chat_history(question, response)
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response))
        
        return response

    # Log to CSV file
    def log_to_csv(self, question, answer, response_time):
        with open(log_filename_csv, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([question, answer, response_time])

    # Log chat entry to TXT file
    def log_chat_history(self, question, answer):
        logging.info(f"User: {question}")
        logging.info(f"Assistant: {answer}")

    # Reset chat history
    def reset_chat_history(self):
        self.chat_history = []

# Main execution
if __name__ == '__main__':
    assistant = Assistant('prepared_data_ver3.txt', 'map navigation')
    print("Hello! Welcome to the Atlas Map Navigation Assistant!")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Thank you for using the Atlas Map Navigation Assistant. Goodbye!")
            break

        try:
            response = assistant.process_chat(user_input)
            print("Assistant:", response)
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Could you rephrase your question?")
