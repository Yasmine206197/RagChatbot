import os, re 
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
from flask import Flask, request, render_template, jsonify
import json

# Set up the Groq API key
os.environ['GROQ_API_KEY'] = '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'

# Initialize Groq client
# initializes the Groq client using the API key retrieved from the environment variable.
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Step 1: Parse the dataset
#This function loads a dataset from a tab-delimited text file (dialogs.txt) and separates it into questions and answers, which are returned as lists.
def load_dataset(file_path):
    data = pd.read_csv(file_path, delimiter='\t', header=None, names=['question', 'answer'])
    questions = data['question'].tolist()
    answers = data['answer'].tolist()
    return questions, answers

questions, answers = load_dataset('dialogs.txt')

# Step 2: Create embeddings and store in a vector database
# A pre-trained sentence transformer model (all-MiniLM-L6-v2) is loaded to create embeddings for the questions. 
model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = model.encode(questions)

# Initialize FAISS index
#The FAISS (Facebook AI Similarity Search) index is initialized. FAISS is a library for efficient similarity search and clustering of dense vectors.
#The dimension d of the embeddings is used to set up the index. The question embeddings are then added to the FAISS index.
d = question_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(d)
faiss_index.add(question_embeddings)

# Function to find the most relevant question
# This function takes a query, creates its embedding, and searches the FAISS index for the most similar questions. 
# It returns the top k matching answers.
def find_best_match(query, k=1):
    query_embedding = model.encode([query])
    D, I = faiss_index.search(query_embedding, k)
    return [answers[i] for i in I[0]]

# Step 3: Define the generation function using Groq API with RAG approach
# It retrieves the most relevant answer using the find_best_match function,
# Constructs a prompt with the retrieved context and the user's query, 
# then sends this prompt to the Groq API to generate a completion. 
def generate_response(query):
    best_match_answers = find_best_match(query, k=1)
    if best_match_answers:
        best_match_answer = best_match_answers[0]
        print(f"Retrieved context: {best_match_answer}")  # Print the retrieved context for debugging
        
        # Create the prompt with context and query
        prompt = f"Context: {best_match_answer}\nUser Query: {query}\nAnswer:"
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content
    else:
        return "Sorry, I don't have an answer for that."
    
    


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Initialize history from the request form, or use an empty list if not present
    history = request.form.get('history', '[]')
    
    try:
        history = json.loads(history)  # Safely parse JSON string to list
    except json.JSONDecodeError:
        history = []  # Default to empty list if parsing fails

    if request.method == 'POST':
        user_query = request.form['query']
        response = generate_response(user_query)
        history.append({'query': user_query, 'response': response})
        # Convert history to JSON string for rendering
        history_json = json.dumps(history)
        return render_template('index.html', history=history, history_json=history_json)
    
    return render_template('index.html', history=history, history_json=json.dumps(history))

if __name__ == '__main__':
    app.run(debug=True)