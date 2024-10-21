from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

app = Flask(__name__)

# Load the FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')  # Load the same model used for embeddings
vectorstore = FAISS.load_local("faiss_store", embeddings, allow_dangerous_deserialization=True)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')

    # Perform similarity search
    results = vectorstore.similarity_search(query)

    # Return the top result
    return jsonify({
        'response': results[0].page_content if results else "No results found."
    })

if __name__ == '__main__':
    app.run(debug=True)
