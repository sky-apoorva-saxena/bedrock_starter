import boto3
import json
import PyPDF2
import io
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

AWS_REGION = "eu-west-1"

bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

def load_and_split_pdf(pdf_path: str, chunk_size: int = 200) -> List[str]:
    """Load PDF and split into chunks"""
    chunks = []
    
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        for page in pdf_reader.pages:
            text = page.extract_text()
            # Simple splitting by sentences
            sentences = text.split('. ')
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
    
    return chunks

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of texts using Bedrock"""
    embeddings = []
    
    for text in texts:
        response = bedrock.invoke_model(
            body=json.dumps({"inputText": text}),
            modelId="amazon.titan-embed-text-v2:0",
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get('body').read())
        embedding = response_body.get("embedding")
        embeddings.append(embedding)
    
    return embeddings

def find_relevant_chunks(query: str, chunks: List[str], embeddings: List[List[float]], k: int = 2) -> List[str]:
    """Find the most relevant chunks using cosine similarity"""
    # Get query embedding
    query_response = bedrock.invoke_model(
        body=json.dumps({"inputText": query}),
        modelId="amazon.titan-embed-text-v2:0",
        accept="application/json",
        contentType="application/json"
    )
    
    query_response_body = json.loads(query_response.get('body').read())
    query_embedding = query_response_body.get("embedding")
    
    # Calculate similarities
    similarities = []
    for embedding in embeddings:
        similarity = cosine_similarity([query_embedding], [embedding])[0][0]
        similarities.append(similarity)
    
    # Get top k chunks
    top_indices = np.argsort(similarities)[-k:][::-1]
    relevant_chunks = [chunks[i] for i in top_indices]
    
    return relevant_chunks

def generate_response(query: str, context: List[str]) -> str:
    """Generate response using Bedrock LLM"""
    context_text = "\n\n".join(context)
    
    prompt = f"""Answer the user's question based on the following context:

Context:
{context_text}

Question: {query}

Answer:"""
    
    response = bedrock.invoke_model(
        body=json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 4096,
                "stopSequences": [],
                "temperature": 0,
                "topP": 1
            }
        }),
        modelId="amazon.titan-text-express-v1",
        accept="application/json",
        contentType="application/json"
    )
    
    response_body = json.loads(response.get('body').read())
    return response_body.get('results')[0].get('outputText')

def main():
    question = "What themes does Gone with the Wind explore?"
    
    # Load and split PDF
    print("Loading and splitting PDF...")
    chunks = load_and_split_pdf("src/assets/books.pdf")
    print(f"Created {len(chunks)} chunks")
    
    # Get embeddings for all chunks
    print("Getting embeddings...")
    embeddings = get_embeddings(chunks)
    
    # Find relevant chunks
    print("Finding relevant chunks...")
    relevant_chunks = find_relevant_chunks(question, chunks, embeddings, k=2)
    
    # Generate response
    print("Generating response...")
    response = generate_response(question, relevant_chunks)
    
    print("\n" + "="*50)
    print("QUESTION:", question)
    print("="*50)
    print("RESPONSE:", response)
    print("="*50)

if __name__ == "__main__":
    main()
