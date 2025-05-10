from dotenv import load_dotenv
import os
import openai  # Use this instead of 'OpenAI'

load_dotenv()

# Load environment variables from .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_answer(query, context):
    prompt = f"""Answer the question based on the context below:
Context: {context}
Question: {query}
Answer:"""

    # Call the OpenAI ChatCompletion API using the openai module
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    # Extract and return the answer from the response
    return response['choices'][0]['message']['content']
