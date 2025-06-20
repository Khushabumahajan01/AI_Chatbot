import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from dotenv import load_dotenv
import groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama3-70b-8192"

app = FastAPI()

class MessageRequest(BaseModel):
    message: str

class MessageResponse(BaseModel):
    response: str
    embedding: list | None = None

class LLMGateway:
    def _init_(self):
        pass

    def embedding(self, text: str):
        payload = {
            "input": text,
            "model": "text-embedding-nomic-embed-text-v1.5@q8_0"
        }
        endpoint = "http://172.52.50.82:3333/v1/embeddings"
        try:
            response = requests.post(endpoint, json=payload, timeout=5)
            response.raise_for_status()
            chat_res = response.json()
            data = chat_res.get("data", [])
            if data:
                return data[0].get("embedding", None)
        except Exception as e:
            print(f"Embedding service error: {e}")
        return None

llm_gateway = LLMGateway()

@app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    try:
        embedding = llm_gateway.embedding(request.message)

        client = groq.Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": request.message},
            ],
            model=MODEL_NAME,
            temperature=0.7,
            max_completion_tokens=1024,
        )

        content = chat_completion.choices[0].message.content
        return MessageResponse(response=content, embedding=embedding)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GROQ Chat API failed: {str(e)}")