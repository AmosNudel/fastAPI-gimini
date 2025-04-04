import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

# Load the Google API key from environment variables (or hardcode it here)
API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyD-uCGPkEnc_fichIw-3-c0RFl0D_SvTIY")  # Replace with your API key if needed

# Initialize the Google Gemini model
llm4 = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=API_KEY)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity, modify as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input model for the question
class QuestionRequest(BaseModel):
    question: str

# Define the POST endpoint to handle user questions
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    # Get the user's question from the request body
    user_question = request.question

    try:
        # Get the response from the Gemini model
        response = llm4.predict(user_question)
        return {"response": response}
    except Exception as e:
        # Handle any errors that occur while getting the response
        raise HTTPException(status_code=500, detail=str(e))

# You can add more endpoints here as needed
