"""
FastAPI Server for Legal Document Analysis
Handles PDF uploads and returns structured analysis results
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import PyPDF2
from io import BytesIO
import logging
from typing import Dict, Any
import os
from dotenv import load_dotenv
import uuid
from datetime import datetime

# Import our backend logic
from backend import process_document

# In-memory storage for documents (in production, use a database)
document_storage = {}

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Legal Document Demystifier API",
    description="AI-powered legal document analysis using Google Gemini",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc
)

# Configure CORS - Allow all origins for development
# In production, specify your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_text_from_pdf_stream(file_stream: BytesIO) -> str:
    """
    Extract text from PDF file stream
    """
    try:
        pdf_reader = PyPDF2.PdfReader(file_stream)
        text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception as e:
                logger.warning(f"Could not extract text from page {page_num + 1}: {e}")
                continue
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read PDF file: {str(e)}"
        )


@app.get("/")
async def root():
    """
    Root endpoint - API health check
    """
    return {
        "message": "Legal Document Demystifier API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    # Check if Google API key is configured
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key or api_key == "your_google_api_key_here":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": "Google API key not configured"
            }
        )
    
    return {
        "status": "healthy",
        "api_configured": True
    }


@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Main endpoint for document analysis
    
    Args:
        file: PDF file to analyze
        
    Returns:
        JSON with extracted legal information
    """
    # Validate file type
    if not file.content_type or file.content_type != 'application/pdf':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload a PDF file."
        )
    
    # Check file size (limit to 10MB)
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    if file_size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large. Maximum size is 10MB."
        )
    
    try:
        logger.info(f"Processing file: {file.filename} ({file_size} bytes)")
        
        # Extract text from PDF
        file_stream = BytesIO(content)
        document_text = extract_text_from_pdf_stream(file_stream)
        
        if not document_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not extract text from PDF. The file may be image-based or corrupted."
            )
        
        logger.info(f"Extracted {len(document_text)} characters from PDF")
        
        # Process the document
        results = process_document(document_text)
        
        # Check if there was an error in processing
        if "error" in results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=results["error"]
            )
        
        # Store document for AI chat reference
        document_id = str(uuid.uuid4())
        document_storage[document_id] = {
            "filename": file.filename,
            "text": document_text,
            "analysis": results,
            "upload_time": datetime.now().isoformat()
        }
        
        logger.info("Document analysis completed successfully")
        
        return {
            "success": True,
            "filename": file.filename,
            "analysis": results,
            "document_id": document_id,  # Return document ID for chat reference
            "metadata": {
                "text_length": len(document_text),
                "processing_time": "completed"
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@app.post("/analyze-text")
async def analyze_text_endpoint(text_data: Dict[str, str]):
    """
    Alternative endpoint for analyzing plain text (for testing)
    
    Args:
        text_data: JSON with "text" field containing the document text
        
    Returns:
        JSON with extracted legal information
    """
    if "text" not in text_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing 'text' field in request body"
        )
    
    document_text = text_data["text"]
    
    if not document_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text field cannot be empty"
        )
    
    try:
        logger.info(f"Processing text input ({len(document_text)} characters)")
        
        results = process_document(document_text)
        
        if "error" in results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=results["error"]
            )
        
        return {
            "success": True,
            "analysis": results,
            "metadata": {
                "text_length": len(document_text),
                "processing_time": "completed"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during text analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@app.post("/chat")
async def chat_endpoint(chat_data: Dict[str, str]):
    """
    AI Chat endpoint for interactive legal document Q&A
    
    Args:
        chat_data: JSON with "question", "context", and optional "document_id" fields
        
    Returns:
        JSON with AI response
    """
    if "question" not in chat_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing 'question' field in request body"
        )
    
    question = chat_data["question"]
    context = chat_data.get("context", "legal_document_analysis")
    document_id = chat_data.get("document_id")
    
    if not question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question field cannot be empty"
        )
    
    try:
        logger.info(f"Processing chat question: {question[:50]}...")
        
        # Create a simple AI response using Gemini
        from backend import analyzer
        
        # Check if we have a document to reference
        document_context = ""
        if document_id and document_id in document_storage:
            doc_data = document_storage[document_id]
            document_context = f"""
DOCUMENT CONTEXT:
Filename: {doc_data['filename']}
Document Text: {doc_data['text'][:2000]}...  # Limit to first 2000 chars
Analysis Results: {doc_data['analysis']}
"""
        else:
            document_context = "No specific document is currently loaded. Please upload a document first to ask specific questions about it."
        
        chat_prompt = f"""You are a helpful legal AI assistant. Answer the following question about legal documents in a clear, simple way.

{document_context}

Question: {question}

Context: {context}

Please provide a helpful, accurate response in plain language. If the question is about a specific document and no document is loaded, explain that the user needs to upload a document first. If you can answer based on the document context provided, do so clearly and specifically.

Response:"""
        
        response = analyzer.model.generate_content(chat_prompt)
        
        if response.text:
            answer = response.text.strip()
        else:
            answer = "I'm sorry, I couldn't generate a response. Please try rephrasing your question."
        
        return {
            "success": True,
            "answer": answer,
            "question": question,
            "context": context,
            "document_id": document_id
        }
        
    except Exception as e:
        logger.error(f"Unexpected error during chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "available_endpoints": ["/", "/health", "/analyze", "/analyze-text"]}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "Please try again later"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)