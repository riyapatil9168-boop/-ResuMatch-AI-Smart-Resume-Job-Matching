from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from extractor import extract_text
from analyzer import parse_job_description, analyze_resume

app = FastAPI(
    title="ResuMatch AI - REST API",
    description="Enterprise API for Resume Parsing, Job Matching & ATS Fraud Detection (Algothon'26 | PS ID: ALG-AI-01)",
    version="1.0.0"
)

# Enable CORS for external frontend or judging tests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobDescriptionRequest(BaseModel):
    text: str

class MatchRequest(BaseModel):
    job_description: str
    resume_text: str
    filename: Optional[str] = "candidate_resume.txt"

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "ResuMatch AI API",
        "hackathon": "Algothon'26",
        "problem_statement": "ALG-AI-01",
        "docs_url": "/docs"
    }

@app.post("/api/v1/parse-jd")
def parse_jd_endpoint(request: JobDescriptionRequest):
    """Extracts required skills, experience, and education from a job description."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Job description text cannot be empty.")
    data = parse_job_description(request.text)
    # Convert set to list for JSON serialization
    data["required_skills"] = sorted(list(data["required_skills"]))
    return {"status": "success", "data": data}

@app.post("/api/v1/match")
def match_text_endpoint(request: MatchRequest):
    """Matches raw resume text against a job description."""
    if not request.job_description.strip() or not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="Job description and resume text are both required.")
    
    jd_data = parse_job_description(request.job_description)
    result = analyze_resume(request.resume_text, request.filename, jd_data)
    # Convert sets to lists in candidate_info
    result["candidate_info"]["skills"] = sorted(list(result["candidate_info"]["skills"]))
    return {"status": "success", "data": result}

@app.post("/api/v1/upload-and-match")
async def upload_and_match(
    job_description: str = Form(...),
    file: UploadFile = File(...)
):
    """Uploads a PDF, DOCX, or TXT resume and matches it against a job description."""
    try:
        content = await file.read()
        extracted_text = extract_text(content, file.filename)
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the provided file.")

        jd_data = parse_job_description(job_description)
        result = analyze_resume(extracted_text, file.filename, jd_data)
        result["candidate_info"]["skills"] = sorted(list(result["candidate_info"]["skills"]))
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
