# app/api/routes.py
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List

# Import our model service
from app.services.deepfake_detector import analyze_image_bytes

router = APIRouter()

class AnalyzeImageResponse(BaseModel):
    fake_probability: float
    risk_level: str
    explanation: str
    model_used: str


class ComplaintRequest(BaseModel):
    incident_description: str
    platform: str
    victim_age: Optional[int] = None
    city: Optional[str] = None
    known_offender: Optional[bool] = None
    additional_details: Optional[str] = None


class ComplaintResponse(BaseModel):
    complaint_text: str


class LegalFaqRequest(BaseModel):
    question: str


class LegalFaqResponse(BaseModel):
    answer: str
    sources: List[str]


@router.get("/health")
async def health_check():
    return {"status": "ok", "message": "SurakshaLens backend is running"}


# ---------------------------------------------------------
# UPDATED: Deepfake Detector Integration
# ---------------------------------------------------------
@router.post("/analyze-image", response_model=AnalyzeImageResponse)
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = analyze_image_bytes(image_bytes)

    return AnalyzeImageResponse(
        fake_probability=result["fake_probability"],
        risk_level=result["risk_level"],
        explanation=result["explanation"],
        model_used=result["model_used"],
    )


@router.post("/generate-complaint", response_model=ComplaintResponse)
async def generate_complaint(req: ComplaintRequest):
    template = f"""
To,
The Cyber Crime Cell,

Subject: Complaint regarding misuse of my images on {req.platform}

Respected Sir/Madam,

I want to file a complaint regarding online harassment and misuse of my images.

Incident:
{req.incident_description}

City: {req.city or 'Not specified'}
Victim Age: {req.victim_age or 'Not specified'}
Known Offender: {"Yes" if req.known_offender else "No or Unknown"}

Additional Details:
{req.additional_details or "None"}

Yours faithfully,
[Your Name]
""".strip()

    return ComplaintResponse(complaint_text=template)


@router.post("/legal-faq", response_model=LegalFaqResponse)
async def legal_faq(req: LegalFaqRequest):
    answer = "This is a placeholder legal answer. RAG module will replace this."
    return LegalFaqResponse(answer=answer, sources=["placeholder"])
