"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from .schemas import GenerateRequest, GenerateResponse, GeneratedImage
from .service import InferenceService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference = InferenceService()
    yield
    app.state.inference = None


app = FastAPI(
    title="Fashion-MNIST DDPM API",
    description="Generate 28×28 Fashion-MNIST-like images from a trained DDPM checkpoint.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    service = getattr(request.app.state, "inference", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Inference model is not loaded.")
    return {"status": "ok", "device": str(service.device)}


@app.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest, request: Request):
    service = getattr(request.app.state, "inference", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Inference model is not loaded.")
    try:
        images, seed = service.generate(payload.num_samples, payload.seed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {exc}") from exc
    return GenerateResponse(
        images=[GeneratedImage(data=image) for image in images],
        num_samples=len(images),
        seed=seed,
    )
