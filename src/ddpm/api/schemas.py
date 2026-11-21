"""Request and response schemas for the inference API."""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    num_samples: int = Field(default=1, ge=1, le=8)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)


class GeneratedImage(BaseModel):
    format: str = "png"
    data: str = Field(description="Base64-encoded PNG image.")


class GenerateResponse(BaseModel):
    images: list[GeneratedImage]
    num_samples: int
    image_size: int = 28
    seed: int
