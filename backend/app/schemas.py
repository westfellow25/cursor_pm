from pydantic import BaseModel, Field


class OpportunitySummary(BaseModel):
    cluster_id: int
    size: int
    theme: str
    representative_feedback: str


class DiscoveryResponse(BaseModel):
    total_feedback_items: int = Field(..., ge=0)
    total_clusters: int = Field(..., ge=0)
    opportunities: list[OpportunitySummary]
