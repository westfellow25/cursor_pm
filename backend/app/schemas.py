from pydantic import BaseModel


class OpportunitySummary(BaseModel):
    cluster_id: int
    size: int
    theme: str
    representative_feedback: str


class AnalyzeResponse(BaseModel):
    run_id: str
    clusters_summary: list[OpportunitySummary]
    top_opportunities: list[dict[str, object]]
    recommended_action: str
    evidence: list[str]
    prd_text: str
    jira_tickets_text: str
