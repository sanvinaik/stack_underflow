from typing import Any, Literal
from pydantic import BaseModel, Field

class Field3(BaseModel):
    value: Any
    status: Literal["known", "inferred", "unknown", "conflicting"]
    evidence_page: int | None = None
    evidence_snippet: str | None = None

class DocCard(BaseModel):
    doc_id: str
    filename: str
    title: Field3
    issuer: Field3
    entities: list[str] = Field(default_factory=list)
    jurisdiction: Field3
    sector: Field3
    doc_type: Field3
    publication_date: Field3
    period_year: Field3
    fiscal_period_end: Field3
    effective_date: Field3
    topics: list[str] = Field(default_factory=list)
    summary: str
    extra_attributes: dict[str, Field3] = Field(default_factory=dict)

class SiblingSet(BaseModel):
    doc_ids: list[str]
    axes: dict[str, dict[str, Any]]

class ScopeField(BaseModel):
    value: Any
    basis: Literal["explicit", "inferred", "unspecified"]

class Scope(BaseModel):
    issuer: ScopeField | None = None
    jurisdiction: ScopeField | None = None
    sector: ScopeField | None = None
    fiscal_period_end: ScopeField | None = None
    entities: ScopeField | None = None
    doc_type: ScopeField | None = None

class Requirement(BaseModel):
    id: str
    text: str
    type: Literal["fact", "number", "list", "definition", "comparison", "condition"]
    exhaustive: bool
    keywords: list[str] = Field(default_factory=list)

class EvidenceContract(BaseModel):
    scope: Scope
    requirements: list[Requirement]

class Support(BaseModel):
    req_id: str
    level: Literal["direct", "qualified", "contradicts"]

class Judgement(BaseModel):
    chunk_id: str
    supports: list[Support]
    material_condition: bool

class ShardResult(BaseModel):
    judgements: list[Judgement]

class AnswerPoint(BaseModel):
    text: str
    cites: list[str]
    req_id: str

class AnswerDraft(BaseModel):
    points: list[AnswerPoint]
    unsupported_req_ids: list[str] = Field(default_factory=list)

class ChunkMetadata(BaseModel):
    page_numbers: list[int]
    document: str

class Chunk(BaseModel):
    chunk_id: str
    content: str
    section_name: str
    metadata: ChunkMetadata

class TraceStep(BaseModel):
    step_name: str
    description: str
    input: Any
    output: Any

class IndexResponse(BaseModel):
    status: Literal["success"] = "success"

class QueryRequest(BaseModel):
    collection_name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    query: str = Field(min_length=1, max_length=4000)

class QueryResponse(BaseModel):
    answer: str
    chunks: list[Chunk]
    trace: list[TraceStep]

# requirements:
# pydantic>=2.0.0

# contract:
# input: None (Schema definitions only).
# output: Pydantic v2 models that enforce the API contract, internal types, and pipeline representations defined in Addendum E.