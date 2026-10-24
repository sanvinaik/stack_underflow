# Fixture data as defined in the SG-ECP MVP Scope Document v2 ENGINEERING_HANDOFF
# Used strictly for mock responses when USE_MOCK_MODELS=true.

FX1_INPUT = "For FY2024, what liquidity coverage ratio does Acme Bank report and what exceptions does it disclose?"

FX1_RESPONSE = {
    "answer": "## Liquidity coverage ratio (R1)\n- Acme Bank reports a liquidity coverage ratio of 128% for FY2024 [acme_fy2024.pdf p.12].\n\n## Disclosed exceptions (R2)\n- The ratio excludes assets held by the overseas subsidiary pending regulatory approval [acme_addendum.pdf p.3].",
    "chunks": [
        {
            "chunk_id": "c_fy2024_p12",
            "content": "The liquidity coverage ratio stood at 128% at year end.",
            "section_name": "Liquidity",
            "metadata": {
                "page_numbers": [12],
                "document": "acme_fy2024.pdf"
            }
        },
        {
            "chunk_id": "c_addendum_p3",
            "content": "The ratio excludes assets held by the overseas subsidiary until regulatory approval is received.",
            "section_name": "Exceptions",
            "metadata": {
                "page_numbers": [3],
                "document": "acme_addendum.pdf"
            }
        }
    ],
    "trace": [
        {
            "step_name": "query_compile",
            "description": "LLM stage 1: scope constraints and evidence requirements",
            "input": FX1_INPUT,
            "output": {
                "scope": {
                    "issuer": {"value": "Acme Bank", "basis": "explicit"},
                    "fiscal_period_end": {"value": 2024, "basis": "explicit"}
                },
                "requirements": [
                    {"id": "R1", "type": "number", "exhaustive": False, "text": "reported liquidity coverage ratio"},
                    {"id": "R2", "type": "list", "exhaustive": False, "text": "disclosed exceptions"}
                ],
                "duration_ms": 0
            }
        },
        {
            "step_name": "scope_gate",
            "description": "Deterministic filter over document cards",
            "input": {"documents": 3},
            "output": {
                "kept": ["acme_fy2024", "acme_addendum"],
                "excluded": [{"doc": "acme_fy2023", "reason": "fiscal_period_end 2023 known, conflicts with explicit 2024"}],
                "unresolved": [],
                "duration_ms": 0
            }
        },
        {
            "step_name": "sibling_check",
            "description": "Sibling sets and discriminating axes over surviving documents",
            "input": {"surviving": ["acme_fy2024", "acme_addendum"]},
            "output": {
                "sets": [
                    {
                        "docs": ["acme_fy2023", "acme_fy2024"],
                        "axes": {"fiscal_period_end": {"acme_fy2023": 2023, "acme_fy2024": 2024}}
                    }
                ],
                "resolution": "pinned_by_explicit_scope",
                "kept": ["acme_fy2024", "acme_addendum"],
                "duration_ms": 0
            }
        },
        {
            "step_name": "candidate_retrieval",
            "description": "Per-requirement hybrid scan of eligible documents with z-normalized cutoffs",
            "input": {"documents": 2, "requirements": 2},
            "output": {
                "candidates": 9,
                "per_requirement": {"R1": 5, "R2": 6},
                "duration_ms": 0
            }
        },
        {
            "step_name": "evidence_adjudication",
            "description": "LLM stage 2: sharded relevance judgment of candidates",
            "input": {"candidates": 9, "shards": 1},
            "output": {
                "c_fy2024_p12": [["R1", "direct"]],
                "c_addendum_p3": [["R2", "qualified"]],
                "adjudication_partial": False,
                "duration_ms": 0
            }
        },
        {
            "step_name": "coverage_selection",
            "description": "Saturating selector with ledger",
            "input": {"budget_tokens": 16000, "max_chunks": 150},
            "output": {
                "selected": ["c_fy2024_p12", "c_addendum_p3"],
                "ledger": {
                    "R1": {"chunks": 1, "stop_reason": "saturated"},
                    "R2": {"chunks": 1, "stop_reason": "saturated"}
                },
                "tokens_used": 310,
                "duration_ms": 0
            }
        },
        {
            "step_name": "answer_generation",
            "description": "LLM stage 3: grounded structured answer",
            "input": {"selected_chunks": 2},
            "output": {
                "points": 2,
                "unsupported_requirements": [],
                "duration_ms": 0
            }
        },
        {
            "step_name": "citation_validation",
            "description": "Verify cited chunk ids and render citations from stored metadata",
            "input": {"points": 2},
            "output": {
                "valid": True,
                "dropped_points": 0,
                "duration_ms": 0
            }
        }
    ]
}

# requirements:
# (none)

# contract:
# input: None
# output: Provides the static FX1 response structure matching app.schemas.QueryResponse to be returned in USE_MOCK_MODELS=true mode.