"""Deterministic and optional LLM explanations for the synthetic dashboard."""
from __future__ import annotations

import json
import os


def deterministic_explanation(analysis: dict) -> dict:
    """Explain only values present in the analysis; safe default for an interview demo."""
    graph = analysis["graph_context"]
    risk = analysis["synthetic_risk_assessment"]
    shipment, product, destination = graph["shipment"], graph["product"], graph["destination"]
    top_features = [item for item in risk["feature_contributions"] if item["coefficient_contribution"] > 0][:3]
    factors = [
        f"{item['feature'].replace('_', ' ')} = {item['value']:g}"
        for item in top_features
    ]
    rule_documents = [rule.get("required_document") for rule in graph["applicable_customs_rules"] if rule.get("required_document")]
    case_issues = [case["issue"] for case in graph["historical_similar_cases"]]
    vector_titles = [doc["metadata"]["title"] for doc in analysis["retrieved_synthetic_documents"]]
    return {
        "headline": f"{risk['risk_band'].lower()}-risk triage for {shipment['shipment_id']}",
        "summary": (
            f"{product['name']} (HS {product['hs_code']}) is travelling to {destination['name']} with status "
            f"{shipment['status']}. The model returned {risk['synthetic_delay_risk_probability']:.1%}. delay risk probability"
        ),
        "risk_drivers": factors or ["No positive synthetic model contributions were identified."],
        "graph_evidence": {
            "applicable_requirements": rule_documents,
            "similar_case_issues": case_issues,
        },
        "vector_evidence": vector_titles,
        "synthetic_next_steps": (
            [f"Review availability of: {', '.join(rule_documents)}."] if rule_documents else []
        ) + ([f"Compare supporting evidence against prior issue: {case_issues[0]}. "] if case_issues else []) + [
            "Use the retrieved synthetic documents as a triage checklist; do not treat them as customs or DHL policy."
        ],
        "disclaimer": "Deterministic explanation derived only from synthetic graph, vector, and model evidence.",
    }


def optional_openai_explanation(analysis: dict, model: str | None = None, api_key: str | None = None) -> str:
    """Optional summary constrained to synthetic evidence; accepts a session-only key."""
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise RuntimeError("Enter an OpenAI API key or set OPENAI_API_KEY to enable the optional LLM explanation.")
    from openai import OpenAI

    payload = json.dumps(analysis, default=str)
    response = OpenAI(api_key=resolved_key).responses.create(
        model=model or os.getenv("OPENAI_MODEL", "gpt-5"),
        instructions=(
            "Summarize only the supplied synthetic DHL customs-delay analysis. Do not claim any real DHL "
            "policy, customs rule, factual compliance requirement, or prediction. Clearly label all findings synthetic. "
            "Cite document IDs, graph rule IDs, and case IDs when used."
        ),
        input=payload,
        store=False,
    )
    return response.output_text
