"""A deliberately small, explainable ML model trained on synthetic labels only."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = (
    "declared_value_eur", "in_customs_clearance", "rule_count", "high_risk_rule_count",
    "similar_case_count", "average_similar_clearance_days",
)


def features_from_context(context: dict) -> dict[str, float]:
    rules = context["applicable_customs_rules"]
    cases = context["historical_similar_cases"]
    return {
        "declared_value_eur": float(context["shipment"]["declared_value_eur"]),
        "in_customs_clearance": float(context["shipment"]["status"] == "CUSTOMS_CLEARANCE"),
        "rule_count": float(len(rules)),
        "high_risk_rule_count": float(sum(rule.get("risk_level") == "HIGH" for rule in rules)),
        "similar_case_count": float(len(cases)),
        "average_similar_clearance_days": float(sum(case["clearance_days"] for case in cases) / len(cases)) if cases else 0.0,
    }


class SyntheticDelayRiskModel:
    """Logistic regression for demonstration, never a real customs or DHL decision model."""

    def __init__(self):
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(random_state=42, max_iter=500)),
        ])
        self.is_fitted = False

    def fit_from_synthetic_data(self, data_dir: Path | str = ROOT / "data"):
        data_dir = Path(data_dir)
        shipments = pd.read_csv(data_dir / "shipments.csv")
        products = pd.read_csv(data_dir / "products.csv", dtype={"hs_code": str})
        rules = pd.read_csv(data_dir / "customs_rules.csv", dtype={"hs_code": str})
        cases = pd.read_csv(data_dir / "customs_cases.csv")
        product_hs = products.set_index("product_id")["hs_code"].to_dict()
        rows, labels = [], []
        for shipment in shipments.to_dict("records"):
            matching_rules = rules[(rules.country_code == shipment["destination_code"]) & (rules.hs_code == product_hs[shipment["product_id"]])].to_dict("records")
            matching_cases = cases[(cases.product_id == shipment["product_id"]) & (cases.country_code == shipment["destination_code"])].to_dict("records")
            context = {
                "shipment": shipment,
                "applicable_customs_rules": matching_rules,
                "historical_similar_cases": matching_cases,
            }
            feature_row = features_from_context(context)
            rows.append(feature_row)
            # Transparent synthetic label construction. It is not observed operational data.
            synthetic_signal = (
                feature_row["in_customs_clearance"]
                + feature_row["high_risk_rule_count"]
                + float(feature_row["declared_value_eur"] >= 10000)
                + float(feature_row["average_similar_clearance_days"] >= 3)
            )
            labels.append(int(synthetic_signal >= 2))
        self.pipeline.fit(pd.DataFrame(rows, columns=FEATURE_NAMES), labels)
        self.is_fitted = True
        return self

    def predict(self, context: dict) -> dict:
        if not self.is_fitted:
            raise RuntimeError("Fit the synthetic risk model before predicting.")
        feature_map = features_from_context(context)
        frame = pd.DataFrame([feature_map], columns=FEATURE_NAMES)
        probability = float(self.pipeline.predict_proba(frame)[0, 1])
        scaler = self.pipeline.named_steps["scale"]
        coefficients = self.pipeline.named_steps["model"].coef_[0]
        scaled = scaler.transform(frame)[0]
        contributions = sorted(
            ({"feature": name, "value": feature_map[name], "coefficient_contribution": round(float(value * coefficient), 4)}
             for name, value, coefficient in zip(FEATURE_NAMES, scaled, coefficients)),
            key=lambda item: abs(item["coefficient_contribution"]), reverse=True,
        )
        return {
            "synthetic_delay_risk_probability": round(probability, 4),
            "risk_band": "HIGH" if probability >= 0.67 else "MEDIUM" if probability >= 0.34 else "LOW",
            "features": feature_map,
            "feature_contributions": contributions,
            "disclaimer": "Synthetic demonstration score trained on synthetic labels; not a DHL, customs, or production decision.",
        }
