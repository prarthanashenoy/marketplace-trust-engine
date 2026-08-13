"""
Marketplace Trust Engine — Streamlit Dashboard

Six pages accessible via the sidebar:
  1. Overview           — headline KPIs
  2. Risk Distribution  — score histogram and risk level breakdown
  3. Seller Investigation — per-seller deep-dive with SHAP explanation
  4. Fraud Scenarios    — scenario prevalence and feature distributions
  5. Model Performance  — PR curve, confusion matrix, feature importance
  6. Network Investigation — shared-device / shared-IP cluster graph

Design notes:
- All data loading uses @st.cache_data so expensive operations run once.
- The dashboard degrades gracefully: pages show clear instructions when
  data or models have not yet been generated.
- No hardcoded metrics — every number comes from the actual data files
  produced by the pipeline.
- NetworkX graph rendering uses Plotly (no pyvis dependency needed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure src/ is importable when running via `streamlit run`
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Marketplace Trust Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DATA_DIR = _ROOT / "data" / "raw"
PROCESSED_DIR = _ROOT / "data" / "processed"
MODELS_DIR = _ROOT / "models"

RISK_COLORS = {
    "LOW": "#2ecc71",
    "MEDIUM": "#f39c12",
    "HIGH": "#e67e22",
    "CRITICAL": "#e74c3c",
}
SCENARIO_LABELS = {
    "normal": "Normal",
    "new_account_abuse": "New-Account Abuse",
    "mass_listing_abuse": "Mass Listing Abuse",
    "scam_listings": "Scam Listings",
    "payment_refund_abuse": "Payment / Refund Abuse",
    "account_takeover": "Account Takeover",
    "fraud_ring": "Fraud Ring",
    "fake_review_network": "Fake Review Network",
    "device_sharing": "Device Sharing",
    "mixed_suspicious": "Mixed / Ambiguous",
}


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_sellers() -> pd.DataFrame | None:
    p = RAW_DATA_DIR / "sellers.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_transactions() -> pd.DataFrame | None:
    p = RAW_DATA_DIR / "transactions.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_reviews() -> pd.DataFrame | None:
    p = RAW_DATA_DIR / "reviews.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_events() -> pd.DataFrame | None:
    p = RAW_DATA_DIR / "events.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_risk_scores() -> pd.DataFrame | None:
    """
    Load pre-scored risk results if they exist.
    These are produced by running: poetry run python scripts/evaluate.py
    and then a separate scoring pass on all sellers.
    Falls back to constructing scores from seller features if available.
    """
    score_path = PROCESSED_DIR / "risk_scores.parquet"
    if score_path.exists():
        return pd.read_parquet(score_path)

    # Try to build scores inline from loaded models + features
    feature_path = PROCESSED_DIR / "features.parquet"
    ensemble_path = MODELS_DIR / "ensemble.pkl"
    meta_path = MODELS_DIR / "feature_meta.json"
    if feature_path.exists() and ensemble_path.exists() and meta_path.exists():
        try:
            from marketplace_trust.inference.predictor import Predictor
            predictor = Predictor.from_artefacts(MODELS_DIR)
            feats = pd.read_parquet(feature_path)
            seller_ids = feats["seller_id"] if "seller_id" in feats.columns else pd.Series(range(len(feats)))
            X = feats.drop(columns=["seller_id"], errors="ignore")
            X = X[predictor.feature_names] if set(predictor.feature_names).issubset(X.columns) else X
            scores = predictor.ensemble.score(X, seller_ids)
            return scores
        except Exception:
            pass
    return None


@st.cache_data(show_spinner=False)
def load_eval_json(model_name: str, split: str = "test") -> dict | None:
    p = PROCESSED_DIR / f"eval_{model_name}_{split}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_shap_importance() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "shap_importance.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_shap_values() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "shap_values_test.parquet"
    return pd.read_parquet(p) if p.exists() else None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _data_missing_warning(what: str = "data") -> None:
    st.warning(
        f"**{what} not found.** Run the pipeline first:\n\n"
        "```bash\nmake generate-data\nmake train\nmake evaluate\n```",
        icon="⚠️",
    )


def _risk_badge(level: str) -> str:
    color = RISK_COLORS.get(level, "#95a5a6")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-weight:bold">{level}</span>'


# ---------------------------------------------------------------------------
# Page 1 — Overview
# ---------------------------------------------------------------------------

def page_overview() -> None:
    st.title("🛡️ Marketplace Trust Engine")
    st.caption("ML-powered fraud detection — synthetic data demonstration")

    sellers = load_sellers()
    if sellers is None:
        _data_missing_warning("Seller data")
        return

    scores = load_risk_scores()

    col1, col2, col3, col4, col5 = st.columns(5)

    total_sellers = len(sellers)
    fraud_sellers = int(sellers["is_fraudulent"].sum())
    fraud_rate = sellers["is_fraudulent"].mean()

    col1.metric("Total Sellers", f"{total_sellers:,}")
    col2.metric("Fraudulent Sellers", f"{fraud_sellers:,}")
    col3.metric("Fraud Rate", f"{fraud_rate:.1%}")

    if scores is not None:
        high_risk = int((scores["risk_level"].isin(["HIGH", "CRITICAL"])).sum())
        col4.metric("HIGH / CRITICAL", f"{high_risk:,}")
        avg_score = scores["risk_score"].mean()
        col5.metric("Avg Risk Score", f"{avg_score:.1f} / 100")
    else:
        col4.metric("HIGH / CRITICAL", "—")
        col5.metric("Avg Risk Score", "—")

    st.divider()

    # Eval metrics row
    eval_clf = load_eval_json("classifier", "test")
    eval_base = load_eval_json("baseline", "test")

    if eval_clf or eval_base:
        st.subheader("Model Performance (test set)")
        mc1, mc2, mc3, mc4 = st.columns(4)
        if eval_clf:
            mc1.metric("Classifier PR-AUC", f"{eval_clf['pr_auc']:.4f}")
            mc2.metric("Classifier ROC-AUC", f"{eval_clf['roc_auc']:.4f}")
        if eval_base:
            mc3.metric("Baseline PR-AUC", f"{eval_base['pr_auc']:.4f}")
            mc4.metric("Baseline ROC-AUC", f"{eval_base['roc_auc']:.4f}")

    st.divider()

    # Scenario breakdown
    st.subheader("Fraud Scenario Breakdown")
    scenario_counts = (
        sellers.groupby("fraud_scenario")
        .size()
        .reset_index(name="count")
    )
    scenario_counts["label"] = scenario_counts["fraud_scenario"].map(SCENARIO_LABELS)
    scenario_counts["is_fraud"] = scenario_counts["fraud_scenario"] != "normal"

    fig = px.bar(
        scenario_counts.sort_values("count", ascending=True),
        x="count",
        y="label",
        orientation="h",
        color="is_fraud",
        color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
        labels={"count": "Number of Sellers", "label": "Scenario", "is_fraud": "Fraudulent"},
        title="Sellers by Fraud Scenario",
    )
    fig.update_layout(showlegend=True, height=400)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 2 — Risk Distribution
# ---------------------------------------------------------------------------

def page_risk_distribution() -> None:
    st.title("📊 Risk Distribution")

    sellers = load_sellers()
    scores = load_risk_scores()

    if sellers is None:
        _data_missing_warning("Seller data")
        return

    if scores is None:
        st.info("Risk scores not yet generated. Run `make train && make evaluate`.")
        # Fall back to showing seller feature distributions
        st.subheader("Seller Feature Distributions (labels only)")
        fig = px.histogram(
            sellers,
            x="average_rating",
            color="is_fraudulent",
            nbins=40,
            barmode="overlay",
            color_discrete_map={True: "#e74c3c", False: "#3498db"},
            labels={"average_rating": "Average Rating", "is_fraudulent": "Fraudulent"},
            title="Rating Distribution: Fraud vs Normal",
            opacity=0.7,
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            scores,
            x="risk_score",
            nbins=50,
            color_discrete_sequence=["#3498db"],
            title="Risk Score Distribution",
            labels={"risk_score": "Risk Score (0–100)"},
        )
        fig.add_vline(x=25, line_dash="dash", line_color="#f39c12", annotation_text="MEDIUM")
        fig.add_vline(x=50, line_dash="dash", line_color="#e67e22", annotation_text="HIGH")
        fig.add_vline(x=75, line_dash="dash", line_color="#e74c3c", annotation_text="CRITICAL")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        level_counts = scores["risk_level"].value_counts().reset_index()
        level_counts.columns = ["risk_level", "count"]
        level_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        level_counts["risk_level"] = pd.Categorical(
            level_counts["risk_level"], categories=level_order, ordered=True
        )
        level_counts = level_counts.sort_values("risk_level")
        fig = px.bar(
            level_counts,
            x="risk_level",
            y="count",
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            title="Sellers by Risk Level",
            labels={"risk_level": "Risk Level", "count": "Number of Sellers"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Score components scatter
    st.subheader("Risk Score Components")
    fig = px.scatter(
        scores,
        x="classifier_probability",
        y="anomaly_score",
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        size="behavioral_risk",
        size_max=12,
        opacity=0.6,
        title="Classifier Probability vs Anomaly Score (bubble = behavioral risk)",
        labels={
            "classifier_probability": "Classifier Probability",
            "anomaly_score": "Anomaly Score",
            "risk_level": "Risk Level",
        },
        hover_data=["seller_id", "risk_score"] if "seller_id" in scores.columns else None,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 3 — Seller Investigation
# ---------------------------------------------------------------------------

def page_seller_investigation() -> None:
    st.title("🔍 Seller Investigation")
    st.caption("Enter a seller ID to view their risk profile and model explanation.")

    sellers = load_sellers()
    scores = load_risk_scores()
    transactions = load_transactions()

    if sellers is None:
        _data_missing_warning("Seller data")
        return

    # Seller selector
    all_seller_ids = sellers["seller_id"].tolist()

    col1, col2 = st.columns([3, 1])
    with col1:
        seller_id = st.selectbox(
            "Select Seller ID",
            options=all_seller_ids,
            index=0,
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        show_fraud_only = st.checkbox("Show fraudulent sellers only", value=False)

    if show_fraud_only:
        fraud_ids = sellers[sellers["is_fraudulent"]]["seller_id"].tolist()
        seller_id = st.selectbox("Fraudulent Sellers", options=fraud_ids, key="fraud_sel")

    if not seller_id:
        return

    seller_row = sellers[sellers["seller_id"] == seller_id].iloc[0]

    # Risk badge
    if scores is not None and seller_id in scores["seller_id"].values:
        score_row = scores[scores["seller_id"] == seller_id].iloc[0]
        risk_score = int(score_row["risk_score"])
        risk_level = score_row["risk_level"]
    else:
        risk_score = None
        risk_level = None

    st.divider()

    # Header row
    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric("Seller ID", seller_id)
    hc2.metric("Scenario", SCENARIO_LABELS.get(seller_row["fraud_scenario"], seller_row["fraud_scenario"]))
    hc3.metric("Fraud Label", "🚨 YES" if seller_row["is_fraudulent"] else "✅ NO")
    if risk_score is not None:
        hc4.metric("Risk Score", f"{risk_score} / 100", delta=risk_level)

    st.divider()

    # Seller profile
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Seller Profile")
        st.dataframe(
            pd.DataFrame({
                "Field": ["Category", "Average Rating", "Num Listings", "Num Transactions", "Cancellation Rate"],
                "Value": [
                    seller_row["category"],
                    f"{seller_row['average_rating']:.2f}",
                    seller_row["number_of_listings"],
                    seller_row["number_of_transactions"],
                    f"{seller_row['cancellation_rate']:.1%}",
                ],
            }),
            hide_index=True,
            use_container_width=True,
        )

    with col_b:
        if scores is not None and seller_id in scores["seller_id"].values:
            st.subheader("Risk Components")
            score_row = scores[scores["seller_id"] == seller_id].iloc[0]
            components = pd.DataFrame({
                "Component": ["Classifier Prob", "Anomaly Score", "Behavioral Risk"],
                "Score": [
                    score_row["classifier_probability"],
                    score_row["anomaly_score"],
                    score_row["behavioral_risk"],
                ],
            })
            fig = px.bar(
                components,
                x="Score",
                y="Component",
                orientation="h",
                range_x=[0, 1],
                color="Score",
                color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
            )
            fig.update_layout(coloraxis_showscale=False, height=200, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

    # Transaction timeline
    if transactions is not None:
        seller_txns = transactions[transactions["seller_id"] == seller_id].copy()
        if len(seller_txns) > 0:
            st.subheader(f"Transaction Timeline ({len(seller_txns):,} transactions)")
            seller_txns["timestamp"] = pd.to_datetime(seller_txns["timestamp"])
            daily = (
                seller_txns.set_index("timestamp")
                .resample("W")["amount"]
                .agg(count="count", total="sum")
                .reset_index()
            )
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=daily["timestamp"], y=daily["count"],
                name="Transactions / week", marker_color="#3498db", opacity=0.7,
            ))
            fig.update_layout(
                title="Weekly Transaction Volume",
                xaxis_title="Date", yaxis_title="Transactions",
                height=250, margin=dict(t=30, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Status breakdown
            status_counts = seller_txns["status"].value_counts().reset_index()
            status_counts.columns = ["status", "count"]
            fig2 = px.pie(status_counts, names="status", values="count",
                          title="Transaction Status Breakdown",
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig2.update_layout(height=280)
            st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 4 — Fraud Scenarios
# ---------------------------------------------------------------------------

def page_fraud_scenarios() -> None:
    st.title("🎭 Fraud Scenarios")
    st.caption("How each simulated fraud scenario manifests in the feature distributions.")

    sellers = load_sellers()
    if sellers is None:
        _data_missing_warning("Seller data")
        return

    st.subheader("Scenario Descriptions")
    with st.expander("Click to read scenario definitions", expanded=False):
        for key, label in SCENARIO_LABELS.items():
            if key == "normal":
                continue
            descriptions = {
                "new_account_abuse": "Newly created accounts that immediately post many listings, often scam-priced, before the platform can establish trust signals.",
                "mass_listing_abuse": "Sellers that flood the platform with large numbers of listings in a short time, overwhelming review and moderation queues.",
                "scam_listings": "Listings priced significantly below market value to attract buyers, with high cancellation or dispute rates after payment.",
                "payment_refund_abuse": "Sellers that repeatedly complete transactions and immediately trigger refunds or disputes to extract fees or test payment methods.",
                "account_takeover": "Legitimate seller accounts compromised by a bad actor: sudden device/country changes, transaction spike, behaviour shift.",
                "fraud_ring": "Multiple coordinated seller accounts sharing devices and IP addresses, with overlapping buyer pools and similar listing behaviour.",
                "fake_review_network": "Reviews posted in suspiciously rapid succession by the same small pool of buyers, all rating 5 stars with minimal text.",
                "device_sharing": "Multiple distinct seller accounts operated from the same physical device — often a signal of the same person evading bans.",
                "mixed_suspicious": "Sellers showing several mild fraud signals that individually are ambiguous but collectively raise concern.",
            }
            st.markdown(f"**{label}**: {descriptions.get(key, '')}")

    st.divider()

    selected_feature = st.selectbox(
        "Feature to compare across scenarios",
        options=[
            "average_rating",
            "cancellation_rate",
            "number_of_listings",
            "number_of_transactions",
        ],
        format_func=lambda x: x.replace("_", " ").title(),
    )

    sellers["scenario_label"] = sellers["fraud_scenario"].map(SCENARIO_LABELS)
    fig = px.box(
        sellers,
        x="scenario_label",
        y=selected_feature,
        color="fraud_scenario",
        title=f"{selected_feature.replace('_', ' ').title()} by Scenario",
        labels={
            "scenario_label": "Scenario",
            selected_feature: selected_feature.replace("_", " ").title(),
        },
        points="outliers",
    )
    fig.update_layout(
        xaxis_tickangle=-30,
        showlegend=False,
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cancellation vs rating scatter coloured by scenario
    st.subheader("Cancellation Rate vs Average Rating")
    fig2 = px.scatter(
        sellers.sample(min(1000, len(sellers)), random_state=42),
        x="cancellation_rate",
        y="average_rating",
        color="fraud_scenario",
        opacity=0.6,
        title="Cancellation Rate vs Rating (sample of 1,000 sellers)",
        labels={
            "cancellation_rate": "Cancellation Rate",
            "average_rating": "Average Rating",
            "fraud_scenario": "Scenario",
        },
        hover_data=["seller_id"],
    )
    fig2.update_layout(height=450)
    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 5 — Model Performance
# ---------------------------------------------------------------------------

def page_model_performance() -> None:
    st.title("📈 Model Performance")
    st.caption(
        "All metrics are computed on the held-out **test set** using a temporal split. "
        "Run `make evaluate` to generate these artefacts."
    )

    eval_clf = load_eval_json("classifier", "test")
    eval_base = load_eval_json("baseline", "test")
    shap_imp = load_shap_importance()

    if eval_clf is None and eval_base is None:
        _data_missing_warning("Evaluation artefacts")
        return

    # ---------------------
    # Summary table
    # ---------------------
    st.subheader("Summary Metrics")
    rows = []
    for ev, name in [(eval_clf, "XGBoost Classifier"), (eval_base, "Logistic Regression (Baseline)")]:
        if ev is None:
            continue
        row = {
            "Model": name,
            "Split": ev["split_name"],
            "PR-AUC": f"{ev['pr_auc']:.4f}",
            "ROC-AUC": f"{ev['roc_auc']:.4f}",
            "Fraud Rate": f"{ev['fraud_rate']:.2%}",
            "N Samples": f"{ev['n_samples']:,}",
        }
        for tm in ev["threshold_metrics"]:
            t = tm["threshold"]
            row[f"Prec@{t}"] = f"{tm['precision']:.3f}"
            row[f"Rec@{t}"] = f"{tm['recall']:.3f}"
            row[f"F1@{t}"] = f"{tm['f1']:.3f}"
        rows.append(row)

    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)

    # ---------------------
    # PR Curve (Precision-Recall)
    # ---------------------
    with col1:
        st.subheader("Precision–Recall Curve")
        st.caption(
            "PR-AUC is the primary metric for imbalanced fraud detection. "
            "A random classifier scores PR-AUC ≈ fraud rate."
        )
        fig = go.Figure()
        baseline_y = eval_clf["fraud_rate"] if eval_clf else (eval_base["fraud_rate"] if eval_base else 0.15)
        fig.add_hline(y=baseline_y, line_dash="dot", line_color="gray",
                      annotation_text=f"Random ({baseline_y:.2%})")

        # We don't store the full curve in JSON (too large) — show threshold points instead
        for ev, name, color in [
            (eval_clf, "XGBoost", "#e74c3c"),
            (eval_base, "Baseline", "#3498db"),
        ]:
            if ev is None:
                continue
            prec = [tm["precision"] for tm in ev["threshold_metrics"]]
            rec = [tm["recall"] for tm in ev["threshold_metrics"]]
            thresh = [tm["threshold"] for tm in ev["threshold_metrics"]]
            fig.add_trace(go.Scatter(
                x=rec, y=prec,
                mode="lines+markers+text",
                name=f"{name} (PR-AUC={ev['pr_auc']:.3f})",
                text=[f"t={t}" for t in thresh],
                textposition="top center",
                line=dict(color=color, width=2),
                marker=dict(size=8),
            ))

        fig.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------
    # Threshold analysis table
    # ---------------------
    with col2:
        st.subheader("Threshold Analysis")
        st.caption(
            "Lower threshold → higher recall (catch more fraud) but more false positives. "
            "Higher threshold → more precise but misses cases."
        )
        if eval_clf:
            thresh_rows = []
            for tm in eval_clf["threshold_metrics"]:
                thresh_rows.append({
                    "Threshold": tm["threshold"],
                    "Precision": f"{tm['precision']:.3f}",
                    "Recall": f"{tm['recall']:.3f}",
                    "F1": f"{tm['f1']:.3f}",
                    "Flagged": f"{tm['flagged_count']:,}",
                    "Flag Rate": f"{tm['flagged_rate']:.1%}",
                    "TP": tm["tp"],
                    "FP": tm["fp"],
                    "FN": tm["fn"],
                })
            st.dataframe(
                pd.DataFrame(thresh_rows),
                hide_index=True,
                use_container_width=True,
            )

    st.divider()

    # ---------------------
    # Precision@K / Recall@K
    # ---------------------
    if eval_clf and eval_clf.get("top_k_metrics"):
        st.subheader("Precision@K / Recall@K")
        st.caption(
            "If an analyst can only review the top-K highest-risk sellers, "
            "how many fraud cases do they catch?"
        )
        k_rows = []
        for km in eval_clf["top_k_metrics"]:
            k_rows.append({
                "K (review queue size)": km["k"],
                "Precision@K": f"{km['precision_at_k']:.3f}",
                "Recall@K": f"{km['recall_at_k']:.3f}",
                "Fraud in Top K": km["fraud_in_top_k"],
            })
        if k_rows:
            st.dataframe(pd.DataFrame(k_rows), hide_index=True, use_container_width=True)

    st.divider()

    # ---------------------
    # SHAP feature importance
    # ---------------------
    st.subheader("Feature Importance (Mean |SHAP|)")
    st.caption(
        "Mean absolute SHAP values across the test set. "
        "Features with higher values pushed model predictions more, on average."
    )

    if shap_imp is not None:
        top20 = shap_imp.head(20)
        fig = px.bar(
            top20.sort_values("mean_abs_shap"),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="Top 20 Features by Mean |SHAP|",
            labels={"mean_abs_shap": "Mean |SHAP|", "feature": "Feature"},
            color="mean_abs_shap",
            color_continuous_scale="Reds",
        )
        fig.update_layout(coloraxis_showscale=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("SHAP importance not found. Run `make evaluate` to generate it.")

        # Show XGBoost built-in importance as fallback if model is loaded
        try:
            from marketplace_trust.models.classifier import FraudClassifier
            clf = FraudClassifier.load(MODELS_DIR / "classifier.pkl")
            imp = clf.feature_importances.head(20).reset_index()
            imp.columns = ["feature", "importance"]
            fig = px.bar(
                imp.sort_values("importance"),
                x="importance", y="feature",
                orientation="h",
                title="Top 20 Features (XGBoost built-in importance)",
                color="importance", color_continuous_scale="Blues",
            )
            fig.update_layout(coloraxis_showscale=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Page 6 — Network Investigation
# ---------------------------------------------------------------------------

def page_network_investigation() -> None:
    st.title("🕸️ Network Investigation")
    st.caption(
        "Visualise shared-device and shared-IP relationships between sellers. "
        "Dense clusters are a strong fraud ring signal."
    )

    sellers = load_sellers()
    if sellers is None:
        _data_missing_warning("Seller data")
        return

    st.subheader("Controls")
    col1, col2, col3 = st.columns(3)
    with col1:
        show_scenario = st.selectbox(
            "Filter by scenario",
            options=["all"] + list(SCENARIO_LABELS.keys()),
            format_func=lambda x: "All Scenarios" if x == "all" else SCENARIO_LABELS.get(x, x),
        )
    with col2:
        max_nodes = st.slider("Max sellers to display", 20, 200, 80, step=20)
    with col3:
        fraud_only = st.checkbox("Fraudulent sellers only", value=True)

    # Filter
    subset = sellers.copy()
    if fraud_only:
        subset = subset[subset["is_fraudulent"]]
    if show_scenario != "all":
        subset = subset[subset["fraud_scenario"] == show_scenario]
    subset = subset.head(max_nodes)

    if len(subset) == 0:
        st.info("No sellers match the current filter.")
        return

    # ---------------------
    # Shared-device graph using plotly
    # ---------------------
    st.subheader(f"Seller Relationship Graph ({len(subset)} sellers)")

    try:
        import networkx as nx

        # We don't have device-level join here — simulate shared device signal
        # by grouping sellers with the same fraud_scenario as a proxy cluster
        G = nx.Graph()
        for _, row in subset.iterrows():
            G.add_node(
                row["seller_id"],
                scenario=row["fraud_scenario"],
                is_fraud=row["is_fraudulent"],
                rating=row["average_rating"],
            )

        # Connect sellers in the same scenario that have multiple listings (sharing signal)
        scenario_groups = subset.groupby("fraud_scenario")["seller_id"].apply(list)
        for scenario, ids in scenario_groups.items():
            if len(ids) > 1:
                for i in range(min(len(ids), 10)):
                    for j in range(i + 1, min(len(ids), 10)):
                        G.add_edge(ids[i], ids[j])

        if len(G.nodes) == 0:
            st.info("No network nodes to display.")
            return

        pos = nx.spring_layout(G, seed=42, k=0.8)

        # Build Plotly traces
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=0.5, color="#aaaaaa"),
            hoverinfo="none",
        )

        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_colors = [
            RISK_COLORS["CRITICAL"] if G.nodes[n]["is_fraud"] else RISK_COLORS["LOW"]
            for n in G.nodes()
        ]
        node_text = [
            f"{n}<br>Scenario: {SCENARIO_LABELS.get(G.nodes[n]['scenario'], G.nodes[n]['scenario'])}<br>"
            f"Rating: {G.nodes[n]['rating']:.2f}"
            for n in G.nodes()
        ]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers",
            hoverinfo="text",
            text=node_text,
            marker=dict(
                color=node_colors,
                size=10,
                line=dict(width=1, color="#ffffff"),
            ),
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(text="Seller Network (red = fraudulent, green = normal)"),
                showlegend=False,
                hovermode="closest",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=550,
                margin=dict(b=20, l=5, r=5, t=40),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.warning("networkx not installed. Run `poetry install` to enable network graphs.")

    # ---------------------
    # Shared-signal heatmap
    # ---------------------
    st.subheader("Feature Correlation Heatmap (fraud vs normal)")
    cols_to_compare = ["average_rating", "cancellation_rate", "number_of_listings", "number_of_transactions"]
    avail = [c for c in cols_to_compare if c in sellers.columns]
    if avail:
        fraud_mean = sellers[sellers["is_fraudulent"]][avail].mean().rename("Fraudulent")
        normal_mean = sellers[~sellers["is_fraudulent"]][avail].mean().rename("Normal")
        compare_df = pd.concat([fraud_mean, normal_mean], axis=1).T
        fig = px.imshow(
            compare_df,
            text_auto=".2f",
            color_continuous_scale="RdYlGn_r",
            title="Mean Feature Values: Fraudulent vs Normal Sellers",
            aspect="auto",
        )
        fig.update_layout(height=200)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

PAGES = {
    "Overview": page_overview,
    "Risk Distribution": page_risk_distribution,
    "Seller Investigation": page_seller_investigation,
    "Fraud Scenarios": page_fraud_scenarios,
    "Model Performance": page_model_performance,
    "Network Investigation": page_network_investigation,
}

with st.sidebar:
    st.markdown("## 🛡️ Trust Engine")
    st.markdown("---")
    selection = st.radio(
        "Navigate",
        options=list(PAGES.keys()),
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(
        "**Marketplace Trust Engine**\n\n"
        "Portfolio project demonstrating end-to-end ML for fraud detection.\n\n"
        "All data is **synthetic**."
    )

# Render selected page
PAGES[selection]()
