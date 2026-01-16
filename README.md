# Healthcare Financial Volatility Index (FVI)

## Overview

This project implements an end-to-end **Financial Volatility Index (FVI)** designed to identify health plan members likely to become **financially unpredictable** in the near term.

Unlike traditional cost prediction models that estimate expected spend, the FVI focuses on **variance and instability**, enabling proactive risk management, budgeting, and care intervention strategies.

---

## Problem Statement

Health insurance organizations face a critical challenge that extends beyond identifying high-cost members.

The core issue is **financial unpredictability** — members who suddenly deviate from their expected cost trajectories due to changes in health status, utilization patterns, or care settings.

These deviations introduce volatility into:
- reserve planning
- actuarial forecasting
- operational decision-making

Standard regression-based cost models are not designed to capture this behavior.

---

## Solution Summary

The Financial Volatility Index quantifies near-term cost instability by separating **risk likelihood** from **risk impact**.

Two complementary models are combined:

- A **classification model** estimating the probability of a sudden cost spike
- A **regression model** estimating the expected magnitude of cost volatility

The final index is defined as:

**FVI = Probability of cost spike × Expected volatility magnitude**

This formulation prioritizes members by **risk-weighted financial instability**, not raw cost alone.

---

## System Design

The system is implemented as a modular, batch-oriented analytics pipeline aligned with real-world healthcare data workflows.

Key components include:

- **Feature Engineering**
  - Rolling cost statistics
  - Utilization and event-based indicators
  - Risk and demographic signals

- **Spike Probability Modeling**
  - Binary classification of sudden cost escalation events
  - Temporal (out-of-time) validation to prevent leakage

- **Volatility Magnitude Modeling**
  - Regression-based estimation of cost dispersion
  - Explicit handling of missing and delayed feature availability

- **Index Construction**
  - Combination of likelihood and magnitude into a single FVI score
  - Risk tiering based on population percentiles

- **Decision Support Interface**
  - Streamlit-based dashboard for filtering, ranking, and exporting results
  - Member-level rollups for operational consumption

This separation between scoring logic and presentation mirrors production healthcare analytics systems.

---

## Validation Approach

Model performance was evaluated using deployment-realistic practices:

- Temporal splits were used instead of random sampling
- A data leakage audit was conducted after detecting unrealistic error metrics
- Label-derived features were identified and removed
- Final metrics reflect performance on unseen future periods only

This process ensures that reported results are representative of real-world usage.

---

## Results

**Spike Probability Model**
- ROC-AUC: 0.792
- Base spike rate: approximately 8%
- Spike rate within the highest FVI tier: approximately 51%
- Approximately six-fold lift over baseline risk

**Volatility Magnitude Model**
- RMSE: approximately 4,300
- Designed to capture dispersion rather than mean cost

The combined index successfully concentrates financial shocks into a small, high-risk subset suitable for proactive action.

---

## Practical Applications

- **Finance and Actuarial Teams**
  - Improved reserve planning through volatility-aware risk estimates
- **Care Management**
  - Targeted outreach to members at risk of destabilizing utilization
- **Population Health Analytics**
  - Transition from reactive cost control to proactive risk stabilization

---

## Interactive Dashboard

An analyst-facing Streamlit application enables:

- Uploading batch-scored FVI outputs
- Filtering by risk tier and spike probability
- Generating downloadable watchlists
- Reviewing member-level risk summaries

The interface is designed to reflect how analytics outputs are consumed in healthcare organizations.

---

## Running the Project Locally

The Financial Volatility Index is generated using a batch scoring pipeline and consumed through an analyst-facing dashboard. To run the project locally, first execute the batch scoring module, which produces a scored output file containing member-level Financial Volatility Index values. Once scoring is complete, launch the Streamlit application and upload the generated CSV file when prompted to explore risk tiers, filter high-volatility members, and export operational watchlists.

```bash
python -m src.modeling.build_fvi
streamlit run app/app.py
```
---

## Technology Stack

The project is implemented using a lightweight but production-representative analytics stack. Core data processing and modeling are performed in Python using Pandas and NumPy, with Scikit-learn for classification and regression workflows. An interactive Streamlit application is used to expose batch-scored outputs through an analyst-facing decision support interface. The overall structure emphasizes modularity, reproducibility, and clear separation of concerns between data preparation, modeling, and presentation.

---

## Design Considerations

The system was designed to reflect real-world healthcare analytics constraints and usage patterns. A batch-first architecture was selected to align with delayed data availability and periodic scoring cycles common in insurance environments. Modeling logic is intentionally decoupled from the presentation layer to ensure that training and scoring pipelines remain reproducible and auditable. Synthetic data is used throughout to mirror compliance requirements while preserving realistic statistical behavior. Emphasis was placed on interpretability, temporal validation, and operational usability rather than purely optimizing predictive metrics.

---

## Data Disclaimer

All data used in this project is fully synthetic and generated exclusively for demonstration purposes. The data does not represent real individuals, claims, or clinical events, and no proprietary or patient-level information is included. This approach reflects common compliance and privacy constraints encountered in healthcare analytics environments.

---

## Future Enhancements

Potential extensions of this work include exposing the scoring logic through an API-based inference service, implementing monitoring to detect index drift over time, and integrating the index with care management or financial planning workflows. Additional feature enrichment using proxy variables for social and behavioral risk could further improve early detection of financial instability. These enhancements would support transition from analytical prototype to production-ready deployment.


