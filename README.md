# Healthcare Cost Volatility Prediction Platform

## Executive Summary
Healthcare organizations struggle not only with high medical costs, but with **unpredictable cost behavior**. Sudden deviations in member spending introduce budgeting risk, reserve pressure, and reactive decision-making.

This project focuses on **predicting healthcare cost volatility** rather than total cost. The goal is to identify members whose future spending is likely to **deviate significantly from their historical cost patterns**, enabling earlier financial planning and targeted review.

The platform demonstrates an end-to-end applied machine learning workflow, including data generation, feature engineering, leakage-aware modeling, and decision-oriented risk scoring.

---

## Why Cost Volatility Instead of Cost Prediction
Traditional healthcare models focus on predicting absolute cost or utilization. While useful, these approaches often fail to capture **instability**, which is what most directly impacts financial planning and forecasting.

This project reframes the problem as:
- Who is likely to experience **unexpected cost changes**, not just high costs?
- Which members introduce **budget risk** due to spending volatility?

This framing aligns closely with the needs of:
- FP&A and budgeting teams  
- Medical economics and actuarial support  
- Strategic healthcare analytics  

---

## Problem Definition
The objective is to predict **future cost volatility at the member level** using historical, claims-like data.

Rather than modeling raw spend, the system estimates deviation from an individual’s **expected cost baseline** over time, capturing instability that may indicate emerging financial or care-related risk.

---

## Data Strategy (Synthetic Data)
Real healthcare claims data cannot be publicly shared due to privacy and regulatory constraints. To address this, the project uses **synthetically generated member-month cost data** designed to mimic the statistical and behavioral properties of real claims data.

The synthetic dataset simulates:
- Member-level baseline risk and demographics  
- Heavy-tailed healthcare cost distributions  
- Utilization-driven costs (e.g., ER and inpatient events)  
- Seasonality effects  
- Rare but impactful cost shock events  

This approach enables full reproducibility while preserving realistic modeling challenges.

---

## System Overview
The diagram below illustrates the end-to-end pipeline used to generate member-level cost volatility risk signals.

![System Overview](docs/system_overview.png)

*Figure: End-to-end batch pipeline for generating financial volatility risk signals from synthetic healthcare cost data.*


---
## Feature Engineering Approach
Features are designed to capture **change, instability, and deviation over time**, rather than static cost snapshots.

Key feature categories include:
- Rolling cost statistics (mean, standard deviation, quantiles) over multiple time windows  
- Member-relative spike indicators (cost exceeding historical percentiles)  
- Global spike indicators (absolute high-cost events)  
- Temporal variability and trend measures  

Early time periods without sufficient historical context are explicitly excluded from modeling to prevent artificial signal and target leakage.

---

## Label Design
The volatility problem is intentionally decomposed into **two complementary targets**, reflecting how real-world financial risk systems operate.

### 1. Spike Classification
A binary indicator representing whether a member experiences a cost spike relative to their historical behavior.

This answers:
> *“Is a volatility event likely to occur?”*

Member-relative and global thresholds are used to distinguish unexpected instability from consistently high-cost behavior.

---

### 2. Volatility Magnitude Regression
A continuous measure representing the **severity of instability** when it occurs.

This answers:
> *“How large could the deviation from expected cost be?”*

Separating event detection from magnitude estimation allows for clearer modeling, evaluation, and operational use.

---

## Modeling Approach
Simple, interpretable baseline models are used intentionally to establish credibility and transparency:

- **Logistic Regression** for spike classification  
- **Linear Regression** for volatility magnitude estimation  

Model selection prioritizes:
- Interpretability  
- Stability on noisy healthcare data  
- Ease of leakage detection and diagnosis  

This baseline-first approach ensures signal is understood before introducing additional complexity.

---

## Validation Strategy
Random train-test splits can leak future information in time-series healthcare data. To avoid this, the project uses **out-of-time validation**:

- Models are trained on earlier time periods  
- Performance is evaluated on later, unseen periods  

This mirrors real-world deployment, where models are trained on historical data and used to score future behavior.

---

## Leakage Detection and Mitigation
During development, unusually perfect model performance triggered further investigation.

Correlation analysis revealed that some rolling statistics used as features were direct proxies for the volatility labels, introducing target leakage.

Corrective actions included:
- Removing label-derived features  
- Re-training models  
- Accepting lower but honest performance  

This process reflects real-world ML practice, where detecting and fixing leakage is critical for trustworthiness.

---

## Financial Volatility Index (FVI)
To support decision-making, probability and impact are combined into a single risk score:

**FVI = P(Spike) × Expected Volatility**

This produces a rankable signal that balances:
- Likelihood of instability  
- Financial severity if instability occurs  

Members are segmented into Low, Medium, and High volatility tiers based on capacity-aware thresholds, enabling practical prioritization.

---

## Evaluation Philosophy
Healthcare cost data is inherently noisy, and point-accuracy metrics alone are insufficient.

Evaluation focuses on:
- Lift and concentration of events in high-risk segments  
- Stability of risk rankings over time  
- Practical usefulness for monitoring and planning  

The goal is **decision support**, not exact dollar prediction.

---

## Design Considerations
The project is structured to support future extension, including:
- Alternative volatility definitions  
- Model replacement or retraining  
- Integration of additional signals (e.g., utilization detail, pharmacy, administrative events)  

The emphasis is on modularity, reproducibility, and clarity rather than overengineering.

---

## Limitations and Future Work
- Synthetic data cannot capture all clinical and operational nuance  
- Volatility is influenced by administrative and external factors not modeled here  
- Future iterations could incorporate richer signals, monitoring logic, and retraining cadence  

---

## Key Takeaways
- Cost volatility is distinct from cost magnitude and represents a critical financial risk signal  
- Leakage-aware feature and label design is essential in healthcare ML  
- Simple, interpretable models can deliver meaningful, decision-ready insights  

