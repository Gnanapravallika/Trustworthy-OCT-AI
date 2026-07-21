# DESIGN FREEZE DOCUMENT

**Project Name:** TrustOCT

**Version:** 1.0

**Status:** DESIGN FROZEN

**Purpose**

This document freezes every major scientific and engineering decision of the TrustOCT project. Any future modifications must be justified through experimental evidence or new peer-reviewed literature.

---

# 1. Research Vision

Develop a trustworthy retinal OCT disease classification framework capable of maintaining reliable performance on unseen datasets while providing uncertainty-aware predictions and interpretable visual explanations.

---

# 2. Research Problem

Current retinal OCT classification models achieve high internal accuracy but often fail to generalize across scanners and datasets. They also lack reliable uncertainty estimation and comprehensive trustworthiness evaluation.

---

# 3. Research Hypothesis

Integrating feature-level domain generalization with uncertainty-aware prediction improves the trustworthiness and cross-dataset robustness of retinal OCT disease classification compared with conventional CNN-based classifiers.

---

# 4. Novelty Statement

TrustOCT is not proposed as a new neural network architecture.

Instead, it is a unified trustworthy AI framework integrating:

- Domain Generalization
- Uncertainty Estimation
- Explainable AI
- Calibration Analysis
- External Cross-Dataset Validation

for retinal OCT disease classification.

The novelty lies in the integration, evaluation methodology, and trustworthiness analysis rather than introducing a new backbone or attention mechanism.

---

# 5. Scientific Contributions

Contribution 1: Develop a modular TrustOCT framework for trustworthy retinal OCT diagnosis.

Contribution 2: Investigate feature-level domain generalization under unseen scanner/domain shifts.

Contribution 3: Integrate evidential uncertainty estimation into retinal OCT classification.

Contribution 4: Perform comprehensive trustworthiness evaluation including calibration, explainability, and external validation.

Contribution 5: Provide reproducible implementation with extensive ablation studies.

---

# 6. Research Scope

Included

- Retinal OCT image classification
- Public datasets
- Cross-dataset evaluation
- Domain Generalization
- Uncertainty Estimation
- Explainable AI
- Calibration Analysis

Excluded

- Segmentation
- Clinical deployment
- Federated learning
- Vision Transformers as primary backbone
- Proprietary clinical datasets

---

# 7. Final Architecture

Input OCT Image

↓

Preprocessing

↓

ResNet50 Backbone

↓

Adaptive Multi-Scale Feature Fusion

↓

CBAM Attention

↓

MixStyle Domain Generalization

↓

Evidential Deep Learning Head

↓

Outputs

• Disease Prediction

• Predictive Uncertainty

↓

LayerCAM

↓

Trustworthiness Evaluation

---

# 8. Final Technical Decisions

## Backbone

ResNet50

Reason: Stable, reproducible, widely adopted in retinal OCT literature, and provides strong comparison with previous work.

---

## Domain Generalization

MixStyle

Reason: Supports single-source training while improving robustness to scanner-specific appearance variations.

---

## Attention

CBAM

Reason: Enhances feature representation with low computational overhead.

---

## Multi-scale Module

Adaptive Multi-Scale Feature Fusion

Reason: Capture retinal lesions appearing at multiple spatial scales.

---

## Uncertainty Estimation

Evidential Deep Learning

Reason: Single-pass uncertainty estimation suitable for deployment-oriented trustworthy AI.

---

## Explainability

Primary: LayerCAM

Baseline: Grad-CAM

Reason: Provide both qualitative interpretation and comparison with the standard explainability approach.

---

# 9. Datasets

Training: Kermany OCT2017

External Evaluation: OCTDL, NEH-UT

Optional: OCTID

---

# 10. Experimental Pipeline

- EXP001: Baseline
- EXP002: + MultiScale
- EXP003: + CBAM
- EXP004: + MixStyle
- EXP005: + Evidential Learning
- EXP006: Full TrustOCT
- EXP007: Cross-Dataset Evaluation
- EXP008: Calibration Analysis
- EXP009: Explainability Comparison
- EXP010: Ablation Study

---

# 11. Evaluation Metrics

Classification

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Robustness

- Internal Accuracy
- External Accuracy
- Performance Drop

Calibration

- Expected Calibration Error
- Brier Score
- Reliability Diagram

Uncertainty

- Selective Prediction
- Confidence Distribution

Explainability

- LayerCAM
- Grad-CAM

---

# 12. Definition of Trustworthiness

Trustworthiness consists of four measurable pillars:

1. Generalization: Performance on unseen external datasets.
2. Calibration: Confidence reflects correctness.
3. Uncertainty Awareness: The model recognizes unreliable predictions.
4. Interpretability: Clinically meaningful visual explanations.

---

# 13. Reproducibility

The project must include:

- Fixed random seeds
- Configuration files
- Experiment logging
- Version control
- Saved checkpoints
- Complete documentation

---

# 14. Paper Target Structure

1. Introduction
2. Related Work
3. Methodology
4. Experimental Setup
5. Results
6. Discussion
7. Limitations
8. Conclusion

---

# 15. Design Freeze Rule

From this point onward:

Architecture changes are not permitted unless supported by:

- Strong experimental evidence
  or
- Newly published peer-reviewed literature demonstrating a superior approach.

Otherwise, implementation proceeds according to this document.
