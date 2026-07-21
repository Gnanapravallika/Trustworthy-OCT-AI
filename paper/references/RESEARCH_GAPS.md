# Research Gaps & TrustOCT Responses

This document maps the technical limitations identified in modern literature to our proposed TrustOCT design choices, providing directly copyable content for the Introduction and Discussion sections of the paper.

---

## 1. Literature Gaps vs. TrustOCT Response Matrix

| Cited Work | Technical Limitation Identified | TrustOCT Methodology Response | Verification Metric |
| :--- | :--- | :--- | :--- |
| **Kim et al. (2024)** (OCTDL) | Baseline CNN models suffer performance drops under device shift due to scanner texture memorization. | Integrates **MixStyle** and **Feature CORAL** layers to learn domain-invariant representations. | Cross-dataset (OCTDL ➔ OCTID) accuracy and Macro-F1. |
| **Sensoy et al. (2018)** (Evidential DL) | Softmax probabilities are overconfident on ambiguous, out-of-distribution (OOD), or noisy images. | Integrates an **Evidential Dirichlet Head** producing continuous uncertainty metrics in a single pass. | Expected Calibration Error (ECE) and OOD AUROC. |
| **Li et al. (2023)** (Clinical AI Review) | AI classifiers make catastrophic errors on borderline clinical samples, causing translation blocks. | Implements **Selective Prediction (Abstention)** using uncertainty thresholds $\tau$ to defer scans. | Accuracy-Coverage Curve and Retained Accuracy. |
| **Qi et al. (2025)** | Standard deep convolutional downsampling dilutes localized spatial biomarker maps. | Implements **Layer 3 + Layer 4 Multi-Scale Feature Fusion** coupled with CBAM attention gates. | LayerCAM attribution maps and Saliency Entropy. |

---

## 2. Copyable Text Drafts

### Introduction (Novelty Formulation)
> *"While modern deep learning backbones show expert-level performance on specific OCT image databases, they suffer from three core clinical limitations: scanner-induced accuracy decay, uncalibrated overconfidence on OOD samples, and coarse, unlocalized visual explanations. Our proposed framework, TrustOCT, bridges these gaps by integrating feature-level statistics mixing (MixStyle) for cross-scanner generalization, an evidential Dirichlet prediction head for single-pass calibration, and a multi-scale feature fusion architecture to capture fine spatial biomarkers. By formalizing a selective prediction loop, the model abstains from high-uncertainty decisions, aligning automated ophthalmic AI with clinical safety standards."*
