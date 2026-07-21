# Paper Summaries Database

This document contains key details of the most critical papers for the **TrustOCT** project, detailing their problem formulations, methodologies, strengths, weaknesses, and direct lessons for our framework.

---

## 1. MixStyle (Zhou et al., ICLR 2021)
- **Problem**: Traditional networks overfit to domain-specific texture styles, resulting in weak generalization on unseen test domains.
- **Methodology**: Mixes style statistics (feature mean and standard deviation) of random instance pairs in intermediate layers of convolutional neural networks (ResNet Bottlenecks) using a Beta distribution during training.
- **Strengths**: Lightweight, requires no target data or adversarial domain classifiers.
- **Weaknesses**: Chiefly tested on standard benchmarks (Office-Home, PACS); clinical efficacy under high noise and speckles was not fully addressed.
- **Direct Lesson for TrustOCT**: Placing MixStyle after ResNet50 Conv1 and Layer 1 structures allows the backbone to learn scanner-invariant retinal shapes (e.g., fluid pockets) rather than scanner-specific resolution details.

---

## 2. Evidential Deep Learning (Sensoy et al., NeurIPS 2018)
- **Problem**: Softmax outputs represent a relative probability distribution, leading to high confidence on out-of-distribution (OOD) images or noisy samples.
- **Methodology**: Places a Dirichlet probability distribution over logits. The network outputs positive parameters $\alpha_k$ for each class, where total evidence $S = \sum \alpha_k$. Epistemic uncertainty $u$ is given by $K/S$, where $K$ is the class count.
- **Strengths**: Computes uncertainty in a single forward pass, optimized directly via a sum-of-squares loss and KL divergence regularizer.
- **Weaknesses**: Training can be unstable initially; requires careful annealing of the KL divergence penalty.
- **Direct Lesson for TrustOCT**: Eliminates the computationally heavy inference loops of MC Dropout (which require 20-50 passes), enabling efficient selective prediction on edge/portable clinical scanners.

---

## 3. LayerCAM (Jiang et al., TIP 2021)
- **Problem**: Classical attribution maps like Grad-CAM lose fine-grained details when pooling gradients from the deepest layer, producing coarse blobs.
- **Methodology**: Computes pixel-level class activation maps by blending activations at multiple CNN layers. Backward gradients are used to weight spatial feature maps at each layer without spatial average pooling.
- **Strengths**: Produces highly detailed attribution maps at shallow layers while maintaining the class specificity of deep layers.
- **Weaknesses**: Can produce fragmented heatmaps if layers are not balanced properly.
- **Direct Lesson for TrustOCT**: Since eye OCT abnormalities (like drusen or thin fluid layers) are structurally small, LayerCAM is superior to Grad-CAM for pinpointing pathology location on B-scans.
