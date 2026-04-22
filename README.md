# Ablation-PCNN-Former
Official PyTorch implementation of **Ablation-PCNN-Former: A Penalty-Based Physics-Guided Neural Network Framework for Irreversible Electroporation Parameter Optimization**

> **Abstract**: Irreversible electroporation (IRE) is a promising non-thermal tumor ablation technique, but traditional finite element simulation (FES) is computationally prohibitive for intraoperative use, and pure data-driven models produce physically inconsistent predictions. We propose Ablation-PCNN-Former, a two-stage hybrid framework that combines a penalty-based constrained neural network (PCNN) for real-time ablation prediction with a Swin-Transformer module for automated label generation from TTC-stained tissue images.

## 🚀 Key Features
- ✅ **Physics-Guided Learning**: Embeds IRE electric field threshold and geometry-adaptive ablation plateau constraints into the loss function
- ✅ **Real-Time Inference**: <0.1s GPU inference time (50,000× faster than standard FES)
- ✅ **Automated Annotation**: Swin-Transformer pipeline achieves 98.2% IoU, reducing labeling time by 95%
- ✅ **Cross-Tissue Generalization**: Partial transfer learning from potato tissue to mouse breast cancer
- ✅ **Inverse Optimization**: Accurately predicts required treatment parameters for target ablation area

