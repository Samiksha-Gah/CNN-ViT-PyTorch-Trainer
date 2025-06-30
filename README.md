# CNN+ViT PyTorch Trainer

Single-file PyTorch trainer for CIFAR-100 or ImageNet-style data with:

- Hybrid CNN stem + Vision Transformer backbone  
- Label-smoothing + focal loss  
- Mixed-precision, cosine LR restarts, gradient accumulation  
- DistributedDataParallel, checkpoint resume/evaluate  
- TensorBoard logging  
- Optuna hyperparameter search  

---

## Install

```bash
pip install torch torchvision timm albumentations optuna tensorboard

Use --help to see all flags.
