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
Usage
bash
Copy
Edit
python advanced_ml_pipeline.py \
  [--dataset cifar100|imagenet] \
  [--data-dir ./data] \
  [--batch-size N] \
  [--epochs N] \
  [--resume path/to.ckpt.pth] \
  [--evaluate] \
  [--search --trials N] \
  [--local_rank $LOCAL_RANK]
Use --help to see all flags.
