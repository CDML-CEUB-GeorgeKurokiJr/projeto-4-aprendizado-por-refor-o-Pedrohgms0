# DCGAN — CelebA Face Generation

Implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) for face image synthesis, trained on a curated subset of the CelebA dataset using PyTorch on Kaggle.

---

## Results

| Metric | Value |
|--------|-------|
| FID (100 epochs) | **13.72** |
| Image resolution | 64 × 64 |
| Training dataset | 20,000 images (curated) |
| Hardware | Kaggle T4 GPU |

---

## Architecture

Based on [Radford et al., 2015](https://arxiv.org/abs/1511.06434) with no structural modifications.

### Generator
Input: latent vector `z ~ N(0,1)` with dimension 100.

```
z (100, 1, 1)
→ ConvTranspose2d → BN → ReLU   (1024, 4, 4)
→ ConvTranspose2d → BN → ReLU   (512,  8, 8)
→ ConvTranspose2d → BN → ReLU   (256, 16,16)
→ ConvTranspose2d → BN → ReLU   (128, 32,32)
→ ConvTranspose2d → Tanh         (3,  64,64)
```

### Discriminator
Input: RGB image 3 × 64 × 64.

```
(3,  64,64) → Conv2d → LeakyReLU(0.2)          (64, 32,32)
(64, 32,32) → Conv2d → BN → LeakyReLU(0.2)    (128, 16,16)
(128,16,16) → Conv2d → BN → LeakyReLU(0.2)    (256,  8, 8)
(256, 8, 8) → Conv2d → BN → LeakyReLU(0.2)    (512,  4, 4)
(512, 4, 4) → Conv2d → Sigmoid                  (1,    1, 1)
```

Weight initialization: `N(0, 0.02)` for Conv layers, `N(1, 0.02)` for BatchNorm.

---

## Dataset

- **Source:** [`jessicali9530/celeba-dataset`](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) on Kaggle
- **Subset:** 20,000 images selected via manual curation (`curated_files.csv`)
- **Rationale:** Curated subset focuses the generator on a narrower facial distribution, reducing mode coverage requirements and improving convergence on limited compute
- **Preprocessing:** `CenterCrop(140)` → `Resize(64)` → `Normalize(mean=0.5, std=0.5)` (output range `[-1, 1]`)

---

## Training

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 100 |
| Batch size | 128 |
| Latent dimension | 100 |
| Optimizer | Adam (β₁=0.5, β₂=0.999) |
| Learning rate | 0.0002 |
| Loss | Binary Cross-Entropy |
| Label smoothing | Real labels = 0.9 (one-sided) |

---

## Evaluation — FID

FID (Fréchet Inception Distance) is computed natively at every epoch using a pre-trained Inception V3 (`transform_input=False`).

- **Real features:** extracted once from 5,000 curated images and kept fixed throughout training
- **Fake features:** 5,000 generator samples per epoch, upsampled to 299 × 299 for Inception V3

> Note: FID scores are internally consistent but not directly comparable to published benchmarks, which use ImageNet-normalized inputs to Inception V3.

---

## Repository Structure

```
celeba-gan20k.ipynb     # Main notebook (data loading → training → evaluation)
output_dcgan/
├── real_samples.png            # Grid of real images from the training set
├── fake_epoch_{N}.png          # Generated samples saved every 5 epochs
├── training_plot.png           # G loss, D loss, and FID over epochs
├── final_generated.png         # 64-image grid from the final generator
├── generator_epoch_{N}.pth     # Generator checkpoint every 10 epochs
└── discriminator_epoch_{N}.pth # Discriminator checkpoint every 10 epochs
```

---

## Requirements

```
torch
torchvision
numpy
pandas
matplotlib
scipy
tqdm
Pillow
```

Tested on Python 3.12, PyTorch 2.x, Kaggle T4 environment.

---

## References

- Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.* [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
- Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). *Deep Learning Face Attributes in the Wild.* [arXiv:1411.7766](https://arxiv.org/abs/1411.7766)
- Heusel, M. et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.* [arXiv:1706.08500](https://arxiv.org/abs/1706.08500) *(FID metric)*
