# Unified Latents for Flow Mismatching Anomaly Detection

## One-line pitch
Train **Unified Latents (UL)** from scratch on normal images, drop the decoder at test time, and score anomalies directly on the compact UL flow prior via **Flow Mismatching**—speeding up inference while testing how latent bitrate ($\sigma_0$) trades off fine textures against large structural defects.

## Background
- **Flow Mismatching** ([arXiv 2605.23070](https://arxiv.org/abs/2605.23070), NeurIPS 2026; [ShengzheC/Flow-Mismatching](https://github.com/ShengzheC/Flow-Mismatching)) trains a single class-conditional (or label-free unified `--no-class_cond`) Flow Matching U-Net exclusively on normal `128×128×3` images for 2,000 epochs. At test time, it avoids reconstruction and ODE integration by probing the learned velocity field $v_\theta(x_t, t)$ along $K$ straight affine paths $x_t^{(k)} = (1-t)x_0^{(k)} + t y$ across $T$ time steps and aggregating the local squared velocity discrepancy $\|v_\theta(x_t^{(k)}, t) - (y - x_0^{(k)})\|_2^2$ via `min` over paths and $t^r$ time weighting.
- Two clear bottlenecks reported in the paper's own Appendix D.2–D.3:
  1. **Large structural / object-level failure mode (Appendix D.2):** Because the model operates directly in pixel space, it captures local textures and edges well (98.7% I-AUROC / 67.5% P-AP on MVTec-AD with class conditioning) but often highlights only the *boundaries* of large structural or logical anomalies instead of the full anomalous region. The authors explicitly point to **latent-space flow formulations** as a natural next step.
  2. **Compute & throughput bottleneck (Appendix D.3):** Training requires 2,000 epochs of a 6-stage `128×128` U-Net, and inference requires $K \times T = 100$ full-resolution forward passes per image.
- **Unified Latents (UL)** ([arXiv 2602.17270](https://arxiv.org/abs/2602.17270), Heek et al., 2026) jointly trains a deterministic encoder $E_\psi(x)$, a **latent diffusion/flow prior** $v_\theta$ anchored at a fixed base noise level $\sigma_0$ (e.g., $\sigma_0 \approx 0.08$), and a **diffusion decoder** $D_\omega$. Linking the encoder's output noise to the prior's noise floor turns the KL penalty into a clean prior-matching MSE loss with an explicit, tunable **latent bitrate bound** controlled by $\sigma_0$.

## The catch, and the fix
Putting UL and Flow Mismatching together—purely from scratch on normal data—introduces three non-obvious catches:

1. **OOD encoder projection (normal-only collapse):** Unlike generative modeling (where train and test come from the same distribution), unsupervised anomaly detection trains UL *only* on normal images. A learned encoder $E_\psi$ trained from scratch could project an anomalous test patch onto the normal latent manifold, masking the defect before the prior sees it.
   - **Fix:** Rely on UL's pixel-diffusion decoder $D_\omega$ (with sigmoid-weighted reconstruction loss) to enforce pixel fidelity in $E_\psi$, **explicitly diagnose OOD collapse** by comparing normal vs. anomalous reconstruction PSNR of the discarded train-only decoder $D_\omega(E_\psi(y))$, and benchmark against a **frozen pretrained SD-VAE** that was never fit solely to normal industrial images.
2. **Noise boundary ($\sigma_0$) and ELBO vs. Flow Mismatching:** UL's prior terminates at base noise $\sigma_0$ (corresponding to affine-path endpoint $t_1 = \alpha_0 / (\alpha_0 + \sigma_0)$ rather than noise-free $t = 1$), and UL's training prior loss averages MSE across noise draws (an ELBO bitrate estimator).
   - **Fix:** Align Flow Mismatching's `t_range=(t0, t1)` in `scoring.py` with UL's exact prior boundary $t_1(\sigma_0)$ (and ablate against `t1 = 1.0`), and directly compare **`aggregation="mean"` (UL's exact per-pixel prior ELBO)** against **`aggregation="min"` (Flow Mismatching's lower-tail false-positive suppressor)** on the exact same UL prior.
3. **Spatial downsampling vs. prior-shaping effects:** Moving from `128×128×3` pixels to `32×32×8` latents simultaneously changes spatial resolution and introduces UL's latent bottleneck.
   - **Fix:** Contrast downsampled UL latents against **full-resolution UL (`128×128×3`, no spatial downsampling)** and against a **decoupled second-stage Flow Matching prior** trained on the frozen UL encoder to cleanly separate spatial downsampling loss, UL latent regularization, and joint vs. two-stage prior training.

## Research questions
On **MVTec-AD** (and **MVTec-LOCO** for large logical/structural anomalies), under strictly **label-free unified multi-class modeling (`--no-class_cond`)** and **without pretrained vision foundation encoders**:
1. Can we train **Unified Latents** from scratch on normal images, discard the diffusion decoder at test time, and score anomalies using **Flow Mismatching directly on the UL prior**—approaching or matching pixel-space Flow Mismatching at a fraction of the training and inference cost?
2. How does UL's **latent bitrate** (controlled via $\sigma_0$) trade off **fine texture defect localization** against **large structural/object-level anomalies**, and how does a prior-shaped UL latent space compare against a generic frozen pretrained VAE?
3. On the exact same UL prior, does Flow Mismatching's `min` aggregation outperform raw UL prior ELBO (`mean` aggregation), and does training a dedicated second-stage prior improve over the jointly trained UL prior?

## Proposed experimental design
All comparisons reuse `ShengzheC/Flow-Mismatching`'s exact data layout, `scoring.py` velocity-mismatch engine, and 7-metric evaluation harness (`evaluate.py`) under matched 1× RTX 4090 compute budgets:

1. **Baselines:**
   - **Pixel-space Flow Mismatching:** Official `128×128×3` pixel-space model under label-free unified `--no-class_cond` (with `--class_cond` paper reference).
   - **Frozen pretrained VAE + Flow Mismatching:** Frozen SD-VAE latents (`256×256` $\to$ `32×32×4`, standardized per channel, **no UL prior regularization**) paired with a latent Flow Matching U-Net. Separates "any compact latent space" from UL's prior-regularized latent space.
2. **Proposed method — Unified Latents from scratch + Flow Mismatching:**
   - **Joint UL model:** Jointly train a deterministic CNN encoder $E_\psi$ (`128×128×3` $\to$ `32×32×8`), a 3-stage Flow Matching prior $v_\theta$, and a lightweight diffusion decoder $D_\omega$ exclusively on normal images with fixed $\sigma_0 = 0.08$. At test time, discard $D_\omega$, encode $z = E_\psi(y)$, and score directly with `anomaly_heatmap(v_theta, z)`.
   - **Scoring & prior ablations:** Directly compare `min` (Flow Mismatching) vs. `mean` (raw UL prior ELBO), $t_1(\sigma_0)$ vs. $t_1 = 1.0$, and the jointly trained UL prior vs. a dedicated second-stage uniform Flow Matching prior trained on frozen $E_\psi(x)$.
   - **Bitrate sweep:** Sweep UL's base noise $\sigma_0 \in \{0.04, 0.08, 0.16\}$ (high $\to$ low latent bitrate) to map the bitrate–accuracy trade-off across textures vs. objects and small vs. large defects.
   - **Full-resolution UL diagnostic:** Train UL at `128×128×3` without spatial downsampling ($\sigma_0 = 0.08$) to isolate UL's $\sigma_0$ latent regularization from 4× spatial downsampling.

## Metrics
- **All 7 official Flow Mismatching metrics** (`evaluate.py`):
  - Image-level: **AUROC**, **AP**, **F1-max**
  - Pixel-level: **AUROC**, **AP**, **F1-max**, **AUPRO** (up to 30% FPR)
- **UL diagnostics:** Latent bitrate (nats/dim from the UL prior ELBO on normal vs. anomalous test sets) + train-only decoder reconstruction PSNR (normal vs. anomalous, diagnosing whether defects survive encoding)
- **Efficiency:** Training wall-clock hours and inference latency (**ms / image** across $K \times T$ sweeps such as $10\times 10$ vs. $4\times 5$)
- **Category & defect-scale breakdown:** MVTec-AD textures (5) vs. objects (10), small-area vs. large-area defects, and MVTec-LOCO structural vs. logical splits

## Closest prior work
- [Flow Mismatching](https://arxiv.org/abs/2605.23070) ([ShengzheC/Flow-Mismatching](https://github.com/ShengzheC/Flow-Mismatching)): base pixel-space velocity-discrepancy AD framework and codebase (NeurIPS 2026)
- [Unified Latents (UL)](https://arxiv.org/abs/2602.17270): joint encoder + diffusion prior ($\sigma_0$-linked ELBO bitrate) + diffusion decoder (Heek et al., 2026)
- [DiAD](https://arxiv.org/abs/2404.06008): latent-diffusion reconstruction for unified multi-class AD (requires test-time decoding/reconstruction)
- [UniAD](https://arxiv.org/abs/2206.03687), [MambaAD](https://arxiv.org/abs/2404.06564), [ViTAD](https://arxiv.org/abs/2312.07495): unified multi-class feature-reconstruction AD baselines using pretrained backbones

## Feasibility and risks
Feasibility is high: `ShengzheC/Flow-Mismatching` is lightweight (`train.py`, `evaluate.py`, and modular `flow_mismatching/`), its `scoring.py` is channel-agnostic (`[1, C, H, W]`), and MVTec-AD is compact (3,629 normal training images), allowing all UL latent experiments to run on a single RTX 4090 (24 GB).
- **Small-defect blurring:** 4× spatial downsampling (`128×128` $\to$ `32×32`) can reduce Pixel AP on thin scratches or tiny defects. We quantify this directly via the full-resolution `128×128` UL diagnostic and separate small- vs. large-defect breakdowns.
- **Normal-only encoder OOD behavior & stochastic textures:** A learned encoder trained only on normal data may compress or smooth high-frequency stochastic textures (e.g., `tile`, `carpet`, `wood`) or partially project OOD defects toward normal latents. Comparing UL against pixel-space Flow Mismatching, frozen SD-VAE, the $\sigma_0$ sweep, and normal-vs.-anomalous decoder PSNR isolates these failure modes cleanly.
- **Negative results are informative:** Mapping the exact **bitrate ($\sigma_0$) vs. texture/object accuracy vs. inference latency** trade-off—and identifying where UL's prior-shaped latents help or fail (e.g., stochastic texture density inversion)—directly answers the pitch.

## Open questions
- [ ] Start with **MVTec-AD** across the core baselines and UL ablations, then extend the best UL configuration to **MVTec-LOCO** (structural vs. logical anomalies) or **VisA**?
- [ ] Keep strictly label-free unified modeling (`--no-class_cond`) as the sole primary setting, or add one `--class_cond` UL run to isolate the cost of dropping class labels?
- [ ] For the latency Pareto curve, test aggressive inference reduction down to $K \times T \in \{1\times 1, 2\times 3, 4\times 5, 10\times 10\}$ on the UL prior?

## Discussion
Our thoughts, pros and cons, and questions for the instructors go here.
