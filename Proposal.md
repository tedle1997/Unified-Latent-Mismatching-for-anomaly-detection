# Unified Latents for Flow Mismatching Anomaly Detection

| Field | Value |
| --- | --- |
| Summary | Replace pixel-space Flow Matching in **Flow Mismatching** (`ShengzheC/Flow-Mismatching`) with **Unified Latents (UL)** trained from scratch on normal data only—without pretrained vision foundation backbones—discarding the UL diffusion decoder at test time and scoring images via velocity discrepancies of the **UL flow prior** directly in latent space, while sweeping UL's noise boundary $\sigma_0$ (latent bitrate) and spatial resolution. |
| Status | In Progress |
| Feasibility | High |
| Key references | Flow Mismatching arXiv 2605.23070 (NeurIPS 2026); Unified Latents (UL) arXiv 2602.17270; Flownomaly arXiv 2610.32964 (WACV 2027); DiAD 2404.06008; UniAD 2206.03687; MambaAD 2404.06564 |
| Novelty | No prior work combines **Unified Latents** (learned encoder + linked noise boundary $\sigma_0$ + joint ELBO flow prior + train-only diffusion decoder) with **Flow Mismatching** velocity-discrepancy anomaly detection. Unlike frozen-backbone methods (e.g., Flownomaly on per-class DINOv2 tokens), we learn a compact, bitrate-controlled 2D latent space purely from normal images under unified label-free modeling (`--no-class_cond`). Crucially, UL's prior ELBO naturally bridges to Flow Mismatching's velocity discrepancy (allowing a direct `mean` ELBO vs. `min` Flow Mismatching comparison on the exact same latent prior). |
| Main risks | A UL encoder trained from scratch only on normal images may project anomalous test patches onto the normal latent manifold (diagnosed via normal vs. anomalous decoder PSNR and compared against frozen SD-VAE); 4×–8× spatial downsampling can blur tiny 1–3 px defects (disentangled by sweeping $\sigma_0$ and testing full-resolution `128×128` UL without spatial downsampling). |

## One-line pitch
Pixel-space Flow Mismatching excels at fine local defects without test-time reconstruction, but struggles on large structural/semantic anomalies and requires 100 full-resolution U-Net forward passes per test image (~268 ms/img). Can **Unified Latents (UL)** trained from scratch on normal images provide a compact, bitrate-controlled latent space whose **jointly trained flow prior directly serves as the test-time Flow Mismatching scorer** (dropping the diffusion decoder at inference)—and how does UL's latent bitrate ($\sigma_0$) govern the trade-off between fine texture defects, large structural anomalies, and inference speed?

## Background
- **Flow Mismatching** ([arXiv 2605.23070](https://arxiv.org/abs/2605.23070), NeurIPS 2026; [ShengzheC/Flow-Mismatching](https://github.com/ShengzheC/Flow-Mismatching)) trains a single class-conditional (or label-free unified `--no-class_cond`) Flow Matching U-Net exclusively on normal `128×128×3` images for 2,000 epochs. At test time, it avoids reconstruction and ODE integration by probing the learned velocity field $v_\theta(x_t, t)$ along $K$ straight affine paths $x_t^{(k)} = (1-t)x_0^{(k)} + t y$ across $T$ time steps and aggregating the local squared velocity discrepancy $\|v_\theta(x_t^{(k)}, t) - (y - x_0^{(k)})\|_2^2$ via `min` over paths and $t^r$ time weighting.
- Two clear bottlenecks reported in the paper's own Appendix D.2–D.3:
  1. **Large structural / object-level failure mode (Appendix D.2):** Because the model operates directly in pixel space, it captures local textures and edges well (98.7% I-AUROC / 67.5% P-AP on MVTec-AD with class conditioning) but often highlights only the *boundaries* of large structural or logical anomalies instead of the full anomalous region. The authors explicitly point to **latent-space flow formulations** as a natural next step.
  2. **Compute & throughput bottleneck (Appendix D.3):** Training requires 2,000 epochs of a 6-stage `128×128` U-Net, and inference requires $K \times T = 100$ full-resolution forward passes per image.
- **Unified Latents (UL)** ([arXiv 2602.17270](https://arxiv.org/abs/2602.17270), Heek et al., 2026) jointly trains a deterministic encoder $E_\psi(x)$, a **latent diffusion/flow prior** $v_\theta$ anchored at a fixed base noise level $\sigma_0$ (e.g., $\sigma_0 \approx 0.08$), and a **diffusion decoder** $D_\omega$. Linking the encoder's output noise to the prior's noise floor turns the KL penalty into a clean prior-matching MSE loss with an explicit, tunable **latent bitrate bound** controlled by $\sigma_0$.

## The catch, and the fix
Putting UL and Flow Mismatching together—purely from scratch on normal data—introduces three non-obvious catches:

1. **OOD encoder projection (normal-only collapse):** Unlike generative modeling (where train and test come from the same distribution), unsupervised anomaly detection trains UL *only* on normal images. A learned encoder $E_\psi$ trained from scratch could project an anomalous test patch onto the normal latent manifold, masking the defect before the prior sees it.
   - **Fix:** Rely on UL's pixel-diffusion decoder $D_\omega$ (with sigmoid-weighted reconstruction loss) to enforce pixel fidelity in $E_\psi$, **explicitly diagnose OOD collapse** by comparing normal vs. anomalous reconstruction PSNR of the discarded train-only decoder $D_\omega(E_\psi(y))$, and benchmark against a **frozen pretrained SD-VAE (`A1-vae`)** that was never fit solely to normal industrial images.
2. **Noise boundary ($\sigma_0$) and ELBO vs. Flow Mismatching:** UL's prior terminates at base noise $\sigma_0$ (corresponding to affine-path endpoint $t_1 = \alpha_0 / (\alpha_0 + \sigma_0)$ rather than noise-free $t = 1$), and UL's training prior loss averages MSE across noise draws (an ELBO bitrate estimator).
   - **Fix:** Align Flow Mismatching's `t_range=(t0, t1)` in `scoring.py` with UL's exact prior boundary $t_1(\sigma_0)$ (and ablate against `t1 = 1.0`), and directly compare **`aggregation="mean"` (UL's exact per-pixel prior ELBO)** against **`aggregation="min"` (Flow Mismatching's lower-tail false-positive suppressor)** on the exact same UL prior.
3. **Spatial downsampling vs. prior-shaping effects:** Moving from `128×128×3` pixels to `32×32×8` latents simultaneously changes spatial resolution and introduces UL's latent bottleneck.
   - **Fix:** Run a **full-resolution UL diagnostic (`128×128×3`, downsample=1)** and a **Stage-2 frozen-encoder Flow Matching ablation** to cleanly separate spatial downsampling loss, UL latent regularization, and joint Stage-1 vs. dedicated Stage-2 prior training.

## Question
On **MVTec-AD** (and **MVTec-LOCO** for large logical/structural anomalies), under strictly **label-free unified multi-class modeling (`--no-class_cond`)** and **without pretrained vision foundation encoders**:
1. Can we train **Unified Latents** from scratch on normal images, discard the diffusion decoder at test time, and score anomalies using **Flow Mismatching directly on the UL prior**—approaching or matching pixel-space Flow Mismatching at a fraction of the training and inference cost?
2. How does UL's **latent bitrate** (controlled via $\sigma_0$) trade off **fine texture defect localization** against **large structural/object-level anomalies** (and how does prior-shaped UL compare to a generic frozen SD-VAE latent space)?
3. On the exact same UL prior, does Flow Mismatching's `min` aggregation outperform raw UL prior ELBO (`mean` aggregation), and does a Stage-2 prior improve over the jointly trained Stage-1 UL prior?

## Experiments
All setups reuse `ShengzheC/Flow-Mismatching`'s exact data layout, `scoring.py` velocity-mismatch engine, and 7-metric evaluation harness (`evaluate.py`) under matched 1× RTX 4090 compute budgets:

1. **Setup A — Baselines (`A0-pixel` & `A1-vae`):**
   - `A0-pixel`: Official pixel-space Flow Mismatching (`128×128×3`) under label-free unified `--no-class_cond` (plus `--class_cond` paper reference).
   - `A1-vae`: Frozen pretrained SD-VAE latents (`256×256` $\to$ `32×32×4`, standardized per channel, **no UL prior regularization**) + Flow Matching + Flow Mismatching. Separates "any compact latent space" from UL's prior-regularized latent space.
2. **Setup B — Unified Latents from scratch + Flow Mismatching (our core method):**
   - `B1` (Stage 1 joint UL): Jointly train UL CNN encoder $E_\psi$ (`128×128×3` $\to$ `32×32×8`) + 3-stage Flow Matching prior $v_\theta$ + lightweight diffusion decoder $D_\omega$ on normal images with $\sigma_0 = 0.08$. Discard $D_\omega$ at test time, encode $z = E_\psi(y)$, and run `anomaly_heatmap(v_theta, z)` comparing `min` vs. `mean` (ELBO) and $t_1(\sigma_0)$ vs. $t_1 = 1.0$.
   - `B-stage2` (Stage 2 prior): Freeze the trained UL encoder $E_\psi$ and train a dedicated uniform Flow Matching prior on $z = E_\psi(x)$ to isolate joint Stage-1 vs. decoupled Stage-2 prior quality.
   - `B-σ` (Bitrate sweep): Sweep UL's base noise $\sigma_0 \in \{0.04, 0.08, 0.16\}$ (high $\to$ low latent bitrate) to measure the bitrate–accuracy curve across textures vs. objects and small vs. large defects.
   - `B-128` (Full-resolution UL diagnostic): Train UL at `128×128×3` (`downsample=1`, $\sigma_0 = 0.08$) to disentangle the effect of spatial downsampling from UL's $\sigma_0$ latent regularization.

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
- [Flownomaly](https://arxiv.org/abs/2610.32964) (WACV 2027): per-class few-shot AD via 1-step conditional-mean recovery on frozen DINOv2 patch tokens with a token-level MLP—distinct from our unified label-free multi-class setting, learned UL latent space, 2D spatial U-Net prior, and multi-path Flow Mismatching / ELBO scoring
- [DiAD](https://arxiv.org/abs/2404.06008): latent-diffusion reconstruction for unified multi-class AD (requires test-time decoding/reconstruction)
- [UniAD](https://arxiv.org/abs/2206.03687), [MambaAD](https://arxiv.org/abs/2404.06564), [ViTAD](https://arxiv.org/abs/2312.07495): unified multi-class feature-reconstruction AD baselines using pretrained backbones

## Feasibility and risks
Feasibility is high: `ShengzheC/Flow-Mismatching` is lightweight (`train.py`, `evaluate.py`, and modular `flow_mismatching/`), its `scoring.py` is channel-agnostic (`[1, C, H, W]`), and MVTec-AD is compact (3,629 normal training images), allowing all UL latent experiments to run on a single RTX 4090 (24 GB).
- **Small-defect blurring:** 4× spatial downsampling (`128×128` $\to$ `32×32`) can reduce Pixel AP on thin scratches or tiny defects. We quantify this directly via the `B-128` full-resolution UL diagnostic and separate small- vs. large-defect breakdowns.
- **Normal-only encoder OOD behavior & stochastic textures:** A learned encoder trained only on normal data may compress or smooth high-frequency stochastic textures (e.g., `tile`, `carpet`, `wood`) or partially project OOD defects toward normal latents. Comparing `B1` against `A0-pixel`, `A1-vae`, `B-σ`, and normal-vs.-anomalous decoder PSNR isolates these failure modes cleanly.
- **Negative results are informative:** Mapping the exact **bitrate ($\sigma_0$) vs. texture/object accuracy vs. inference latency** trade-off—and identifying where UL's prior-shaped latents help or fail (e.g., stochastic texture density inversion)—directly answers the pitch.

## Open questions
- [ ] Start with **MVTec-AD** across all ablations (`A0`, `A1`, `B1`, `B-stage2`, `B-σ`, `B-128`), then extend the best UL configuration to **MVTec-LOCO** (structural vs. logical anomalies) or **VisA**?
- [ ] Keep strictly label-free unified modeling (`--no-class_cond`) as the sole primary setting, or add one `--class_cond` UL run to isolate the cost of dropping class labels?
- [ ] For the latency Pareto curve, test aggressive inference reduction down to $K \times T \in \{1\times 1, 2\times 3, 4\times 5, 10\times 10\}$ on `B1`?

## Discussion
Our thoughts, pros and cons, and questions for the instructors go here.
