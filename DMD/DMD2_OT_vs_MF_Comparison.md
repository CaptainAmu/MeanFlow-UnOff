# DMD2_OT vs MF_DMD_CIFAR10: Implementation Comparison

This document compares the DMD distillation implementations in **DMD2_OT** (EDM-based, `demo/cifar10_distill.py`) and **MF_DMD_CIFAR10** (Flow Matching-based) to identify differences that may cause the student to generate images not alike the teacher.

---

## 1. Distribution Matching (DM) Gradient

### DMD2_OT (EDM)
- **Noise**: `x_t = x0 + σ_t·ε` (additive, Karras σ schedule)
- **α_t**: `α_t = 1` (EDM uses sigma-only, no alpha_bar)
- **Weight**: `denom = ‖pred_real - x0‖₁ + ε`, `weight = (C·S) / denom`
- **Clamp**: `weight = clamp(weight, max=100)` ← **prevents gradient explosion when pred_real ≈ x0**
- **Coef**: `coef = (pred_real - pred_fake) * weight` (no α_t factor)
- **Loss**: `loss = (coef.detach() * x0).mean()`

### MF_DMD_CIFAR10 (Flow Matching)
- **Noise**: `x_t = (1-t)x0 + t·ε` (flow interpolation)
- **α_t**: `α_t = 1 - t` (Reflow)
- **Weight**: `weight_factor = mean(|p_real|) + ε` (uses **mean**, not sum)
- **Clamp**: **None** ← when `|p_real|` is small, gradient can explode
- **Grad**: `grad = α_t * (p_real - p_fake) / weight_factor`
- **Loss**: `loss = 0.5 * MSE(x0, (x0 - grad).detach())`

### Differences
| Item | DMD2_OT | MF_DMD_CIFAR10 |
|------|---------|----------------|
| α_t in gradient | No (α_t=1) | Yes (α_t=1-t) |
| Weight normalization | `(C·S) / ‖·‖₁` (sum) | `1 / mean(|·|)` (mean) |
| Weight clamp | `max=100` | **None** |
| Scale | C·S=384 (3×128) | No explicit C·S factor |

---

## 2. Fake Score (μ_fake) Denoising Loss

### DMD2_OT
- **Noise**: `x_t = x0 + σ_t·ε`
- **Weight**: `weights = σ_t^{-2} + 1/σ_data²` (SNR + 1/σ_data²)
- **Timestep range**: `[min_step, max_step]` over Karras sigmas (e.g. 2%–98% of 1000 steps)

### MF_DMD_CIFAR10
- **Noise**: `x_t = (1-t)x0 + t·ε`
- **Weight**: `get_denoising_weight()`: `w = (1-t)^{-2}/σ_data² + t^{-2}`, capped at `40/σ_data²`
- **Timestep range**: `t ∈ [0.01, 0.99]` (clamped)

### Differences
- Different noise parameterization (EDM σ vs FM t)
- Different weight formulas; MF uses a cap, DMD2_OT does not
- Both avoid extreme timesteps

---

## 3. GAN / Discriminator

### DMD2_OT (main EDM training)
- **λ_gan**: `gen_cls_loss_weight = 3e-3` (ImageNet)
- **Diffusion GAN**: Optional `diffusion_gan=True` — adds noise to image before D: `x = x + σ·ε`
- Discriminator can see **noisy** images for stability

### MF_DMD_CIFAR10
- **λ_gan**: `lambda_gan = 5e-4` (config) — **lower** than DMD2_OT’s 3e-3
- **Diffusion GAN**: **None** — discriminator always sees **clean** images (`t_clean=0`)
- No noise injection before D

---

## 4. Regression Loss

### DMD2_OT
- **DMD2 removes** the original DMD regression loss (LPIPS matching teacher trajectories)
- README: *"We eliminate the regression loss and the need for expensive dataset construction"*
- Trade-off: stability vs. pointwise teacher–student correspondence

### MF_DMD_CIFAR10
- **No regression loss** at all

### Impact
- Without regression, only **distribution matching** is used
- Distributions can match while the **mapping** G(z)→x0 differs from the teacher
- Same z can produce different images; student is not forced to mimic teacher outputs
- This likely explains images that are "not alike" the teacher

---

## 5. Recommended Changes for MF_DMD_CIFAR10

### High priority

1. **Add weight clamping** in `distribution_matching_loss`:
   ```python
   weight_factor = torch.abs(p_real).mean(...) + eps
   weight_factor = torch.clamp(weight_factor, min=1/100.0)  # or clamp weight
   ```
   Or mirror DMD2_OT: use `(C*S)/denom` with `denom = |p_real|.sum(...) + eps` and `weight = clamp(weight, max=100)`.

2. **Add optional regression loss** (LPIPS or L2) to match teacher outputs:
   - Sample z, get `x0_teacher = teacher_sample(z, c)` (multi-step) and `x0_student = student(z, c)`
   - `L_reg = LPIPS(x0_student, x0_teacher)` or `MSE(x0_student, x0_teacher)`
   - Start with small weight (e.g. 0.01) to avoid destabilizing DM

### Medium priority

3. **Align weight normalization** with DMD2_OT:
   - Use `denom = |p_real|.sum(dim=[1,2,3], keepdim=True) + eps`
   - Use `weight = (C*S) / denom` with C=3, S=128 (or similar)
   - Add `weight = clamp(weight, max=100)`

4. **Revisit λ_gan**: Try `lambda_gan = 3e-3` to match DMD2_OT.

5. **Optional Diffusion GAN**: Add noise to images before the discriminator to improve stability.

### Lower priority

6. **α_t scaling**: For Reflow, α_t=1-t may be correct for flow matching. If results stay off, consider ablating α_t (e.g. set to 1) to match EDM behavior.

---

## 6. Summary Table

| Component | DMD2_OT | MF_DMD_CIFAR10 |
|-----------|---------|----------------|
| DM weight clamp | Yes (max=100) | No |
| DM weight norm | (C·S)/‖·‖₁ | 1/mean(‖·‖) |
| α_t in DM grad | No | Yes |
| λ_gan | 3e-3 | 5e-4 |
| Diffusion GAN | Optional | No |
| Regression loss | Removed in DMD2 | None |
| Fake score weight | σ⁻² + 1/σ_data² | EDM-style + cap |
