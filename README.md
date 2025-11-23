# Mini Panacea Reproduction – Chat 3-Dimensional SFT (Simplified)

This repository contains a scaled-down reproduction of one experiment from the paper:

**Panacea: Parameter-Efficient Multi-Objective Alignment**  
https://arxiv.org/pdf/2402.02030

The original paper introduces **Panacea**, a method that allows one model to smoothly adjust behavior along multiple objectives using a **preference vector** and an **SVD-LoRA adapter**. This notebook reproduces the **Chat 3-Dimensional SFT experiment** from Section 5.3 in a lightweight, Colab-friendly form.

---

## ✔️ Experiment Reproduced: Chat 3-Dimensional SFT

The Panacea paper trains Llama-3-8B-Instruct to respond in different styles (helpful, humorous, philosophical) using:

- A **3-dim preference vector** λ = [λ_helpful, λ_humorous, λ_philosophical]
- A **Panacea SVD-LoRA** module inserted across model layers
- A multi-style SFT dataset generated from Alpaca instructions

This notebook recreates the core idea using a much lighter pipeline:

### What this reproduction preserves

- Preference-conditioned SVD-LoRA  
- 3-dimensional preference vector λ  
- Multi-style SFT with Alpaca prompts  
- A single model producing multiple behaviors  
- Smooth interpolation between styles when λ changes  

### What is simplified for Colab

| Component         | Original Paper (Full Scale)            | This Reproduction              |
| ----------------- | -------------------------------------- | ------------------------------ |
| Base model        | Llama-3-8B-Instruct                    | Phi-3 Mini 4k Instruct (~3.8B) |
| Panacea insertion | All attention + MLP layers             | Final LM head only             |
| Data              | Llama-3 generated 3–10 style responses | Template-generated 3 styles    |
| Compute           | Multi-GPU (8× A800-80GB)               | Google Colab T4 / A100         |
| Metrics           | Full MOO metrics                       | Qualitative style comparison   |

Despite these simplifications, the **core Panacea behavior emerges**:  
**changing λ changes the model’s output style without retraining.**

---

## 🚀 Running the Notebook

1. Open the notebook in Google Colab.  
2. Go to `Runtime → Change runtime type → GPU`.  
3. Run all cells top-to-bottom.  
4. Test different preference vectors:

```python
generate_with_pref("Explain black holes to a 10 year old.", [1.0, 0.0, 0.0])  # helpful
generate_with_pref("Explain black holes to a 10 year old.", [0.0, 1.0, 0.0])  # humorous
generate_with_pref("Explain black holes to a 10 year old.", [0.0, 0.0, 1.0])  # philosophical
generate_with_pref("Explain black holes to a 10 year old.", [0.3, 0.3, 0.4])  # mixed
