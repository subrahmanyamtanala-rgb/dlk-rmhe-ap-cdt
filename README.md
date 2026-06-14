# DLK-RMHE for Cyber-Resilient Artificial Pancreas Digital Twins

**Deep-Learned Kernel Robust Moving Horizon Estimation for Cyber-Resilient Closed-Loop Artificial Pancreas Digital Twins Under Composite FDI/DoS/Replay Attacks**

*Subrahmanyam Tanala, Member, IEEE*
Department of Electrical and Electronics Engineering, ANITS, Visakhapatnam, India
Department of AI Convergence, Woosong University, Daejeon, South Korea

---

## Abstract

This repository contains the paper source, figures, and simulation code for the DLK-RMHE framework — a cyber-resilient state estimator for artificial pancreas (AP) clinical digital twin (CDT) systems. The framework integrates a contrastively-trained deep kernel network within a sparse L1-regularized Moving Horizon Estimator (MHE) to achieve provably resilient glucose state estimation under simultaneous **False Data Injection (FDI)**, **Denial-of-Service (DoS)**, and **replay attacks** on the CGM sensor channel.

### Key Results (UVa/Padova T1DM Simulator, 300 Monte Carlo runs)

| Metric | DLK-RMHE (V4) | EKF (V1) | Reduction |
|---|---|---|---|
| MAGE under composite attack | **4.2 ± 1.1 mg/dL** | 31.7 ± 6.4 mg/dL | **86.8%** |
| Time-in-Range (TIR) | **78.6 ± 4.3%** | 52.3 ± 8.1% | +26.3 pp |
| Time-in-Hypoglycemia (TIH) | **2.1 ± 0.9%** | 11.4 ± 3.2% | −9.3 pp |
| Attack Detection AUROC | **0.983 ± 0.008** | — | — |
| Compute per CGM step | **183 ± 24 ms** | — | 3.1% of 5-min interval |

---

## Repository Structure

```
dlk-rmhe-ap-cdt/
├── paper/
│   ├── main.tex                  # IEEEtran LaTeX source (600 lines)
│   ├── refs.bib                  # BibTeX database (61 references)
│   ├── DLK_RMHE_AP_CDT.pdf       # Compiled manuscript (6 pages)
│   └── figures/
│       ├── fig1_architecture.png  # AP-CDT four-layer architecture
│       ├── fig2_kernel.png        # Deep kernel training workflow + t-SNE
│       ├── fig3_framework.png     # DLK-RMHE block diagram
│       ├── fig4_attacks.png       # Attack scenario timelines
│       ├── fig5_trajectories.png  # 72h glucose trajectories (AS#07, Scenario C)
│       ├── fig6_roc.png           # ROC curves + detection latency
│       └── fig7_timing.png        # Computational timing profile
├── simulation/
│   └── gen_figures.py            # Matplotlib figure generation script
├── supplementary/
│   └── (Monte Carlo seed tables and per-subject logs — to be added upon acceptance)
└── README.md
```

---

## Reproducing the Paper Figures

```bash
pip install matplotlib numpy scipy
python simulation/gen_figures.py
# Outputs: paper/figures/fig1_*.png through fig7_*.png
```

## Compiling the LaTeX

```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Requires: `texlive-publishers` (IEEEtran), `texlive-science` (algorithm), `texlive-latex-extra`.

---

## Deep Kernel Network Specification

| Layer | In | Out | Activation | Dropout | Batch Norm |
|---|---|---|---|---|---|
| FC-1 | 24 | 128 | ReLU | 0.20 | Yes |
| FC-2 | 128 | 64 | ReLU | 0.20 | Yes |
| FC-3 | 64 | 32 | ReLU | 0.20 | No |
| FC-4 | 32 | 16 | Linear | — | No |

**Total parameters:** 14,992 | **Training:** AdamW, lr=1e-3→1e-5 (cosine), 200 epochs, batch=256, PyTorch 2.1, RTX 3090 (~47 min)

---

## ISS Stability Guarantee

Under Assumptions A1–A3 (uniform detectability δ_d=0.18, Rademacher kernel approximation ε_max≤0.797, bounded T_max=18-sparse attacks ā=2.5 mmol/L), the estimation error satisfies:

```
‖e_k‖ ≤ ‖e_0‖·exp(−δ_d·k/2) + γ_a·sup‖aᵢ‖ + γ_ε·sup‖εᵢ‖
```

**Conservative theoretical bounds:** γ_a = 6.08, γ_ε = 2.20
**Empirical gain:** γ_a^emp ≈ 1.7 (from 300 Monte Carlo runs)

Proof uses BPDN stable recovery (Candès & Tao 2006) with explicit constants C₁=2.0, δ_s=0 (orthogonal single-channel CGM operator), recovery bound ‖ê_k−a_k‖≤3.32 mmol/L.

---

## Citation

```bibtex
@article{tanala2025dlkrmhe,
  author  = {Tanala, Subrahmanyam},
  title   = {Deep-Learned Kernel Robust Moving Horizon Estimation for
             Cyber-Resilient Closed-Loop Artificial Pancreas Digital Twins
             Under Composite {FDI/DoS/Replay} Attacks},
  journal = {IEEE Transactions on Biomedical Engineering},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

This repository is made available for academic reproducibility purposes.
© 2025 Subrahmanyam Tanala. All rights reserved.
