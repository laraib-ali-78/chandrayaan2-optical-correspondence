---
title: Chandrayaan-2 Optical Image Correspondence
emoji: 🌕
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Multi-modal, Sun-Angle and Scale-Invariant Lunar Image Correspondence (SIH 26166)
---

# Chandrayaan-2 Optical Image Correspondence Pipeline (SIH 26166)

Multi-modal, Sun-Angle, and Scale-Invariant Lunar Image Correspondence and Registration for Chandrayaan-2 (OHRC, TMC-2, IIRS) and NASA LRO (NAC, WAC).

### Key Highlights:
- **Adaptive Matcher Controller:** Dynamic cascade across LightGlue, LoFTR, and RoMa.
- **MAGSAC++ Geometric Verification:** Noise-scale marginalized consensus with spatial 8x8 grid uniformity.
- **Sub-Pixel Refinement:** ECC affine patch alignment with parabolic peak-fit fallback (<0.5 px accuracy).
- **8-Stage Ablation Matrix:** Methodical proof comparing baseline through full controller.

