"""Convert research/PAPER.md to a pandoc-ready markdown (ASCII math) and
then to research/PAPER.tex. Full pipeline:

    python3 research/make_tex.py
    pandoc -f markdown+tex_math_single_backslash research/.PAPER_clean.md \
        -o research/PAPER.tex -s --shift-heading-level-by=-1 \
        -V geometry:margin=1in -V colorlinks=true -V linkcolor=blue -V urlcolor=blue
    sed -i '' 's/\\{\\}/{}/g' research/PAPER.tex   # un-escape inserted {} groups
    cd research && tectonic PAPER.tex
"""

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
src = (HERE / "PAPER.md").read_text()

lines = src.split("\n")
title = lines[0].lstrip("# ").strip()
body = "\n".join(lines[1:])

# Ordered replacements: composite tokens first, single glyphs last.
R = [
    ("λ̂₊", r"\(\hat{\lambda}_+\)"),
    ("σ*_upper", r"\(\sigma^*_{\mathrm{upper}}\)"),
    ("u_i v_iᵀ", r"\(u_i v_i^{\mathsf{T}}\)"),
    ("cosθ_u", r"\(\cos\theta_u\)"),
    ("cosθ_v", r"\(\cos\theta_v\)"),
    ("τ_GD", r"\(\tau_{\mathrm{GD}}\)"),
    ("σ_min", r"\(\sigma_{\min}\)"),
    ("t*(σ)", r"\(t^*(\sigma)\)"),
    ("ρ(σ)", r"\(\rho(\sigma)\)"),
    ("τ(σ)", r"\(\tau(\sigma)\)"),
    ("τ_i", r"\(\tau_i\)"),
    ("σ_i", r"\(\sigma_i\)"),
    ("σ*", r"\(\sigma^*\)"),
    ("λ₊", r"\(\lambda_+\)"),
    ("ℝ^{m×n}", r"\(\mathbb{R}^{m\times n}\)"),
    ("ℝ", r"\(\mathbb{R}\)"),
    ("σ^p", r"\(\sigma^p\)"),
    ("3^k", r"\(3^k\)"),
    ("√(smoothing\nradius)", r"\(\sqrt{\text{smoothing radius}}\)"),
    ("ℓ", r"\(\ell\)"),
    ("√w", r"\(\sqrt{w}\)"),
    ("√m", r"\(\sqrt{m}\)"),
    ("√n", r"\(\sqrt{n}\)"),
    ("√B", r"\(\sqrt{B}\)"),
    ("√2", r"\(\sqrt{2}\)"),
    ("ŝ", r"\(\hat{s}\)"),
    ("ŵ", r"\(\hat{w}\)"),
    ("ᵀ", r"\(^{\mathsf{T}}\)"),
    ("²", r"\(^2\)"),
    ("⁴", r"\(^4\)"),
    ("₊", r"\(_+\)"),
    ("σ", r"\(\sigma\)"),
    ("ρ", r"\(\rho\)"),
    ("τ", r"\(\tau\)"),
    ("λ", r"\(\lambda\)"),
    ("θ", r"\(\theta\)"),
    ("β", r"\(\beta\)"),
    ("Φ", r"\(\Phi\)"),
    ("Σ", r"\(\Sigma\)"),
    ("Δ", r"\(\Delta\)"),
    ("≈", r"\(\approx\)"),
    ("≲", r"\(\lesssim\)"),
    ("≤", r"\(\leq\)"),
    ("≥", r"\(\geq\)"),
    ("≠", r"\(\neq\)"),
    ("≫", r"\(\gg\)"),
    ("≡", r"\(\equiv\)"),
    ("∝", r"\(\propto\)"),
    ("∩", r"\(\cap\)"),
    ("±", r"\(\pm\)"),
    ("·", r"\(\cdot\)"),
    ("→", r"\(\to\)"),
    ("⇒", r"\(\Rightarrow\)"),
    ("✓", r"\(\checkmark\)"),
    ("✗", r"\(\times\)"),
    ("×", r"\(\times\)"),
    ("−", "-"),
    ("…", "..."),
]

for old, new in R:
    body = body.replace(old, new)

# bare t* — but not the "t*" inside emphasis like *not*
body = re.sub(r"(?<![A-Za-z])t\*", r"\(t^*\)", body)

# pandoc refuses \(...\) math whose closing \) is immediately followed by
# an alphanumeric; an empty group keeps it parseable without adding space.
body = re.sub(r"\\\)(?=[A-Za-z0-9])", r"\){}", body)

# vector figures for the PDF build
body = body.replace(".png)", ".pdf)")

leftover = sorted({c for c in body if ord(c) > 127})
if leftover:
    print("WARNING non-ASCII left:", " ".join(leftover), file=sys.stderr)

header = f"""---
title: "{title}"
date: "Working draft v3 --- 2026-08-25"
---

"""
(HERE / ".PAPER_clean.md").write_text(header + body)
print("wrote", HERE / ".PAPER_clean.md")
