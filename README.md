# Infinite Limit

This repository contains the source for **infinitelimit.ai** — a personal collection of notes and essays on systems, scaling, architecture, and limits.

The focus is on where theory meets physical constraint: performance ceilings, scaling regimes, memory and bandwidth limits, and the structural properties that shape real systems.

Most writing here is exploratory and technical.

---

## Site

* **Live site:** [https://infinitelimit.ai](https://infinitelimit.ai)
* **Publishing:** Quarto → GitHub Pages
* **Format:** Markdown with LaTeX (MathJax / TikZ)

---

## Structure

```
.
├── _quarto.yml        # Site configuration
├── index.qmd          # Homepage
├── posts/             # Essays and notes
├── images/            # Figures and diagrams
├── CNAME              # Custom domain for GitHub Pages
├── LICENSE             # Content license
└── README.md
```

---

## Writing & Rendering

Local preview:

```bash
quarto preview
```

Render the site:

```bash
quarto render
```

The generated site is published automatically via GitHub Pages.

---

## Scope

Topics may include, but are not limited to:

* Scaling laws and asymptotic behavior
* System architecture and topology
* Memory, bandwidth, and latency constraints
* Performance ceilings and regime changes
* Distributed systems and AI infrastructure

Content reflects ongoing thinking rather than finished conclusions.

---

## License

Unless otherwise noted, content is licensed under
**Creative Commons Attribution 4.0 International (CC BY 4.0)**.

---

## Notes

This repository is intentionally minimal.
Design, tooling, and structure favor durability over novelty.
