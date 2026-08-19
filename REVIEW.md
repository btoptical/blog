# Quarto template cleanup review

This bundle contains a side-by-side copy of the current core files and a proposed cleanup.  It is designed for Quarto 1.10.18 or later.

## Main changes

1. Give the site a real semantic title while suppressing the duplicate navbar title.
2. Add site description, favicon, Open Graph metadata, Twitter Card metadata, RSS, and `llms.txt` output.
3. Replace the three overlapping homepage-listing CSS implementations with one supported custom listing template.
4. Reduce the navbar logo from 100 px to 56 px and remove the 2 rem artificial top margin.
5. Make custom colors work in both Quarto light and dark modes by keying them to `body.quarto-dark`.
6. Remove global execution, table-of-contents, code-copy, MathJax, and JavaScript smooth-scrolling settings that are unnecessary on every page.
7. Load the existing `print.css`, which is currently present but unused.
8. Remove the inactive sub-navigation include and its stale `/writing.html` link.
9. Use `description` and `image-alt` in post metadata for listings, feeds, social cards, and accessibility.
10. Clean up the contact form markup and add intentional form styling.
11. Align the footer license with `LICENSE.md` and `README.md`, which both say CC BY 4.0.

## Apply

Review the unified patch first:

```bash
git apply --check quarto-template-cleanup.patch
git apply quarto-template-cleanup.patch
```

Then upgrade Quarto and render:

```bash
quarto --version
quarto render
quarto preview
```

The proposed configuration uses `llms-txt`, so Quarto 1.9 or newer is required.  Quarto 1.10.18 is the current stable release used as the target for this review.

## Deliberately not changed

- The light and dark Bootstrap themes remain `default` and `darkly`.
- The article prose is unchanged except for removing the redundant plain-text “About Infinite Limit” line.
- Search is disabled for the current one-post site.  Re-enable it with `navbar: search: true` when the archive is large enough to justify it.
- A dedicated 1200 × 630 PNG social card is not included.  That would make link previews more reliable than relying on per-post SVG images.

## Preview files

- `preview-light.png`
- `preview-dark.png`
- `preview-mobile.png`

These are static HTML mockups of the proposed CSS and listing structure, not full Quarto renders.

## Validation note

The YAML files were parsed locally, and the CSS was exercised against a static HTML mockup.  A full Quarto render was not run in this environment because the Quarto CLI binary was unavailable here.  Run `quarto render` locally before committing the generated `docs/` output.
