# Static GitHub Pages Version

This directory contains a fully static version of the Dash app that can be hosted on GitHub Pages.

## Build

From `web/` run:

```bash
python static_site/build_static_site.py
```

This generates:

- `static_site/manifest.json`
- `static_site/figures/*.json`

## Preview locally

From `web/static_site/` run:

```bash
python -m http.server 8000
```

Then open `http://127.0.0.1:8000`.

## Deploy to GitHub Pages

1. Commit `web/static_site/` including generated `manifest.json` and `figures/*.json`.
2. In GitHub repo settings, set Pages source to the folder you publish (for example root/docs or gh-pages branch).
3. If needed, copy `web/static_site/` content into your selected Pages folder.

No Python backend is needed at runtime; all plots are loaded from static JSON files.
