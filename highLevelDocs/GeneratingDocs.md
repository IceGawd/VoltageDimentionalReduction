# How to Generate the MKDocs Documentation

We currently do not have GitHub Actions set up for automatic documentation generation. The best way to generate and serve the documentation is to do it locally using the following steps.

---

## Step 1: Install Required Dependencies

To build the documentation locally, you will need:

- Python 3.7 or higher
- `pip` (Python package manager)
- `mkdocs`
- Any additional MKDocs plugins your project may be using (e.g., `mkdocs-material`, `mkdocs-mermaid2-plugin`, etc.)

You can install the basic dependencies with:

```bash
pip install mkdocs
pip install mkdocs-material
pip install mkdocstrings[python]
```

## Step 2: Preview the Docs Locally

To serve the docs locally and preview changes in real-time:

```bash
mkdocs serve
```

This will start a local development server (usually at http://127.0.0.1:8000/). Open that URL in your browser to see your docs.

## Step 3: Deploy to Github Pages

```bash
mkdocs gh-deploy
```

This will push the contents of the ``site/`` folder to the ``gh-pages`` branch of the repository which will automatically be displayed in the [Github Pages](https://icegawd.github.io/VoltageDimentionalReduction/reference/)