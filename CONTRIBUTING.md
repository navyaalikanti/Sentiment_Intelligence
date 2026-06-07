# Contributing to Sentiment Intelligence Dashboard

Thank you for helping improve this project.

## Development Guidelines

1. Create a feature branch from `main`.
2. Keep changes focused and avoid unrelated refactors.
3. Do not commit secrets, local credentials, or environment files.
4. Run the backend and frontend checks before opening a pull request.
5. Update documentation when behavior, routes, or setup steps change.

## Recommended Checks

- Backend tests: `python -m unittest`
- Backend syntax check: `python -m py_compile main.py sentiment_analyzer.py roberta_model.py visualization.py backend/ml_model.py`
- Frontend build: `npm run build` from the `frontend` directory

## Pull Request Expectations

- Describe what changed and why.
- Mention any data or model artifacts affected.
- Include screenshots when UI changes are involved.
- Call out any compatibility or migration notes.

## Security

- Never commit API keys, tokens, passwords, or private certificates.
- Use `.env` locally and keep `.env.example` updated for shared configuration.

