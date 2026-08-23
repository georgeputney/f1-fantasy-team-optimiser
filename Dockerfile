# builds the React frontend, then serves it (as static files) plus the API from a single
# Python/uvicorn process - see api/main.py's WEB_DIST mount
FROM node:20-slim AS web-build
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm install
COPY web/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /srv

COPY pyproject.toml ./
COPY app/ ./app/
COPY api/ ./api/
COPY data/ ./data/
COPY reports/ ./reports/
RUN pip install --no-cache-dir -e ".[api]"

COPY --from=web-build /web/dist ./web/dist

EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
