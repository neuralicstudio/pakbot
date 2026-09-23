FROM python:3.11-slim

# Enable bytecode compilation and copy-mode linking for uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Install dependencies from lockfile (no project install, no dev deps)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Copy application code
COPY ./bot.py bot.py
COPY ./persona.py persona.py

EXPOSE 7860

CMD ["uv", "run", "python", "bot.py", "--host", "0.0.0.0", "--port", "7860"]
