FROM --platform=linux/arm64 python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Phase 3: agent-invoker + agent_action
COPY . .

EXPOSE 8080

CMD ["opentelemetry-instrument", "python", "main.py"]
