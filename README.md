# transfer-insight-agent

Transfer analytics assistant on GreenNode AgentBase. Loads transfer CSV into RAM (pandas), pre-computes executive metrics at startup, and answers questions via 50 LangChain tools + LLM brain.

## Data schema

```csv
order_no,app_id,sender_id,peer_id,product_code,amount,order_time
```

- `order_time`: epoch milliseconds
- Active user = `sender_id`, recipient = `peer_id`

Place your file at `data/transfers.csv` (supports up to ~100k rows in RAM).

## Metrics

**Pre-computed at startup:** TPV, DAU/WAU/MAU, DAU/MAU ratio, R7/R30 transfer retention, D1/D7/D30 retention, churn, repeat sender rate, trends, product/app breakdowns.

**R7/R30:** % of all senders who made a 2nd transfer within 7 / 30 days of their first transfer.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Configure LLM via /agentbase-llm or set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
```

## Run locally

```bash
python main.py
```

- Dashboard: http://127.0.0.1:8080/
- Chat UI: http://127.0.0.1:8080/chat
- API: `POST /api/chat` with `{"message": "..."}`
- AgentBase: `POST /invocations` with `{"message": "..."}`
- Zalo webhook: `POST /webhooks/zalo`

## Example questions

- "What is TPV and transfer count?"
- "How many unique senders?"
- "R7 and R30 transfer retention?"
- "Total transactions of user 160516000000516"
- "Executive dashboard summary"
- "Senders with more than 3 peers"

## Deploy

Use `/agentbase-deploy` to build Docker image and create a Custom Agent runtime.

## Environment

| Variable | Description |
|----------|-------------|
| `TRANSFERS_CSV_PATH` | Path to CSV (default `data/transfers.csv`) |
| `MAX_ROWS` | Optional row cap for dev |
| `LLM_*` | OpenAI-compatible LLM config |
