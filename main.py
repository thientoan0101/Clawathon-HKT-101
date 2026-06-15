"""Application entrypoint: load data, pre-compute, start agent server."""

from __future__ import annotations

import logging
import os
import time

from dotenv import load_dotenv
from greennode_agentbase import GreenNodeAgentBaseApp, PingStatus, RequestContext

from agent.chat_service import handle_message
from agent.llm_config import ensure_env_loaded, llm_config_status
from analytics.loader import load_transfers
from analytics.precompute import precompute_all
from analytics.store import initialize_store, is_ready
from web.routes import register_web_routes

load_dotenv()
ensure_env_loaded()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = GreenNodeAgentBaseApp()
_data_loaded = False


def bootstrap_data() -> None:
    """Load CSV into RAM and pre-compute all dashboard statistics."""
    global _data_loaded
    if _data_loaded:
        return

    max_rows = os.environ.get("MAX_ROWS")
    max_rows_int = int(max_rows) if max_rows else None

    t0 = time.time()
    logger.info("Loading transfer data into RAM...")
    df = load_transfers(max_rows=max_rows_int)
    logger.info("Loaded %s rows in %.1fs", len(df), time.time() - t0)

    t1 = time.time()
    logger.info("Pre-computing statistics...")
    precomputed = precompute_all(df)
    initialize_store(df, precomputed)
    logger.info("Pre-compute done in %.1fs", time.time() - t1)

    _data_loaded = True
    logger.info("DataStore ready. %s transactions in memory.", len(df))


@app.ping
def health_check() -> PingStatus:
    if not is_ready():
        return PingStatus.HEALTHY_BUSY
    return PingStatus.HEALTHY


@app.entrypoint
def handler(payload: dict, context: RequestContext) -> dict:
    """AgentBase SDK entrypoint — same brain as web/Zalo chat."""
    message = payload.get("message", "").strip()
    if not message:
        return {"status": "error", "error": "message is required"}

    return handle_message(
        message,
        channel="api",
        user_id=context.user_id,
        session_id=context.session_id,
    )


def main() -> None:
    bootstrap_data()
    llm_status = llm_config_status()
    if not all(llm_status.values()):
        missing = [k for k, ok in llm_status.items() if not ok]
        logger.warning(
            "LLM not fully configured (%s). Chat will return 503 until .env is updated.",
            ", ".join(missing),
        )
    else:
        logger.info("LLM configuration OK (model=%s)", os.environ.get("LLM_MODEL", ""))
    register_web_routes(app)
    app.run(port=8080, host="0.0.0.0")


if __name__ == "__main__":
    main()
