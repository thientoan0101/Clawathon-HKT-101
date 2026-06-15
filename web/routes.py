"""SSR web routes and REST chat API."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from agent.chat_service import handle_message
from analytics.store import get_store

TEMPLATES_DIR = Path(__file__).parent / "templates"
env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=select_autoescape(["html"]))


def _dashboard_context() -> dict:
    store = get_store()
    p = store.precomputed
    return {
        "stats": {
            "tpv": p["transfer"]["tpv"],
            "transfer_count": p["transfer"]["transfer_count"],
            "unique_senders": p["global"]["unique_senders"],
            "dau": p["activity"]["latest_dau"],
            "mau": p["activity"]["latest_mau"],
            "dau_mau": p["activity"]["dau_mau_ratio_pct"],
            "r7": p["retention"]["r7_transfer_retention_pct"],
            "r30": p["retention"]["r30_transfer_retention_pct"],
            "d30": p["retention"]["d30_retention_pct"],
            "avg_amount": p["transfer"]["average_transfer_value"],
            "row_count": p["meta"]["row_count"],
        }
    }


async def dashboard_page(request: Request) -> HTMLResponse:
    template = env.get_template("dashboard.html")
    return HTMLResponse(template.render(request=request, **_dashboard_context()))


async def chat_page(request: Request) -> HTMLResponse:
    template = env.get_template("chat.html")
    session_id = request.cookies.get("session_id") or str(uuid.uuid4())
    response = HTMLResponse(template.render(request=request, session_id=session_id))
    response.set_cookie("session_id", session_id, httponly=True, samesite="lax")
    return response


async def api_chat(request: Request) -> JSONResponse:
    body = await request.json()
    message = body.get("message", "").strip()
    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)

    session_id = (
        body.get("session_id")
        or request.headers.get("X-GreenNode-AgentBase-Session-Id")
        or str(uuid.uuid4())
    )
    user_id = body.get("user_id") or request.headers.get("X-GreenNode-AgentBase-User-Id") or "web-user"
    channel = body.get("channel", "web")

    result = handle_message(message, channel=channel, user_id=user_id, session_id=session_id)
    if result.get("status") == "error":
        return JSONResponse(result, status_code=503)
    return JSONResponse(result)


async def zalo_webhook(request: Request) -> JSONResponse:
    """Normalize Zalo OA webhook payload → unified chat handler."""
    body = await request.json()
    # Zalo payload shapes vary; extract message text from common fields
    message = (
        body.get("message", {}).get("text", "")
        if isinstance(body.get("message"), dict)
        else body.get("text", body.get("message", ""))
    )
    if not message:
        return JSONResponse({"status": "ignored", "reason": "no message text"})

    user_id = str(body.get("sender", {}).get("id", body.get("user_id", "zalo-user")))
    session_id = f"zalo-{user_id}"
    result = handle_message(str(message), channel="zalo", user_id=user_id, session_id=session_id)
    return JSONResponse(result)


async def api_dashboard(request: Request) -> JSONResponse:
    """Return all precomputed dashboard data as JSON for the SPA frontend."""
    store = get_store()
    p = store.precomputed
    return JSONResponse({
        "stats": {
            "tpv": p["transfer"]["tpv"],
            "transfer_count": p["transfer"]["transfer_count"],
            "unique_senders": p["global"]["unique_senders"],
            "unique_peers": p["global"]["unique_peers"],
            "avg_amount": p["transfer"]["average_transfer_value"],
            "median_amount": p["global"]["median_amount"],
            "dau": p["activity"]["latest_dau"],
            "mau": p["activity"]["latest_mau"],
            "wau": p["activity"]["latest_wau"],
            "dau_mau": p["activity"]["dau_mau_ratio_pct"],
            "txn_per_user": p["activity"]["transactions_per_active_user"],
            "row_count": p["meta"]["row_count"],
            "date_start": p["meta"]["date_range_start"],
            "date_end": p["meta"]["date_range_end"],
        },
        "retention": {
            "d1": p["retention"]["d1_retention_pct"],
            "d7": p["retention"]["d7_retention_pct"],
            "d30": p["retention"]["d30_retention_pct"],
            "r7": p["retention"]["r7_transfer_retention_pct"],
            "r30": p["retention"]["r30_transfer_retention_pct"],
            "churn": p["retention"]["churn_rate_pct"],
            "repeat_rate": p["retention"]["repeat_sender_rate_pct"],
        },
        "growth": {
            "new_users": p["growth"]["new_users_total"],
            "growth_rate": p["growth"]["user_growth_rate_pct"],
            "new_by_month": p["growth"]["new_users_by_month"],
            "forecast": p["growth"]["forecasting_next_month"],
        },
        "trends": {
            "daily": p["trends"]["daily_volume"],
        },
        "product": p["product"],
        "app": p["app"],
        "activity": {
            "time_of_day": p["activity"]["time_of_day_breakdown"],
            "daily_active": p["activity"]["daily_active_users"],
        },
    })


def register_web_routes(app) -> None:
    """Register SSR and chat API routes on GreenNodeAgentBaseApp."""
    app.add_route("/", dashboard_page, methods=["GET"])
    app.add_route("/chat", chat_page, methods=["GET"])
    app.add_route("/api/chat", api_chat, methods=["POST"])
    app.add_route("/api/dashboard", api_dashboard, methods=["GET"])
    app.add_route("/webhooks/zalo", zalo_webhook, methods=["POST"])
