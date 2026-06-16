"""SSR web routes and REST chat API."""

from __future__ import annotations

import hmac
import json
import os
import uuid
from pathlib import Path
from urllib import error, request

from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from agent.chat_service import handle_message
from analytics.store import get_store

TEMPLATES_DIR = Path(__file__).parent / "templates"
env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=select_autoescape(["html"]))
ZALO_BOT_API_BASE_URL = "https://bot-api.zaloplatforms.com"
ZALO_SECRET_TOKEN_HEADER = "x-bot-api-secret-token"


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


def _send_zalo_text(chat_id: str, text: str) -> dict:
    bot_token = os.environ.get("ZALO_BOT_TOKEN", "").strip()
    if not bot_token:
        return {"status": "skipped", "reason": "ZALO_BOT_TOKEN is not configured"}

    payload = {
        "chat_id": chat_id,
        "text": text[:2000],
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{ZALO_BOT_API_BASE_URL}/bot{bot_token}/sendMessage",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=10) as resp:
            response_body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        response_body = exc.read().decode("utf-8")
        return {"status": "error", "http_status": exc.code, "body": response_body}
    except error.URLError as exc:
        return {"status": "error", "error": str(exc.reason)}

    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError:
        parsed = {"raw": response_body}
    return {"status": "sent", "body": parsed}


def _is_authorized_zalo_webhook(request: Request) -> bool:
    expected_token = os.environ.get("ZALO_WEBHOOK_SECRET_TOKEN", "").strip()
    if not expected_token:
        return True

    actual_token = request.headers.get(ZALO_SECRET_TOKEN_HEADER, "").strip()
    return bool(actual_token) and hmac.compare_digest(actual_token, expected_token)


async def zalo_webhook(request: Request) -> JSONResponse:
    """Handle Zalo Bot Platform webhooks, answer with the shared chat brain, then reply in Zalo."""
    """print request log"""
    print(request.method)
    print(request.headers)
    print(request.body)
    if request.method == "GET":
        return JSONResponse({"status": "ok"})

    raw_body = await request.body()
    if not raw_body.strip():
        return JSONResponse({"status": "ok"})

    if not _is_authorized_zalo_webhook(request):
        return JSONResponse({"status": "error", "error": "invalid Zalo webhook secret token"}, status_code=401)

    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        return JSONResponse({"status": "error", "error": "invalid JSON body"}, status_code=400)

    result_data = body.get("result", {})
    event_name = result_data.get("event_name", "")
    zalo_message = result_data.get("message", {})
    message = zalo_message.get("text", "")
    if not message:
        return JSONResponse({
            "status": "ignored",
            "reason": "no message text",
            "event_name": event_name,
        })

    user_id = str(zalo_message.get("from", {}).get("id", "zalo-user"))
    chat_id = str(zalo_message.get("chat", {}).get("id", user_id))
    session_id = f"zalo-{user_id}"
    result = handle_message(str(message), channel="zalo", user_id=user_id, session_id=session_id)
    if result.get("status") == "success":
        result["zalo_delivery"] = _send_zalo_text(chat_id, result.get("reply", ""))
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
    app.add_route("/webhooks", zalo_webhook, methods=["GET", "POST"])
