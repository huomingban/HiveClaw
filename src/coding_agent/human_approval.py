from __future__ import annotations

"""Human-in-the-loop approval for high-risk tool calls."""

import asyncio
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from agent_core.types import BeforeToolCallContext, BeforeToolCallResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ApprovalRequest:
    confirmation_id: str
    session_id: str | None
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    risk_level: str
    reason: str
    status: str = "pending"
    requested_at: str = ""
    decided_at: str | None = None
    decided_by: str | None = None
    decision_reason: str | None = None


class HumanApprovalManager:
    """Async approval gate driven by CLI, IM, or an HTTP UI."""

    def __init__(self, *, session_id: str | None = None, timeout_seconds: float = 300) -> None:
        self.session_id = session_id
        self.timeout_seconds = timeout_seconds
        self._pending: dict[str, ApprovalRequest] = {}
        self._waiters: dict[str, asyncio.Future[bool]] = {}
        self._audit: Callable[[dict[str, Any]], None] | None = None
        self._request_handler: Callable[[dict[str, Any]], bool | Awaitable[bool]] | None = None

    def attach_audit_sink(self, sink: Callable[[dict[str, Any]], None]) -> None:
        self._audit = sink

    def attach_request_handler(self, handler: Callable[[dict[str, Any]], bool | Awaitable[bool]]) -> None:
        self._request_handler = handler

    def pending(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self._pending.values() if item.status == "pending"]

    def _record(self, event: str, request: ApprovalRequest) -> None:
        if self._audit:
            self._audit({"type": event, **asdict(request)})

    @staticmethod
    def assess(ctx: BeforeToolCallContext) -> tuple[str, str] | None:
        if ctx.tool_call.name in {"write", "write_file", "edit", "edit_file"}:
            return "medium", "file content will be changed"
        if ctx.tool_call.name in {"bash", "shell", "run_command"}:
            command = str(ctx.args.get("command", ""))
            if re.search(r"\b(rm|del|rmdir|format|drop|truncate|shutdown|git\s+reset)\b", command, re.I):
                return "critical", "command may be destructive or irreversible"
            return "high", "a shell command will be executed"
        return None

    async def before_tool_call(self, ctx: BeforeToolCallContext, signal: Any | None = None) -> BeforeToolCallResult | None:
        _ = signal
        risk = self.assess(ctx)
        if risk is None:
            return None
        risk_level, reason = risk
        confirmation_id = f"confirm_{uuid.uuid4().hex[:12]}"
        request = ApprovalRequest(confirmation_id, self.session_id, ctx.tool_call.id, ctx.tool_call.name,
                                  dict(ctx.args), risk_level, reason, requested_at=_now())
        waiter: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending[confirmation_id] = request
        self._waiters[confirmation_id] = waiter
        self._record("approval_requested", request)
        if self._request_handler is not None:
            decision = self._request_handler(asdict(request))
            if asyncio.iscoroutine(decision) or asyncio.isfuture(decision):
                decision = await decision
            self.decide(confirmation_id, approved=bool(decision))
        try:
            approved = await asyncio.wait_for(waiter, timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            request.status = "expired"
            request.decided_at = _now()
            self._record("approval_expired", request)
            return BeforeToolCallResult(block=True, reason=f"Approval expired: {confirmation_id}")
        finally:
            self._waiters.pop(confirmation_id, None)
        if approved:
            return None
        return BeforeToolCallResult(block=True, reason=f"Tool call rejected: {confirmation_id}")

    def decide(self, confirmation_id: str, *, approved: bool, decided_by: str = "user", reason: str = "") -> dict[str, Any]:
        request = self._pending.get(confirmation_id)
        if request is None:
            raise KeyError(f"Unknown confirmation id: {confirmation_id}")
        if request.status != "pending":
            raise ValueError(f"Confirmation is already {request.status}")
        request.status = "approved" if approved else "rejected"
        request.decided_at = _now()
        request.decided_by = decided_by
        request.decision_reason = reason or None
        self._record("approval_decided", request)
        waiter = self._waiters.get(confirmation_id)
        if waiter and not waiter.done():
            waiter.set_result(approved)
        return asdict(request)
