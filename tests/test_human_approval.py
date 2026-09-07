from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from ai.models import get_model
from ai.types import ToolCall
from agent_core.types import AgentContext, AssistantMessage, BeforeToolCallContext
from coding_agent.human_approval import HumanApprovalManager
from coding_agent.session_store import SessionStore


class HumanApprovalTests(unittest.IsolatedAsyncioTestCase):
    async def test_approval_blocks_until_decision_and_audits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(tmp, "s1")
            store.ensure_initialized(model_id="m", provider="p", system_prompt="")
            manager = HumanApprovalManager(session_id="s1", timeout_seconds=1)
            manager.attach_audit_sink(store.append_event)
            ctx = BeforeToolCallContext(
                assistant_message=AssistantMessage(),
                tool_call=ToolCall(id="tc1", name="bash", arguments={"command": "pytest -q"}),
                args={"command": "pytest -q"},
                context=AgentContext(system_prompt="", messages=[]),
            )
            task = asyncio.create_task(manager.before_tool_call(ctx))
            await asyncio.sleep(0)
            pending = manager.pending()
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["risk_level"], "high")
            manager.decide(pending[0]["confirmation_id"], approved=True, decided_by="test")
            self.assertIsNone(await task)
            events = store.events_file.read_text(encoding="utf-8")
            self.assertIn("approval_requested", events)
            self.assertIn("approval_decided", events)

    async def test_rejection_blocks_tool(self) -> None:
        manager = HumanApprovalManager(timeout_seconds=1)
        ctx = BeforeToolCallContext(
            assistant_message=AssistantMessage(),
            tool_call=ToolCall(id="tc1", name="write", arguments={"path": "a.txt"}),
            args={"path": "a.txt"},
            context=AgentContext(system_prompt="", messages=[]),
        )
        task = asyncio.create_task(manager.before_tool_call(ctx))
        await asyncio.sleep(0)
        cid = manager.pending()[0]["confirmation_id"]
        manager.decide(cid, approved=False)
        result = await task
        self.assertTrue(result and result.block)


if __name__ == "__main__":
    unittest.main()
