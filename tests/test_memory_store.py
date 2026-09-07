from __future__ import annotations

import tempfile
import unittest

from coding_agent.memory_store import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def test_persists_and_retrieves_project_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(tmp)
            item = store.add("项目使用 pytest -q 运行测试", memory_type="project_fact")
            self.assertTrue(item.memory_id.startswith("memory_"))
            results = store.search("如何运行项目测试", limit=3, min_score=-1)
            self.assertEqual(results[0][0].memory_id, item.memory_id)
            self.assertIn("pytest", store.format_results("项目测试命令", limit=3))

    def test_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(tmp)
            item = store.add("keep this fact")
            self.assertTrue(store.delete(item.memory_id))
            self.assertFalse(store.delete(item.memory_id))

    def test_backend_can_be_injected_without_model_dependencies(self) -> None:
        class FakeBackend:
            name = "fake-embedding"

            def encode(self, texts, *, query=False):
                return [[1.0, 0.0] if (query or texts[0].startswith("semantic")) else [0.0, 1.0] for _ in texts]

        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(tmp)
            store.embedding_backend = FakeBackend()
            item = store.add("semantic project policy")
            self.assertEqual(item.embedding_model, "fake-embedding")
            self.assertEqual(store.search("anything", min_score=-1)[0][0].memory_id, item.memory_id)


if __name__ == "__main__":
    unittest.main()
