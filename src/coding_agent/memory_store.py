from __future__ import annotations

"""Project-scoped long-term memory with a dependency-free vector index."""

import hashlib
import json
import math
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokens(text: str) -> list[str]:
    tokens = re.findall(r"[\w./:-]+", text.lower())
    # Character bigrams make local retrieval useful for Chinese project notes.
    for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(chunk[index:index + 2] for index in range(max(0, len(chunk) - 1)))
    return tokens


def _embed(text: str, dimensions: int = 256) -> list[float]:
    vector = [0.0] * dimensions
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0 if digest[4] % 2 else -1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class LocalOnnxEmbeddingBackend:
    """Lazy BGE-small-zh ONNX backend; dependencies are optional."""

    name = "bge-small-zh-v1.5-onnx"
    query_prefix = "为这个句子生成表示以用于检索相关文档："

    def __init__(self, model_dir: str | Path, *, allow_download: bool = False, batch_size: int = 32) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.batch_size = max(1, batch_size)
        model_file = self.model_dir / "onnx" / "model_quantized.onnx"
        required = (self.model_dir / "tokenizer.json", model_file)
        if any(not path.is_file() for path in required) and allow_download:
            from huggingface_hub import hf_hub_download
            self.model_dir.mkdir(parents=True, exist_ok=True)
            for filename in ("tokenizer.json", "onnx/model_quantized.onnx"):
                hf_hub_download("Xenova/bge-small-zh-v1.5", filename=filename,
                                revision="75c43b069aac4d136ba6bc1122f995fedcfd2781",
                                local_dir=self.model_dir)
        if any(not path.is_file() for path in required):
            raise FileNotFoundError("BGE ONNX model is incomplete; set HIVECLAW_EMBEDDING_MODEL_ROOT")
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(model_file), sess_options=session_options,
                                             providers=["CPUExecutionProvider"])
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._np = np

    def encode(self, texts: Sequence[str], *, query: bool = False) -> list[list[float]]:
        values = [self.query_prefix + str(text).strip() if query else str(text).strip() for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(values), self.batch_size):
            encodings = self.tokenizer.encode_batch(values[start:start + self.batch_size])
            feeds: dict[str, Any] = {}
            if "input_ids" in self.input_names:
                feeds["input_ids"] = self._np.asarray([item.ids for item in encodings], dtype=self._np.int64)
            if "attention_mask" in self.input_names:
                feeds["attention_mask"] = self._np.asarray([item.attention_mask for item in encodings], dtype=self._np.int64)
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = self._np.asarray([item.type_ids for item in encodings], dtype=self._np.int64)
            output = self._np.asarray(self.session.run(None, feeds)[0], dtype=self._np.float32)
            values_out = output[:, 0, :] if output.ndim == 3 else output
            norms = self._np.linalg.norm(values_out, axis=1, keepdims=True)
            vectors.extend((values_out / self._np.maximum(norms, 1e-12)).tolist())
        return vectors


@dataclass
class Memory:
    content: str
    memory_type: str = "project_fact"
    scope: str = "workspace"
    source_session_id: str | None = None
    confidence: float = 1.0
    memory_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    embedding: list[float] | None = None
    embedding_model: str = "hash-256"


class MemoryStore:
    def __init__(self, workspace_dir: str | Path) -> None:
        self.path = Path(workspace_dir) / ".hiveclaw" / "memories.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_backend = self._load_embedding_backend()

    @property
    def embedding_model(self) -> str:
        return getattr(self.embedding_backend, "name", "hash-256")

    def _load_embedding_backend(self) -> Any:
        import os
        if os.getenv("HIVECLAW_EMBEDDING_ENABLED", "false").lower() not in {"1", "true", "yes", "on"}:
            return None
        try:
            return LocalOnnxEmbeddingBackend(
                os.getenv("HIVECLAW_EMBEDDING_MODEL_ROOT", str(self.path.parent / "models" / "embeddings")),
                allow_download=os.getenv("HIVECLAW_EMBEDDING_ALLOW_DOWNLOAD", "false").lower() in {"1", "true", "yes", "on"},
            )
        except Exception as exc:
            import logging
            logging.getLogger("hiveclaw.coding_agent.memory").warning("embedding backend unavailable, using hash fallback: %s", exc)
            return None

    def _encode(self, texts: Sequence[str], *, query: bool = False) -> list[list[float]]:
        if self.embedding_backend is not None:
            return self.embedding_backend.encode(texts, query=query)
        return [_embed(text) for text in texts]

    def _load(self) -> list[Memory]:
        if not self.path.exists():
            return []
        return [Memory(**json.loads(line)) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def add(self, content: str, *, memory_type: str = "project_fact", scope: str = "workspace",
            source_session_id: str | None = None, confidence: float = 1.0) -> Memory:
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty")
        now = _now()
        item = Memory(content, memory_type, scope, source_session_id, max(0.0, min(1.0, confidence)),
                      f"memory_{uuid.uuid4().hex[:12]}", now, now, self._encode([content])[0], self.embedding_model)
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
        return item

    def search(self, query: str, *, limit: int = 5, min_score: float = 0.08) -> list[tuple[Memory, float]]:
        q = self._encode([query], query=True)[0]
        scored = []
        for item in self._load():
            vector = item.embedding
            if not vector or item.embedding_model != self.embedding_model:
                vector = self._encode([item.content])[0]
            score = sum(a * b for a, b in zip(q, vector))
            if score >= min_score:
                scored.append((item, score))
        return sorted(scored, key=lambda pair: pair[1], reverse=True)[:max(0, limit)]

    def delete(self, memory_id: str) -> bool:
        memories = self._load()
        kept = [item for item in memories if item.memory_id != memory_id]
        if len(kept) == len(memories):
            return False
        self.path.write_text("".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in kept), encoding="utf-8")
        return True

    def format_results(self, query: str, *, limit: int = 5) -> str:
        return "\n".join(f"- [{item.memory_type} score={score:.2f}] {item.content}" for item, score in self.search(query, limit=limit))
