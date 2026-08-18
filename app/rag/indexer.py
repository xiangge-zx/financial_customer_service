"""从 knowledge/ 构建或加载 FAISS 向量索引。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

if TYPE_CHECKING:
    from app.config import Settings

SUPPORTED_GLOBS = ("**/*.md", "**/*.txt")
MANIFEST_NAME = "manifest.json"
INDEX_NAME = "index"


def _create_embeddings(model_name: str) -> Embeddings:
    import os

    from langchain_huggingface import HuggingFaceEmbeddings

    # 即使 hub 常量已在 import 时读过环境变量，也强制只读本地缓存，避免 HEAD 探测外网。
    offline = os.getenv("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes"}
    model_kwargs = {"local_files_only": True} if offline else {}
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)


def _list_source_files(knowledge_dir: Path) -> list[Path]:
    if not knowledge_dir.is_dir():
        return []

    files: set[Path] = set()
    for pattern in SUPPORTED_GLOBS:
        files.update(path for path in knowledge_dir.glob(pattern) if path.is_file())
    return sorted(files)


def _file_fingerprint(path: Path) -> dict[str, float | int | str]:
    stat = path.stat()
    return {
        "path": str(path),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
    }


def _build_manifest(
    *,
    knowledge_dir: Path,
    source_files: list[Path],
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
) -> dict:
    return {
        "knowledge_dir": str(knowledge_dir),
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "files": [_file_fingerprint(path) for path in source_files],
        "content_hash": hashlib.sha256(
            json.dumps(
                [_file_fingerprint(path) for path in source_files],
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _manifest_path(index_dir: Path) -> Path:
    return index_dir / MANIFEST_NAME


def _load_manifest(index_dir: Path) -> dict | None:
    path = _manifest_path(index_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save_manifest(index_dir: Path, manifest: dict) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    _manifest_path(index_dir).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _index_is_fresh(index_dir: Path, expected_manifest: dict) -> bool:
    existing = _load_manifest(index_dir)
    if existing is None:
        return False
    faiss_file = index_dir / f"{INDEX_NAME}.faiss"
    pkl_file = index_dir / f"{INDEX_NAME}.pkl"
    if not faiss_file.is_file() or not pkl_file.is_file():
        return False
    return (
        existing.get("content_hash") == expected_manifest.get("content_hash")
        and existing.get("embedding_model") == expected_manifest.get("embedding_model")
        and existing.get("chunk_size") == expected_manifest.get("chunk_size")
        and existing.get("chunk_overlap") == expected_manifest.get("chunk_overlap")
    )


def _load_documents(knowledge_dir: Path) -> list:
    documents = []
    for pattern in SUPPORTED_GLOBS:
        loader = DirectoryLoader(
            str(knowledge_dir),
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
            use_multithreading=False,
        )
        documents.extend(loader.load())
    return documents


def _split_documents(
    documents: list,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def _create_empty_store(embeddings: Embeddings) -> FAISS:
    return FAISS.from_texts(
        texts=["知识库暂无可用文档。"],
        embedding=embeddings,
        metadatas=[{"source": "empty"}],
    )


def build_or_load_vector_store(
    settings: Settings,
    *,
    embeddings: Embeddings | None = None,
) -> FAISS:
    """构建或复用 FAISS 索引；知识库为空时返回可检索的占位索引。"""
    knowledge_dir = settings.resolve_knowledge_dir()
    index_dir = settings.resolve_index_dir()
    source_files = _list_source_files(knowledge_dir)
    expected_manifest = _build_manifest(
        knowledge_dir=knowledge_dir,
        source_files=source_files,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        embedding_model=settings.embedding_model,
    )

    resolved_embeddings = embeddings or _create_embeddings(settings.embedding_model)

    if _index_is_fresh(index_dir, expected_manifest):
        return FAISS.load_local(
            str(index_dir),
            resolved_embeddings,
            index_name=INDEX_NAME,
            allow_dangerous_deserialization=True,
        )

    if not source_files:
        store = _create_empty_store(resolved_embeddings)
        index_dir.mkdir(parents=True, exist_ok=True)
        store.save_local(str(index_dir), index_name=INDEX_NAME)
        _save_manifest(index_dir, expected_manifest)
        return store

    documents = _load_documents(knowledge_dir)
    chunks = _split_documents(
        documents,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    if not chunks:
        store = _create_empty_store(resolved_embeddings)
    else:
        store = FAISS.from_documents(chunks, resolved_embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir), index_name=INDEX_NAME)
    _save_manifest(index_dir, expected_manifest)
    return store
