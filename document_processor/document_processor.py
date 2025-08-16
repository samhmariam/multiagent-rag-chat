from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.constant import MAX_FILE_SIZE, MAX_TOTAL_SIZE, ALLOWED_TYPES

@dataclass
class ProcessedDocument:
    """Container for a processed document ready for indexing.

    Fields:
    - source_path: Original file path.
    - file_size: Size in bytes.
    - file_hash: SHA256 hash of the file contents.
    - markdown: Extracted markdown content.
    - chunks: List of dicts with keys: content, metadata.
    - meta: Additional processing metadata (e.g., tool versions, params).
    """

    source_path: str
    file_size: int
    file_hash: str
    markdown: str
    chunks: List[Dict[str, Any]]
    meta: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class DocumentProcessor:
    """Handles document validation, parsing (Docling), caching, and chunking.

    Contract:
    - Input: path to a document file (pdf, docx, pptx, md, txt, etc.)
    - Output: ProcessedDocument with markdown and header-aware chunks.
    - Errors: ValueError for size/validation issues; RuntimeError for parse failures.
    """

    def __init__(
        self,
        cache_dir: os.PathLike | str = ".cache/document_processor",
        max_file_size_mb: Optional[int] = None,
        headers_to_split_on: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Use configured constant by default; allow override in MB for convenience
        self.max_file_size_bytes = (
            MAX_FILE_SIZE if max_file_size_mb is None else max_file_size_mb * 1024 * 1024
        )
        self.headers_to_split_on = headers_to_split_on or [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ]

    # -------------------------- Public API ---------------------------
    def process(self, file_path: os.PathLike | str) -> ProcessedDocument:
        """Full pipeline: validate -> cache hit? -> parse -> chunk -> cache -> return."""
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise ValueError(f"File not found: {path}")

        # Validate allowed file types (normalize .markdown to .md)
        ext = path.suffix.lower()
        if ext == ".markdown":
            ext = ".md"
        if ext not in ALLOWED_TYPES:
            allowed = ", ".join(ALLOWED_TYPES)
            raise ValueError(f"Unsupported file type '{path.suffix}'. Allowed: {allowed}")

        file_size = path.stat().st_size
        self._validate_file_size(file_size)

        file_hash = self._hash_file(path)
        cache_key = self._cache_key(file_hash)
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        markdown = self._extract_markdown_with_docling(path)
        chunks = self._chunk_markdown(markdown)

        processed = ProcessedDocument(
            source_path=str(path.resolve()),
            file_size=file_size,
            file_hash=file_hash,
            markdown=markdown,
            chunks=chunks,
            meta={
                "created_at": datetime.now(timezone.utc).isoformat(),
                "headers_to_split_on": self.headers_to_split_on,
                "max_file_size_bytes": self.max_file_size_bytes,
                "tooling": {
                    "docling": self._safe_pkg_version("docling"),
                    "langchain-text-splitters": self._safe_pkg_version(
                        "langchain-text-splitters"
                    ),
                },
            },
        )

        self._save_cache(cache_key, processed)
        return processed

    # ------------------------- Implementation ------------------------
    def _validate_file_size(self, size_bytes: int) -> None:
        if size_bytes > self.max_file_size_bytes:
            mb = size_bytes / (1024 * 1024)
            limit_mb = self.max_file_size_bytes / (1024 * 1024)
            raise ValueError(
                f"File too large: {mb:.2f} MB, limit is {limit_mb:.2f} MB."
            )

    def _hash_file(self, path: Path) -> str:
        sha = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def _cache_key(self, file_hash: str) -> str:
        # Include chunking params to avoid mismatched cache entries if headers change
        params_json = json.dumps({"headers": self.headers_to_split_on}, sort_keys=True)
        params_hash = hashlib.sha256(params_json.encode("utf-8")).hexdigest()[:12]
        return f"{file_hash}-{params_hash}"

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _load_cache(self, cache_key: str) -> Optional[ProcessedDocument]:
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ProcessedDocument(**data)
        except Exception:
            # Corrupt cache; ignore and recompute
            return None

    def _save_cache(self, cache_key: str, processed: ProcessedDocument) -> None:
        path = self._cache_path(cache_key)
        path.write_text(processed.to_json(), encoding="utf-8")

    def _extract_markdown_with_docling(self, path: Path) -> str:
        """Use Docling to extract structured Markdown; fallback to plain text for .txt/.md."""
        suffix = path.suffix.lower()

        # Lightweight fast-path for .md/.txt
        if suffix in {".md", ".markdown", ".txt"}:
            return path.read_text(encoding="utf-8", errors="ignore")

        try:
            # Lazy import so unit tests without packages can still import the module
            from docling.document_converter import DocumentConverter  # type: ignore

            converter = DocumentConverter()
            result = converter.convert(path)

            # Try markdown first, then text
            doc = getattr(result, "document", None)
            if doc is not None:
                export_md = getattr(doc, "export_to_markdown", None)
                if callable(export_md):
                    return export_md()
                export_txt = getattr(doc, "export_to_text", None)
                if callable(export_txt):
                    return export_txt()

            # Some versions return directly usable fields
            export_md = getattr(result, "export_to_markdown", None)
            if callable(export_md):
                return export_md()

            raise RuntimeError("Docling conversion succeeded but no export method found")

        except ImportError as e:
            raise RuntimeError(
                "Docling is not installed. Please add 'docling' to your dependencies."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to parse document with Docling: {e}") from e

    def _chunk_markdown(self, markdown: str) -> List[Dict[str, Any]]:
        try:
            # Imported here to keep imports light for environments missing the splitter
            from langchain_text_splitters import (
                MarkdownHeaderTextSplitter,  # type: ignore
            )

            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=self.headers_to_split_on,
                strip_headers=False,
            )
            docs = splitter.split_text(markdown)

            chunks: List[Dict[str, Any]] = []
            for d in docs:
                # Support both BaseDocument and simple dict-like objects
                content = getattr(d, "page_content", None)
                if content is None and isinstance(d, dict):
                    content = d.get("page_content")
                metadata = getattr(d, "metadata", None)
                if metadata is None and isinstance(d, dict):
                    metadata = d.get("metadata", {})
                chunks.append({"content": content or "", "metadata": metadata or {}})

            return chunks
        except ImportError as e:
            raise RuntimeError(
                "langchain-text-splitters is not installed. Please add 'langchain-text-splitters' to your dependencies."
            ) from e

    @staticmethod
    def _safe_pkg_version(pkg: str) -> Optional[str]:
        try:
            import importlib.metadata as im

            return im.version(pkg)  # type: ignore
        except Exception:
            return None

    # ------------------------- Utilities ----------------------------
    @staticmethod
    def validate_total_size(paths: List[os.PathLike | str]) -> None:
        """Validate aggregate size of multiple files using MAX_TOTAL_SIZE.

        Raises ValueError if total exceeds the configured limit.
        """
        total = 0
        for p in paths:
            st = Path(p).stat()
            total += st.st_size
        if total > MAX_TOTAL_SIZE:
            mb = total / (1024 * 1024)
            limit_mb = MAX_TOTAL_SIZE / (1024 * 1024)
            raise ValueError(
                f"Total upload size too large: {mb:.2f} MB, limit is {limit_mb:.2f} MB."
            )


__all__ = ["DocumentProcessor", "ProcessedDocument"]
