"""Utilities for segmenting plain-text documents into database chunks.

Currently provides a single public helper `chunk_doc` that:

1. Reads a cleaned `.txt` file (already extracted & normalized elsewhere).
2. Splits it into pages (form-feed ``\f`` boundaries) and paragraphs.
3. Aggregates paragraphs into size‑bounded chunks with optional character overlap.
4. Persists each chunk using `infrastructure.database.persist_chunk`.

Design notes:
* Chunk sizing is character based (not tokens) to avoid extra deps.
* Overlap helps preserve context across boundaries (default 150 chars).
* Each chunk records page_start / page_end (1-based indices of the spanned pages).
* A very small heuristic attempts to derive a section label from the first
  non-empty line that looks like a heading (ALL CAPS or Title Case and short).
* Idempotency: `persist_chunk` REPLACE logic + hash-derived chunk_id prevents
  uncontrolled duplication if the same text is reprocessed.

Future improvements (not required now):
* Token-based sizing using a tokenizer (e.g., tiktoken) when available.
* Heading detection via regex / numbering patterns (e.g., 1., 1.1, Section 2).
* Optional semantic boundary preservation (sentence splitter).
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any, Iterable
import os
import re
import json
from pathlib import Path

from .logger import get_logger
from .vector_search import upsert_datapoints
from .embeddings import generate_embeddings


try:  # Avoid hard import failure if heavy deps (openai, fitz) not installed during chunk-only operations
	from .document_processor import normalize  # type: ignore
except Exception:  # pragma: no cover - fallback path
	def normalize(s: str) -> str:  # minimal fallback
		s = s.replace("\r\n", "\n").replace("\r", "\n")
		s = re.sub(r"\n{3,}", "\n\n", s)
		return s

logger = get_logger()

API_KEY = os.getenv("OPENAI_API_KEY", "")

# Optional embeddings import with fallback to avoid import errors
try:
	from .embeddings import generate_embeddings_for_chunks  # type: ignore
except Exception:  # pragma: no cover
	def generate_embeddings_for_chunks(texts):
		logger.warning("Embeddings module unavailable; returning empty embeddings list")
		return []


def _split_pages(text: str) -> List[str]:
	"""Split text into pages on form-feed markers inserted during PDF extraction.

	If no form feed exists, the entire text is a single page.
	"""
	if "\f" not in text:
		return [text]
	# Allow light whitespace around form-feed separator pattern used in extraction
	parts = re.split(r"\n*\f\n*", text)
	return [p.strip() for p in parts if p.strip()]


def _paragraphs(page_text: str) -> List[str]:
	"""Return logical paragraphs (double-newline or blank-line separated)."""
	paras = re.split(r"\n{2,}", page_text)
	return [p.strip() for p in paras if p.strip()]


def _looks_like_heading(line: str) -> bool:
	line_stripped = line.strip()
	if not line_stripped:
		return False
	# Short & all caps (allow digits and basic punctuation)
	if (len(line_stripped) <= 80 and
		re.fullmatch(r"[A-Z0-9 .,'()/-]+", line_stripped) and
		any(c.isalpha() for c in line_stripped)):
		return True
	# Title Case with few words
	words = line_stripped.split()
	if 1 <= len(words) <= 8 and all(w[:1].isupper() for w in words if w):
		return True
	return False


def _derive_section(paragraph: str) -> str | None:
	first_line = paragraph.splitlines()[0].strip()
	return first_line if _looks_like_heading(first_line) else None



def _chunk_paragraphs(paragraphs: List[Tuple[int, str]], max_chars: int, overlap: int) -> List[Tuple[str, int, int]]:
	"""Aggregate (page_index, paragraph_text) into sized chunks.

	Returns list of (chunk_text, page_start, page_end).
	Overlap is applied on character basis between consecutive chunks.
	"""
	chunks: List[Tuple[str, int, int]] = []
	buf: List[str] = []
	buf_pages: List[int] = []
	current_len = 0
	last_tail = ""

	for page_idx, para in paragraphs:
		para_len = len(para)
		if current_len and current_len + para_len + 2 > max_chars:
			# Flush current buffer
			chunk_text = "\n\n".join(buf).strip()
			if chunk_text:
				chunks.append((chunk_text, min(buf_pages), max(buf_pages)))
				# Prepare overlap tail
				last_tail = chunk_text[-overlap:] if overlap > 0 else ""
			buf, buf_pages = [], []
			current_len = 0
		# Add overlap to new buffer if starting fresh and have tail
		if not buf and last_tail:
			buf.append(last_tail)
			buf_pages.append(page_idx)  # associate with current page for simplicity
			current_len = len(last_tail)
		buf.append(para)
		buf_pages.append(page_idx)
		current_len += para_len + 2  # account for join newlines

	# Final flush
	if buf:
		chunk_text = "\n\n".join(buf).strip()
		if chunk_text:
			chunks.append((chunk_text, min(buf_pages), max(buf_pages)))
	return chunks


try:  # Optional tokenizer (tiktoken) for accurate token counts
	import tiktoken  # type: ignore
	_TT_ENC = tiktoken.get_encoding("cl100k_base")
	def _encode_tokens(text: str) -> List[int]:
		return _TT_ENC.encode(text)  # type: ignore[union-attr]
	def _decode_tokens(tokens: List[int]) -> str:
		return _TT_ENC.decode(tokens)  # type: ignore[union-attr]
	TOKENIZER_NAME = "cl100k_base"
except Exception:  # pragma: no cover
	_TT_ENC = None  # type: ignore
	_FALLBACK_VOCAB: Dict[str, int] = {}
	_FALLBACK_ID2WORD: Dict[int, str] = {}
	def _encode_tokens(text: str) -> List[int]:  # fallback naive whitespace word -> int mapping
		tokens: List[int] = []
		for w in re.findall(r"\S+", text):
			if w not in _FALLBACK_VOCAB:
				idx = len(_FALLBACK_VOCAB) + 1
				_FALLBACK_VOCAB[w] = idx
				_FALLBACK_ID2WORD[idx] = w
			tokens.append(_FALLBACK_VOCAB[w])
		return tokens
	def _decode_tokens(tokens: List[int]) -> str:
		return " ".join(_FALLBACK_ID2WORD.get(t, "") for t in tokens).strip()
	TOKENIZER_NAME = "fallback_simple_vocab"


def _split_large_paragraph_tokens(tokens: List[int], max_tokens: int) -> Iterable[List[int]]:
	for i in range(0, len(tokens), max_tokens):
		yield tokens[i:i+max_tokens]


def _chunk_paragraphs_tokens(paragraphs: List[Tuple[int, str]], max_tokens: int, overlap_tokens: int) -> List[Tuple[str, int, int]]:
	"""Token-based chunk aggregation.

	paragraphs: list of (page_index, text)
	Returns list of (chunk_text, page_start, page_end)
	"""
	# Pre-tokenize paragraphs (cache by object id / value) to avoid repeats
	tokenized: List[Tuple[int, List[int], str]] = []  # (page_idx, token_ids, original_text)
	for page_idx, para in paragraphs:
		toks = _encode_tokens(para)
		# If single para exceeds max_tokens, split hard
		if len(toks) > max_tokens:
			for slice_tokens in _split_large_paragraph_tokens(toks, max_tokens):
				tokenized.append((page_idx, list(slice_tokens), _decode_tokens(list(slice_tokens))))
		else:
			tokenized.append((page_idx, toks, para))

	chunks: List[Tuple[str, int, int]] = []
	buf_tokens: List[int] = []
	buf_pages: List[int] = []
	tail_tokens: List[int] = []

	for page_idx, toks, original in tokenized:
		needed = len(toks)
		if buf_tokens and (len(buf_tokens) + needed) > max_tokens:
			# Flush current
			chunk_text = _decode_tokens(buf_tokens).strip()
			if chunk_text:
				chunks.append((chunk_text, min(buf_pages), max(buf_pages)))
				tail_tokens = buf_tokens[-overlap_tokens:] if overlap_tokens > 0 else []
			buf_tokens, buf_pages = [], []
		if not buf_tokens and tail_tokens:
			buf_tokens.extend(tail_tokens)
			buf_pages.append(page_idx)
		buf_tokens.extend(toks)
		buf_pages.append(page_idx)

	if buf_tokens:
		chunk_text = _decode_tokens(buf_tokens).strip()
		if chunk_text:
			chunks.append((chunk_text, min(buf_pages), max(buf_pages)))
	return chunks

def chunk_doc(file_path: str, doc_id: str, max_chars: int = 1200, overlap: int = 150, *, max_tokens: Optional[int] = None, token_overlap: Optional[int] = None) -> List[str]:
	"""Chunk a cleaned text file, persist chunks, metadata, embeddings, and index.

	Returns list of chunk IDs.
	"""
	# Validate modes
	use_tokens = max_tokens is not None
	if use_tokens:
		if max_tokens is None or max_tokens <= 0:
			raise ValueError("max_tokens must be positive when provided")
		if token_overlap is None:
			# default 12% of max_tokens capped at 200
			token_overlap = min(max(0, int(max_tokens * 0.12)), 200)
		if token_overlap < 0:
			raise ValueError("token_overlap cannot be negative")
		if token_overlap >= max_tokens:
			logger.warning("token_overlap (%d) >= max_tokens (%d); reducing", token_overlap, max_tokens)
			token_overlap = max(0, max_tokens // 5)
	else:
		if max_chars <= 0:
			raise ValueError("max_chars must be positive")
		if overlap < 0:
			raise ValueError("overlap cannot be negative")
		if overlap >= max_chars:
			logger.warning("overlap (%d) >= max_chars (%d); reducing overlap", overlap, max_chars)
			overlap = max(0, max_chars // 4)  # soften

	if not os.path.isfile(file_path):
		raise FileNotFoundError(file_path)
	if not file_path.lower().endswith(".txt"):
		raise ValueError("chunk_doc currently expects a .txt file")

	with open(file_path, "r", encoding="utf-8") as f:
		raw = f.read()
	text = normalize(raw)

	pages = _split_pages(text)
	logger.info("Chunking doc_id=%s pages=%d (max_chars=%d overlap=%d)", doc_id, len(pages), max_chars, overlap)
	doc_iden = doc_id  # avoid shadowing
	# Flatten paragraphs with page indices
	para_with_pages: List[Tuple[int, str]] = []
	for p_idx, page_text in enumerate(pages, start=1):
		for para in _paragraphs(page_text):
			para_with_pages.append((p_idx, para))

	if not para_with_pages:
		logger.warning("No paragraphs detected; using raw text as single chunk for %s", doc_id)
		para_with_pages = [(1, text)]

	if use_tokens:
		chunk_specs = _chunk_paragraphs_tokens(para_with_pages, max_tokens=max_tokens, overlap_tokens=token_overlap or 0)  # type: ignore[arg-type]
		logger.info("Token chunking mode tokenizer=%s max_tokens=%s overlap_tokens=%s", TOKENIZER_NAME, max_tokens, token_overlap)
	else:
		chunk_specs = _chunk_paragraphs(para_with_pages, max_chars=max_chars, overlap=overlap)

	from . import database, vector_search
	db = database.get_db()
	db_session = db
	chunk_ids: List[str] = []

	# Preload manifest metadata; only include these keys in output metadata
	manifest_meta: Dict[str, Optional[str]] = {
		"doc_type": None,
		"party_role": None,
		"jurisdiction": None,
		"governing_law": None,
		"industry": None,
	}
	try:
		manifest_path = Path(os.getenv('METADATA_FOLDER', './data/metadata')) / 'manifest.jsonl'
		if manifest_path.exists():
			with open(manifest_path, 'r', encoding='utf-8') as mf:
				for line in mf:
					try:
						rec = json.loads(line)
						if rec.get('doc_id') == doc_id:
							for k in manifest_meta.keys():
								if k in rec:
									manifest_meta[k] = rec.get(k)
							break
					except Exception:  # skip malformed lines
						continue
	except Exception as e:
		logger.warning("Failed reading manifest metadata: %s", e)

	def _extract_section_number_and_title(heading: str) -> Tuple[Optional[str], Optional[str]]:
		# Patterns like '9.2 Title', 'Section 9.2 Title', '9 Title'
		m = re.match(r'(?i)\s*(?:section\s+)?((?:\d+)(?:\.\d+)*)\s+(.{1,120})', heading.strip())
		if m:
			num = m.group(1)
			title = m.group(2).strip()
			# Strip trailing punctuation likely not part of title
			title = title.rstrip(' .:-')
			return num, title if title else None
		return None, heading.strip() if heading else None

	def _detect_definition_terms(text_block: str) -> List[str]:
		terms = set()
		# Quoted capitalized phrases
		for m in re.finditer(r'"([A-Z][A-Za-z0-9 ]{1,40})"', text_block):
			phrase = m.group(1).strip()
			if len(phrase.split()) <= 5:
				terms.add(phrase)
		# Single ALLCAPS words (likely defined terms) length >=3
		for m in re.finditer(r'\b([A-Z]{3,30})\b', text_block):
			w = m.group(1)
			if not w.isdigit() and w.upper() == w and w.lower() != w:
				terms.add(w)
		# Terms preceding "means" pattern
		for m in re.finditer(r'\b([A-Z][A-Za-z0-9]{2,40})\b\s+means', text_block):
			terms.add(m.group(1))
		# Return sorted list for determinism
		return sorted(terms)[:50]

	chunk_metadata_records: List[Dict[str, Any]] = []

	for idx, (chunk_text, page_s, page_e) in enumerate(chunk_specs):
		heading = _derive_section(chunk_text)
		sec_number, section_title = _extract_section_number_and_title(heading) if heading else (None, None)


		new_chunk = database.add_chunk(
			doc_id=int(doc_iden),
			chunk_index=idx,
			text=chunk_text,
			page_start=page_s,
			page_end=page_e,
			section=sec_number,
		)
		chunk_ids.append(new_chunk.id)
		chunk_metadata_records.append({
			'id': new_chunk.id,
			'doc_id': doc_id,
			'text': chunk_text,
			'metadata': {k: v for k, v in manifest_meta.items() if v is not None},
			'prev_id': None,  # fill later
			'next_id': None,  # fill later
		})

	# Link prev/next
	for i, rec in enumerate(chunk_metadata_records):
		if i > 0:
			rec['prev_id'] = chunk_metadata_records[i-1]['id']
		if i < len(chunk_metadata_records) - 1:
			rec['next_id'] = chunk_metadata_records[i+1]['id']

	# Write / upsert JSONL file
	try:
		metadata_dir = Path(os.getenv('METADATA_FOLDER', './data/metadata'))
		metadata_dir.mkdir(parents=True, exist_ok=True)
		chunks_jsonl = metadata_dir / 'chunks.jsonl'
		existing: List[Dict[str, Any]] = []
		if chunks_jsonl.exists():
			with open(chunks_jsonl, 'r', encoding='utf-8') as cf:
				for line in cf:
					try:
						obj = json.loads(line)
						if obj.get('doc_id') != doc_id:  # drop existing lines for this doc_id
							existing.append(obj)
					except Exception:
						continue
		# Write updated file excluding old records for this doc_id and adding new ones
		with open(chunks_jsonl, 'w', encoding='utf-8') as cf:
			for obj in existing + chunk_metadata_records:
				cf.write(json.dumps(obj, ensure_ascii=False) + '\n')
	except Exception as e:
		logger.warning("Failed to update chunks.jsonl for %s: %s", doc_id, e)
	chunk_texts = [spec[0] for spec in chunk_specs]
	# Generate embeddings and upsert to Vertex AI
	try:
		embeddings = generate_embeddings(chunk_texts)

		if embeddings:
			datapoints = [{'id': cid, 'embedding': emb} for cid, emb in zip(chunk_ids, embeddings)]
			upsert_datapoints(datapoints)
			logger.info('Upserted %d embeddings to Vertex AI for doc_id=%s', len(embeddings), doc_id)
		else:
			logger.warning('No embeddings generated (module missing or empty) for doc_id=%s', doc_id)

	except Exception as e:  # pragma: no cover
		logger.error('Embedding/indexing step failed for %s: %s', doc_id, e)
		embeddings = generate_embeddings(chunk_texts)

		if embeddings:
			datapoints = [{'id': cid, 'embedding': emb} for cid, emb in zip(chunk_ids, embeddings)]
			vector_search.upsert_datapoints(datapoints)
			logger.info('Upserted %d embeddings to Vertex AI for doc_id=%s', len(embeddings), doc_id)
		else:
			logger.warning('No embeddings generated (module missing or empty) for doc_id=%s', doc_id)

	db_session.commit()
	db_session.close()
	logger.info("Persisted %d chunks for doc_id=%s", len(chunk_ids), doc_id)
	return chunk_ids


__all__ = ["chunk_doc"]

