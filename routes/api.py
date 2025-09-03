import os
import json
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app
import uuid
from werkzeug.utils import secure_filename
from infrastructure.logger import get_logger
from infrastructure.database import get_db, upsert_document
from infrastructure.embeddings import generate_and_store_embeddings, build_chroma_index
from infrastructure.document_processor import extract_text, extract_title, get_manifest_info, sha256_text
from infrastructure.chunker import chunk_doc

logger = get_logger()

def allowed_file(filename: str) -> bool:
	"""Return True if the filename has an allowed extension.

	Note: Docstring for the route advertises support for txt as well, so include it here.
	"""
	return "." in filename and filename.rsplit('.', 1)[1].lower() in ["pdf", "docx", "txt"]


api_bp = Blueprint('api_bp', __name__, url_prefix='/api')

@api_bp.route('/upload', methods=['POST'])
def add_doc():
	"""Accept a single pdf, docx, or txt document upload and store it locally.

	Request: multipart/form-data with field name 'file'.
	Response: JSON { message, filename }
	"""
	if 'file' not in request.files:
		logger.warning("Upload attempted without 'file' in form data")
		return jsonify({"error": "No file part"}), 400

	file = request.files['file']

	if not file.filename:
		logger.warning("Upload attempted with empty filename")
		return jsonify({"error": "No selected file"}), 400

	filename_value = file.filename  # type: ignore[assignment]
	if not allowed_file(filename_value):
		logger.warning("Rejected file with disallowed extension: %s", file.filename)
		return jsonify({"error": "Unsupported file type. Allowed: PDF, TXT, DOCX"}), 400

	# Ensure uploads directory exists (configurable via UPLOAD_FOLDER or default to ./uploads)
	upload_folder = os.getenv('UPLOAD_FOLDER', None) or os.path.join(current_app.root_path, '.data/corpus_raw')
	clean_folder = os.getenv('CLEAN_FOLDER', None) or os.path.join(current_app.root_path, '.data/corpus_clean')
	metadata_folder = os.getenv('METADATA_FOLDER', None) or os.path.join(current_app.root_path, '.data/metadata')
	os.makedirs(upload_folder, exist_ok=True)
	os.makedirs(clean_folder, exist_ok=True)
	os.makedirs(metadata_folder, exist_ok=True)

	doc_id = str(uuid.uuid4())
	base_name = secure_filename(filename_value)
	mtime_epoch = int(datetime.now(timezone.utc).timestamp())  # seconds since epoch fits in 64-bit
	name, ext = os.path.splitext(base_name)			# noqa: F841 (may be used later)
	stored_filename = f"{doc_id}{ext.lower()}"  
	file_path = os.path.join(upload_folder, stored_filename)


	try:
		file.save(file_path)
	except Exception as e:
		logger.exception("Failed to save uploaded file: %s", e)
		return jsonify({"error": "Failed to save file"}), 500

	logger.info("Stored uploaded document: %s", doc_id)
	
	text = extract_text(file_path)
	title = extract_title(text[:1000])
	manifest_info = get_manifest_info(file_path)
	logger.info("Extracted title: %s", title)
	logger.info("Extracted manifest info: %s", manifest_info)

	clean_text_path = os.path.join(clean_folder, f"{doc_id}.txt")
	try:
		with open(clean_text_path, "w", encoding="utf-8") as tf:
			tf.write(text)
		logger.info("Wrote cleaned text for %s", doc_id)
	except Exception as e:
		logger.error("Failed to write cleaned text for %s: %s", doc_id, e)

	source_url = manifest_info.get("source_url")
	license_ = manifest_info.get("license")
	doc_type = manifest_info.get("doc_type")
	party_role = manifest_info.get("party_role")
	jurisdiction = manifest_info.get("jurisdiction")
	governing_law = manifest_info.get("governing_law")
	industry = manifest_info.get("industry")
	effective_date = manifest_info.get("effective_date")
	manifest_jsonl_path = os.path.join(metadata_folder, "manifest.jsonl")

	manifest_record = {
		"doc_id": doc_id,
		"source_url": source_url,
		"license": license_,
		"doc_type": doc_type,
		"party_role": party_role,
		"jurisdiction": jurisdiction,
		"governing_law": governing_law,
		"industry": industry,
		"effective_date": effective_date,
		"last_retrieved": datetime.now(timezone.utc).isoformat(),
		"hash_sha256": sha256_text(text)
	}

	try:
		with open(manifest_jsonl_path, "a", encoding="utf-8") as mf:
			mf.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")
		logger.info("Appended manifest metadata for %s", doc_id)
	except Exception as e:
		logger.exception("Failed writing manifest metadata for %s: %s", doc_id, e)

	upsert_document(db=get_db(), doc_id=doc_id, source_path=file_path, title=title, norm_text=text, mtime=mtime_epoch)
	
	chunk_doc(file_path=clean_text_path, doc_id=doc_id)

	return jsonify({
		"message": "Document uploaded successfully.",
		"filename": doc_id
	}), 201


@api_bp.route('/embeddings/generate', methods=['POST'])
def generate_embeddings_endpoint():
	"""Generate embeddings (and optionally index in Chroma) for chunks.

	Request JSON (all optional):
		{
			"doc_id": "<document id to restrict>",
			"reindex": true | false  # if true, rebuild Chroma index for that scope
		}

	Response JSON:
		{"generated": <int>, "indexed": <int>, "doc_id": <str|None>, "model": <str>, "token_mode": <bool>}
	"""
	payload = {}
	if request.is_json:
		try:
			payload = request.get_json() or {}
		except Exception:
			return jsonify({"error": "Invalid JSON"}), 400
	doc_id = payload.get('doc_id') if isinstance(payload, dict) else None
	reindex = bool(payload.get('reindex')) if isinstance(payload, dict) else False

	model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large')
	try:
		generated = generate_and_store_embeddings(doc_id=doc_id, model=model)
	except Exception as e:
		logger.exception("Embedding generation failed")
		return jsonify({"error": f"embedding generation failed: {e}"}), 500

	indexed = 0
	if reindex:
		try:
			indexed = build_chroma_index(doc_id=doc_id, model=model)
		except Exception as e:
			logger.error("Index build failed: %s", e)
			# still return success for generated; include error detail
			return jsonify({
				"generated": generated,
				"indexed": indexed,
				"doc_id": doc_id,
				"model": model,
				"index_error": str(e)
			}), 207  # 207 Multi-Status style signal

	return jsonify({
		"generated": generated,
		"indexed": indexed,
		"doc_id": doc_id,
		"model": model
	}), 200

@api_bp.route('/query', methods=['POST'])
def query():
	"""RAG query endpoint.

	Request JSON:
	{
	  "query": "...",            # required user question
	  "top_k": 5,                 # optional number of chunks to retrieve (default 5, max 20)
	  "docType": "License",      # optional filter (matches manifest doc_type exact / case-insensitive)
	  "jurisdiction": "US-CA"    # optional filter (exact / case-insensitive)
	}

	Response JSON:
	{
	  "query": str,
	  "answer": str | null,
	  "matches": [ { "chunk_id", "doc_id", "section", "score", "preview" } ... ],
	  "used_model": str | null,          # chat / completion model used for answer
	  "embedding_model": str | null,
	  "filters": { ... },
	  "timing_ms": { "retrieval": int, "answer": int },
	  "error": str (optional)
	}

	If the OpenAI client / key is unavailable, returns matches only (no generated answer).
	"""
	from time import time
	import math
	import sqlite3
	import array

	if not request.is_json:
		return jsonify({"error": "Expected application/json"}), 400
	payload = request.get_json(silent=True) or {}
	question = (payload.get('query') or '').strip()
	if not question:
		return jsonify({"error": "'query' is required"}), 400

	max_q_len = int(os.getenv('MAX_QUERY_LENGTH', '1000'))
	if len(question) > max_q_len:
		return jsonify({"error": f"query too long (>{max_q_len} chars)"}), 400

	top_k = payload.get('top_k') or 5
	try:
		top_k = int(top_k)
	except Exception:
		return jsonify({"error": "top_k must be int"}), 400
	if top_k <= 0:
		top_k = 5
	if top_k > 20:
		top_k = 20

	filter_doc_type = (payload.get('docType') or '').strip() or None
	filter_juris = (payload.get('jurisdiction') or '').strip() or None

	# Load manifest metadata (doc_type / jurisdiction mapping)
	manifest_map: dict[str, dict[str, str | None]] = {}
	manifest_path = os.getenv('METADATA_FOLDER', None) or os.path.join(current_app.root_path, '.data/metadata')
	manifest_file = os.path.join(manifest_path, 'manifest.jsonl')
	if os.path.isfile(manifest_file):
		try:
			with open(manifest_file, 'r', encoding='utf-8') as mf:
				for line in mf:
					try:
						obj = json.loads(line)
						mid = obj.get('doc_id')
						if mid:
							manifest_map[mid] = {
								'doc_type': obj.get('doc_type'),
								'jurisdiction': obj.get('jurisdiction')
							}
					except Exception:
						continue
		except Exception as e:
			logger.warning('Failed reading manifest: %s', e)

	# Determine allowed doc_ids after filters
	allowed_doc_ids = None  # None = no restriction
	if filter_doc_type or filter_juris:
		allowed_doc_ids = []
		for did, meta in manifest_map.items():
			if filter_doc_type and (meta.get('doc_type') or '').lower() != filter_doc_type.lower():
				continue
			if filter_juris and (meta.get('jurisdiction') or '').lower() != filter_juris.lower():
				continue
			allowed_doc_ids.append(did)
		if allowed_doc_ids and len(allowed_doc_ids) == 0:
			return jsonify({"query": question, "matches": [], "answer": None, "filters": {"docType": filter_doc_type, "jurisdiction": filter_juris}, "embedding_model": None, "used_model": None}), 200

	db = get_db()

	# Embedding model + client
	embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large')
	try:
		from openai import OpenAI  # type: ignore
		api_key = os.getenv('OPENAI_API_KEY')
		client = OpenAI(api_key=api_key) if api_key else None
	except Exception:
		client = None  # type: ignore

	query_vec: list[float] | None = None
	if client:
		try:
			qresp = client.embeddings.create(model=embedding_model, input=[question])  # type: ignore[attr-defined]
			query_vec = qresp.data[0].embedding  # type: ignore
		except Exception as e:
			logger.error('Query embedding failed: %s', e)
			query_vec = None

	if query_vec is None:
		# Cannot compute similarity without embedding; return error
		return jsonify({"error": "embedding generation unavailable (missing openai client or key)", "query": question}), 500

	def _deserialize(blob: bytes) -> list[float]:
		arr = array.array('f')
		arr.frombytes(blob)
		return list(arr)

	retrieval_start = time()
	params: list[str] = []
	where = ''
	if allowed_doc_ids is not None:
		if not allowed_doc_ids:  # no docs satisfy filter
			return jsonify({"query": question, "matches": [], "answer": None, "filters": {"docType": filter_doc_type, "jurisdiction": filter_juris}}), 200
		ph = ','.join(['?'] * len(allowed_doc_ids))
		where = f'WHERE c.doc_id IN ({ph})'
		params.extend(allowed_doc_ids)

	rows: list[sqlite3.Row] = db.execute(
		f'''SELECT e.chunk_id, e.vector, e.dim, c.doc_id, c.section, c.text
			FROM embeddings e
			JOIN chunks c ON c.chunk_id = e.chunk_id
			{where}''', params).fetchall()

	if not rows:
		return jsonify({"query": question, "matches": [], "answer": None, "filters": {"docType": filter_doc_type, "jurisdiction": filter_juris}, "embedding_model": embedding_model}), 200

	# Precompute norms
	q_norm = math.sqrt(sum(v*v for v in query_vec)) or 1.0
	scored: list[tuple[float, sqlite3.Row]] = []
	for r in rows:
		vec = _deserialize(r['vector'])
		if len(vec) != r['dim']:
			continue
		vdot = sum(a*b for a,b in zip(query_vec, vec))
		vnorm = math.sqrt(sum(x*x for x in vec)) or 1.0
		cos = vdot / (q_norm * vnorm)
		scored.append((cos, r))

	# Select top_k
	scored.sort(key=lambda x: x[0], reverse=True)
	top = scored[:top_k]

	# Get document titles for the matches
	doc_ids_in_matches = list(set(r['doc_id'] for _, r in top))
	doc_titles = {}
	if doc_ids_in_matches:
		title_rows = db.execute(
			f"SELECT doc_id, title FROM documents WHERE doc_id IN ({','.join(['?'] * len(doc_ids_in_matches))})",
			doc_ids_in_matches
		).fetchall()
		doc_titles = {row['doc_id']: row['title'] for row in title_rows}

	matches = []
	for score, r in top:
		text_val = r['text']
		preview = text_val[:300].replace('\n', ' ') + ('...' if len(text_val) > 300 else '')
		doc_id = r['doc_id']
		
		# Get document title and doc_type
		doc_title = doc_titles.get(doc_id, 'Unknown Title')
		doc_meta = manifest_map.get(doc_id, {})
		doc_type = doc_meta.get('doc_type', 'Unknown Type')
		
		matches.append({
			'chunk_id': r['chunk_id'],
			'doc_id': doc_id,
			'doc_title': doc_title,
			'doc_type': doc_type,
			'section': r['section'],
			'score': round(float(score), 5),
			'preview': preview
		})
	retrieval_ms = int((time() - retrieval_start) * 1000)

	# Build answer using chat model (optional)
	answer_text = None
	chat_model = os.getenv('OPENAI_MODEL', 'gpt-5')
	answer_start = time()
	if client and matches:
		context_parts = []
		for m in matches:
			# fetch full text again from rows for reliability
			for _s, r in top:
				if r['chunk_id'] == m['chunk_id']:
					context_parts.append(f"[Chunk {m['chunk_id']} section={m['section']}]\n{r['text']}\n")
					break
		context_text = '\n---\n'.join(context_parts)
		prompt = (
			"You are a legal assistant specialized in technology transactions. "
			"Answer the user's question ONLY using the provided chunk context. "
			"If the answer is not contained, say you cannot answer based on the corpus. "
			"Cite relevant section numbers if present."
		)
		try:
			resp = client.chat.completions.create(  # type: ignore[attr-defined]
				model=chat_model,
				messages=[
					{"role": "system", "content": prompt},
					{"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
				],
				reasoning_effort="high",
				verbosity="high"
			)
			answer_text = (resp.choices[0].message.content or '').strip() if resp.choices else None  # type: ignore
		except Exception as e:
			logger.error('Answer generation failed: %s', e)
	answer_ms = int((time() - answer_start) * 1000) if answer_text is not None else 0

	return jsonify({
		'query': question,
		'answer': answer_text,
		'matches': matches,
		'used_model': chat_model if answer_text else None,
		'embedding_model': embedding_model,
		'filters': {
			'docType': filter_doc_type,
			'jurisdiction': filter_juris
		},
		'timing_ms': {
			'retrieval': retrieval_ms,
			'answer': answer_ms
		},
		'total_candidates': len(rows)
	}), 200
