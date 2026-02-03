import express from 'express';
import type { Request, Response as ExpressResponse } from 'express';
import fetch from 'node-fetch';
import axios from 'axios';
import dotenv from 'dotenv';  // Add this import

// Load .env file
dotenv.config();

import bodyParser from 'body-parser';
import cors from 'cors';
import multer from 'multer';
import * as fs from 'node:fs';
import * as path from 'node:path';
import AWS from 'aws-sdk';

// Helper: remove keys with null values recursively from objects/arrays
function stripNulls(obj: any): any {
	if (Array.isArray(obj)) return obj.map(stripNulls);
	if (obj && typeof obj === 'object') {
		const out: any = {};
		for (const [k, v] of Object.entries(obj)) {
			if (v === null || v === undefined) continue;
			const cleaned = stripNulls(v);
			// skip values that became empty objects/arrays
			if (cleaned === null || cleaned === undefined) continue;
			if (typeof cleaned === 'object') {
				if (Array.isArray(cleaned) && cleaned.length === 0) continue;
				if (!Array.isArray(cleaned) && Object.keys(cleaned).length === 0) continue;
			}
			out[k] = cleaned;
		}
		return out;
	}
	return obj;
}

// -------------------- NEW: Hugging Face helper & analyzeImage --------------------

// helper to call Hugging Face Inference API via router with better diagnostics
async function hfRequest(model: string, buffer: Buffer) {
	// require HUGGINGFACE_API_KEY in env
	const key = process.env.HUGGINGFACE_API_KEY;
	if (!key) throw new Error('Hugging Face API key not set in HUGGINGFACE_API_KEY');

	// single supported endpoint: router (old api-inference is deprecated)
	const endpoints = [
		`https://router.huggingface.co/hf-inference/models/${encodeURIComponent(model)}`
	];

	// pick fetch implementation: prefer global fetch (Node 18+), else imported node-fetch
	const fetchImpl: typeof fetch = (globalThis as any).fetch ?? fetch;

	let lastErr: any = null;
	for (const url of endpoints) {
		try {
			let res: any;
			try {
				res = await fetchImpl(url, {
					method: 'POST',
					headers: {
						Authorization: `Bearer ${key}`,
						'Content-Type': 'application/octet-stream'
					},
					body: buffer,
					// small timeout not built-in for fetch; network errors will surface as thrown errors
				});
			} catch (netErr: any) {
				// network-level failure (DNS, TLS, proxy, etc.)
				const msg = `Network error when fetching ${url}: ${netErr?.message ?? netErr}`;
				console.error(msg);
				lastErr = new Error(msg);
				// try next endpoint
				await new Promise(r => setTimeout(r, 250));
				continue;
			}

			// model loading - retry a few times
			if (res.status === 503) {
				const info = await parseResponse(res).catch(() => ({} as { estimated_time?: number }));
				const waitMs = ((info as any).estimated_time ?? 20) * 1000;
				await new Promise(r => setTimeout(r, waitMs));
				// do one more try for this endpoint
				const retryRes = await fetchImpl(url, {
					method: 'POST',
					headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/octet-stream' },
					body: buffer
				});
				if (!retryRes.ok) {
					const txt = await retryRes.text().catch(() => String(retryRes.status));
					throw new Error(`HF ${model} endpoint ${url} error after retry: ${retryRes.status} ${txt}`);
				}
				return await parseResponse(retryRes);
			}

			// explicit deprecation warning previously observed
			if (res.status === 410) {
				const txt = await (res.text ? res.text().catch(() => String(res.status)) : Promise.resolve(String(res.status)));
				throw new Error(`HF API endpoint deprecated (410) from ${url}: ${txt}`);
			}

			if (res.status === 401 || res.status === 403) {
				const txt = await (res.text ? res.text().catch(() => String(res.status)) : Promise.resolve(String(res.status)));
				throw new Error(`HF auth error ${res.status} from ${url}: ${txt}`);
			}

			if (res.status === 404) {
				// model not found or not accessible on router
				lastErr = new Error(`HF model not found or inaccessible (404) at ${url}. Ensure the model exists and your token has access.`);
				continue;
			}

			if (!res.ok) {
				const txt = await (res.text ? res.text().catch(() => String(res.status)) : Promise.resolve(String(res.status)));
				throw new Error(`HF ${model} error from ${url}: ${res.status} ${txt}`);
			}

			return await parseResponse(res);
		} catch (err: any) {
			// record last error and try next endpoint
			console.error('hfRequest try error for model', model, 'endpoint', url, err);
			lastErr = err;
			// small delay before next endpoint
			await new Promise(r => setTimeout(r, 250));
			continue;
		}
	}
	throw lastErr ?? new Error('HF request failed for unknown reason');
}

// helper to parse JSON or text responses (safe for different fetch Response types)
async function parseResponse(res: any) {
	try {
		const ct = (res?.headers && typeof res.headers.get === 'function') ? res.headers.get('content-type') : (res?.headers?.['content-type'] || '');
		if (ct && ct.includes('application/json')) {
			return await res.json();
		} else {
			// return raw text if not JSON (models sometimes return plain text)
			return await res.text();
		}
	} catch (err: unknown) {
		// if parsing fails, throw informative error (safe access to message)
		const msg = (err && typeof err === 'object' && 'message' in err) ? (err as any).message : String(err);
		throw new Error(`Failed to parse HF response: ${msg}`);
	}
}

// Try a list of candidate models, return first successful result or throw aggregated error
async function tryModels(candidates: string[], buffer: Buffer) {
	const errors: any[] = [];
	for (const model of candidates) {
		try {
			const out = await hfRequest(model, buffer);
			// basic validation — some endpoints return strings for captions, arrays for classifiers, accept both
			if (out != null) return { model, result: out };
		} catch (err) {
			errors.push({ model, error: String(err) });
		}
	}
	throw new Error(`All models failed: ${JSON.stringify(errors)}`);
}

// analyzeImage now uses the local Python model service for embeddings and NN
async function analyzeImage(buffer: Buffer, contentType?: string) {
	const out: any = {
		labels: null,
		text: null,
		faces: null,
		objects: null,
		safeSearch: null,
		embedding: null,
		nn: null
	};

	// --- Domain detection helpers ---
	function detectDomainFromBuffer(buf: Buffer, ct?: string) {
		try {
			const s = buf.toString('latin1', 0, Math.min(buf.length, 2048));
			// JPEG with Exif metadata is usually from a camera/real photo
			if (/Exif/.test(s) || /JFIF/.test(s)) return { domain: 'Real Photo', reason: 'contains EXIF/JFIF' };
			// heuristic: many generated images live in PNG without EXIF
			if (ct && /png/i.test(ct)) return { domain: 'Possibly Synthetic', reason: 'PNG and no EXIF detected' };
			return { domain: 'Unknown', reason: 'no EXIF marker found' };
		} catch (e) {
			return { domain: 'Unknown', reason: 'detection error' };
		}
	}

	function detectDomainFromPath(p: string) {
		if (!p || typeof p !== 'string') return { domain: 'unknown', reason: 'no path' };
		const lower = p.toLowerCase();
		if (/hvae_generated|generated|synthetic|hvae|generated_ip/i.test(lower)) return { domain: 'Generated', reason: 'path indicates generated gallery' };
		if (/real|photo|images|dataset|camera|iphone|oppo|ip16promax/i.test(lower)) return { domain: 'Real Photo', reason: 'path indicates real dataset or camera' };
		return { domain: 'Unknown', reason: 'no domain marker in path' };
	}

	// compute domain metadata for the input image
	out.domain_metadata = { input: detectDomainFromBuffer(buffer, contentType) };

	// Forward to local Python /embed
	try {
		const resp = await axios.post(`${PY_SERVICE}/embed`, buffer, {
			headers: { 'Content-Type': contentType || 'application/octet-stream' },
			timeout: 30000,
		});
		out.embedding = resp.data;
	} catch (err: any) {
		console.error('Local embed error:', err?.message ?? err);
		out.embedding = { error: err?.message ?? String(err) };
	}

	// Forward to local Python /nn (k=5)
	try {
		const resp2 = await axios.post(`${PY_SERVICE}/nn?k=5`, buffer, {
			headers: { 'Content-Type': contentType || 'application/octet-stream' },
			timeout: 30000,
		});
		// enrich NN results with domain metadata and attach to output
		const rawNN = resp2.data;
		const nnList: any[] = rawNN && Array.isArray(rawNN.nn) ? rawNN.nn : [];
		const enrichedNN = nnList.map((it: any) => {
			const p = it.path || it[0] || '';
			const dom = detectDomainFromPath(p);
			return Object.assign({}, it, { domain: dom.domain, domain_reason: dom.reason });
		});
		out.nn = Object.assign({}, rawNN, { nn: enrichedNN });

		// derive predicted class + confidence from nearest-neighbors returned by Python
		try {
			// use the enriched list for scoring
			
			if (nnList.length > 0) {
				// helper to extract class name from a gallery file path
				const extractClassFromPath = (p: string) => {
					if (!p) return null;
					const parts = p.split(/\\|\//).filter(Boolean);
					// try to find a folder named like hvae_generated or gallery and take next segment
					const markerIdx = parts.findIndex(s => /hvae_generated|gallery|generated/i.test(s));
					if (markerIdx >= 0 && parts.length > markerIdx + 1) return parts[markerIdx + 1];
					// fallback: parent folder of the file's directory (two levels up)
					if (parts.length >= 3) return parts[parts.length - 3];
					if (parts.length >= 2) return parts[parts.length - 2];
					return parts[parts.length - 1] || null;
				};

				// accumulate weighted scores per class
				const scoreSums: Record<string, number> = {};
				const counts: Record<string, number> = {};
				let totalScore = 0;
				for (const it of enrichedNN) {
					const p = it.path || it[0] || '';
					const score = Number(it.score ?? 0) || 0;
					const cls = extractClassFromPath(p) || 'unknown';
					scoreSums[cls] = (scoreSums[cls] || 0) + score;
					counts[cls] = (counts[cls] || 0) + 1;
					totalScore += score;
				}

				// choose class with highest summed score
				let predClass: string | null = null;
				let bestScore = -Infinity;
				for (const [cls, s] of Object.entries(scoreSums)) {
					if (s > bestScore) { bestScore = s; predClass = cls; }
				}
				if (predClass) {
					const confidence = totalScore > 0 ? (scoreSums[predClass] / totalScore) : (counts[predClass] / nnList.length);
					out.predicted_class = predClass;
					out.predicted_confidence = Number(confidence.toFixed(4));
					out.predicted_count = counts[predClass] || 0;
				}
				// domain diversity summary for neighbors
				try {
					const domainCounts: Record<string, number> = {};
					for (const n of enrichedNN) {
						domainCounts[n.domain || 'unknown'] = (domainCounts[n.domain || 'unknown'] || 0) + 1;
					}
					out.domain_metadata = out.domain_metadata || {};
					out.domain_metadata.neighbor_domain_counts = domainCounts;
					out.domain_metadata.neighbor_domains = Object.keys(domainCounts);
					out.domain_metadata.domain_diverse = Object.keys(domainCounts).length > 1;
				} catch (e) {
					// ignore
				}
			}
		} catch (e) {
			console.error('Failed to compute predicted class from NN:', e);
		}
	} catch (err: any) {
		console.error('Local nn error:', err?.message ?? err);
		out.nn = { error: err?.message ?? String(err) };
	}

	// Remove raw embedding vectors from the response to avoid sending large arrays
	// to the browser. If you need the raw vector, call the Python service directly
	// with the `?include_raw=1` flag.
	try {
		if (out.embedding && typeof out.embedding === 'object') {
			// If the Python response included an `embedding` array, delete it
			if ('embedding' in out.embedding && Array.isArray(out.embedding.embedding)) {
				delete out.embedding.embedding;
			}
		}
	} catch (e) {
		// ignore
	}

	// Strip null/empty fields before returning to frontend
	return stripNulls(out);
}

// Configure uploads folder
const UPLOAD_DIR = path.join(__dirname, 'uploads');
if (!fs.existsSync(UPLOAD_DIR)) {
	fs.mkdirSync(UPLOAD_DIR, { recursive: true });
}
// Heatmap output folder (inside uploads)
const HEATMAP_DIR = path.join(UPLOAD_DIR, 'heatmaps');
if (!fs.existsSync(HEATMAP_DIR)) {
	fs.mkdirSync(HEATMAP_DIR, { recursive: true });
}

// Configure AWS S3 if env vars are present
const S3_BUCKET = process.env.S3_BUCKET || '';
let s3: AWS.S3 | null = null;
if (process.env.AWS_ACCESS_KEY_ID && process.env.AWS_SECRET_ACCESS_KEY && S3_BUCKET) {
	AWS.config.update({
		accessKeyId: process.env.AWS_ACCESS_KEY_ID,
		secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
		region: process.env.AWS_REGION || 'us-east-1',
	});
	s3 = new AWS.S3();
	console.log('S3 configured. Bucket:', S3_BUCKET);
} else {
	console.log('S3 not configured. Falling back to local disk storage.');
}

// create express app and port before using app
const app = express();
const port = Number(process.env.PORT) || 3000;

app.use(bodyParser.json({ limit: '10mb' }));
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cors());

// Python model service URL
const PY_SERVICE = process.env.PY_SERVICE_URL || 'http://localhost:8001';

// Multer for multipart/form-data
const storage = multer.memoryStorage();
const upload = multer({ storage });

// Health check
app.get('/health', (_req: Request, res: ExpressResponse) => res.json({ ok: true }));

// Accept JSON with base64 image: { filename?: string, data: 'data:image/jpeg;base64,...' }
app.post('/upload', async (req: Request, res: ExpressResponse) => {
	try {
		const { filename = `photo_${Date.now()}.jpg`, data } = req.body;
		if (!data) return res.status(400).json({ error: 'No data provided' });

		// support data URLs
		const matches = data.match(/^data:(.+);base64,(.+)$/);
		let buffer: Buffer;
		let contentType = 'image/jpeg';
		if (matches) {
			contentType = matches[1];
			buffer = Buffer.from(matches[2], 'base64');
		} else {
			// assume raw base64
			buffer = Buffer.from(data, 'base64');
		}

		// Analyze image with local Python model service (embedding / NN)
		let analysis: any = null;
		try {
			analysis = await analyzeImage(buffer, contentType);
		} catch (aiErr) {
			console.error('Local analysis failed:', aiErr);
			analysis = { error: String(aiErr) };
		}

		if (s3) {
			const key = `${Date.now()}_${filename}`;
			await s3.upload({ Bucket: S3_BUCKET, Key: key, Body: buffer, ContentType: contentType }).promise();
			const url = `https://${S3_BUCKET}.s3.amazonaws.com/${encodeURIComponent(key)}`;
			return res.json({ ok: true, url, analysis });
		} else {
			const outPath = path.join(UPLOAD_DIR, `${Date.now()}_${filename}`);
			fs.writeFileSync(outPath, buffer);
			return res.json({ ok: true, path: outPath, analysis });
		}
	} catch (err: any) {
		console.error('Upload error:', err);
		return res.status(500).json({ error: err.message || String(err) });
	}
});

// Accept multipart/form-data with field 'photo'
app.post('/upload-form', upload.single('photo'), async (req: Request & { file?: { originalname?: string; buffer: Buffer; mimetype?: string } }, res: ExpressResponse) => {
	try {
		if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
		const filename = req.file.originalname || `photo_${Date.now()}.jpg`;
		const buffer = req.file.buffer;
		const contentType = req.file.mimetype || 'application/octet-stream';

		// Analyze image with local Python model service (embedding / NN)
		let analysis: any = null;
		try {
			analysis = await analyzeImage(buffer, contentType);
		} catch (aiErr) {
			console.error('Local analysis failed:', aiErr);
			analysis = { error: String(aiErr) };
		}

		if (s3) {
			const key = `${Date.now()}_${filename}`;
			await s3.upload({ Bucket: S3_BUCKET, Key: key, Body: buffer, ContentType: contentType }).promise();
			const url = `https://${S3_BUCKET}.s3.amazonaws.com/${encodeURIComponent(key)}`;
			return res.json({ ok: true, url, analysis });
		} else {
			const outPath = path.join(UPLOAD_DIR, `${Date.now()}_${filename}`);
			fs.writeFileSync(outPath, buffer);
			return res.json({ ok: true, path: outPath, analysis });
		}
	} catch (err: any) {
		console.error('Upload-form error:', err);
		return res.status(500).json({ error: err.message || String(err) });
	}
});

app.listen(port, () => {
	console.log(`Backend listening on http://localhost:${port}`);
});

// --- New: endpoints that forward to Python model service ---
app.post('/model/embed', upload.single('photo'), async (req: Request & { file?: { originalname?: string; buffer: Buffer; mimetype?: string } }, res: ExpressResponse) => {
	try {
		if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
		const resp = await axios.post(`${PY_SERVICE}/embed`, req.file.buffer, {
			headers: { 'Content-Type': req.file.mimetype || 'application/octet-stream' },
			timeout: 30000,
		});
		return res.json(resp.data);
	} catch (err: any) {
		console.error('Error forwarding to PY /embed:', err?.message ?? err);
		return res.status(500).json({ error: err?.message ?? String(err) });
	}
});

app.post('/model/heatmap', upload.single('photo'), async (req: Request & { file?: { originalname?: string; buffer: Buffer; mimetype?: string } }, res: ExpressResponse) => {
	try {
		if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
 		// Forward optional query params `cv` and `colormap` to the Python service
 		const useCv = String(req.query.cv || '0').toLowerCase();
 		const colormap = String(req.query.colormap || 'jet');

 		const pyUrl = `${PY_SERVICE}/heatmap${(useCv === '1' || useCv === 'true' || useCv === 'yes') ? `?cv=1&colormap=${encodeURIComponent(colormap)}` : ''}`;

		// Propagate optional heatmap mode/patch params
		const mode = String(req.query.mode || '').toLowerCase();
		const patchSize = req.query.patch_size ? `&patch_size=${encodeURIComponent(String(req.query.patch_size))}` : '';
		const stride = req.query.stride ? `&stride=${encodeURIComponent(String(req.query.stride))}` : '';
		const topK = req.query.top_k ? `&top_k=${encodeURIComponent(String(req.query.top_k))}` : '';

		const pyUrlWithMode = mode === 'patch' || mode === 'similarity' ? `${pyUrl}${patchSize}${stride}${topK}` : pyUrl;

		const resp = await axios.post(pyUrlWithMode, req.file.buffer, {
			headers: { 'Content-Type': req.file.mimetype || 'application/octet-stream' },
			timeout: 60000,
		});
		// If the Python service returned a base64 heatmap, save it locally
		try {
			const data = resp.data;
			if (data && typeof data.heatmap_base64 === 'string' && data.heatmap_base64.length > 0) {
				// build safe filename from originalname
				const orig = req.file.originalname || `photo_${Date.now()}`;
				const base = path.parse(orig).name.replace(/[^a-zA-Z0-9-_]/g, '_');
				const heatmapName = `${Date.now()}_${base}.png`;
				const heatmapPath = path.join(HEATMAP_DIR, heatmapName);
				const buf = Buffer.from(data.heatmap_base64, 'base64');
				fs.writeFileSync(heatmapPath, buf);
				// include the server-side path in the response for convenience
				data.heatmap_path = heatmapPath;
			}
			return res.json(data);
		} catch (saveErr: any) {
			console.error('Failed to save heatmap file:', saveErr);
			// still return the python response even if saving fails
			return res.json(resp.data);
		}
	} catch (err: any) {
		console.error('Error forwarding to PY /heatmap:', err?.message ?? err);
		return res.status(500).json({ error: err?.message ?? String(err) });
	}
});

export {};
