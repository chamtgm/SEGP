import express from 'express';
import type { Request, Response as ExpressResponse } from 'express';
import dotenv from 'dotenv';
import fetch from 'node-fetch';

// Load .env file
dotenv.config();

import bodyParser from 'body-parser';
import cors from 'cors';
import multer from 'multer';
import * as fs from 'node:fs';
import * as path from 'node:path';
import AWS from 'aws-sdk';

// Python model service base URL
const PY_SERVICE = process.env.PYTHON_BACKEND_URL || process.env.PY_SERVICE_URL || 'http://127.0.0.1:8001';

// Helper: remove keys with null/undefined and empty objects/arrays
function stripNulls(obj: any): any {
	if (Array.isArray(obj)) return obj.map(stripNulls);
	if (obj && typeof obj === 'object') {
		const out: any = {};
		for (const [k, v] of Object.entries(obj)) {
			if (v === null || v === undefined) continue;
			const cleaned = stripNulls(v);
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

// Analyze image: call Python /embed and /nn, enrich with domain metadata and predicted class
async function analyzeImage(buffer: Buffer, contentType?: string): Promise<any> {
	const out: any = {
		labels: null,
		nn: null,
		embedding: null,
		domain_metadata: null,
		predicted_class: null,
		predicted_confidence: null,
	};

	function detectDomainFromPath(p: string) {
		if (!p || typeof p !== 'string') return { domain: 'unknown', reason: 'no path' };
		const lower = p.toLowerCase();
		if (/hvae_generated|generated|synthetic|hvae|generated_ip/i.test(lower)) return { domain: 'Generated', reason: 'path indicates generated gallery' };
		if (/real|photo|images|dataset|camera|iphone|oppo|ip16promax/i.test(lower)) return { domain: 'Real Photo', reason: 'path indicates real dataset or camera' };
		return { domain: 'Unknown', reason: 'no domain marker in path' };
	}

	// /embed
	try {
		const embedRes = await fetch(`${PY_SERVICE}/embed`, {
			method: 'POST',
			headers: { 'Content-Type': contentType || 'application/octet-stream' },
			body: buffer,
		});
		if (embedRes.ok) {
			const data = await embedRes.json();
			if (data && typeof data === 'object' && Array.isArray(data.embedding)) delete data.embedding;
			out.embedding = data;
		} else {
			out.embedding = { error: `embed returned ${embedRes.status}` };
		}
	} catch (err: any) {
		out.embedding = { error: err?.message ?? String(err) };
	}

	// /nn?k=5
	try {
		const nnRes = await fetch(`${PY_SERVICE}/nn?k=5`, {
			method: 'POST',
			headers: { 'Content-Type': contentType || 'application/octet-stream' },
			body: buffer,
		});
		if (!nnRes.ok) {
			out.nn = { error: `nn returned ${nnRes.status}` };
			return stripNulls(out);
		}
		const rawNN = await nnRes.json();
		const nnList: any[] = rawNN && Array.isArray(rawNN.nn) ? rawNN.nn : [];
		const enrichedNN = nnList.map((it: any) => {
			const p = it.path || it[0] || '';
			const dom = detectDomainFromPath(p);
			return Object.assign({}, it, { domain: dom.domain, domain_reason: dom.reason });
		});
		out.nn = Object.assign({}, rawNN, { nn: enrichedNN });

		// predicted class from neighbor votes
		if (nnList.length > 0) {
			const extractClassFromPath = (p: string) => {
				if (!p) return null;
				const parts = p.split(/\\|\//).filter(Boolean);
				const markerIdx = parts.findIndex(s => /hvae_generated|gallery|generated/i.test(s));
				if (markerIdx >= 0 && parts.length > markerIdx + 1) return parts[markerIdx + 1];
				if (parts.length >= 3) return parts[parts.length - 3];
				if (parts.length >= 2) return parts[parts.length - 2];
				return parts[parts.length - 1] || null;
			};
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
			let predClass: string | null = null;
			let bestScore = -Infinity;
			for (const [cls, s] of Object.entries(scoreSums)) {
				if (s > bestScore) { bestScore = s; predClass = cls; }
			}
			if (predClass) {
				const confidence = totalScore > 0 ? scoreSums[predClass] / totalScore : (counts[predClass] || 0) / nnList.length;
				out.predicted_class = predClass;
				out.predicted_confidence = Number(confidence.toFixed(4));
			}
			const domainCounts: Record<string, number> = {};
			for (const n of enrichedNN) {
				domainCounts[n.domain || 'unknown'] = (domainCounts[n.domain || 'unknown'] || 0) + 1;
			}
			out.domain_metadata = { neighbor_domain_counts: domainCounts, neighbor_domains: Object.keys(domainCounts), domain_diverse: Object.keys(domainCounts).length > 1 };
		}
	} catch (err: any) {
		out.nn = { error: err?.message ?? String(err) };
	}

	return stripNulls(out);
}

// Configure uploads folder
const UPLOAD_DIR = path.join(__dirname, 'uploads');
if (!fs.existsSync(UPLOAD_DIR)) {
	fs.mkdirSync(UPLOAD_DIR, { recursive: true });
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

// Multer for multipart/form-data
const storage = multer.memoryStorage();
const upload = multer({ storage });

// // Health check
// app.get('/health', (_req: Request, res: ExpressResponse) => res.json({ ok: true }));

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

		// Call analyzer service (Python /nn) and include analysis in response
		let analysis: any = { labels: null, tsne_coordinates: null };
		try {
			const analyzerUrl = process.env.PYTHON_BACKEND_URL || 'http://127.0.0.1:8001/nn';
			const aRes = await fetch(analyzerUrl, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: buffer as any });
			if (aRes.ok) {
				const aJson = await aRes.json();
				const labels = Array.isArray(aJson.nn) ? aJson.nn.map((n: any) => ({ description: n.path, confidence: n.score })) : [];
				analysis = { labels, tsne_coordinates: aJson.tsne_coordinates ?? (aJson.tsne_plot_path ? { plot_path: aJson.tsne_plot_path } : null), raw: aJson };
			} else {
				analysis = { error: `Analyzer returned status ${aRes.status}` };
			}
		} catch (e: any) {
			console.error('Analyzer call failed:', e);
			analysis = { error: String(e) };
		}

		if (s3) {
			const key = `${Date.now()}_${filename}`;
			await s3.upload({ Bucket: S3_BUCKET, Key: key, Body: buffer, ContentType: contentType }).promise();
			const url = `https://${S3_BUCKET}.s3.amazonaws.com/${encodeURIComponent(key)}`;
			console.log('File uploaded to S3:', url);
			return res.json({ ok: true, url, analysis });
		} else {
			const outPath = path.join(UPLOAD_DIR, `${Date.now()}_${filename}`);
			fs.writeFileSync(outPath, buffer);
			console.log('File saved locally:', outPath);
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

		let analysis: any;
		try {
			analysis = await analyzeImage(buffer, contentType);
			// Shape for frontend: analysis.labels and analysis.tsne_coordinates (same as Python /nn response)
			if (analysis.nn && analysis.nn.nn) {
				analysis.labels = analysis.nn.nn.map((n: any) => ({ description: n.path, confidence: n.score }));
				analysis.tsne_coordinates = analysis.nn.tsne_coordinates ?? null;
			}
		} catch (e: any) {
			console.error('Analyzer call failed:', e);
			analysis = { error: String(e) };
		}

		if (s3) {
			const key = `${Date.now()}_${filename}`;
			await s3.upload({ Bucket: S3_BUCKET, Key: key, Body: buffer, ContentType: contentType }).promise();
			const url = `https://${S3_BUCKET}.s3.amazonaws.com/${encodeURIComponent(key)}`;
			console.log('File uploaded to S3:', url);
			return res.json({ ok: true, url, analysis });
		} else {
			const outPath = path.join(UPLOAD_DIR, `${Date.now()}_${filename}`);
			fs.writeFileSync(outPath, buffer);
			console.log('File saved locally:', outPath);
			return res.json({ ok: true, path: outPath, analysis });
		}
	} catch (err: any) {
		console.error('Upload-form error:', err);
		return res.status(500).json({ error: err.message || String(err) });
	}
});

// Proxy POST /model/heatmap to the Python heatmap endpoint; forward cv, colormap, alpha
app.post('/model/heatmap', upload.single('photo'), async (req: Request & { file?: { originalname?: string; buffer: Buffer; mimetype?: string } }, res: ExpressResponse) => {
	try {
		if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
		const buffer = req.file.buffer;
		const useCv = String(req.query.cv || '0').toLowerCase();
		const colormap = String(req.query.colormap || 'jet');
		const alpha = String(req.query.alpha ?? '0.7');
		const cv = (useCv === '1' || useCv === 'true' || useCv === 'yes') ? '1' : '0';
		const qs = new URLSearchParams({ cv, colormap, alpha });
		const heatUrl = `${PY_SERVICE}/heatmap?${qs.toString()}`;
		const hRes = await fetch(heatUrl, { method: 'POST', headers: { 'Content-Type': req.file.mimetype || 'application/octet-stream' }, body: buffer as any });
		if (hRes.ok) {
			const hJson = await hRes.json();
			return res.json(hJson);
		} else {
			return res.status(502).json({ error: `Heatmap service returned status ${hRes.status}` });
		}
	} catch (e: any) {
		console.error('Heatmap proxy failed:', e);
		return res.status(500).json({ error: String(e) });
	}
});

// Serve images from uploads folder
app.get('/uploads/:filename', (req: Request, res: ExpressResponse) => {
	try {
		const filename = Array.isArray(req.params.filename) ? req.params.filename[0] : req.params.filename;
		if (!filename) {
			return res.status(400).json({ error: 'Filename required' });
		}
		const filePath = path.join(UPLOAD_DIR, filename);
		
		// Security: prevent directory traversal
		if (!fs.existsSync(filePath)) {
			return res.status(404).json({ error: 'File not found' });
		}
		
		// Check if file is actually in the uploads directory
		const resolvedPath = path.resolve(filePath);
		const resolvedUploadDir = path.resolve(UPLOAD_DIR);
		if (!resolvedPath.startsWith(resolvedUploadDir)) {
			return res.status(403).json({ error: 'Access denied' });
		}
		
		res.sendFile(resolvedPath);
	} catch (err: any) {
		console.error('Error serving file:', err);
		return res.status(500).json({ error: err.message || String(err) });
	}
});

// List all images in uploads folder
app.get('/api/images', (req: Request, res: ExpressResponse) => {
	try {
		if (!fs.existsSync(UPLOAD_DIR)) {
			return res.json({ images: [] });
		}
		
		const files = fs.readdirSync(UPLOAD_DIR);
		const imageFiles = files
			.filter(file => {
				const ext = path.extname(file).toLowerCase();
				return ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'].includes(ext);
			})
			.map(file => ({
				filename: file,
				url: `/uploads/${file}`,
				path: path.join(UPLOAD_DIR, file)
			}))
			.sort((a, b) => {
				// Sort by filename (which includes timestamp) descending (newest first)
				return b.filename.localeCompare(a.filename);
			});
		
		return res.json({ images: imageFiles });
	} catch (err: any) {
		console.error('Error listing images:', err);
		return res.status(500).json({ error: err.message || String(err) });
	}
});

app.listen(port, () => {
	console.log(`Backend listening on http://localhost:${port}`);
});

export {};
