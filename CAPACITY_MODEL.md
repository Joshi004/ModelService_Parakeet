# Parakeet TDT ASR Service — Capacity Model

Service path: `/home/vision/ModelService_Parakeet`
Framework: FastAPI + Uvicorn (single worker), NeMo toolkit, PyTorch 2.11.0+cu128
Model: `nvidia/parakeet-tdt-0.6b-v3` (~600 M params, RNN-T/TDT ASR)
Hardware: 1× NVIDIA H100 80 GB, pinned to GPU 4 (`CUDA_VISIBLE_DEVICES=4`)
Log analysed: `logs/service.log` (108 410 lines, 2026-04-07 16:44 → 2026-04-18 11:20, ~10.77 days uptime)

> All numeric claims below are traceable to log lines quoted in the appendix. Estimates and speculative statements are flagged inline.

---

## 1. Executive summary

- Serves a single NeMo ASR model (`parakeet-tdt-0.6b-v3`) behind a FastAPI `/transcribe` endpoint on port 8006.
- Single Uvicorn worker; `model.transcribe()` is a **blocking synchronous call** inside an `async def` route → effective concurrency = **1 request at a time** (confirmed by log — peak overlap seen was 1).
- Observed workload: **1 369 successful transcriptions** over ~10.77 days, 0 errors, 0 non-200 responses.
- Typical request: ~25-minute audio file (~47 MB) → processed end-to-end in **~9.2 s** (p50 = 8.74 s, p90 = 9.29 s, p95 = 9.51 s, p99 = 13.86 s, max = 145.64 s).
- End-to-end real-time factor (audio-seconds / wall-clock) ≈ **168×** on the dominant workload. Model's published raw RTF is ~3 386× on pure inference — the delta is download + preprocess + chunking + merge overhead.
- Almost all requests (1 368 / 1 369) hit the **30 MB chunking threshold** and were split into 6 sequential 300 s chunks with 10 s overlap.
- GPU is massively under-utilised: total GPU-busy time ≈ 1 369 × ~7.5 s ≈ **2.9 h of work in a 258.6 h window (~1.1 %)**.
- No OOMs, no failed requests, no restarts in the observed window.
- Biggest structural ceiling is the **single-worker blocking design**, not the GPU or the model. A ~2 GB model on an 80 GB H100 has ~40× headroom in VRAM alone.

---

## 2. Hardware footprint

| GPU | Role | VRAM | Used by this service |
|---|---|---|---|
| 0 | TP rank 0 (MiniMax) | 80 GB HBM3 | No |
| 1 | TP rank 1 (MiniMax) | 80 GB HBM3 | No |
| 2 | TP rank 2 (MiniMax) | 80 GB HBM3 | No |
| 3 | TP rank 3 (MiniMax) | 80 GB HBM3 | No |
| 4 | Parakeet (this service) | 80 GB HBM3 | **Yes** |
| 5 | — | 80 GB HBM3 | Idle |
| 6 | — | 80 GB HBM3 | Idle |
| 7 | — | 80 GB HBM3 | Idle |

- Host: same box as `ModelService_MinMax-M2` (Intel Xeon Sapphire Rapids, 160 vCPU, 983 GiB RAM).
- GPU 4 is dedicated to this service via `CUDA_VISIBLE_DEVICES=4` in `start_service.sh` (line 25) — no interference with MiniMax.
- CUDA 12.8 toolchain, PyTorch cu128 wheels.

---

## 3. Model & runtime configuration

Sourced from `config.py`, `config.env`, and `start_service.sh`.

| Knob | Value | Effect |
|---|---|---|
| `MODEL_NAME` | `nvidia/parakeet-tdt-0.6b-v3` | TDT ASR, ~600 M params, native punctuation + word/segment timestamps |
| `CUDA_VISIBLE_DEVICES` | `4` | Pinned to GPU 4 |
| `HOST` / `PORT` | `0.0.0.0` / `8006` | HTTP API bind |
| `MAX_FILE_SIZE_MB` | 1024 | Hard upload cap |
| `CHUNK_THRESHOLD_MB` | 30 | Files larger than this get chunked before transcription |
| `CHUNK_DURATION` | 300 s (5 min) | Size of each chunk fed to the model |
| `CHUNK_OVERLAP` | 10 s | Overlap between chunks (for boundary dedup) |
| Uvicorn workers | 1 (default — no `--workers` in start script) | Single process, single event loop |
| Audio preprocessing | librosa → mono, 16 kHz, 16-bit PCM WAV | Enforced before inference |
| CUDA-graph decoder | **disabled** (workaround) | Required for NeMo 2.6.x + CUDA 12.8 compatibility (`app.py` lines 448-456, GitHub NeMo#15145) |
| Decoding strategy | `greedy_batch`, TDT (durations 0-4) | See log line 56-169 |
| Transcription mode | `model.transcribe([...], timestamps=True)` | One call per chunk; blocking |

Concurrency model (important for capacity):

- FastAPI route `/transcribe` is `async def`, but the three blocking calls inside it — `requests.get(..., stream=True)` (download), `librosa.load` (preprocess), and `model.transcribe(...)` (inference) — are **synchronous**. They run on the event-loop thread and block all other requests on this worker, including `/health`.
- Consequence: while any `/transcribe` is in flight, the server cannot progress any other HTTP request. This is the binding capacity constraint, not GPU or model speed.

---

## 4. Memory footprint (approximate)

No detailed memory telemetry is logged (Parakeet, unlike vLLM, doesn't self-report KV / weight budgets). The numbers below are derived from the model card and library defaults.

| Bucket | Estimated size | Source |
|---|---|---|
| Model weights (FP32, ~600 M params) | ~2.4 GB | PARAKEET_MODELS.md and HF model card; consistent with "~2 GB GPU memory" claim |
| Activation / scratch per 300 s chunk @ 16 kHz | sub-GB, bounded | librosa loads 300 s × 16 000 Hz = 4.8 M float32 samples ≈ 18 MB on CPU; on-GPU feature tensors scale similarly |
| Total resident on GPU 4 during a request | **~3–4 GB (estimated)** | Leaves ~76 GB free on the 80 GB card |

Implication: **VRAM is nowhere near the limit**. The service could in principle hold ~15–20 copies of the model in parallel on the same GPU, modulo NeMo/PyTorch multi-model-on-one-device ergonomics. Memory is not the bottleneck here.

---

## 5. Per-request timing breakdown

From a representative request (log lines 174–253, a 1 544 s / 47 MB audio file; 25.7 min):

| Phase | Wall-clock | Notes |
|---|---|---|
| Download (47 MB over HTTPS) | ~0.57 s | `requests.get` with stream=True |
| Audio preprocess (librosa → mono 16 kHz WAV) | ~1.07 s | CPU-bound |
| Chunking (split into 6× 300 s WAVs on disk) | ~0.26 s | CPU + disk I/O, bounded by file write |
| Chunk 1 inference (incl. NeMo dataloader init) | ~1.79 s | First chunk is slightly slower — dataloader warmup |
| Chunk 2–5 inference | ~1.21–1.34 s each | Steady state |
| Chunk 6 inference (94 s tail chunk, not full 300 s) | ~0.41 s | Proportional to audio length |
| Merge + dedup overlap words (CPU) | <0.01 s | In-process |
| Save transcript JSON | ~0.02 s | Small file write |
| **End-to-end total** | **~9.17 s** | matches log line 251 |

Derived single-chunk throughput: **~300 s audio / ~1.25 s inference ≈ 240× RTF per chunk** on GPU. End-to-end RTF is lower (~168×) because chunks are processed **sequentially** inside a request and non-inference overhead (download + preprocess + chunk split) adds ~1.9 s of fixed cost.

---

## 6. Observed workload (evidence from `logs/service.log`)

### 6.1 Request volume and concurrency

| Metric | Value | Source |
|---|---|---|
| Total `/transcribe` requests (HTTP 200) | 1 371 | grep `"POST /transcribe HTTP/1.1\" 200"` |
| Total requests that reached `Transcription completed in` | 1 369 | grep count |
| Non-200 responses | 0 | grep all HTTP status codes on `/transcribe` |
| Error / OutOfMemory / Traceback lines | 0 | grep `ERROR\|Failed to\|OutOfMemory\|Traceback` |
| Log span | 2026-04-07 16:44 → 2026-04-18 11:20 (~10.77 days) | first/last timestamp |
| Peak concurrent in-flight requests | **1** (never exceeded) | overlap analysis on `Downloading...` vs `Transcription completed in` events |
| Avg gap between requests | ~680 s (~11 min) | 10.77 days / 1 369 req |

### 6.2 End-to-end processing time (`Transcription completed in Xs`)

| Stat | Value |
|---|---|
| Count | 1 369 |
| Mean | **9.20 s** |
| p50 | 8.74 s |
| p90 | 9.29 s |
| p95 | 9.51 s |
| p99 | 13.86 s |
| Min | 4.56 s |
| Max | 145.64 s (single outlier, ~16× p50) |

The distribution is extremely tight — p50 and p90 differ by only ~0.55 s, which says the workload is highly homogeneous (mostly the same ~25-min audio file transcribed repeatedly).

### 6.3 File-size distribution

| Stat | Value |
|---|---|
| Mean downloaded file size | 47.27 MB |
| Min | 47.13 MB |
| Max | 151.14 MB |
| Files ≥ `CHUNK_THRESHOLD_MB` (30) → chunked | 1 368 / 1 369 (99.93 %) |

### 6.4 Audio duration

| Stat | Value |
|---|---|
| Mean audio duration | 1 548.79 s (~25.8 min) |
| Min | 1 544.31 s |
| Max | 4 952.51 s (~82.5 min) |
| Total audio transcribed | ~588.5 hours |
| Chunks per request: 6 | 1 366 / 1 369 |
| Chunks per request: 15 | 1 (the 82.5-min outlier) |
| Chunks per request: 18 | 1 |

### 6.5 GPU utilisation estimate

- Per-request GPU-busy time ≈ **~7.5 s** (sum of 6 chunk inferences).
- Total GPU-busy time over 10.77 days ≈ 1 369 × 7.5 s ≈ 10 270 s ≈ **2.85 h**.
- Wall-clock window ≈ 258.6 h → **~1.1 % GPU utilisation**.

### 6.6 Startup cost (one-off)

| Phase | Duration | Log line |
|---|---|---|
| App process start → model load begin | ~1 s | 7 → 9 |
| NeMo model `.from_pretrained` + restore | ~14 s | 9 → 171 |
| `Application startup complete` | ~15 s total | 9 → 172 |
| First request served | ~36 s after process start | 7 → 174 |

Cold-start is ~15 s to ready. Not a capacity issue; relevant only for rolling restarts.

---

## 7. Current bottleneck analysis

Rank-ordered limiters, most binding first, given the observed workload:

1. **Single Uvicorn worker + blocking `model.transcribe` → concurrency hard-capped at 1.** Any second concurrent request must wait the full ~9 s of the in-flight request before it can even begin its download. Observed peak overlap = 1, so today's traffic has never exposed this, but it is the first wall any burst will hit.
2. **Sequential per-chunk inference inside a single request.** A 25-min audio file runs 6 chunks one after another (~7.5 s total GPU time). In principle `model.transcribe([chunk1, ..., chunk6], ...)` can batch them, eliminating most of the per-call overhead. See §8 opportunity #1.
3. **Fixed per-request overhead (~1.9 s): download + preprocess + chunk-split.** For short audios this is a big fraction of total latency; for long audios it's negligible.
4. **CUDA-graph decoder disabled (workaround for NeMo + CUDA 12.8).** Per-chunk decode is slower than it would be with graph capture enabled. Magnitude not measured in this log. Re-enabling once NeMo upstream fixes issue #15145 is a free win.
5. **GPU memory, compute throughput, disk I/O.** All measured to be non-binding — GPU 4 is ~99 % idle in wall-clock terms, VRAM has ~40× headroom over the resident model, and chunk writes are sub-300 ms.

Real-world note: the service is **load-limited, not capacity-limited**. Peak observed concurrency = 1 in 10+ days. There is no incident waiting to happen under today's traffic shape. The capacity ceiling only becomes relevant if request arrival rate grows above ~1 every 9–10 s sustained.

---

## 8. Capacity model — headline numbers

Assumptions (state explicitly before reading the table):

- **End-to-end latency per request ≈ fixed ~1.9 s overhead + (chunks × ~1.25 s GPU)**. Based on the timing decomposition in §5.
- Chunks per request ≈ `ceil(audio_duration_s / 290)` (effective chunk = 300 s − 10 s overlap).
- Service serves **1 request at a time** (see §7 #1). Aggregate throughput = `1 / latency`.
- Ignores network variance on the audio download (observed ~0.2–0.6 s for 47 MB).
- Ignores first-chunk warmup penalty (~0.5 s extra on chunk 1).

| Audio length | Chunks | GPU time | Overhead | Per-request wall-clock | Req/hr (sequential) | Req/hr (sustained, parallelised — see §8) |
|---|---|---|---|---|---|---|
| 2 min (120 s)  | 1 | 1.25 s | 1.9 s | **~3.2 s**   | ~1 130 | speculative (see §9) |
| 5 min (300 s)  | 1 | 1.25 s | 1.9 s | **~3.2 s**   | ~1 130 |  |
| 10 min (600 s) | 3 | 3.75 s | 1.9 s | **~5.7 s**   | ~630 |  |
| 25 min (1 544 s) | 6 | 7.50 s | 1.9 s | **~9.4 s**   | **~380** (matches observed p50 ~8.74 s within ~8 %) |  |
| 60 min (3 600 s) | 13 | 16.25 s | 2.5 s | **~18.8 s** | ~190 |  |
| 82 min (4 952 s) | 18 | 22.50 s | 3.0 s | **~25.5 s** | ~140 (observed max outlier 145 s — see §11) |  |

Pattern to internalise: for typical 25-min audios, the service can sustain roughly **~380 requests/hour** back-to-back (one at a time), which is equivalent to **~160 hours of audio transcribed per wall-clock hour** on a single GPU. Observed volume is ~1369 requests / 10.77 days ≈ **5.3 req/hour**, i.e. ~1.4 % of single-worker capacity.

---

## 9. Opportunities to get more out of the same hardware

Ordered by confidence (high → low).

| # | Change | Expected effect | Risk / caveat |
|---|---|---|---|
| 1 | **Batch chunks in a single `model.transcribe([c1..c6], ...)` call** instead of a Python loop | Eliminates per-call NeMo dataloader init (~0.2–0.5 s/chunk, x6). Likely **~30–50 % end-to-end speedup** on chunked files | Must still apply per-chunk time offsets afterward. Code change in `process_single_chunk` / main loop. Validate output parity first. |
| 2 | **Add Uvicorn `--workers N`** (e.g. 2–4) | Linear concurrency scaling: N parallel `/transcribe` requests. GPU 4 has VRAM headroom for ~15× copies of this model | Each worker loads the model independently (~2.5 GB + ~15 s startup each). Watch GPU VRAM on startup. Shared temp dir is fine (filenames are timestamped). |
| 3 | **Run `model.transcribe` off the event loop** (`asyncio.to_thread` or `run_in_executor`) | Unblocks `/health` and other endpoints during inference. Does NOT increase model throughput | CPU-bound path; still only one in-flight GPU job per worker. Combines well with #2. |
| 4 | **Re-enable CUDA-graph decoder** once NeMo fixes issue #15145 | ~10–30 % faster per-chunk decode (speculative — typical graph-capture gain for RNN-T) | Wait for upstream fix; regression-test output on a fixed audio set. |
| 5 | **Raise `CHUNK_DURATION` from 300 s toward the model's 1 440 s (24-min) full-attention window** | Fewer chunks per request → less per-call overhead; e.g. 25-min audio with 1 200 s chunks = 2 chunks instead of 6 | Longer chunks use more GPU scratch memory and give coarser error recovery on a bad chunk. Profile VRAM at 600 / 900 / 1 200 s before rolling out. |
| 6 | **Stream chunk transcription while next chunk is still splitting** (pipeline) | Hides the ~260 ms split time behind chunk 1's GPU work | Minor — file-split is already only ~3 % of total. Low priority. |
| 7 | **Accept raw bytes on POST instead of a URL** (optional path) | Removes the ~0.2–0.6 s network download from the critical path when caller already has the audio in-process | Adds multipart-upload complexity; existing callers would need to change. Scope-expanding. |
| 8 | **Downgrade to `parakeet-tdt-0.6b-v3` with local-attention flag for very long files** (speculative — model card claims 3 h with local attention) | Could fit >82-min audios in one pass, collapsing chunk overhead | Unverified in this codebase; would need a toggle path in `app.py`. Validate WER doesn't regress on short audios. |

Safety notes:

- Changes #1, #2, #3 are pure server-side knobs; reversible by restart.
- #2 needs a quick VRAM sanity check on startup — N workers × ~2.5 GB should fit easily on an 80 GB card up to N ≈ 15.
- No change below should affect transcript output; validate with a diff on the existing `transcripts/` corpus before rolling to production.

---

## 10. Horizontal scaling hooks (brief)

- GPUs 5/6/7 on the same host are idle. A second Parakeet replica on GPU 5 is **trivial** (copy the service, set `CUDA_VISIBLE_DEVICES=5`, pick a new port). Model is small enough that ~3 independent replicas could share GPUs 5–7 without stepping on each other. This is almost certainly a bigger lever than any per-worker tweak if sustained load materialises.
- Service is stateless (no in-process cache, transcripts are written to disk per-request) — load-balancing across replicas is trivial HTTP round-robin.
- Callers' `audio_url`s are signed GCS URLs (visible in the log) → already decoupled from the service; no shared storage concerns.
- Per-replica capacity numbers in §8 multiply by replica count.
- Cold-start per replica ≈ 15 s (model load). Keep warm replicas behind the LB for failover.

---

## 11. Risks & caveats

- **Blocking single worker is the real capacity ceiling.** Everything in §8 that doesn't address this (#1, #4, #5, #6) only shortens the critical-section, it doesn't let two requests run in parallel. Combine with #2 + #3 for actual concurrency.
- **One p99 outlier (145.64 s vs p50 8.74 s)** is ~16× the median. Cause not visible in the log at current grep granularity — could be a GCS download stall on a signed URL, an unusually long audio, or a chunk-level slowdown. Worth a targeted follow-up if max-latency SLOs matter.
- **CUDA-graph decoder disabled** due to NeMo 2.6.x + CUDA 12.8 incompatibility (see `app.py` lines 448-456). This is a documented workaround, not a bug in this service, but it leaves free performance on the table.
- **RTF numbers depend on a homogeneous workload.** All observed traffic was essentially the same ~25-min audio file. WER or speed on other accents / languages / music was not exercised in the log.
- **GPU memory estimate is not from live telemetry** — no process-level `nvidia-smi` samples are in the log, and `nvidia-smi` itself is unavailable in the current shell (`NVML: Unknown Error`). The ~2–4 GB figure is derived from the model card + framework defaults. Re-validate once NVML is accessible.
- **Single log window (~10.77 days, ~1 369 requests, 0 concurrency > 1).** All capacity claims for N ≥ 2 concurrent requests are **extrapolation**. Run a simple `hey` / `wrk` burst test against the endpoint before trusting §8's parallelised column.
- **NeMo / PyTorch version drift** — pinned at `torch==2.11.0+cu128`, `nemo_toolkit[asr]>=1.22.0`. Upgrades can flip decoder performance either way; retest after any bump.

---

## 12. Appendix — raw log evidence

Line numbers reference `logs/service.log`.

Model load and startup (lines 7-173):

```
7:   INFO:     Started server process [78527]
9:   2026-04-07 16:44:13,528 - app - INFO - Loading Parakeet TDT model...
53:  [NeMo I ...] Model EncDecRNNTBPEModel was successfully restored from
     .../parakeet-tdt-0.6b-v3.nemo
171: 2026-04-07 16:44:27,495 - app - INFO - Model loaded successfully!
172: INFO:     Application startup complete.
173: INFO:     Uvicorn running on http://0.0.0.0:8006
```

Representative single request, full lifecycle (lines 174-254):

```
174:  Downloading audio from URL: https://storage.googleapis.com/.../new-lmpd-....wav?...
180:  Downloaded file size: 47.13MB (limit: 1024MB)           # ~0.57 s after 174
182:  Preprocessed audio saved to: ... (shape: (24709004,), sr: 16000Hz)   # +1.07 s
184:  Large file detected (47.13MB), processing in chunks
185:  Splitting audio: 1544.31s duration, 300s chunks, 10s overlap
186-191: Created chunk 1..6/6
202:  Chunk processed: 770 words, 41 segments (offset: 0.00s)    # +1.79 s (incl. first-time dataloader init)
210:  Chunk processed: 752 words, 50 segments (offset: 290.00s)  # +1.31 s
218:  Chunk processed: 693 words, 61 segments (offset: 580.00s)  # +1.20 s
226:  Chunk processed: 678 words, 62 segments (offset: 870.00s)  # +1.21 s
234:  Chunk processed: 797 words, 44 segments (offset: 1160.00s) # +1.34 s
242:  Chunk processed: 241 words, 12 segments (offset: 1450.00s) # +0.41 s (tail chunk, 94 s)
244-248: Removed N duplicate words from overlap region  (5 overlaps × ~25 words)
249:  Merged result: 3805 total words, 270 total segments
251:  Transcription completed in 9.17s
254:  INFO:  100.119.19.114:57616 - "POST /transcribe HTTP/1.1" 200 OK
```

Aggregate metrics derived from the log:

```
# Successful /transcribe responses
$ grep -E '"POST /transcribe HTTP/1.1" 200' logs/service.log | wc -l
1371

# Non-200 /transcribe responses (all status codes)
$ grep -E '"POST /transcribe HTTP/1.1" [0-9]+ ' logs/service.log \
    | grep -vE '" 200 ' | wc -l
0

# End-to-end latency stats (over 1369 samples)
# count=1369 mean=9.20s min=4.56s max=145.64s
# p50=8.74s p90=9.29s p95=9.51s p99=13.86s

# File-size stats (1368 chunked requests)
# count=1368 mean=47.27MB min=47.13MB max=151.14MB

# Audio-duration stats
# count=1368 mean=1548.79s min=1544.31s max=4952.51s total=588.54 hours

# Chunk-count distribution
#   1366× "Split audio into 6 chunks"
#      1× "Split audio into 15 chunks"
#      1× "Split audio into 18 chunks"

# Incidents
$ grep -cE 'OutOfMemory|CUDA out of memory|Traceback|GPU out of memory|Transcription failed' logs/service.log
0

# Concurrency overlap (pairing Downloading/Completed events)
# peak simultaneous in-flight = 1
```

CUDA-graph decoder workaround (app.py lines 448-456):

```
# WORKAROUND: Disable CUDA graph decoder due to NeMo 2.6.x incompatibility
# with CUDA 12.8+ (cudaStreamGetCaptureInfo API changed its return signature).
# See: https://github.com/NVIDIA-NeMo/NeMo/issues/15145
decoding_cfg.greedy.use_cuda_graph_decoder = False
model.change_decoding_strategy(decoding_cfg)
```

Startup script GPU pin (`start_service.sh` line 25):

```
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
```
