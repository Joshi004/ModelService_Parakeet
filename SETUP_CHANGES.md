# Parakeet ASR Service — Changes Applied

## System Facts (This Machine)

| Item | Value |
|------|-------|
| Current user | `vision` — scripts now use `${HOME}` (dynamic) |
| Home directory | `/home/vision` |
| GPUs | 8× NVIDIA H100 80GB HBM3 — **640 GB total VRAM** |
| CUDA toolkit | `12.4` at `/usr/local/cuda-12.4` (symlinked: `/usr/local/cuda`) |
| CUDA driver | 570.211.01 |
| Root disk `/` | ~535 GB free |
| Data disk `/mnt/hf-cache` | ~467 GB free — model weights stored here |

---

## Resource Assessment

| Requirement | Needed | Available | Status |
|-------------|--------|-----------|--------|
| GPU VRAM | ~2 GB | 80 GB (1× H100, GPU 4) | ✅ |
| Model disk space | ~600 MB | 467 GB on `/mnt/hf-cache` | ✅ |
| Python venv disk space | ~3–5 GB | 535 GB on `/` | ✅ |

**GPU assignment**: Parakeet uses GPU 4 (`CUDA_VISIBLE_DEVICES=4`). MiniMax-M2 uses GPUs 0-3.

---

## Changes Applied

### `start_service.sh` (edited)

| Item | Old | New |
|------|-----|-----|
| `CUDA_HOME` | `/usr/local/cuda-12.9` | `/usr/local/cuda` |
| `VENV_PATH` | `/home/naresh/venvs/parakeet-service` | `${HOME}/venvs/parakeet-service` |
| `mkdir -p` for venvs dir | `/home/naresh/venvs` | `${HOME}/venvs` |
| `cd` into service dir (×2) | `/home/naresh/parakeet-service` | `${SCRIPT_DIR}` |
| Datasets HTTP server block | Present (lines 48–91) | **Removed entirely** |
| Logs mkdir | `/home/naresh/parakeet-service/logs` | `${SCRIPT_DIR}/logs` |
| Tee log path | `/home/naresh/parakeet-service/logs/service.log` | `${SCRIPT_DIR}/logs/service.log` |
| `HF_HOME` export | _(missing)_ | `/mnt/hf-cache/huggingface` |
| `HUGGINGFACE_HUB_CACHE` export | _(missing)_ | `/mnt/hf-cache/huggingface/hub` |
| `NEMO_CACHE_DIR` export | _(missing)_ | `/mnt/hf-cache/nemo` |
| `SCRIPT_DIR` | _(missing)_ | `$(cd "$(dirname "$0")" && pwd)` |
| Auto-venv creation | Present | Kept and updated with dynamic paths |
| Config loading | `cat config.env` (relative) | `cat "${SCRIPT_DIR}/config.env"` |

### `config.env` (created — was missing)

New file created at `ModelService_Parakeet/config.env`:

| Variable | Value | Purpose |
|----------|-------|---------|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8006` | API port |
| `MODEL_NAME` | `nvidia/parakeet-tdt-0.6b-v3` | NeMo model identifier |
| `CUDA_VISIBLE_DEVICES` | `4` | Assign GPU 4 to Parakeet |
| `MAX_FILE_SIZE_MB` | `100` | Max upload size |
| `CHUNK_THRESHOLD_MB` | `50` | Files above this size are chunked |
| `CHUNK_DURATION` | `600` | Seconds per chunk (10 min) |
| `CHUNK_OVERLAP` | `10` | Overlap between chunks (seconds) |
| `TEMP_DIR` | `/tmp/parakeet_uploads` | Temp file location |

---

## Files Not Changed

| File | Reason |
|------|--------|
| `app.py` | No hardcoded user paths — uses `config.*` throughout |
| `config.py` | Fully dynamic — uses `os.getenv()` and `BASE_DIR` |
| `segment_utils.py` | Pure logic, no filesystem paths |
| `requirements.txt` | Package list only |

---

## How to Use

### First Time — Start Service

The Parakeet model (~600 MB) downloads automatically on first start via NeMo.
No manual model download needed.

```bash
cd ~/ModelService_Parakeet
./start_service.sh
# venv auto-created, dependencies installed, model downloaded, server starts
```

### Subsequent Starts

```bash
cd ~/ModelService_Parakeet
./start_service.sh
```

### Health Check

```bash
curl http://localhost:8006/health
```

### Transcribe Audio

```bash
curl -X POST http://localhost:8006/transcribe \
  -H "Content-Type: application/json" \
  -d '{"audio_url": "http://example.com/audio.wav"}'
```

### API Docs

Once running: http://localhost:8006/docs
