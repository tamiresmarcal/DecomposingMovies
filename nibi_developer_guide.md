# Cinematic Surprise Pipeline — Nibi Cluster Developer Guide

**Last updated:** March 2026
**Cluster:** Nibi (Digital Research Alliance of Canada, University of Waterloo — replaced Graham)
**Author:** Tamires Marcal

---

## Overview

This document captures hard-won lessons from setting up and debugging the `cinematic_surprise` pipeline on the Nibi HPC cluster. It covers container configuration, GPU issues, Jupyter setup, and common pitfalls. Read this before changing anything.

---

## Directory Structure

```
/home/tamires/projects/rpp-aevans-ab/tamires/
├── singularity/
│   └── cinematic_surprise1.sif        # Singularity container
├── audiovisual_stimuli/
│   ├── 12_years_a_slave.mp4           # Input videos
│   └── 12_years_a_slave_transcript.csv
├── DecomposingMovies/
│   └── outputs/                       # Pipeline output files
└── run_pipeline.sh                    # SBATCH job script
```

---

## Container (Singularity / Apptainer)

### Base Image

```
Bootstrap: docker
From: nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04
```

**Why 12.6 and not 12.1?** The Nibi H100 nodes run driver version **580.82.07** (CUDA 13.0 capable). The original container used CUDA 12.1, which was too old for this driver and caused `cuInit: UNKNOWN ERROR (303)`. CUDA 12.6 is the sweet spot — new enough for the driver, and the newest version with PyTorch prebuilt wheels.

**Why not CUDA 13.0?** PyTorch does not have prebuilt wheels for CUDA 13.0 yet. The driver is backward-compatible, so 12.6 works fine under a 13.0-capable driver.

### Key Dependencies

The container includes two deep learning frameworks that can conflict:

- **PyTorch** (CLIP, ResNet, sentence-transformers) — uses CUDA via its own runtime
- **TensorFlow** (pulled in by DeepFace) — uses CUDA via a separate runtime

These two frameworks fight over CUDA initialization. See the GPU section below.

### Critical Install Details

- **NumPy must be pinned to 1.x** (`numpy==1.26.4`). Several compiled packages (OpenCV, etc.) were built against NumPy 1.x and will crash with NumPy 2.x. The container force-reinstalls NumPy at the end to undo any upgrades pulled in by other packages.
- **opencv-python-headless** is used (not `opencv-python`) to avoid GUI library dependencies.
- **pyarrow** must be explicitly installed for parquet file support — pandas does not include it.
- **pytest** is safe to add and has no dependency conflicts.

### Renaming the .sif File

Safe to do. The `.sif` is self-contained. Just update paths in your scripts.

### Rebuilding the Container

The container is built from a `.def` file. To rebuild after changes:

```bash
singularity build --fakeroot cinematic_surprise1.sif cinematic_surprise.def
```

---

## GPU Configuration — The Critical Part

### The Problem

TensorFlow (imported by DeepFace) attempts to initialize CUDA at import time. On Nibi's H100 nodes, this fails with:

```
CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
```

When TensorFlow's CUDA init fails, **it poisons the CUDA context for PyTorch too**, making `torch.cuda.is_available()` return `False`. This means the entire pipeline runs on CPU (~18s per second of video instead of ~1.5s).

### The Solution (Two Parts)

**1. Import order matters.** PyTorch must grab the GPU *before* TensorFlow is imported:

```python
import torch
torch.zeros(1).cuda()  # Lock PyTorch onto the GPU FIRST
# Only now import the pipeline (which imports TensorFlow)
```

**2. Tell TensorFlow to use CPU only:**

```python
os.environ['TF_CUDA_VISIBLE_DEVICES'] = '-1'
```

This prevents TensorFlow from attempting CUDA init at all. DeepFace's face detection runs fine on CPU — the heavy GPU work (CLIP, ResNet) is all PyTorch.

### The LD_LIBRARY_PATH Trap

**NEVER set `--env LD_LIBRARY_PATH=""`** in your `singularity exec` call.

The `--nv` flag works by injecting the host's NVIDIA driver libraries into the container via `LD_LIBRARY_PATH`. Blanking this variable wipes out those paths, and PyTorch cannot find any NVIDIA driver at all:

```
RuntimeError: Found no NVIDIA driver on your system.
```

The original SBATCH script had this flag, and it was the root cause of all GPU failures.

### Required Environment Variables

```python
os.environ.pop('SSL_CERT_FILE', None)           # Remove broken SSL cert path from Alliance
os.environ['CURL_CA_BUNDLE'] = ''               # Disable SSL verification for model downloads
os.environ['REQUESTS_CA_BUNDLE'] = ''           # Same for requests library
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'       # Suppress oneDNN warnings
os.environ['TF_CUDA_VISIBLE_DEVICES'] = '-1'    # Force TensorFlow to CPU only
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # Prevent TF from grabbing all VRAM
os.environ['CUDA_VISIBLE_DEVICES'] = '0'        # Tell PyTorch which GPU to use
```

### For Login Node (No GPU)

When running on the login node (e.g., Jupyter without GPU):

```python
os.environ.pop('SSL_CERT_FILE', None)
os.environ['TF_CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''          # Empty = no GPU
```

Do NOT use `torch.zeros(1).cuda()` on the login node — it will crash.

---

## SBATCH Job Script (GPU Pipeline)

```bash
#!/bin/bash
#SBATCH --job-name=cinematic_surprise
#SBATCH --account=def-aevans
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:30:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
module load StdEnv/2023 apptainer/1.4.5
singularity exec --nv \
    --env PYTHONNOUSERSITE=1 \
    /home/tamires/projects/rpp-aevans-ab/tamires/singularity/cinematic_surprise1.sif python3 -u -c "
import os
os.environ.pop('SSL_CERT_FILE', None)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
torch.zeros(1).cuda()
print('PyTorch CUDA locked:', torch.cuda.get_device_name(0))

try:
    import cinematic_surprise.config as cfg
    cfg.CLIP_DEVICE      = 'cuda'
    cfg.RESNET_DEVICE    = 'cuda'
    cfg.DEEPFACE_BACKEND = 'retinaface'
    cfg.BATCH_SIZE       = 16
    from cinematic_surprise import CinematicSurprisePipeline
    movie           = '12_years_a_slave'
    video_path      = f'/home/tamires/projects/rpp-aevans-ab/tamires/audiovisual_stimuli/{movie}.mp4'
    transcript_path = f'/home/tamires/projects/rpp-aevans-ab/tamires/audiovisual_stimuli/{movie}_transcript.csv'
    out_path        = f'/home/tamires/projects/rpp-aevans-ab/tamires/DecomposingMovies/outputs/{movie}_surprise_uncertainty.csv'
    pipe = CinematicSurprisePipeline()
    print('Pipeline created, starting run...')
    df   = pipe.run(video_path, transcript=transcript_path)
    df.to_csv(out_path, index=False)
    print('saved:', out_path)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f'FAILED: {e}')
"
```

### Resource Estimates

Based on benchmarks with a 60-second test clip:

| Video Length | Processing Time | Recommended `--time` | Recommended `--mem` |
|---|---|---|---|
| 1 minute | ~2 minutes | 10 min | 16G |
| 90 minutes | ~2.7 hours | 3:30:00 | 32G |
| 2+ hours | ~4+ hours | 5:00:00 | 32G |

Processing speed: **~1.5–1.8 seconds per second of video** on H100 with GPU.
Without GPU (CPU fallback): **~16–18 seconds per second of video** (10x slower).

### Notes

- **`--env PYTHONNOUSERSITE=1`** prevents Python from loading packages from `~/.local/`, which can conflict with container packages (e.g., a different OpenCV or NumPy version).
- **4 CPUs** is sufficient — the GPU does the heavy lifting. CPUs handle video decoding.
- **`cfg.BATCH_SIZE = 16`** works well. You can try 32 or 64 given the 80GB VRAM, but the bottleneck is typically CPU-side video decoding.
- The `.err` file will contain TensorFlow warnings about libdevice and PTX compilation — these are harmless. The inline Python code may also appear repeated in the logs due to a TF directory scanner quirk; this is cosmetic only.

---

## Jupyter on GPU Node

### Launch

```bash
#!/bin/bash
#SBATCH --job-name=jupyter_gpu
#SBATCH --account=def-aevans
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
module load StdEnv/2023 apptainer/1.4.5
echo "Node: $(hostname)"
echo "Port: 8888"
singularity exec --nv --env PYTHONNOUSERSITE=1 \
    /home/tamires/projects/rpp-aevans-ab/tamires/singularity/cinematic_surprise1.sif \
    jupyter lab --no-browser --port=8888 --ip=0.0.0.0
```

### Connect

1. Check `.err` for the Jupyter URL with token
2. From your local machine:
   ```bash
   ssh -N -L 8888:<node_name>:8888 tamires@nibi.alliancecan.ca
   ```
3. Open `http://localhost:8888` and paste the token

### Jupyter on Login Node (No GPU)

```bash
singularity exec --env PYTHONNOUSERSITE=1 \
    /home/tamires/projects/rpp-aevans-ab/tamires/singularity/cinematic_surprise1.sif \
    jupyter lab --no-browser --port=8880
```

In your notebook, use the login-node env vars (no GPU):

```python
import os
os.environ.pop('SSL_CERT_FILE', None)
os.environ['TF_CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

---

## Common Pitfalls

### "CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)"
TensorFlow's CUDA init failed. Check that:
1. `--env LD_LIBRARY_PATH=""` is NOT in your singularity command
2. `TF_CUDA_VISIBLE_DEVICES` is set to `-1`
3. PyTorch is imported and `.cuda()` is called before the pipeline import

### "Found no NVIDIA driver on your system"
You blanked `LD_LIBRARY_PATH`. Remove `--env LD_LIBRARY_PATH=""`.

### "numpy.core.multiarray failed to import" / "_ARRAY_API not found"
NumPy version mismatch. You're probably loading packages from `~/.local/` instead of the container. Make sure `--env PYTHONNOUSERSITE=1` is set.

### "Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'"
pyarrow is not installed in the container. Either add it to the `.def` file or save as CSV instead of parquet.

### "Out Of Memory" on interactive srun
The default `srun` memory allocation is very small. Always specify `--mem=16G` (or more).

### "singularity: No such file or directory"
You forgot to load the module. Run `module load StdEnv/2023 apptainer/1.4.5` first.

### Noisy .err logs with repeated code blocks
Harmless. TensorFlow's libdevice directory scanner dumps the content of `python3 -c "..."` into its search output. Move your code to a `.py` file if it bothers you.

---

## Useful Commands

```bash
# Check running jobs
squeue -u $USER

# Cancel a job
scancel <job_id>

# Check job resource usage after completion
sacct -j <job_id> --format=JobID,State,ExitCode,MaxRSS,Elapsed,TimelimitRaw

# Check GPU driver on a compute node
srun --gres=gpu:h100:1 --account=def-aevans --time=0:05:00 --mem=16G nvidia-smi

# Quick PyTorch GPU test
module load StdEnv/2023 apptainer/1.4.5
srun --gres=gpu:h100:1 --account=def-aevans --time=0:05:00 --mem=16G \
    singularity exec --nv \
    /home/tamires/projects/rpp-aevans-ab/tamires/singularity/cinematic_surprise1.sif \
    python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
