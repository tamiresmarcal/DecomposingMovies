#!/bin/bash
#SBATCH --account=def-aevans
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=8:00:00

MOVIE=$1

if [ -z "$MOVIE" ]; then
    echo "ERROR: no movie name provided"
    exit 1
fi

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

import sys
sys.path.insert(0, '/home/tamires/projects/rpp-aevans-ab/tamires/DecomposingMovies')
import cinematic_surprise
print('Origin cinematic surprise library:')
print(cinematic_surprise.__file__)

import torch
torch.zeros(1).cuda()
print('PyTorch CUDA locked:', torch.cuda.get_device_name(0))

import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')

try:
    import cinematic_surprise.config as cfg
    cfg.CLIP_DEVICE      = 'cuda'
    cfg.RESNET_DEVICE    = 'cuda'
    cfg.DEEPFACE_BACKEND = 'retinaface'
    cfg.BATCH_SIZE       = 16

    from cinematic_surprise import CinematicSurprisePipeline

    movie           = '$MOVIE'
    video_path      = f'/home/tamires/projects/rpp-aevans-ab/tamires/audiovisual_stimuli/{movie}.mp4'
    transcript_path = f'/home/tamires/projects/rpp-aevans-ab/tamires/audiovisual_stimuli/{movie}_transcript.csv'
    out_path1       = f'/home/tamires/projects/rpp-aevans-ab/tamires/DecomposingMovies/outputs/{movie}_surprise_uncertainty.parquet'
    out_path2       = f'/home/tamires/projects/rpp-aevans-ab/tamires/DecomposingMovies/outputs/{movie}_features.parquet'

    t0 = time.time()
    pipe = CinematicSurprisePipeline(batch_size=16)
    t_load = time.time() - t0
    print(f'MODEL LOAD TIME: {t_load:.1f}s')

    t0 = time.time()
    df1, df2 = pipe.run(video_path, transcript=transcript_path)
    t_run = time.time() - t0

    df1.to_parquet(out_path1, index=False)
    print('saved:', out_path1)
    df2.to_parquet(out_path2, index=False)
    print('saved:', out_path2)

    n_seconds = len(df1)
    per_second = t_run / max(n_seconds, 1)

    print(f'')
    print(f'========== PROFILING RESULTS ==========')
    print(f'Movie:                {movie}')
    print(f'Seconds processed:    {n_seconds}')
    print(f'Total run time:       {t_run:.1f}s')
    print(f'Per-second cost:      {per_second:.2f}s')
    print(f'Model load time:      {t_load:.1f}s')
    print(f'Shape df1:            {df1.shape}')
    print(f'Shape df2:            {df2.shape}')
    print(f'')
    print(f'──── MEMORY ────')
    import psutil
    mem = psutil.Process().memory_info()
    print(f'RSS (current):        {mem.rss / 1e9:.2f} GB')
    if torch.cuda.is_available():
        print(f'GPU mem allocated:    {torch.cuda.memory_allocated()/1e9:.2f} GB')
        print(f'GPU mem peak:         {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
    print(f'========================================')

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f'FAILED: {e}')
"
