#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --account=def-aevans
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
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
    out_path        = f'/home/tamires/projects/rpp-aevans-ab/tamires/DecomposingMovies/outputs/{movie}_surprise_uncertainty.parquet'
    pipe = CinematicSurprisePipeline() #CinematicSurprisePipeline(max_seconds=60)
    print('Pipeline created, starting run...')
    df   = pipe.run(video_path, transcript=transcript_path)
    df.to_parquet(out_path, index=False)
    print('saved:', out_path)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f'FAILED: {e}')

"