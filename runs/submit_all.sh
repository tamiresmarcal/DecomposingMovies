#!/bin/bash
# submit_all.sh — Submit one SLURM job per movie
# Usage: bash submit_all.sh

mkdir -p slurm_logs

MOVIES=(
    #"12_years_a_slave"
    "500_days_of_summer"
    "back_to_the_future"
    "citizenfour"
    "little_miss_sunshine"
    "pulp_fiction"
    "split"
    "the_prestige"
    "the_shawshank_redemption"
    "the_usual_suspects"
)

for movie in "${MOVIES[@]}"; do
    echo "Submitting: $movie"
    sbatch \
        --job-name="surprise_${movie}" \
        --output="slurm_logs/%j_${movie}.out" \
        --error="slurm_logs/%j_${movie}.err" \
        --export=ALL \
        /home/tamires/projects/rpp-aevans-ab/tamires/DecomposingMovies/runs/run_movie.sh "$movie"
done

echo ""
echo "Submitted ${#MOVIES[@]} jobs. Check with: squeue -u \$USER"
