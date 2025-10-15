# MNIST Array Job for USC Carc Cluster

This directory contains an array job setup for running MNIST training tasks on the USC Carc cluster.

## Files

- `array_job.slurm` - SLURM array job script
- `mnist_classify.py` - MNIST training script

## Usage

Submit the array job:
```bash
sbatch array_job.slurm
```

This will run 11 tasks with epochs 3-13.

## Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Cancel job
scancel <JOB_ID>
```

## Output

Each task creates output files:
- `mnist_<JOB_ID>_<TASK_ID>.out` - Standard output
- `mnist_<JOB_ID>_<TASK_ID>.err` - Error output
