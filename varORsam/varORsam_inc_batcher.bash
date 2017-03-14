#! /usr/bin/env bash

sbatch incremental_main.slurm 20 log variational
sbatch incremental_main.slurm 40 log variational
sbatch incremental_main.slurm 60 log variational
sbatch incremental_main.slurm 80 log variational
sbatch incremental_main.slurm 100 log variational
sbatch incremental_main.slurm 20 log sampling
sbatch incremental_main.slurm 40 log sampling
sbatch incremental_main.slurm 60 log sampling
sbatch incremental_main.slurm 80 log sampling
sbatch incremental_main.slurm 100 log sampling
