#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=preprocess
#SBATCH --output=slurm-out-preprocess/%j-preprocess.out  # %j for jobID
#SBATCH --mail-user=alan.diaz@uhnresearch.ca
#SBATCH --mail-type=ALL
module load python/3.9
source venv/bin/activate

PYTHONPATH=. python projects/gnn_to_soz/preprocess.py
EOT
