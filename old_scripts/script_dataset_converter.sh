#!/bin/bash -l

CODE_SRC_DIR="/proj/sc_ml/users/x_hazko/scuphr_git/scuphr"
DATA_DIR="/proj/sc_ml/users/x_hazko/data_simulator/data"
RESULT_DIR="/proj/sc_ml/users/x_hazko/data_simulator/results"

SEED_VAL=1
NUM_CELLS=50

MUT_RATIO=1
CHR_ID=1
READ_LENGTH=2

for AVG_MUT in 6 12
do
	for COVERAGE in 5 10 20
	do
		for DATA_PADO in 0 0.1 0.2
		do
			for DATA_PAE in 0.001 0.01
			do
				EXP_NAME="seed_${SEED_VAL}_cells_${NUM_CELLS}_avgmut_${AVG_MUT}_ratio_${MUT_RATIO}_ado_${DATA_PADO}_ae_${DATA_PAE}_cov_${COVERAGE}"
				mkdir ${DATA_DIR}/${EXP_NAME}
				job_file="${DATA_DIR}/${EXP_NAME}/dataset_converter.sh"

				echo "#!/bin/bash -l
#SBATCH -A snic2020-5-278
#SBATCH -n 1
#SBATCH -t 2:00:00
#SBATCH -J ${EXP_NAME}
#SBATCH -e ${DATA_DIR}/${EXP_NAME}/dataset_converter_err.txt
#SBATCH -o ${DATA_DIR}/${EXP_NAME}/dataset_converter_out.txt
echo \"$(date) Running on: $(hostname)\"

module load Python/3.7.0-anaconda-5.3.0-extras-nsc1
source activate /proj/sc_ml/users/x_hazko/scuphr_conda/scuphr_env
wait
echo \"$(date) Modules / environments are loaded\"

cd $CODE_SRC_DIR/src/
wait
echo \"$(date) Directory is changed\"

python3 dataset_converter_pairs_to_singles.py ${DATA_DIR}/${EXP_NAME}/ 
wait
echo \"$(date) Dataset conversion is finished\"

echo \"$(date) All done!\" >&2" > $job_file

				sbatch $job_file
			done
		done
	done
done
