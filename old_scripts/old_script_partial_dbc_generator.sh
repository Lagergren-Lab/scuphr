#!/bin/bash -l

CODE_SRC_DIR="/proj/sc_ml/users/x_hazko/scuphr_git/scuphr"
DATA_DIR="/proj/sc_ml/users/x_hazko/data_simulator/data"
RESULT_DIR="/proj/sc_ml/users/x_hazko/data_simulator/results"

SEED_VAL=1
NUM_CELLS=50

AVG_MUT=12
MUT_RATIO=1
COVERAGE=20

TEST_PADO=0.1
TEST_PAE=0.01

DATA_PAE=0.01
for DATA_PADO in 0 0.1 0.2 
do
	EXP_NAME="seed_${SEED_VAL}_cells_${NUM_CELLS}_avgmut_${AVG_MUT}_ratio_${MUT_RATIO}_ado_${DATA_PADO}_ae_${DATA_PAE}_cov_${COVERAGE}"
	job_file="${DATA_DIR}/${EXP_NAME}/dbc_partial_12.sh"

	echo "#!/bin/bash -l
#SBATCH -A snic2020-5-278
#SBATCH -n 10
#SBATCH -t 24:00:00
#SBATCH -J ${EXP_NAME}
#SBATCH -e ${DATA_DIR}/${EXP_NAME}/dbc_partial_12_err.txt
#SBATCH -o ${DATA_DIR}/${EXP_NAME}/dbc_partial_12_out.txt
echo \"$(date) Running on: $(hostname)\"

module load Python/3.7.0-anaconda-5.3.0-extras-nsc1
source activate /proj/sc_ml/users/x_hazko/scuphr_conda/scuphr_env
wait
echo \"$(date) Modules / environments are loaded\"

cd $CODE_SRC_DIR/src/
wait
echo \"$(date) Directory is changed\"

python3 analyse_dbc_json.py ${DATA_DIR}/${EXP_NAME}/ --seed_val $SEED_VAL --data_type synthetic --p_ado $TEST_PADO --p_ae $TEST_PAE --print_status 0 --pos_range_min 1200 --pos_range_max 0 --output_dir ${RESULT_DIR}/${EXP_NAME}/${TEST_PADO}_${TEST_PAE}/ 
wait
echo \"$(date) Distance between cells is finished\"

echo \"$(date) All done!\" >&2" > $job_file

	sbatch $job_file

done
