#!/bin/bash -l

CODE_SRC_DIR="/proj/sc_ml/users/x_hazko/scuphr_git/scuphr"
DATA_DIR="/proj/sc_ml/users/x_hazko/data_simulator/data"
RESULT_DIR="/proj/sc_ml/users/x_hazko/data_simulator/results"

SEED_VAL=1
NUM_CELLS=50

MUT_RATIO=10
SCUPHR_STRATEGY="paired"
PHASED_FREQ=0.25

TEST_PADO=0.1
TEST_PAE=0.01

for AVG_MUT in 12
do
	for COVERAGE in 5 10 20
	do
		for DATA_PADO in 0 0.1 0.2
		do
			for DATA_PAE in 0.001 
			do
				EXP_NAME="seed_${SEED_VAL}_cells_${NUM_CELLS}_avgmut_${AVG_MUT}_ratio_${MUT_RATIO}_phasedfreq_${PHASED_FREQ}_ado_${DATA_PADO}_ae_${DATA_PAE}_cov_${COVERAGE}"
				job_file="${DATA_DIR}/${EXP_NAME}/dbc_all.sh"

				echo "#!/bin/bash -l
#SBATCH -A snic2020-5-280
#SBATCH -n 10
#SBATCH -t 24:00:00
#SBATCH -J ${EXP_NAME}
#SBATCH -e ${DATA_DIR}/${EXP_NAME}/dbc_all_err.txt
#SBATCH -o ${DATA_DIR}/${EXP_NAME}/dbc_all_out.txt
echo \"$(date) Running on: $(hostname)\"

module load Python/3.7.0-anaconda-5.3.0-extras-nsc1
source activate /proj/sc_ml/users/x_hazko/scuphr_conda/scuphr_env
wait
echo \"$(date) Modules / environments are loaded\"

cd $CODE_SRC_DIR/src/
wait
echo \"$(date) Directory is changed\"

python3 analyse_dbc_json.py ${DATA_DIR}/${EXP_NAME}/ --seed_val $SEED_VAL --data_type synthetic --p_ado $TEST_PADO --p_ae $TEST_PAE --print_status 0 --scuphr_strategy $SCUPHR_STRATEGY --output_dir ${RESULT_DIR}/${EXP_NAME}/${TEST_PADO}_${TEST_PAE}/ 
wait
echo \"$(date) Distance between cells is finished\"

python3 analyse_dbc_combine_json.py ${DATA_DIR}/${EXP_NAME}/ --data_type synthetic --scuphr_strategy $SCUPHR_STRATEGY --output_dir ${RESULT_DIR}/${EXP_NAME}/${TEST_PADO}_${TEST_PAE}/ 
wait
echo \"$(date) Combine distances is finished\"

cp ${DATA_DIR}/${EXP_NAME}/truth/real.tre ${RESULT_DIR}/${EXP_NAME}/${TEST_PADO}_${TEST_PAE}/
wait
echo \"$(date) Copying real tree is finished\"

python3 lineage_trees.py ${RESULT_DIR}/${EXP_NAME}/${TEST_PADO}_${TEST_PAE}/ --data_type synthetic 
wait
echo \"$(date) Lineage trees is finished\"

echo \"$(date) All done!\" >&2" > $job_file

				sbatch $job_file
			done
		done
	done
done
