#!/bin/bash -l

CODE_SRC_DIR="/proj/sc_ml/users/x_hazko/scuphr_git/scuphr"
DATA_DIR="/proj/sc_ml/users/x_hazko/data_simulator/data"
RESULT_DIR="/proj/sc_ml/users/x_hazko/data_simulator/results"

SEED_VAL=1

SCUPHR_STRATEGY="hybrid"

DATA_TYPE="real"

CNV_RATIO=0
for TEST_PADO in 0.064
do
	for TEST_PAE in 0.00006
	do
		for NUM_CELLS in 50 
		do
			for AVG_MUT in 20
			do
				for PHASED_FREQ in 0.01
				do
					for DATA_PADO in 0
					do
						for DATA_PAE in 0.00001
						do
							for COVERAGE in 10
							do
								EXP_NAME="seed_${SEED_VAL}_cells_${NUM_CELLS}_avgmut_${AVG_MUT}_phasedfreq_${PHASED_FREQ}_ado_${DATA_PADO}_ae_${DATA_PAE}_cov_${COVERAGE}"
								job_file="${DATA_DIR}/${EXP_NAME}/dbc_all.sh"

								echo "#!/bin/bash -l
#SBATCH -A snic2021-5-258
#SBATCH -n 10
#SBATCH -t 48:00:00
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

python3 analyse_dbc_json.py ${DATA_DIR}/${EXP_NAME}/ --seed_val $SEED_VAL --data_type ${DATA_TYPE} --p_ado $TEST_PADO --p_ae $TEST_PAE --print_status 0 --scuphr_strategy $SCUPHR_STRATEGY --output_dir ${RESULT_DIR}/${EXP_NAME}/${TEST_PADO}_${TEST_PAE}/ 
wait
echo \"$(date) Distance between cells is finished\"

python3 analyse_dbc_combine_json.py ${DATA_DIR}/${EXP_NAME}/ --data_type ${DATA_TYPE} --scuphr_strategy $SCUPHR_STRATEGY --output_dir ${RESULT_DIR}/${EXP_NAME}/${TEST_PADO}_${TEST_PAE}/ 
wait
echo \"$(date) Combine distances is finished\"

cp ${DATA_DIR}/${EXP_NAME}/truth/real.tre ${RESULT_DIR}/${EXP_NAME}/${TEST_PADO}_${TEST_PAE}/
wait
echo \"$(date) Copying real tree is finished\"

python3 lineage_trees.py ${RESULT_DIR}/${EXP_NAME}/${TEST_PADO}_${TEST_PAE}/ --data_type $DATA_TYPE
wait
echo \"$(date) Lineage trees is finished\"

tar -czf ${RESULT_DIR}/${EXP_NAME}/${TEST_PADO}_${TEST_PAE}/commonZstatus.tar.gz ${RESULT_DIR}/${EXP_NAME}/${TEST_PADO}_${TEST_PAE}/commonZstatus
wait
echo \"$(date) Compressing commonZstatus folder is finished\"

tar -czf ${RESULT_DIR}/${EXP_NAME}/${TEST_PADO}_${TEST_PAE}/matrix.tar.gz ${RESULT_DIR}/${EXP_NAME}/${TEST_PADO}_${TEST_PAE}/matrix
wait
echo \"$(date) Compressing matrix folder is finished\"

echo \"$(date) All done!\" >&2" > $job_file

								sbatch $job_file
							done
						done
					done
				done
			done
		done
	done
done
