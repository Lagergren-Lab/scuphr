#!/bin/bash -l

CODE_SRC_DIR="/proj/sc_ml/users/x_hazko/scuphr_git/scuphr"
DATA_DIR="/proj/sc_ml/users/x_hazko/data_simulator/data"
RESULT_DIR="/proj/sc_ml/users/x_hazko/data_simulator/results"

SEED_VAL=1

SCUPHR_STRATEGY="hybrid"

DATA_TYPE="real"

NUM_CHAINS=3
MAX_ITER=5000
BURNIN=20

POS_RANGE_MIN=0
POS_RANGE_MAX=20

CNV_RATIO=0

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
						job_file="${DATA_DIR}/${EXP_NAME}/param_est.sh"

						echo "#!/bin/bash -l
#SBATCH -A snic2021-5-258
#SBATCH -n 10
#SBATCH -t 24:00:00
#SBATCH -J ${EXP_NAME}
#SBATCH -e ${DATA_DIR}/${EXP_NAME}/param_est_err.txt
#SBATCH -o ${DATA_DIR}/${EXP_NAME}/param_est_out.txt
echo \"$(date) Running on: $(hostname)\"

module load Python/3.7.0-anaconda-5.3.0-extras-nsc1
source activate /proj/sc_ml/users/x_hazko/scuphr_conda/scuphr_env
wait
echo \"$(date) Modules / environments are loaded\"

cd $CODE_SRC_DIR/src/
wait
echo \"$(date) Directory is changed\"

python3 run_parameter_estimation.py ${DATA_DIR}/${EXP_NAME}/ --seed_val $SEED_VAL --data_type ${DATA_TYPE} --num_chains $NUM_CHAINS --max_iter $MAX_ITER --burnin $BURNIN --pos_range_min $POS_RANGE_MIN --pos_range_max $POS_RANGE_MAX --print_status 0 --scuphr_strategy $SCUPHR_STRATEGY --output_dir ${RESULT_DIR}/${EXP_NAME}/ 
wait
echo \"$(date) Parameter estimation is finished\"

echo \"$(date) All done!\" >&2" > $job_file

						sbatch $job_file
					done
				done
			done
		done
	done
done

