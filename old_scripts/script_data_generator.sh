#!/bin/bash -l

CODE_SRC_DIR="/proj/sc_ml/users/x_hazko/scuphr_git/scuphr"
DATA_DIR="/proj/sc_ml/users/x_hazko/data_simulator/data"
RESULT_DIR="/proj/sc_ml/users/x_hazko/data_simulator/results"

SEED_VAL=1
NUM_CELLS=50

MUT_RATIO=10
CHR_ID=1
READ_LENGTH=2
SCUPHR_STRATEGY="hybrid"
PHASED_FREQ=0.25

for AVG_MUT in 12
do
	for COVERAGE in 10 20
	do
		for DATA_PADO in 0 0.1
		do
			for DATA_PAE in 0.01 
			do
				EXP_NAME="seed_${SEED_VAL}_cells_${NUM_CELLS}_avgmut_${AVG_MUT}_ratio_${MUT_RATIO}_phasedfreq_${PHASED_FREQ}_ado_${DATA_PADO}_ae_${DATA_PAE}_cov_${COVERAGE}"
				mkdir ${DATA_DIR}/${EXP_NAME}
				job_file="${DATA_DIR}/${EXP_NAME}/simulation.sh"

				echo "#!/bin/bash -l
#SBATCH -A snic2020-5-278
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -J ${EXP_NAME}
#SBATCH -e ${DATA_DIR}/${EXP_NAME}/simulation_err.txt
#SBATCH -o ${DATA_DIR}/${EXP_NAME}/simulation_out.txt
echo \"$(date) Running on: $(hostname)\"

module load Python/3.7.0-anaconda-5.3.0-extras-nsc1
source activate /proj/sc_ml/users/x_hazko/scuphr_conda/scuphr_env
wait
echo \"$(date) Modules / environments are loaded\"

cd $CODE_SRC_DIR/src/
wait
echo \"$(date) Directory is changed\"

#python3 data_simulation_pairs.py --global_dir ${DATA_DIR}/${EXP_NAME}/ --num_cells $NUM_CELLS --p_ado $DATA_PADO --p_ae $DATA_PAE --mean_coverage $COVERAGE --seed_val $SEED_VAL --avg_mut_per_branch $AVG_MUT --no_mut_to_mut_ratio $MUT_RATIO --chr_id $CHR_ID --read_length $READ_LENGTH --phased_freq $PHASED_FREQ
wait
echo \"$(date) Data simulation is finished\"

#python3 site_detection.py ${DATA_DIR}/${EXP_NAME}/ $NUM_CELLS --read_length $READ_LENGTH --scuphr_strategy $SCUPHR_STRATEGY
wait
echo \"$(date) Site detection is finished\"

python3 generate_dataset_json.py ${DATA_DIR}/${EXP_NAME}/ $NUM_CELLS  --seed_val $SEED_VAL --chr_id $CHR_ID --read_length $READ_LENGTH --data_type synthetic
wait
echo \"$(date) Dataset generation is finished\"

echo \"$(date) All done!\" >&2" > $job_file

				sbatch $job_file
			done
		done
	done
done
