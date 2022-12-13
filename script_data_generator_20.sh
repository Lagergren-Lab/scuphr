#!/bin/bash -l

CODE_SRC_DIR="/proj/sc_ml/users/x_hazko/scuphr_git/scuphr"
DATA_DIR="/proj/sc_ml/users/x_hazko/data_simulator/data"
RESULT_DIR="/proj/sc_ml/users/x_hazko/data_simulator/results"

NUM_CELLS=20

SEED_VAL=1
CHR_ID=1
READ_LENGTH=2
GENOME_LENGTH=1999000
SCUPHR_STRATEGY="hybrid"

CNV_RATIO=0

for AVG_MUT in 10 20
do
	for PHASED_FREQ in 1
	do
		for DATA_PADO in 0 0.1 0.2
		do
			for DATA_PAE in 0.00001
			do
				for COVERAGE in 5 10 
				do
					EXP_NAME="seed_${SEED_VAL}_cells_${NUM_CELLS}_avgmut_${AVG_MUT}_phasedfreq_${PHASED_FREQ}_ado_${DATA_PADO}_ae_${DATA_PAE}_cov_${COVERAGE}"
					mkdir ${DATA_DIR}/${EXP_NAME}
					job_file="${DATA_DIR}/${EXP_NAME}/simulation.sh"

					echo "#!/bin/bash -l
#SBATCH -A snic2021-5-258
#SBATCH -n 5
#SBATCH -t 48:00:00
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

python3 data_simulation_pairs.py --global_dir ${DATA_DIR}/${EXP_NAME}/ --num_cells $NUM_CELLS --p_ado $DATA_PADO --p_ae $DATA_PAE --mean_coverage $COVERAGE --seed_val $SEED_VAL --avg_mut_per_branch $AVG_MUT --chr_id $CHR_ID --read_length $READ_LENGTH --phased_freq $PHASED_FREQ --genome_length $GENOME_LENGTH
wait
echo \"$(date) Data simulation is finished\"

samtools mpileup --fasta-ref ${DATA_DIR}/${EXP_NAME}/truth/bulk_ref.fasta --no-BAQ --output ${DATA_DIR}/${EXP_NAME}/bam/cells_pile.mpileup ${DATA_DIR}/${EXP_NAME}/bam/bulk.bam   ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_0.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_1.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_2.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_3.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_4.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_5.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_6.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_7.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_8.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_9.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_10.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_11.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_12.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_13.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_14.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_15.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_16.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_17.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_18.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_19.bam  
wait
echo "$(date) Pileup is finished"

python3 site_detection.py ${DATA_DIR}/${EXP_NAME}/ $NUM_CELLS --read_length $READ_LENGTH --scuphr_strategy $SCUPHR_STRATEGY --genome_length $GENOME_LENGTH
wait
echo \"$(date) Site detection is finished\"

python3 generate_dataset_json.py ${DATA_DIR}/${EXP_NAME}/ $NUM_CELLS  --seed_val $SEED_VAL --chr_id $CHR_ID --read_length $READ_LENGTH --data_type real
wait
echo \"$(date) Dataset generation is finished\"

echo \"$(date) All done!\" >&2" > $job_file

					sbatch $job_file
				done
			done
		done
	done
done
