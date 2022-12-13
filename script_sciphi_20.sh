#!/bin/bash -l

CODE_SRC_DIR="/proj/sc_ml/users/x_hazko/scuphr_git/scuphr"
DATA_DIR="/proj/sc_ml/users/x_hazko/data_simulator/data"
RESULT_DIR="/proj/sc_ml/users/x_hazko/data_simulator/results"

SEED_VAL=1
NUM_CELLS=20

TEST_PADO=0.1
TEST_PAE=0.01

CNV_RATIO=0.4

for PHASED_FREQ in 1
do
	for AVG_MUT in 10
	do
		for COVERAGE in 10
		do
			for DATA_PAE in 0.001
			do
				for DATA_PADO in 0 0.1 0.2
				do
					EXP_NAME="seed_${SEED_VAL}_cells_${NUM_CELLS}_avgmut_${AVG_MUT}_phasedfreq_${PHASED_FREQ}_ado_${DATA_PADO}_ae_${DATA_PAE}_cov_${COVERAGE}_cnv_${CNV_RATIO}"
					job_file="${DATA_DIR}/${EXP_NAME}/script_sciphi.sh"

					folder_name=${RESULT_DIR}/${EXP_NAME}
					mkdir "$folder_name"
					
					echo "#!/bin/bash -l
#SBATCH -A snic2020-5-280
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -J ${EXP_NAME}
#SBATCH -e ${RESULT_DIR}/${EXP_NAME}/sciphi_err.txt
#SBATCH -o ${RESULT_DIR}/${EXP_NAME}/sciphi_out.txt
echo \"$(date) Running on: $(hostname)\"

module load Python/3.7.0-anaconda-5.3.0-extras-nsc1
source activate /proj/sc_ml/users/x_hazko/scuphr_conda/scuphr_env
wait
echo \"$(date) Modules / environments are loaded\"

cd $CODE_SRC_DIR/benchmark/sciphi/
wait
echo \"$(date) Directory is changed\"

#samtools mpileup --fasta-ref ${DATA_DIR}/${EXP_NAME}/truth/bulk_ref.fasta --no-BAQ --output ${DATA_DIR}/${EXP_NAME}/bam/cells_pile.mpileup ${DATA_DIR}/${EXP_NAME}/bam/bulk.bam   ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_0.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_1.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_2.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_3.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_4.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_5.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_6.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_7.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_8.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_9.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_10.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_11.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_12.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_13.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_14.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_15.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_16.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_17.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_18.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_19.bam  
wait
echo "$(date) Pileup is finished"

#sciphi -o ${RESULT_DIR}/${EXP_NAME}/sciphi_result --in ${DATA_DIR}/cellNames_${NUM_CELLS}.txt --seed ${SEED_VAL} ${DATA_DIR}/${EXP_NAME}/bam/cells_pile.mpileup 

sciphi -o ${RESULT_DIR}/${EXP_NAME}/sciphi_result --in ${DATA_DIR}/cellNames_${NUM_CELLS}.txt --seed ${SEED_VAL} ${DATA_DIR}/${EXP_NAME}/bam/cells_pile.mpileup --inc ${DATA_DIR}/${EXP_NAME}/bam/inclusion_list.vcf --ex ${DATA_DIR}/${EXP_NAME}/bam/exclusion_list.vcf
wait
echo "$(date) Sciphi is finished"

python3 sciphi_result_parser.py ${RESULT_DIR}/${EXP_NAME}/sciphi_result.gv $NUM_CELLS --output_filepath ${RESULT_DIR}/${EXP_NAME}/sciphi_processed_result.tre --output_filepath_normed ${RESULT_DIR}/${EXP_NAME}/sciphi_processed_result_normed.tre 
wait
echo \"$(date) Sciphi result parsing is finished\"

cp ${DATA_DIR}/${EXP_NAME}/truth/real.tre ${RESULT_DIR}/${EXP_NAME}/
wait
echo \"$(date) Copying real tree is finished\"

python3 lineage_trees_sciphi.py ${RESULT_DIR}/${EXP_NAME}/sciphi_processed_result.tre  ${RESULT_DIR}/${EXP_NAME}/sciphi_tree_comparison.txt --real_newick ${RESULT_DIR}/${EXP_NAME}/real.tre --scuphr_newick ${RESULT_DIR}/${EXP_NAME}/${TEST_PADO}_${TEST_PAE}/inferred_w_bulk_root.tre --sciphi_tsv ${RESULT_DIR}/${EXP_NAME}/sciphi_result_mut2Sample.tsv --data_dir ${DATA_DIR}/${EXP_NAME}/
wait
echo \"$(date) Tree comparison is finished\"

echo \"$(date) All done!\" >&2" > $job_file

					sbatch $job_file
				done
			done
		done
	done
done
