#!/bin/bash -l

CODE_SRC_DIR="/proj/sc_ml/users/x_hazko/scuphr_git/scuphr"
DATA_DIR="/proj/sc_ml/users/x_hazko/data_simulator/data"
RESULT_DIR="/proj/sc_ml/users/x_hazko/data_simulator/results"

SEED_VAL=1
NUM_CELLS=10

AVG_MUT=12
MUT_RATIO=1
COVERAGE=5

TEST_PADO=0.1
TEST_PAE=0.01

DATA_PAE=0.1
for DATA_PADO in 0 0.1 0.2
do
	EXP_NAME="seed_${SEED_VAL}_cells_${NUM_CELLS}_avgmut_${AVG_MUT}_ratio_${MUT_RATIO}_ado_${DATA_PADO}_ae_${DATA_PAE}_cov_${COVERAGE}"
	job_file="${DATA_DIR}/${EXP_NAME}/script_sciphi.sh"

	echo "#!/bin/bash -l
#SBATCH -A snic2020-5-278
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -J ${EXP_NAME}
#SBATCH -e ${RESULT_DIR}/pairs_results_seed_${SEED_VAL}_cells_${NUM_CELLS}/${EXP_NAME}/sciphi_err.txt
#SBATCH -o ${RESULT_DIR}/pairs_results_seed_${SEED_VAL}_cells_${NUM_CELLS}/${EXP_NAME}/sciphi_out.txt
echo \"$(date) Running on: $(hostname)\"

module load Python/3.7.0-anaconda-5.3.0-extras-nsc1
source activate /proj/sc_ml/users/x_hazko/scuphr_conda/scuphr_env
wait
echo \"$(date) Modules / environments are loaded\"

cd $CODE_SRC_DIR/benchmark/sciphi/
wait
echo \"$(date) Directory is changed\"

#samtools mpileup --fasta-ref ${DATA_DIR}/${EXP_NAME}/truth/bulk_ref.fasta --no-BAQ --output ${DATA_DIR}/${EXP_NAME}/bam/cells_pile.mpileup ${DATA_DIR}/${EXP_NAME}/bam/bulk.bam   ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_0.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_1.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_2.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_3.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_4.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_5.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_6.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_7.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_8.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_9.bam   
wait
echo "$(date) Pileup is finished"

#sciphi -o ${RESULT_DIR}/${EXP_NAME}/sciphi_result --in ${DATA_DIR}/cellNames_${NUM_CELLS}.txt --seed ${SEED_VAL} ${DATA_DIR}/${EXP_NAME}/bam/cells_pile.mpileup
wait
echo "$(date) Sciphi is finished"

#python3 sciphi_result_parser.py ${RESULT_DIR}/${EXP_NAME}/sciphi_result.gv $NUM_CELLS --output_filepath ${RESULT_DIR}/${EXP_NAME}/sciphi_processed_result.tre --output_filepath_normed ${RESULT_DIR}/${EXP_NAME}/sciphi_processed_result_normed.tre 
wait
echo \"$(date) Sciphi result parsing is finished\"

#cp ${DATA_DIR}/${EXP_NAME}/truth/real.tre ${RESULT_DIR}/${EXP_NAME}/
wait
echo \"$(date) Copying real tree is finished\"

python3 lineage_trees_sciphi.py ${RESULT_DIR}/pairs_results_seed_${SEED_VAL}_cells_${NUM_CELLS}/${EXP_NAME}/sciphi_processed_result.tre  ${RESULT_DIR}/pairs_results_seed_${SEED_VAL}_cells_${NUM_CELLS}/${EXP_NAME}/sciphi_tree_comparison.txt --real_newick ${RESULT_DIR}/pairs_results_seed_${SEED_VAL}_cells_${NUM_CELLS}/${EXP_NAME}/real.tre --scuphr_newick ${RESULT_DIR}/pairs_results_seed_${SEED_VAL}_cells_${NUM_CELLS}/${EXP_NAME}/${TEST_PADO}_${TEST_PAE}/inferred_w_bulk_root.tre --sciphi_tsv ${RESULT_DIR}/pairs_results_seed_${SEED_VAL}_cells_${NUM_CELLS}/${EXP_NAME}/sciphi_result_mut2Sample.tsv --data_dir ${DATA_DIR}/${EXP_NAME}/
wait
echo \"$(date) Tree comparison is finished\"

echo \"$(date) All done!\" >&2" > $job_file

	sbatch $job_file

done
