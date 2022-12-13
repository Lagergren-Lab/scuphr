#!/bin/bash -l

CODE_SRC_DIR="/proj/sc_ml/users/x_hazko/scuphr_git/scuphr"
DATA_DIR="/proj/sc_ml/users/x_hazko/data_simulator/data"
RESULT_DIR="/proj/sc_ml/users/x_hazko/data_simulator/results"

SEED_VAL=1
NUM_CELLS=50

TEST_PADO=0.1
TEST_PAE=0.01

for PHASED_FREQ in 0.01 
do
	for AVG_MUT in 10
	do
		for COVERAGE in 10
		do
			for DATA_PAE in 0.001
			do
				for DATA_PADO in 0.2
				do
					EXP_NAME="seed_${SEED_VAL}_cells_${NUM_CELLS}_avgmut_${AVG_MUT}_phasedfreq_${PHASED_FREQ}_ado_${DATA_PADO}_ae_${DATA_PAE}_cov_${COVERAGE}"
					job_file="${DATA_DIR}/${EXP_NAME}/script_sciphi_normal_v2.sh"

					folder_name=${RESULT_DIR}/${EXP_NAME}
					mkdir "$folder_name"

					echo "#!/bin/bash -l
#SBATCH -A snic2021-5-258
#SBATCH -n 1
#SBATCH -t 168:00:00
#SBATCH -J ${EXP_NAME}
#SBATCH -e ${RESULT_DIR}/${EXP_NAME}/sciphi_normal_v2_err.txt
#SBATCH -o ${RESULT_DIR}/${EXP_NAME}/sciphi_normal_v2_out.txt
echo \"$(date) Running on: $(hostname)\"

module load Python/3.7.0-anaconda-5.3.0-extras-nsc1
source activate /proj/sc_ml/users/x_hazko/scuphr_conda/scuphr_env
wait
echo \"$(date) Modules / environments are loaded\"

cd $CODE_SRC_DIR/benchmark/sciphi/
wait
echo \"$(date) Directory is changed\"

#samtools mpileup --fasta-ref ${DATA_DIR}/${EXP_NAME}/truth/bulk_ref.fasta --no-BAQ --output ${DATA_DIR}/${EXP_NAME}/bam/cells_pile.mpileup ${DATA_DIR}/${EXP_NAME}/bam/bulk.bam   ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_0.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_1.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_2.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_3.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_4.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_5.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_6.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_7.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_8.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_9.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_10.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_11.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_12.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_13.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_14.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_15.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_16.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_17.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_18.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_19.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_20.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_21.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_22.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_23.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_24.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_25.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_26.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_27.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_28.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_29.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_30.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_31.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_32.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_33.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_34.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_35.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_36.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_37.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_38.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_39.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_40.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_41.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_42.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_43.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_44.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_45.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_46.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_47.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_48.bam ${DATA_DIR}/${EXP_NAME}/bam/cell_idx_49.bam   
#wait
#echo "$(date) Pileup is finished"

sciphi -o ${RESULT_DIR}/${EXP_NAME}/sciphi_result_v2 --in ${DATA_DIR}/cellNames_${NUM_CELLS}.txt --seed ${SEED_VAL} ${DATA_DIR}/${EXP_NAME}/bam/cells_pile.mpileup
wait
echo "$(date) Sciphi is finished"

python3 sciphi_result_parser.py ${RESULT_DIR}/${EXP_NAME}/sciphi_result_v2.gv $NUM_CELLS --output_filepath ${RESULT_DIR}/${EXP_NAME}/sciphi_processed_result.tre --output_filepath_normed ${RESULT_DIR}/${EXP_NAME}/sciphi_processed_result_normed.tre 
wait
echo \"$(date) Sciphi result parsing is finished\"

cp ${DATA_DIR}/${EXP_NAME}/truth/real.tre ${RESULT_DIR}/${EXP_NAME}/
wait
echo \"$(date) Copying real tree is finished\"

python3 lineage_trees_sciphi.py ${RESULT_DIR}/${EXP_NAME}/sciphi_processed_result.tre  ${RESULT_DIR}/${EXP_NAME}/sciphi_tree_comparison.txt --real_newick ${RESULT_DIR}/${EXP_NAME}/real.tre --sciphi_tsv ${RESULT_DIR}/${EXP_NAME}/sciphi_result_mut2Sample.tsv --data_dir ${DATA_DIR}/${EXP_NAME}/
wait
echo \"$(date) Tree comparison is finished\"

echo \"$(date) All done!\" >&2" > $job_file

					sbatch $job_file
				done
			done
		done
	done
done
