#!/bin/bash -l

CODE_SRC_DIR="/proj/sc_ml/users/x_hazko/scuphr_git/scuphr/benchmark"
DATA_DIR="/proj/sc_ml/users/x_hazko/data_simulator/data"
RESULT_DIR="/proj/sc_ml/users/x_hazko/data_simulator/results"

SEED_VAL=1
CHR_ID=1
GENOME_LENGTH=1000000

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
					job_file="${DATA_DIR}/${EXP_NAME}/script_sciphi_ours.sh"
					
					folder_name=${RESULT_DIR}/${EXP_NAME}
					mkdir "$folder_name"

					echo "#!/bin/bash -l
#SBATCH -A snic2021-5-259
#SBATCH -n 1
#SBATCH -t 168:00:00
#SBATCH -J ${EXP_NAME}
#SBATCH -e ${RESULT_DIR}/${EXP_NAME}/sciphi_our_sites_err.txt
#SBATCH -o ${RESULT_DIR}/${EXP_NAME}/sciphi_our_sites_out.txt
echo \"$(date) Running on: $(hostname)\"

module load Python/3.7.0-anaconda-5.3.0-extras-nsc1
source activate /proj/sc_ml/users/x_hazko/scuphr_conda/scuphr_env
wait
echo \"$(date) Modules / environments are loaded\"

cd ${DATA_DIR}/${EXP_NAME}/
wait
echo \"$(date) Directory is changed\"

# Assumed pileup already done
#wait
#echo "$(date) Pileup is finished"

#tar -xzf ${DATA_DIR}/${EXP_NAME}/truth.tar.gz
#mv ${DATA_DIR}/${EXP_NAME}/${EXP_NAME}/truth .
#wait
#echo "$(date) Untar truth folder is finished"

#cat ${DATA_DIR}/${EXP_NAME}/bam/cells_pile.mpileup | python3 ${CODE_SRC_DIR}/MonoVar/src/monovar.py -p 0.002 -a 0.2 -t 0.05 -m 1 -f ${DATA_DIR}/${EXP_NAME}/truth/bulk_ref.fasta -b ${DATA_DIR}/sifit_bam_filenames_${NUM_CELLS}.txt -o ${RESULT_DIR}/${EXP_NAME}/cells_vcf.vcf
#wait
#echo "$(date) Monovar is finished"

cd $CODE_SRC_DIR/sciphi/
wait
echo \"$(date) Directory is changed\"

python3 sciphi_our_vcf.py ${DATA_DIR}/${EXP_NAME}/ ${RESULT_DIR}/${EXP_NAME}/cells_vcf.vcf --chr_id $CHR_ID --genome_length $GENOME_LENGTH
wait
echo \"$(date) Sciphi inclusion and exclusion file creation is finished\"

sciphi -o ${RESULT_DIR}/${EXP_NAME}/sciphi_result_our_sites_v2 --in ${DATA_DIR}/cellNames_${NUM_CELLS}.txt --seed ${SEED_VAL} ${DATA_DIR}/${EXP_NAME}/bam/cells_pile.mpileup --inc ${DATA_DIR}/${EXP_NAME}/bam/inclusion_list.vcf --ex ${DATA_DIR}/${EXP_NAME}/bam/exclusion_list.vcf
wait
echo "$(date) Sciphi is finished"

python3 sciphi_result_parser.py ${RESULT_DIR}/${EXP_NAME}/sciphi_result_our_sites_v2.gv $NUM_CELLS --output_filepath ${RESULT_DIR}/${EXP_NAME}/sciphi_processed_result_our_sites.tre --output_filepath_normed ${RESULT_DIR}/${EXP_NAME}/sciphi_processed_result_normed_our_sites.tre 
wait
echo \"$(date) Sciphi result parsing is finished\"

cp ${DATA_DIR}/${EXP_NAME}/truth/real.tre ${RESULT_DIR}/${EXP_NAME}/
wait
echo \"$(date) Copying real tree is finished\"

python3 lineage_trees_sciphi.py ${RESULT_DIR}/${EXP_NAME}/sciphi_processed_result_our_sites.tre  ${RESULT_DIR}/${EXP_NAME}/sciphi_tree_comparison_our_sites.txt --real_newick ${RESULT_DIR}/${EXP_NAME}/real.tre --sciphi_tsv ${RESULT_DIR}/${EXP_NAME}/sciphi_result_our_sites_mut2Sample.tsv --data_dir ${DATA_DIR}/${EXP_NAME}/
wait
echo \"$(date) Tree comparison is finished\"

echo \"$(date) All done!\" >&2" > $job_file

					sbatch $job_file
				done
			done
		done
	done
done
