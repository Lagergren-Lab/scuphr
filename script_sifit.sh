#!/bin/bash -l

CODE_SRC_DIR="/proj/sc_ml/users/x_hazko/scuphr_git/scuphr"
DATA_DIR="/proj/sc_ml/users/x_hazko/data_simulator/data"
RESULT_DIR="/proj/sc_ml/users/x_hazko/data_simulator/results"

SEED_VAL=1

TEST_PADO=0.1
TEST_PAE=0.01

for NUM_CELLS in 10 
do
	for AVG_MUT in 20
	do
		for PHASED_FREQ in 0.001 
		do
			for DATA_PADO in 0 0.1 0.2
			do
				for DATA_PAE in 0.00001 0.001
				do
					for COVERAGE in 5 10 
					do
						EXP_NAME="seed_${SEED_VAL}_cells_${NUM_CELLS}_avgmut_${AVG_MUT}_phasedfreq_${PHASED_FREQ}_ado_${DATA_PADO}_ae_${DATA_PAE}_cov_${COVERAGE}"
						job_file="${DATA_DIR}/${EXP_NAME}/script_sifit.sh"

						NUM_SNVS=$(< ${RESULT_DIR}/${EXP_NAME}/cells_bin.txt wc -l)

						echo "#!/bin/bash -l
#SBATCH -A snic2020-5-280
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -J ${EXP_NAME}
#SBATCH -e ${RESULT_DIR}/${EXP_NAME}/sifit_err.txt
#SBATCH -o ${RESULT_DIR}/${EXP_NAME}/sifit_out.txt
echo \"$(date) Running on: $(hostname)\"

module load Python/3.7.0-anaconda-5.3.0-extras-nsc1 Java/1.8.0_181-nsc1
source activate /proj/sc_ml/users/x_hazko/scuphr_conda/scuphr_env
wait
echo \"$(date) Modules / environments are loaded\"

cd ${DATA_DIR}/${EXP_NAME}/
wait
echo \"$(date) Directory is changed\"

wait
echo \"$(date) Number of SNVs: ${NUM_SNVS}\"

java -jar ${CODE_SRC_DIR}/benchmark/sifit/SiFit.jar -m ${NUM_CELLS} -n ${NUM_SNVS} -iter 1000000 -df 0 -r 1 -ipMat ${RESULT_DIR}/${EXP_NAME}/cells_bin.txt -cellNames ${DATA_DIR}/sifit_cellNames_${NUM_CELLS}.txt > ${RESULT_DIR}/${EXP_NAME}/sifit_log.txt
wait 
echo \"$(date) SiFit is finished\"

cp ${RESULT_DIR}/${EXP_NAME}/cells_bin_mlTree.newick ${RESULT_DIR}/${EXP_NAME}/sifit_tree.tre
wait
echo \"$(date) Renaming SiFit tree is finished\"

cp ${DATA_DIR}/${EXP_NAME}/truth/real.tre ${RESULT_DIR}/${EXP_NAME}/
wait
echo \"$(date) Copying real tree is finished\"

python3 ${CODE_SRC_DIR}/benchmark/sciphi/lineage_trees_sciphi.py ${RESULT_DIR}/${EXP_NAME}/sifit_tree.tre ${RESULT_DIR}/${EXP_NAME}/sifit_tree_comparison.txt --real_newick ${RESULT_DIR}/${EXP_NAME}/real.tre --scuphr_newick ${RESULT_DIR}/${EXP_NAME}/${TEST_PADO}_${TEST_PAE}/inferred_w_bulk_root.tre --data_dir ${DATA_DIR}/${EXP_NAME}/
wait
echo \"$(date) Tree comparison is finished\"

echo \"$(date) All done!\" >&2" > $job_file

						sbatch $job_file
					done
				done
			done
		done
	done
done
