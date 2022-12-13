#!/bin/bash -l

CODE_SRC_DIR="/proj/sc_ml/users/x_hazko/scuphr_git/scuphr/benchmark"
DATA_DIR="/proj/sc_ml/users/x_hazko/data_simulator/data"
RESULT_DIR="/proj/sc_ml/users/x_hazko/data_simulator/results"

SEED_VAL=1

TEST_PADO=0.1
TEST_PAE=0.01

CNV_RATIO=0.4

REMOVE_GSNV=0  # Don't remove gSNVs: 0, Remove gSNVs: 1

for NUM_CELLS in 20
do
	for AVG_MUT in 10 20
	do
		for PHASED_FREQ in 1 
		do
			for DATA_PADO in 0 0.1 0.2
			do
				for DATA_PAE in 0.001
				do
					for COVERAGE in 10
					do
						EXP_NAME="seed_${SEED_VAL}_cells_${NUM_CELLS}_avgmut_${AVG_MUT}_phasedfreq_${PHASED_FREQ}_ado_${DATA_PADO}_ae_${DATA_PAE}_cov_${COVERAGE}_cnv_${CNV_RATIO}"
						job_file="${DATA_DIR}/${EXP_NAME}/script_monovar.sh"

						folder_name=${RESULT_DIR}/${EXP_NAME}
						mkdir "$folder_name"

						echo "#!/bin/bash -l
#SBATCH -A snic2020-5-280
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -J ${EXP_NAME}
#SBATCH -e ${RESULT_DIR}/${EXP_NAME}/script_monovar_err.txt
#SBATCH -o ${RESULT_DIR}/${EXP_NAME}/script_monovar_out.txt
echo \"$(date) Running on: $(hostname)\"

module load Python/3.7.0-anaconda-5.3.0-extras-nsc1
source /proj/sc_ml/users/x_hazko/data_simulator/myownvirtualenv/bin/activate
wait
echo \"$(date) Modules / environments are loaded\"

cd ${DATA_DIR}/${EXP_NAME}/
wait
echo \"$(date) Directory is changed\"

cat ${DATA_DIR}/${EXP_NAME}/bam/cells_pile.mpileup | python3 ${CODE_SRC_DIR}/MonoVar/src/monovar.py -p 0.002 -a 0.2 -t 0.05 -m 1 -f ${DATA_DIR}/${EXP_NAME}/truth/bulk_ref.fasta -b ${DATA_DIR}/sifit_bam_filenames_${NUM_CELLS}.txt -o ${RESULT_DIR}/${EXP_NAME}/cells_vcf.vcf
wait
echo "$(date) Monovar is finished"

python3 ${CODE_SRC_DIR}/sifit/binarizeVCF.py ${RESULT_DIR}/${EXP_NAME}/cells_vcf.vcf ${RESULT_DIR}/${EXP_NAME}/cells_bin.txt
wait
echo "$(date) Binarize VCF is finished"

python3 ${CODE_SRC_DIR}/sifit/parse_genotype_matrix.py ${RESULT_DIR}/${EXP_NAME}/cells_bin.txt --remove_gsnv ${REMOVE_GSNV} --data_dir ${DATA_DIR}/${EXP_NAME}/
wait
echo "$(date) Parsing genotype matrix is finished"

echo \"$(date) All done!\" >&2" > $job_file

						sbatch $job_file
					done
				done
			done
		done
	done
done
