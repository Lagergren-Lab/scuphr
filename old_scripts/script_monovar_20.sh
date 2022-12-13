#!/bin/bash -l

CODE_SRC_DIR="/proj/sc_ml/users/x_hazko/scuphr_git/scuphr/benchmark"
DATA_DIR="/proj/sc_ml/users/x_hazko/data_simulator/data"
RESULT_DIR="/proj/sc_ml/users/x_hazko/data_simulator/results"

SEED_VAL=1
NUM_CELLS=20

AVG_MUT=12
MUT_RATIO=10

TEST_PADO=0.1
TEST_PAE=0.01

for PHASED_FREQ in 0.25 0.5 1
do
	for COVERAGE in 5 10 20
	do
		for DATA_PAE in 0.001 0.01
		do
			for DATA_PADO in 0 0.1 0.2
			do
				EXP_NAME="seed_${SEED_VAL}_cells_${NUM_CELLS}_avgmut_${AVG_MUT}_ratio_${MUT_RATIO}_phasedfreq_${PHASED_FREQ}_ado_${DATA_PADO}_ae_${DATA_PAE}_cov_${COVERAGE}"
				job_file="${DATA_DIR}/${EXP_NAME}/script_monovar.sh"

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

#cat ${DATA_DIR}/${EXP_NAME}/bam/cells_pile.mpileup | python3 ${CODE_SRC_DIR}/MonoVar/src/monovar.py -p 0.002 -a 0.2 -t 0.05 -m 1 -f ${DATA_DIR}/${EXP_NAME}/truth/bulk_ref.fasta -b ${DATA_DIR}/sifit_bam_filenames_${NUM_CELLS}.txt -o ${RESULT_DIR}/${EXP_NAME}/cells_vcf.vcf
wait
echo "$(date) Monovar is finished"

#python3 ${CODE_SRC_DIR}/sifit/binarizeVCF.py ${RESULT_DIR}/${EXP_NAME}/cells_vcf.vcf ${RESULT_DIR}/${EXP_NAME}/cells_bin.txt
wait
echo "$(date) Binarize VCF is finished"

python3 ${CODE_SRC_DIR}/sifit/parse_genotype_matrix.py ${RESULT_DIR}/${EXP_NAME}/cells_bin.txt --remove_gsnv 1 --data_dir ${DATA_DIR}/${EXP_NAME}/
wait
echo "$(date) Binarize VCF is finished"

echo \"$(date) All done!\" >&2" > $job_file

				sbatch $job_file
			done
		done
	done
done
