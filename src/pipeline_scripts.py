import os
import datetime
import argparse


def get_parameters():
    simulation_params = ['ado_poisson_rate', 'ado_type', 'amp_method', 'chr_id', 'genome_length', 'global_dir',
                         'gsnv_rate', 'is_flat', 'mut_poisson_rate', 'num_cells', 'num_iter', 'num_max_mid_points',
                         'num_rep_amp_bias', 'p_ado', 'p_ae', 'phred_type', 'read_length',  'seed_val']

    site_detection_params_fixed = ['global_dir', 'num_cells']
    site_detection_params = ['bulk_depth_threshold', 'cell_depth_threshold', 'chr_id', 'het_ratio_threshold',
                             'min_line', 'max_line', 'nuc_depth_threshold', 'print_status', 'read_length', 'seed_val']

    generate_dataset_params_fixed = ['global_dir', 'num_cells']
    generate_dataset_params = ['chr_id', 'data_type', 'min_read_count', 'max_read_count', 'max_site_count',
                               'min_cell_count', 'output_dict_dir', 'read_length', 'seed_val']

    dbc_params_fixed = ['global_dir']
    dbc_params = ['a_g', 'b_g', 'data_type', 'output_dir', 'p_ado', 'p_ae', 'pos_range_min', 'pos_range_max',
                  'print_status', 'seed_val']

    dbc_combine_params_fixed = ['global_dir']
    dbc_combine_params = ['data_type', 'pos_range_min', 'pos_range_max', 'output_dir']

    return simulation_params, site_detection_params_fixed, site_detection_params, \
           generate_dataset_params_fixed, generate_dataset_params, \
           dbc_params_fixed, dbc_params, dbc_combine_params_fixed, dbc_combine_params


def write_combined_data_script(args, dir_src, script_data_dir, log_data_dir, all_sim_parameters,
                               all_site_parameters, all_dataset_parameters):
    job_name = "00_combined_data_par"

    filename = script_data_dir + "script_" + job_name + ".sh"
    file = open(filename, "w")

    file.write("#!/bin/bash -l\n")
    file.write("#SBATCH -A %s\n" % args.project_name)
    file.write("#SBATCH -p core\n")
    file.write("#SBATCH -n 16\n")
    file.write("#SBATCH -t 72:00:00\n")
    file.write("#SBATCH -J %s\n" % job_name)
    file.write("#SBATCH -e %slogs_%s_err.txt\n" % (log_data_dir, job_name))
    file.write("#SBATCH -o %slogs_%s_out.txt\n" % (log_data_dir, job_name))

    file.write("echo \"$(date) Running on: $(hostname)\"\n\n")
    file.write("module load python3 bioinfo-tools pysam samtools/1.9\n")
    file.write("wait\n")
    file.write("echo \"$(date) Modules are loaded\"\n\n")

    file.write("cd %s\n" % dir_src)
    file.write("wait\n")
    file.write("echo \"$(date) Directory is changed\"\n\n")

    #####
    sub_job_name = "01_data_simulation"

    params = ""
    for arg in vars(args):
        if str(arg) in all_sim_parameters:
            params = params + " --" + str(arg) + " " + str(getattr(args, arg))
    file.write("python3 data_simulator.py %s > %slogs_%s_log.txt\n" % (params, log_data_dir, sub_job_name))
    file.write("wait\n")
    file.write("echo \"$(date) Data simulation is finished\"\n\n")
    #####

    sub_job_name = "02_site_detection"

    params = "" + str(args.global_dir) + " " + str(args.num_cells)
    for arg in vars(args):
        if str(arg) in all_site_parameters:
            params = params + " --" + str(arg) + " " + str(getattr(args, arg)) + " "
    file.write("python3 site_detection.py %s > %slogs_%s_log.txt\n" % (params, log_data_dir, sub_job_name))
    file.write("wait\n")
    file.write("echo \"$(date) Site detection is finished\"\n\n")
    #####

    sub_job_name = "03_generate_dataset"

    params = "" + str(args.global_dir) + " " + str(args.num_cells)
    for arg in vars(args):
        if str(arg) in all_dataset_parameters:
            if str(arg) == 'output_dict_dir' and str(getattr(args, arg)) == "":
                params = params + " --" + str(arg) + " \"\" "
            else:
                params = params + " --" + str(arg) + " " + str(getattr(args, arg)) + " "
    file.write("python3 generate_dataset.py %s > %slogs_%s_log.txt\n" % (params, log_data_dir, sub_job_name))
    file.write("wait\n")
    file.write("echo \"$(date) Dataset generation is finished\"\n\n")

    file.write("echo \"$(date) All done!\" >&2\n")
    file.close()
    print("\nScript is saved to ", filename)


def write_data_simulation_script(args, dir_src, script_data_dir, log_data_dir, all_parameters):
    job_name = "01_data_simulation"

    filename = script_data_dir + "script_" + job_name + ".sh"
    file = open(filename, "w")

    file.write("#!/bin/bash -l\n")
    file.write("#SBATCH -A %s\n" % args.project_name)
    file.write("#SBATCH -p core\n")
    file.write("#SBATCH -n 1\n")
    file.write("#SBATCH -t 72:00:00\n")
    file.write("#SBATCH -J %s\n" % job_name)
    file.write("#SBATCH -e %slogs_%s_err.txt\n" % (log_data_dir, job_name))
    file.write("#SBATCH -o %slogs_%s_out.txt\n" % (log_data_dir, job_name))

    file.write("echo \"$(date) Running on: $(hostname)\"\n\n")
    file.write("module load python3 bioinfo-tools pysam samtools/1.9\n")
    file.write("wait\n")
    file.write("echo \"$(date) Modules are loaded\"\n\n")

    file.write("cd %s\n" % dir_src)
    file.write("wait\n")
    file.write("echo \"$(date) Directory is changed\"\n\n")

    params = ""
    for arg in vars(args):
        if str(arg) in all_parameters:
            params = params + " --" + str(arg) + " " + str(getattr(args, arg))

    file.write("python3 data_simulator.py %s > %slogs_%s_log.txt\n" % (params, log_data_dir, job_name))
    file.write("wait\n")
    file.write("echo \"$(date) Data simulation is finished\"\n\n")

    file.write("echo \"$(date) All done!\" >&2\n")
    file.close()
    print("\nScript is saved to ", filename)


def write_site_detection_script(args, dir_src, script_data_dir, log_data_dir, all_parameters):
    job_name = "02_site_detection"

    filename = script_data_dir + "script_" + job_name + ".sh"
    file = open(filename, "w")

    file.write("#!/bin/bash -l\n")
    file.write("#SBATCH -A %s\n" % args.project_name)
    file.write("#SBATCH -p core\n")
    file.write("#SBATCH -n 16\n")
    file.write("#SBATCH -t 72:00:00\n")
    file.write("#SBATCH -J %s\n" % job_name)
    file.write("#SBATCH -e %slogs_%s_err.txt\n" % (log_data_dir, job_name))
    file.write("#SBATCH -o %slogs_%s_out.txt\n" % (log_data_dir, job_name))

    file.write("echo \"$(date) Running on: $(hostname)\"\n\n")
    file.write("module load python3 bioinfo-tools pysam\n")
    file.write("wait\n")
    file.write("echo \"$(date) Modules are loaded\"\n\n")

    file.write("cd %s\n" % dir_src)
    file.write("wait\n")
    file.write("echo \"$(date) Directory is changed\"\n\n")

    params = "" + str(args.global_dir) + " " + str(args.num_cells)

    for arg in vars(args):
        if str(arg) in all_parameters:
            params = params + " --" + str(arg) + " " + str(getattr(args, arg)) + " "

    file.write("python3 site_detection.py %s > %slogs_%s_log.txt\n" % (params, log_data_dir, job_name))
    file.write("wait\n")
    file.write("echo \"$(date) Site detection is finished\"\n\n")

    file.write("echo \"$(date) All done!\" >&2\n")
    file.close()
    print("\nScript is saved to ", filename)


def write_generate_dataset_script(args, dir_src, script_data_dir, log_data_dir, all_parameters):
    job_name = "03_generate_dataset"

    filename = script_data_dir + "script_" + job_name + ".sh"
    file = open(filename, "w")

    file.write("#!/bin/bash -l\n")
    file.write("#SBATCH -A %s\n" % args.project_name)
    file.write("#SBATCH -p core\n")
    file.write("#SBATCH -n 1\n")
    file.write("#SBATCH -t 72:00:00\n")
    file.write("#SBATCH -J %s\n" % job_name)
    file.write("#SBATCH -e %slogs_%s_err.txt\n" % (log_data_dir, job_name))
    file.write("#SBATCH -o %slogs_%s_out.txt\n" % (log_data_dir, job_name))

    file.write("echo \"$(date) Running on: $(hostname)\"\n\n")
    file.write("module load python3\n")
    file.write("wait\n")
    file.write("echo \"$(date) Modules are loaded\"\n\n")

    file.write("cd %s\n" % dir_src)
    file.write("wait\n")
    file.write("echo \"$(date) Directory is changed\"\n\n")

    params = "" + str(args.global_dir) + " " + str(args.num_cells)

    for arg in vars(args):
        if str(arg) in all_parameters:
            if str(arg) == 'output_dict_dir' and str(getattr(args, arg)) == "":
                params = params + " --" + str(arg) + " \"\" "
            else:
                params = params + " --" + str(arg) + " " + str(getattr(args, arg)) + " "

    file.write("python3 generate_dataset.py %s > %slogs_%s_log.txt\n" % (params, log_data_dir, job_name))
    file.write("wait\n")
    file.write("echo \"$(date) Dataset generation is finished\"\n\n")

    file.write("echo \"$(date) All done!\" >&2\n")
    file.close()
    print("\nScript is saved to ", filename)


def write_distance_between_cells_script(args, dir_src, script_dir, log_dir, all_parameters):
    job_name = "04_dbc"

    filename = script_dir + "script_" + job_name + ".sh"
    file = open(filename, "w")

    file.write("#!/bin/bash -l\n")
    file.write("#SBATCH -A %s\n" % args.project_name)
    file.write("#SBATCH -p core\n")
    file.write("#SBATCH -n 16\n")
    file.write("#SBATCH -t 72:00:00\n")
    file.write("#SBATCH -J %s\n" % job_name)
    file.write("#SBATCH -e %slogs_%s_err.txt\n" % (log_dir, job_name))
    file.write("#SBATCH -o %slogs_%s_out.txt\n" % (log_dir, job_name))

    file.write("echo \"$(date) Running on: $(hostname)\"\n\n")
    file.write("module load python3\n")
    file.write("wait\n")
    file.write("echo \"$(date) Modules are loaded\"\n\n")

    file.write("cd %s\n" % dir_src)
    file.write("wait\n")
    file.write("echo \"$(date) Directory is changed\"\n\n")

    params = "" + str(args.global_dir) + " "

    for arg in vars(args):
        if str(arg) in all_parameters:
            params = params + " --" + str(arg) + " " + str(getattr(args, arg)) + ""

    file.write("python3 analyse_dbc.py %s > %slogs_%s_log.txt\n" % (params, log_dir, job_name))
    file.write("wait\n")
    file.write("echo \"$(date) Distance between cell computation is finished\"\n\n")

    file.write("echo \"$(date) All done!\" >&2\n")
    file.close()
    print("\nScript is saved to ", filename)


def write_distance_between_cells_combine_script(args, dir_src, script_dir, log_dir, all_parameters):
    job_name = "05_dbc_combine"

    filename = script_dir + "script_" + job_name + ".sh"
    file = open(filename, "w")

    file.write("#!/bin/bash -l\n")
    file.write("#SBATCH -A %s\n" % args.project_name)
    file.write("#SBATCH -p core\n")
    file.write("#SBATCH -n 1\n")
    file.write("#SBATCH -t 72:00:00\n")
    file.write("#SBATCH -J %s\n" % job_name)
    file.write("#SBATCH -e %slogs_%s_err.txt\n" % (log_dir, job_name))
    file.write("#SBATCH -o %slogs_%s_out.txt\n" % (log_dir, job_name))

    file.write("echo \"$(date) Running on: $(hostname)\"\n\n")
    file.write("module load python3\n")
    file.write("wait\n")
    file.write("echo \"$(date) Modules are loaded\"\n\n")

    file.write("cd %s\n" % dir_src)
    file.write("wait\n")
    file.write("echo \"$(date) Directory is changed\"\n\n")

    params = "" + str(args.global_dir) + " "

    for arg in vars(args):
        if str(arg) in all_parameters:
            params = params + " --" + str(arg) + " " + str(getattr(args, arg)) + ""

    file.write("python3 analyse_dbc_combine.py %s > %slogs_%s_log.txt\n" % (params, log_dir, job_name))
    file.write("wait\n")
    file.write("echo \"$(date) Distance between cell combine is finished\"\n\n")

    file.write("echo \"$(date) All done!\" >&2\n")
    file.close()
    print("\nScript is saved to ", filename)


def write_lineage_tree_script(args, dir_src, script_dir, log_dir):
    job_name = "06_lineage_tree"

    filename = script_dir + "script_" + job_name + ".sh"
    file = open(filename, "w")

    file.write("#!/bin/bash -l\n")
    file.write("#SBATCH -A %s\n" % args.project_name)
    file.write("#SBATCH -p core\n")
    file.write("#SBATCH -n 1\n")
    file.write("#SBATCH -t 72:00:00\n")
    file.write("#SBATCH -J %s\n" % job_name)
    file.write("#SBATCH -e %slogs_%s_err.txt\n" % (log_dir, job_name))
    file.write("#SBATCH -o %slogs_%s_out.txt\n" % (log_dir, job_name))

    file.write("echo \"$(date) Running on: $(hostname)\"\n\n")
    file.write("module load python3\n")
    file.write("wait\n")
    file.write("echo \"$(date) Modules are loaded\"\n\n")

    file.write("cd %s\n" % dir_src)
    file.write("wait\n")
    file.write("echo \"$(date) Directory is changed\"\n\n")

    params = "" + str(args.output_dir) + " --data_type " + str(args.data_type)

    file.write("python3 lineage_trees.py %s > %slogs_%s_log.txt\n" % (params, log_dir, job_name))
    file.write("wait\n")
    file.write("echo \"$(date) Lineage tree is finished\"\n\n")
    file.write("echo \"$(date) All done!\" >&2\n")

    file.close()
    print("\nScript is saved to ", filename)


def main():
    dt = datetime.datetime.now()
    default_data_dir = "%s_%s_%s/" % (dt.year, dt.month, dt.day)
    default_result_dir = "%s_%s_%s/" % (dt.year, dt.month, dt.day)

    # Code to process command line arguments
    parser = argparse.ArgumentParser(description='Generate scripts for the pipeline.')

    # Mandatory arguments.
    # Project name is required for -A parameter for job allocation.
    parser.add_argument('project_name', help="Specify the project name. Needed for -A tag in bash script.", type=str)
    # Project directory is the source code directory.
    parser.add_argument('project_dir', help="Specify the source code directory, i.e. /proj/scuphr/", type=str)

    # Arguments shared with multiple scripts
    parser.add_argument('--chr_id', help="Specify the chromosome number. Default: 1", type=int, default=1)
    parser.add_argument('--data_type', help="Specify the data type. Default: real", type=str, default="real")
    parser.add_argument('--global_dir', help="Specify the directory.", type=str, default=default_data_dir)
    parser.add_argument('--num_cells', help="Specify the number of cells. Default: 10", type=int, default=10)
    parser.add_argument('--output_dir', help="Specify the output directory.", type=str, default=default_result_dir)
    parser.add_argument('--p_ado', help="Specify the allelic dropout probability of a base. Default: 0.005", type=float,
                        default=0.005)
    parser.add_argument('--p_ae', help="Specify the amplification error probability of a base. Default: 0.00001",
                        type=float, default=0.00001)
    parser.add_argument('--pos_range_min', help="Specify the position range (min value).", type=int, default=0)
    parser.add_argument('--pos_range_max', help="Specify the position range (max value).", type=int, default=0)
    parser.add_argument('--print_status', help="Specify the print (0 for do not print, 1 for print). Default: 0",
                        type=int, default=0)
    parser.add_argument('--read_length', help="Specify the read length. Default: 100", type=int, default=100)
    parser.add_argument('--seed_val', help="Specify the seed. Default: 123", type=int, default=123)

    # Data simulation specific arguments
    parser.add_argument('--ado_poisson_rate', help="Specify the ado poisson rate. Default: 0", type=float, default=0)
    parser.add_argument('--ado_type',
                        help="Specify the allelic dropout type "
                             "(0 for no ado, 1 for random Beta, 2 for even smaller Beta, 3 for fixed. Default: 3).",
                        type=int, default=3)
    parser.add_argument('--amp_method', help="Specify the amplification method. Default: mda", type=str, default="mda")
    parser.add_argument('--genome_length', help="Specify the length of genome. Default: 50000", type=int, default=50000)
    parser.add_argument('--gsnv_rate', help="Specify the gSNV rate. Default: 0.005", type=float, default=0.005)
    parser.add_argument('--is_flat',
                        help="Specify the amplification bias type (True for flat, False for sine wave). Default: False",
                        default=False)
    parser.add_argument('--mut_poisson_rate', help="Specify the mutation poisson rate. Default: 3", type=float,
                        default=3)
    parser.add_argument('--num_iter', help="Specify the number of iteration. Default: 5000", type=int, default=5000)
    parser.add_argument('--num_max_mid_points',
                        help="Specify the number of division points of amplification bias. Default: 10", type=int,
                        default=10)
    parser.add_argument('--num_rep_amp_bias',
                        help="Specify the number of repetitions of amplification bias. Default: 3", type=int, default=3)
    parser.add_argument('--phred_type',
                        help="Specify the phred score type "
                             "(0 for no errors, 1 for all 42, 2 for truncated normal). Default: 2",
                        type=int, default=2)

    # Site detection specific arguments
    parser.add_argument('--bulk_depth_threshold', help="Specify the bulk depth threshold. Default: 20", type=int,
                        default=20)
    parser.add_argument('--cell_depth_threshold', help="Specify the cell depth threshold. Default: 0", type=int,
                        default=0)
    parser.add_argument('--het_ratio_threshold', help="Specify the bulk heterozygous ratio threshold. Default: 0.2",
                        type=float, default=0.2)
    parser.add_argument('--min_line', help="Specify the line number of min het position. Default: 0", type=int,
                        default=0)
    parser.add_argument('--max_line', help="Specify the line number of max het position. Default: 0", type=int,
                        default=0)
    parser.add_argument('--nuc_depth_threshold', help="Specify the minimum number of valid reads. Default: 2",
                        type=int, default=2)

    # Generate dataset specific arguments
    parser.add_argument('--min_read_count', help="Specify the minimum read count. Default: 0", type=int, default=0)
    parser.add_argument('--max_read_count', help="Specify the maximum read count (0 for all). Default: 0", type=int,
                        default=0)
    parser.add_argument('--min_cell_count', help="Specify the minimum cell count. Default: 2", type=int, default=2)
    parser.add_argument('--output_dict_dir', help="Specify the output dictionary directory.", type=str, default="")

    # Distance between cells specific arguments
    parser.add_argument('--a_g', help="Specify the alpha prior of mutation probability. Default: 1", type=float,
                        default=1)
    parser.add_argument('--b_g', help="Specify the beta prior of mutation probability. Default: 1", type=float,
                        default=1)

    # Distance between cells combine specific arguments
    # Note: It only has shared arguments

    # Lineage tree specific arguments
    # Note: It only has shared arguments

    args = parser.parse_args()

    #####
    print("\nCreate necessary folders")

    dir_src = args.project_dir + "src/"

    data_dir = args.project_dir + "data/" + args.global_dir
    args.global_dir = data_dir

    result_dir = args.project_dir + "results/" + args.output_dir
    args.output_dir = result_dir

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("\tData directory is created")

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print("\tResult directory is created")

    script_data_dir = data_dir + "scripts/"
    if not os.path.exists(script_data_dir):
        os.makedirs(script_data_dir)
        print("\tScript directory is created")

    log_data_dir = data_dir + "logs/"
    if not os.path.exists(log_data_dir):
        os.makedirs(log_data_dir)
        print("\t Log directory is created")

    script_dir = result_dir + "scripts/"
    if not os.path.exists(script_dir):
        os.makedirs(script_dir)
        print("\tScript directory is created")

    log_dir = result_dir + "logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print("\tLog directory is created")
        #####

    #####
    print("\nGet all parameters")
    simulation_params, site_detection_params_fixed, site_detection_params, \
        generate_dataset_params_fixed, generate_dataset_params, \
        dbc_params_fixed, dbc_params, dbc_combine_params_fixed, dbc_combine_params = get_parameters()
    #####

    #####
    print("\nWriting data simulation script")
    write_data_simulation_script(args, dir_src, script_data_dir, log_data_dir, simulation_params)

    print("\nWriting site detection par script")
    write_site_detection_script(args, dir_src, script_data_dir, log_data_dir, site_detection_params)

    print("\nWriting generate dataset script")
    write_generate_dataset_script(args, dir_src, script_data_dir, log_data_dir, generate_dataset_params)

    print("\nWriting distance between cells script")
    write_distance_between_cells_script(args, dir_src, script_dir, log_dir, dbc_params)

    print("\nWriting distance between cells combine script")
    write_distance_between_cells_combine_script(args, dir_src, script_dir, log_dir, dbc_combine_params)

    print("\nWriting lineage tree script")
    write_lineage_tree_script(args, dir_src, script_dir, log_dir)
    #####

    print("\nWriting combined data par simulation script")
    write_combined_data_script(args, dir_src, script_data_dir, log_data_dir, simulation_params,
                                   site_detection_params, generate_dataset_params)


if __name__ == "__main__":
    main()
