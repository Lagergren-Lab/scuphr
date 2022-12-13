import os
import sys
import time
import json
import pickle
import datetime
import argparse
import matplotlib
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from lambda_iter import partition_lambda
from read_probability import precompute_reads
from compute_zcy import compute_zcy_log_dict_pos
from fragment_enumerator import enumerate_fragment3
from fragment_probability import fragment_log_probability
from mutation_type_probability import compute_mutation_probabilities_log_dp
from distance_between_cells_probability import distance_between_cells_all_configs_log_dp

from analyse_dbc_json_paired_site import analyse_infer_dbc_one_pos_pool as analyse_infer_dbc_one_pos_pool_paired
from analyse_dbc_json_single_site import analyse_infer_dbc_one_pos_pool as analyse_infer_dbc_one_pos_pool_singleton

def save_json(filename, cell_dict):
    with open(filename, 'w') as fp:
        json.dump(cell_dict, fp)


def load_json(filename):
    with open(filename) as fp:
        cell_dict = json.load(fp)
    return cell_dict


def save_dictionary(filename, cell_dict):
    with open(filename, 'wb') as fp:
        pickle.dump(cell_dict, fp)


def load_dictionary(filename):
    with open(filename, 'rb') as fp:
        cell_dict = pickle.load(fp)
    return cell_dict


def analyse_real_dbc_one_pos_pool(pos, dataset_sample, matrix_dir):
    start_time = time.time()

    print("\n******\nAnalysing Real DBC")
    print("Pos idx of process is: ", pos)
    print("Process id: ", os.getpid(), ". Uname: ", os.uname())

    print("\n***********\nPosition: ", pos)

    cell_list = dataset_sample['cell_list']
    num_cells = len(cell_list)

    real_dist_matrix = np.ones((num_cells + 1, num_cells + 1))
    for i in range(num_cells):
        for j in range(num_cells):
            if i <= j:
                continue
            else:
                # Compute distance between cell probabilities
                #if cell_list[i].Y != cell_list[j].Y:
                if cell_list[i]['y'] != cell_list[j]['y']:
                    real_dist_matrix[i][j] = 0
                    real_dist_matrix[j][i] = 0

                # Set distances with bulk as well
                #if cell_list[i].Y != 0:
                if cell_list[i]['y'] != 0:
                    real_dist_matrix[i][num_cells] = 0
                    real_dist_matrix[num_cells][i] = 0
                #if cell_list[j].Y != 0:
                if cell_list[j]['y'] != 0:
                    real_dist_matrix[j][num_cells] = 0
                    real_dist_matrix[num_cells][j] = 0

    title_str = "matrix_real_similarity_pos_" + str(pos)
    filename = matrix_dir + str(title_str) + ".out"
    np.savetxt(filename, real_dist_matrix)

    end_time = time.time()
    print("\nTotal time of position ", pos, " is: ", end_time - start_time)
    sys.stdout.flush()
    return pos, real_dist_matrix


def analyse_real_dbc_one_pos(pos, output, dataset_sample, matrix_dir):
    start_time = time.time()

    print("\n******\nAnalysing Real DBC")
    print("Pos idx of process is: ", pos)
    print("Process id: ", os.getpid(), ". Uname: ", os.uname())

    print("\n***********\nPosition: ", pos)

    cell_list = dataset_sample['cell_list']
    num_cells = len(cell_list)

    real_dist_matrix = np.ones((num_cells + 1, num_cells + 1))
    for i in range(num_cells):
        for j in range(num_cells):
            if i <= j:
                continue
            else:
                # Compute distance between cell probabilities
                if cell_list[i].Y != cell_list[j].Y:
                    real_dist_matrix[i][j] = 0
                    real_dist_matrix[j][i] = 0

                # Set distances with bulk as well
                if cell_list[i].Y != 0:
                    real_dist_matrix[i][num_cells] = 0
                    real_dist_matrix[num_cells][i] = 0
                if cell_list[j].Y != 0:
                    real_dist_matrix[j][num_cells] = 0
                    real_dist_matrix[num_cells][j] = 0

    title_str = "matrix_real_similarity_pos_" + str(pos)
    filename = matrix_dir + str(title_str) + ".out"
    np.savetxt(filename, real_dist_matrix)

    end_time = time.time()
    print("\nTotal time of position ", pos, " is: ", end_time - start_time)
    sys.stdout.flush()
    output.put((pos, real_dist_matrix))


def main():
    # np.random.seed(123) # I set the seed to select same positions for the experiments. If you want, remove this part.

    dt = datetime.datetime.now()
    default_dir = "../results/%s_%s_%s/" % (dt.year, dt.month, dt.day)

    # code to process command line arguments
    parser = argparse.ArgumentParser(description='Distance between cells computation.')
    parser.add_argument('global_dir', help="Specify the data directory.", type=str)
    parser.add_argument('--a_g', help="Specify the alpha prior of mutation probability. Default: 1", type=float,
                        default=1)
    parser.add_argument('--b_g', help="Specify the beta prior of mutation probability. Default: 1", type=float,
                        default=1)
    parser.add_argument('--data_type', help="Specify the data type. Default: real", type=str, default="real")
    parser.add_argument('--output_dir', help="Specify the output directory.", type=str, default=default_dir)
    parser.add_argument('--p_ado', help="Specify the initial allelic dropout probability of a base. Default: 0.1",
                        type=float, default=0.1)
    parser.add_argument('--p_ae', help="Specify the initial amplification error probability of a base. Default: 0.001",
                        type=float, default=0.001)
    parser.add_argument('--pos_range_min', help="Specify the position range (min value). Default: 0", type=int,
                        default=0)
    parser.add_argument('--pos_range_max', help="Specify the position range (max value). Default: 0", type=int,
                        default=0)
    parser.add_argument('--print_status', help="Specify the print (0 for do not print, 1 for print). Default: 0",
                        type=int, default=0)
    parser.add_argument('--scuphr_strategy',
                        help="Specify the strategy for Scuphr (paired, singleton, hybrid). Default: paired",
                        type=str, default="paired")
    parser.add_argument('--seed_val', help="Specify the seed. Default: 123", type=int, default=123)

    args = parser.parse_args()

    if args.print_status == 1:
        print_results = True
    else:
        print_results = False

    proc_dict_dir = args.global_dir + "processed_data_dict/"
    read_prob_dir = proc_dict_dir + "read_probabilities/"

    if not os.path.exists(read_prob_dir):
        os.makedirs(read_prob_dir)
        print("Read probability directory is created")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print("Output directory is created")

    matrix_dir = args.output_dir + "matrix/"
    if not os.path.exists(matrix_dir):
        os.makedirs(matrix_dir)
        print("Matrix directory is created")

    common_z_dir = args.output_dir + "commonZstatus/"
    if not os.path.exists(common_z_dir):
        os.makedirs(common_z_dir)
        print("Common Z status directory is created")

    fig_dir = args.output_dir + "figures_pos/"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        print("Figures directory is created")

    print("\n*****\nLoad the dataset and read probabilities...")

    dataset = load_json(proc_dict_dir + "data.txt")

    num_cells = len(dataset['0']['cell_list'])
    num_total_positions = len(dataset)
    print("\nThe dataset is loaded. Number of cells: %d, number of position pairs: %d"
          % (num_cells, num_total_positions))

    # filename = read_prob_dir + "read_dict.pickle"
    # read_dicts = loadDictionary(filename)

    if args.pos_range_max == 0:
        args.pos_range_max = num_total_positions

    print("\nPosition range: [", args.pos_range_min, ", ", args.pos_range_max, ")")
    print("Number of positions in range: ", args.pos_range_max - args.pos_range_min)

    # This part selects relevant sites to the Scuphr strategy.
    # hybrid accepts all sites & site-pairs,
    # paired accepts only the site-pairs, singleton accepts only the single-sites.
    print("Strategy: ", args.scuphr_strategy)
    positions_orig = np.arange(args.pos_range_min, args.pos_range_max)
    positions_paired = []
    positions_singleton = []
    for pos in positions_orig:
        cur_bulk = np.array(dataset[str(pos)]["bulk"])
        if cur_bulk.shape == (2, 2):
            positions_paired.append(pos)
        elif cur_bulk.shape == (2, ):
            positions_singleton.append(pos)
    print("\tTotal number of paired positions: \t", len(positions_paired))
    print("\tTotal number of singleton positions: \t", len(positions_singleton))

    if args.scuphr_strategy == "paired":
        positions = positions_paired
    elif args.scuphr_strategy == "singleton":
        positions = positions_singleton
    else:
        positions = positions_orig
    positions = np.array(positions)
    print("Number of valid positions in range: ", len(positions))
    print(positions)

    if len(positions) == 0:
        print("\nWARNING! There are no positions to calculate distance.")
        sys.exit()

    ###
    # If the data is synthetic
    if args.data_type == "synthetic":
        print("\n*****\nCalculating real distances...")

        pool = mp.Pool()
        real_results = pool.starmap(analyse_real_dbc_one_pos_pool,
                                    [(int(pos), dataset[str(pos)], matrix_dir) for pos in positions])
        pool.close()
        pool.join()

        print("\n*****\nCalculating real distances finished...")
        sys.stdout.flush()

    ###
    print("\n*****\nCalculating inferred distances...")
    if args.scuphr_strategy == "paired":
        pool = mp.Pool()
        infer_results = pool.starmap(analyse_infer_dbc_one_pos_pool_paired,
                                     [(int(pos), dataset[str(pos)], args.p_ae, args.p_ado,
                                       args.a_g, args.b_g, args.data_type,
                                       read_prob_dir, matrix_dir, common_z_dir,
                                       print_results) for pos in positions_paired])
        pool.close()
        pool.join()
    elif args.scuphr_strategy == "singleton":
        pool = mp.Pool()
        infer_results = pool.starmap(analyse_infer_dbc_one_pos_pool_singleton,
                                     [(int(pos), dataset[str(pos)], args.p_ae, args.p_ado,
                                       args.a_g, args.b_g, args.data_type,
                                       read_prob_dir, matrix_dir, common_z_dir,
                                       print_results) for pos in positions_singleton])
        pool.close()
        pool.join()
    else:  #args.scuphr_strategy == "hybrid":
        pool = mp.Pool()
        infer_results_paired = pool.starmap(analyse_infer_dbc_one_pos_pool_paired,
                                     [(int(pos), dataset[str(pos)], args.p_ae, args.p_ado,
                                       args.a_g, args.b_g, args.data_type,
                                       read_prob_dir, matrix_dir, common_z_dir,
                                       print_results) for pos in positions_paired])
        pool.close()
        pool.join()

        pool = mp.Pool()
        infer_results_singleton = pool.starmap(analyse_infer_dbc_one_pos_pool_singleton,
                                     [(int(pos), dataset[str(pos)], args.p_ae, args.p_ado,
                                       args.a_g, args.b_g, args.data_type,
                                       read_prob_dir, matrix_dir, common_z_dir,
                                       print_results) for pos in positions_singleton])
        pool.close()
        pool.join()

        infer_results = infer_results_paired + infer_results_singleton

    print("\n*****\nCalculating inferred distances finished...")
    sys.stdout.flush()

    ###
    if False:
        print("\n*****\nPlotting distance matrices...")

        # Save and plot results
        for res_idx in range(len(infer_results)):
            # pos = str(infer_results[res_idx][0])
            # corresp_infer_results = [item for item in infer_results if item[0] == int(pos)][0]
            # dist_matrix_dict_pos = corresp_infer_results[1]

            pos = str(infer_results[res_idx][0])
            dist_matrix_dict_pos = infer_results[res_idx][1]

            if args.data_type == "real":  # If the data is real-world
                plt.figure(figsize=(20, 12))

                plt.imshow(dist_matrix_dict_pos, interpolation='nearest', cmap='viridis', vmin=0, vmax=1)
                plt.colorbar()
                plt.xlabel("Cells")
                plt.ylabel("Cells")
                title_str = "matrix_infer_similarity_pos_" + str(pos)
                plt.title(title_str)

                title_str = "dbc_pos_" + str(pos)
                filename = fig_dir + str(title_str) + ".png"
                plt.savefig(filename)
                plt.close()

            else:  # If the data is synthetic
                corresp_real_results = [item for item in real_results if item[0] == int(pos)][0]
                real_dist_matrix_dict_pos = corresp_real_results[1]

                diff_matrix = abs(real_dist_matrix_dict_pos - dist_matrix_dict_pos)
                title_str = "matrix_absolute_difference_pos_" + str(pos)
                filename = args.output_dir + "/matrix/" + str(title_str) + ".out"
                np.savetxt(filename, diff_matrix)

                diff_matrix_dict_pos = diff_matrix

                plt.figure(figsize=(20, 12))
                plt.subplot(131)
                plt.imshow(real_dist_matrix_dict_pos, interpolation='nearest', cmap='viridis', vmin=0, vmax=1)
                plt.colorbar()
                plt.xlabel("Cells")
                plt.ylabel("Cells")
                title_str = "matrix_real_similarity_pos_" + str(pos)
                plt.title(title_str)

                plt.subplot(132)
                plt.imshow(dist_matrix_dict_pos, interpolation='nearest', cmap='viridis', vmin=0, vmax=1)
                plt.colorbar()
                plt.xlabel("Cells")
                plt.ylabel("Cells")
                title_str = "matrix_infer_similarity_pos_" + str(pos)
                plt.title(title_str)

                plt.subplot(133)
                plt.imshow(diff_matrix_dict_pos, interpolation='nearest', cmap='viridis', vmin=0, vmax=1)
                plt.colorbar()
                plt.xlabel("Cells")
                plt.ylabel("Cells")
                title_str = "matrix_absolute_difference_pos_" + str(pos)
                plt.title(title_str)

                title_str = "dbc_pos_" + str(pos)
                filename = fig_dir + str(title_str) + ".png"
                plt.savefig(filename)
                plt.close()

        sys.stdout.flush()


if __name__ == "__main__":
    main()
