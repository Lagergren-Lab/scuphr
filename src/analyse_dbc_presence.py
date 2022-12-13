# This file contains the distance calculation code with presence.
# It also saves the presence matrices, which will be used while combining the matrices in the future steps.

import os
import sys
import time
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
from compute_zcy_presence import compute_zcyp_log_dict_pos
from fragment_enumerator import enumerate_fragment3
from fragment_probability import fragment_probability, fragment_log_probability
from mutation_type_probability_presence import compute_mutation_probabilities_log_dp
from distance_between_cells_probability_presence import distance_between_cells_all_configs_log_dp


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

    real_dist_matrix = np.zeros((num_cells + 1, num_cells + 1))
    for i in range(num_cells):
        for j in range(num_cells):
            if i <= j:
                continue
            else:
                # Compute distance between cell probabilities
                if cell_list[i].Y != cell_list[j].Y:
                    real_dist_matrix[i][j] = 1
                    real_dist_matrix[j][i] = 1

                # Set distances with bulk as well
                if cell_list[i].Y != 0:
                    real_dist_matrix[i][num_cells] = 1
                    real_dist_matrix[num_cells][i] = 1
                if cell_list[j].Y != 0:
                    real_dist_matrix[j][num_cells] = 1
                    real_dist_matrix[num_cells][j] = 1

    title_str = "matrix_real_difference_pos_" + str(pos)
    filename = matrix_dir + str(title_str) + ".out"
    np.savetxt(filename, real_dist_matrix)

    end_time = time.time()
    print("\nTotal time of position ", pos, " is: ", end_time - start_time)
    sys.stdout.flush()
    return pos, real_dist_matrix


def analyse_infer_dbc_one_pos_pool(pos, dataset, p_ae, p_ado, a_g, b_g, alpha_list, data_type, read_prob_dir,
                                   matrix_dir, common_z_dir, print_results=False):
    start_time = time.time()

    print("\n******\nAnalysing DBC for parameters: ", p_ae, p_ado, a_g, b_g)
    print("Pos idx of process is: ", pos)
    print("Process id: ", os.getpid(), ". Uname: ", os.uname())

    print("\n***********\nPosition: ", pos)

    cell_list = dataset['cell_list']
    num_cells = len(cell_list)

    bulk = dataset['bulk']
    z_list = dataset['z_list']

    filename = read_prob_dir + "read_dict.pickle"

    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print("\nLoading read dictionary...")
        read_dicts = load_dictionary(filename)
        read_dicts = read_dicts[int(pos)]
    else:
        filename = read_prob_dir + "read_dict_" + str(pos) + ".pickle"
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            print("\nLoading read position dictionary...")
            read_dicts = load_dictionary(filename)
        else:
            print("\nError. No read dictionary for pos ", str(pos))
            print("Calculating read probability...")
            read_dicts = precompute_reads(cell_list, z_list, bulk, print_results)
            save_dictionary(filename, read_dicts)

    # Compute read probabilities
    # read_dicts = precomputeReads(cell_list, z_list, bulk, printResults=False)

    # Compute mutation probabilities with beta prior dp
    # log_zcy_dict_prior = computeZCYLogDict(cell_list, bulk, z_list, read_dicts, p_m, p_ado, p_ae, printResults)
    # output =  mp.Queue()
    # pos, log_zcy_dict_prior = computeZCYLogDict(pos, output, dataset, read_dicts, p_ado, p_ae, printResults)

    log_zcy_filename = read_prob_dir + "log_zcyp_" + str(p_ado) + "_" + str(p_ae) + ".pickle"
    if os.path.exists(log_zcy_filename) and os.path.getsize(log_zcy_filename) > 0:
        print("\nLoading log_ZCYP dictionary...")
        log_zcy_dict_prior = load_dictionary(log_zcy_filename)
        log_zcy_dict_prior = log_zcy_dict_prior[int(pos)]
    else:
        log_zcy_pos_filename = read_prob_dir + "log_zcyp_" + str(p_ado) + "_" + str(p_ae) + "_" + str(pos) + ".pickle"
        if os.path.exists(log_zcy_pos_filename) and os.path.getsize(log_zcy_pos_filename) > 0:
            print("\nLoading log_ZCYP position dictionary...")
            log_zcy_dict_prior = load_dictionary(log_zcy_pos_filename)
        else:
            log_zcy_dict_prior = compute_zcyp_log_dict_pos(dataset, read_dicts, p_ado, p_ae, print_results)
            save_dictionary(log_zcy_pos_filename, log_zcy_dict_prior)

    print("\nLOG_ZCY")
    print(log_zcy_dict_prior)

    normed_prob_z, highest_z, highest_z_prob, max_key, general_lookup_table, log_prob_z \
        = compute_mutation_probabilities_log_dp(cell_list, z_list, alpha_list, log_zcy_dict_prior, a_g, b_g, p_ado,
                                                print_results=False)
    print("\n***Common Z results:\n")
    print("\nNormalized probabilities of common mutation type Z: \n", normed_prob_z)
    print("\nMaximum probability: ", highest_z_prob, ". Dict key: ", max_key)
    print("\nCorresponding mutation type: ", highest_z[0], highest_z[1])
    print("\nLog probabilities of common mutation type Z: \n", log_prob_z)
    print("\nsum Z: \n", sum(np.exp(log_prob_z)))

    gen_filename = read_prob_dir + "gen_lookup_" + str(p_ado) + "_" + str(p_ae) + "_" + str(pos) + ".pickle"
    #save_dictionary(gen_filename, general_lookup_table)

    filename2 = common_z_dir + "commonZstatus_" + str(pos) + ".txt"
    out = open(filename2, "w")

    is_correct = False
    if data_type == "synthetic":  # If the data is synthetic
        if dataset['commonZ'][0][0] == highest_z[0][0] and dataset['commonZ'][0][1] == highest_z[0][1] and \
                dataset['commonZ'][1][0] == highest_z[1][0] and dataset['commonZ'][1][1] == highest_z[1][1]:
            is_correct = True

    # print("\nIs correct: ", is_correct)
    sys.stdout.flush()

    # Homozygous
    if highest_z[0][1] == highest_z[1][1]:
        if is_correct:
            out.write(str(0))
        else:
            out.write(str(2))
            # Heterozygous
    else:
        if is_correct:
            out.write(str(1))
        else:
            out.write(str(3))
    out.close()

    dist_matrix = np.zeros((num_cells + 1, num_cells + 1))
    presence_matrix = np.zeros((num_cells + 1, num_cells + 1))
    np.fill_diagonal(presence_matrix, 1)

    for i in range(num_cells):
        for j in range(num_cells):  # num_cells
            if i <= j:
                continue
            else:
                # if True:
                if cell_list[i].lc > 0 and cell_list[j].lc > 0:

                    # call_noCall = False

                    # Call / No Call activated
                    # if call_noCall:
                    # if cell_list[i].X == 1:
                    # print("\n The Cell with ado1 reads for pos: ", pos, " is : ", i)
                    #    dist_matrix[i,:] = np.nan
                    #    dist_matrix[:,i] = np.nan
                    # elif cell_list[j].X == 1:
                    # print("\n The Cell with ado1 reads for pos: ", pos, " is : ", j)
                    #    dist_matrix[j,:] = np.nan
                    #    dist_matrix[:,j] = np.nan

                    # else:
                    # if True:
                    # Compute distance between cell probabilities
                    constant_cell_ids = [i, j]

                    # DBC with beta prior and DP
                    normed_prob_dists, prob_dists, _, _, _ = distance_between_cells_all_configs_log_dp(
                        constant_cell_ids,
                        z_list, cell_list,
                        alpha_list,
                        log_zcy_dict_prior,
                        general_lookup_table,
                        a_g, b_g, p_ado,
                        print_results)


                    # print("\n****")
                    # print("\nCells ", i, j)
                    # print("\nNormed prob dist: ", normed_prob_dists)
                    # print("\nProb_dists: ", prob_dists)
                    # print("\nsum Prob_dists: ", sum(prob_dists))

                    # Cells have same genotype and both have presence
                    diff_prob = normed_prob_dists[7] + normed_prob_dists[11]
                    pres_prob = normed_prob_dists[3] + normed_prob_dists[7] + normed_prob_dists[11] + normed_prob_dists[15]

                    # Sanity check
                    # same_prob = normed_prob_dists[0] + normed_prob_dists[1] + normed_prob_dists[2] \
                    #            + normed_prob_dists[3] + normed_prob_dists[12] + normed_prob_dists[13] \
                    #            + normed_prob_dists[14] + normed_prob_dists[15]

                    dist_matrix[i][j] = diff_prob
                    dist_matrix[j][i] = diff_prob

                    presence_matrix[i][j] = pres_prob
                    presence_matrix[j][i] = pres_prob

                    # Set distances with bulk as well
                    dist_cell_i_bulk = normed_prob_dists[10] + normed_prob_dists[11] \
                                       + normed_prob_dists[14] + normed_prob_dists[15]
                    dist_matrix[i][num_cells] = dist_cell_i_bulk
                    dist_matrix[num_cells][i] = dist_cell_i_bulk

                    pres_cell_i_bulk = normed_prob_dists[2] + normed_prob_dists[3] + normed_prob_dists[6] \
                                       + normed_prob_dists[7] + normed_prob_dists[10] + normed_prob_dists[11] \
                                       + normed_prob_dists[14] + normed_prob_dists[15]
                    presence_matrix[i][num_cells] = pres_cell_i_bulk
                    presence_matrix[num_cells][i] = pres_cell_i_bulk

                    dist_cell_j_bulk = normed_prob_dists[5] + normed_prob_dists[7] \
                                       + normed_prob_dists[13] + normed_prob_dists[15]
                    dist_matrix[j][num_cells] = dist_cell_j_bulk
                    dist_matrix[num_cells][j] = dist_cell_j_bulk

                    pres_cell_j_bulk = normed_prob_dists[1] + normed_prob_dists[3] + normed_prob_dists[5] \
                                       + normed_prob_dists[7] + normed_prob_dists[9] + normed_prob_dists[11] \
                                       + normed_prob_dists[13] + normed_prob_dists[15]
                    presence_matrix[j][num_cells] = pres_cell_j_bulk
                    presence_matrix[num_cells][j] = pres_cell_j_bulk

                    print("\n Difference. Cells with reads for pos: ", pos, " are ids: ", i, j, " dist: ", diff_prob)
                    print("\n Presence. Cells with reads for pos: ", pos, " are ids: ", i, j, " dist: ", pres_prob)

                # This part is to avoid calculations for cells with no reads at this position
                else:
                    # print("\n Cell with no reads for pos: ", pos, " are ids: ", i, j)
                    if cell_list[i].lc == 0:
                        # print("\n The Cell with no reads for pos: ", pos, " is : ", i)
                        dist_matrix[i, :] = np.nan
                        dist_matrix[:, i] = np.nan

                        presence_matrix[i, :] = 0
                        presence_matrix[:, i] = 0
                    else:
                        # print("\n The Cell with no reads for pos: ", pos, " is : ", j)
                        dist_matrix[j, :] = np.nan
                        dist_matrix[:, j] = np.nan

                        presence_matrix[j, :] = 0
                        presence_matrix[:, j] = 0

    title_str = "matrix_infer_difference_pos_" + str(pos)
    filename = matrix_dir + str(title_str) + ".out"
    np.savetxt(filename, dist_matrix)

    title_str = "matrix_infer_presence_pos_" + str(pos)
    filename = matrix_dir + str(title_str) + ".out"
    np.savetxt(filename, presence_matrix)

    end_time = time.time()
    print("\nTotal time of position ", pos, " is: ", end_time - start_time)
    sys.stdout.flush()
    return pos, dist_matrix


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
    parser.add_argument('--print_status', help="Specify the print (0 for do not print, 1 for print). Default: 1",
                        type=int, default=1)
    parser.add_argument('--seed_val', help="Specify the seed. Default: 123", type=int, default=123)

    args = parser.parse_args()

    if args.print_status == 1:
        print_results = True
    else:
        print_results = False

    alpha_list = np.ones(12)

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
    dataset = load_dictionary(proc_dict_dir + "data.pickle")
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
    positions = np.arange(args.pos_range_min, args.pos_range_max)

    ###
    # If the data is synthetic
    if args.data_type == "synthetic":
        print("\n*****\nCalculating real distances...")

        pool = mp.Pool()
        real_results = pool.starmap(analyse_real_dbc_one_pos_pool,
                                    [(int(pos), dataset[str(pos)], matrix_dir) for pos in positions])
        # real_results = [pool.apply(analyseRealDBC_onepos_pool, args=(int(pos), dataset[str(pos)], matrix_dir,
        # print_results)) for pos in positions]
        pool.close()
        pool.join()

        print("\n*****\nCalculating real distances finished...")
        sys.stdout.flush()

    ###

    print("\n*****\nCalculating inferred distances...")

    pool = mp.Pool()
    infer_results = pool.starmap(analyse_infer_dbc_one_pos_pool, [(int(pos), dataset[str(pos)], args.p_ae, args.p_ado,
                                                                   args.a_g, args.b_g, alpha_list, args.data_type,
                                                                   read_prob_dir, matrix_dir, common_z_dir,
                                                                   print_results) for pos in positions])
    # infer_results = [pool.apply(analyseInferDBC_onepos_pool, args=(int(pos), dataset[str(pos)], args.p_ae,
    # args.p_ado, args.a_g, args.b_g, alpha_list, args.data_type, read_prob_dir, matrix_dir, common_z_dir,
    # print_results)) for pos in positions]
    pool.close()
    pool.join()

    print("\n*****\nCalculating inferred distances finished...")
    sys.stdout.flush()

    ###
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
            title_str = "matrix_infer_difference_pos_" + str(pos)
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
            title_str = "matrix_real_difference_pos_" + str(pos)
            plt.title(title_str)

            plt.subplot(132)
            plt.imshow(dist_matrix_dict_pos, interpolation='nearest', cmap='viridis', vmin=0, vmax=1)
            plt.colorbar()
            plt.xlabel("Cells")
            plt.ylabel("Cells")
            title_str = "matrix_infer_difference_pos_" + str(pos)
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
