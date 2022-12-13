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

from read_probability_single_site import precompute_reads
from compute_zcy_single_site import compute_zcy_log_dict_pos
from mutation_type_probability import compute_mutation_probabilities_log_dp
from distance_between_cells_probability import distance_between_cells_all_configs_log_dp


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


def analyse_infer_dbc_one_pos_pool(pos, dataset, p_ae, p_ado, a_g, b_g, data_type, read_prob_dir,
                                   matrix_dir, common_z_dir, print_results=False):
    start_time = time.time()

    #print("\n******\nAnalysing DBC for parameters: ", p_ae, p_ado, a_g, b_g)
    #print("Pos idx of process is: ", pos)
    #print("Process id: ", os.getpid(), ". Uname: ", os.uname())

    #print("\n***********\nPosition: ", pos)
    cell_list = dataset['cell_list']
    num_cells = len(cell_list)

    bulk = dataset['bulk']
    z_list = dataset['z_list']
    alpha_list = np.ones(len(z_list))
    filename = read_prob_dir + "read_dict.pickle"

    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        #print("\nLoading read dictionary...")
        read_dicts = load_json(filename)
        read_dicts = read_dicts[int(pos)]
    else:
        filename = read_prob_dir + "read_dict_" + str(pos) + ".pickle"
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            #print("\nLoading read position dictionary...")
            read_dicts = load_json(filename)
        else:
            #print("\nError. No read dictionary for pos ", str(pos))
            #print("Calculating read probability...")
            read_dicts = precompute_reads(cell_list, z_list, bulk, print_results)
            save_json(filename, read_dicts)

    # Compute read probabilities
    # read_dicts = precomputeReads(cell_list, z_list, bulk, printResults=False)

    # Compute mutation probabilities with beta prior dp
    # log_zcy_dict_prior = computeZCYLogDict(cell_list, bulk, z_list, read_dicts, p_m, p_ado, p_ae, printResults)
    # output =  mp.Queue()
    # pos, log_zcy_dict_prior = computeZCYLogDict(pos, output, dataset, read_dicts, p_ado, p_ae, printResults)

    log_zcy_filename = read_prob_dir + "log_zcy_" + str(p_ado) + "_" + str(p_ae) + ".pickle"
    if os.path.exists(log_zcy_filename) and os.path.getsize(log_zcy_filename) > 0:
        #print("\nLoading log_ZCY dictionary...")
        log_zcy_dict_prior = load_json(log_zcy_filename)
        log_zcy_dict_prior = log_zcy_dict_prior[int(pos)]
    else:
        log_zcy_pos_filename = read_prob_dir + "log_zcy_" + str(p_ado) + "_" + str(p_ae) + "_" + str(pos) + ".pickle"
        if os.path.exists(log_zcy_pos_filename) and os.path.getsize(log_zcy_pos_filename) > 0:
            #print("\nLoading log_ZCY position dictionary...")
            log_zcy_dict_prior = load_json(log_zcy_pos_filename)
        else:
            log_zcydda_dict = None

            # Check if ZCYDDA file exists
            log_zcydda_pos_filename = read_prob_dir + "log_zcydda_" + str(pos) + ".pickle"
            if os.path.exists(log_zcydda_pos_filename) and os.path.getsize(log_zcydda_pos_filename) > 0:
                #print("\nLoading log_ZCYDDA position dictionary...")
                log_zcydda_dict = load_json(log_zcydda_pos_filename)
                log_zcy_dict_prior, _ = compute_zcy_log_dict_pos(dataset, read_dicts, p_ado, p_ae, print_results, log_zcydda_dict)
                #save_json(log_zcy_pos_filename, log_zcy_dict_prior)
            else:
                log_zcy_dict_prior, log_zcydda_dict = compute_zcy_log_dict_pos(dataset, read_dicts, p_ado, p_ae, print_results, log_zcydda_dict)
                save_json(log_zcydda_pos_filename, log_zcydda_dict)
                #save_json(log_zcy_pos_filename, log_zcy_dict_prior)

    # print("\nLOG_ZCY")
    # print(log_zcy_dict_prior)

    normed_prob_z, highest_z, highest_z_prob, max_key, general_lookup_table, log_prob_z\
        = compute_mutation_probabilities_log_dp(cell_list, z_list, alpha_list, log_zcy_dict_prior, a_g, b_g, p_ado,
                                                print_results=False)
    if print_results:
        print("\n***Common Z results:")
        print("Normalized probabilities of common mutation type Z: \n", normed_prob_z)
        print("Maximum probability: ", highest_z_prob, ". Dict key: ", max_key)
        print("Corresponding mutation type: ", highest_z[0], highest_z[1])
        print("Log probabilities of common mutation type Z: \n", log_prob_z)
        print("sum Z: \n", sum(np.exp(log_prob_z)))

    gen_filename = read_prob_dir + "gen_lookup_" + str(p_ado) + "_" + str(p_ae) + "_" + str(pos) + ".pickle"
    #save_dictionary(gen_filename, general_lookup_table)

    filename2 = common_z_dir + "commonZstatus_" + str(pos) + ".txt"
    out = open(filename2, "w")

    is_correct = False
    if data_type == "synthetic":  # If the data is synthetic
        #if dataset['commonZ'][0][0] == highest_z[0][0] and dataset['commonZ'][0][1] == highest_z[0][1] and \
        #        dataset['commonZ'][1][0] == highest_z[1][0] and dataset['commonZ'][1][1] == highest_z[1][1]:

        # Since it is single site, the order doesn't matter
        if dataset['common_z'][0] == highest_z[0] and dataset['common_z'][1] == highest_z[1]:
            is_correct = True
        elif dataset['common_z'][1] == highest_z[0] and dataset['common_z'][0] == highest_z[1]:
            is_correct = True

    # print("\nIs correct: ", is_correct)
    sys.stdout.flush()

    # Homozygous
    if highest_z[0] == highest_z[1]:
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

    dist_matrix = np.ones((num_cells + 1, num_cells + 1))

    for i in range(num_cells):
        for j in range(num_cells):  # num_cells
            if i <= j:
                continue
            else:
                # if True:
                #if cell_list[i].lc > 0 and cell_list[j].lc > 0:
                if cell_list[i]['lc'] > 0 and cell_list[j]['lc'] > 0:

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

                    same_prob = normed_prob_dists[0] + normed_prob_dists[3]

                    dist_matrix[i][j] = same_prob
                    dist_matrix[j][i] = same_prob

                    # Set distances with bulk as well
                    dist_matrix[i][num_cells] = normed_prob_dists[0] + normed_prob_dists[1]
                    dist_matrix[num_cells][i] = normed_prob_dists[0] + normed_prob_dists[1]

                    dist_matrix[j][num_cells] = normed_prob_dists[0] + normed_prob_dists[2]
                    dist_matrix[num_cells][j] = normed_prob_dists[0] + normed_prob_dists[2]

                    if print_results:
                        print("Similarity. Cells with reads for pos: ", pos, " are ids: ", i, j, " dist: ", same_prob)

                # This part is to avoid calculations for cells with no reads at this position
                else:
                    # print("\n Cell with no reads for pos: ", pos, " are ids: ", i, j)
                    if cell_list[i]['lc'] == 0:
                        # print("\n The Cell with no reads for pos: ", pos, " is : ", i)
                        dist_matrix[i, :] = np.nan
                        dist_matrix[:, i] = np.nan
                    else:
                        # print("\n The Cell with no reads for pos: ", pos, " is : ", j)
                        dist_matrix[j, :] = np.nan
                        dist_matrix[:, j] = np.nan

    #dist_matrix = np.ones((num_cells + 1, num_cells + 1))

    title_str = "matrix_infer_similarity_pos_" + str(pos)
    filename = matrix_dir + str(title_str) + ".out"
    np.savetxt(filename, dist_matrix)

    end_time = time.time()
    #print("\nTotal time of position ", pos, " is: ", end_time - start_time)
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

    #dataset = load_dictionary(proc_dict_dir + "data.pickle")
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
    positions = np.arange(args.pos_range_min, args.pos_range_max)

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

    pool = mp.Pool()
    infer_results = pool.starmap(analyse_infer_dbc_one_pos_pool, [(int(pos), dataset[str(pos)], args.p_ae, args.p_ado,
                                                                   args.a_g, args.b_g, args.data_type,
                                                                   read_prob_dir, matrix_dir, common_z_dir,
                                                                   print_results) for pos in positions])
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
