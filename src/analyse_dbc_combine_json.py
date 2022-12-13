import os
import sys
import time
import json
import pickle
import argparse
import datetime
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def save_json(filename, cell_dict):
    with open(filename, 'w') as fp:
        json.dump(cell_dict, fp)


def load_json(filename):
    with open(filename) as fp:
        cell_dict = json.load(fp)
    return cell_dict

def load_dictionary(filename):
    with open(filename, 'rb') as fp:
        cell_dict = pickle.load(fp)
    return cell_dict


def load_dicts(global_dir, positions, data_type):
    start_time = time.time()

    real_dist_matrix_dict = {}
    dist_matrix_dict = {}
    num_invalid = 0

    for pos in positions:
        try:
            if data_type == "synthetic":
                title_str = "matrix_real_similarity_pos_" + str(pos)
                filename = global_dir + "matrix/" + str(title_str) + ".out"
                real_dist_matrix_dict[str(pos)] = np.loadtxt(filename)

            title_str = "matrix_infer_similarity_pos_" + str(pos)
            filename = global_dir + "matrix/" + str(title_str) + ".out"
            dist_matrix_dict[str(pos)] = np.loadtxt(filename)
        except:
            num_invalid += 1

    end_time = time.time()
    print("\nTotal time of loading dictionaries: ", end_time - start_time)
    print("Number of invalid dictionaries: ", num_invalid)

    return real_dist_matrix_dict, dist_matrix_dict


def combine_multiple_sites(global_dir, real_dist_matrix_dict, dist_matrix_dict, num_cells, positions, data_type):
    start_time = time.time()

    print("\n*****\nCombining multiple sites")

    marg_sum_list = []
    fro_norm_list = []

    real_res_sub_sum = np.zeros((num_cells + 1, num_cells + 1))
    res_sub_sum = np.zeros((num_cells + 1, num_cells + 1))

    real_res_mask = np.zeros((num_cells + 1, num_cells + 1))
    res_mask = np.zeros((num_cells + 1, num_cells + 1))

    sub = 1

    num_het = 0
    num_homoz = 0
    num_correct_het = 0
    num_correct_homoz = 0
    skipped_positions = []
    num_invalid = 0

    for pos in positions:
        try:
            # Check commonZ status
            filename2 = global_dir + "commonZstatus/commonZstatus_" + str(pos) + ".txt"
            out = open(filename2)
            status = int(out.read())
            out.close()

            # He site is homozygous

            if status == 0 or status == 2:
                num_homoz += 1
                skipped_positions.append(pos)
                print("Skipping position ", pos, " (He is homozygous in inferredZ)")
                # Correct guess
                if status == 0:
                    num_correct_homoz += 1

            # He site is heterozygous
            else:
                num_het += 1
                # Correct guess
                if status == 1:
                    num_correct_het += 1

                inferred_dist_matrix = np.copy(dist_matrix_dict[str(pos)])
                # inferred_temp = np.ma.masked_less(inferred_dist_matrix,0)
                inferred_temp = np.ma.masked_invalid(inferred_dist_matrix)
                inferred_mask = (~inferred_temp.mask) * 1
                inferred_temp = inferred_temp.filled(fill_value=0)

                res_sub_sum = res_sub_sum + inferred_temp
                res_mask = res_mask + inferred_mask

                if data_type == "synthetic":  # If the dataset is synthetic
                    real_dist_matrix = np.copy(real_dist_matrix_dict[str(pos)])
                    # real_temp = np.ma.masked_less(real_dist_matrix,0)
                    real_temp = np.ma.masked_invalid(real_dist_matrix)
                    real_mask = (~real_temp.mask) * 1
                    real_temp = real_temp.filled(fill_value=0)

                    real_res_sub_sum = real_res_sub_sum + real_temp
                    real_res_mask = real_res_mask + real_mask

                #print("Using ", sub, " positions")

                res_sub = np.divide(res_sub_sum, res_mask)

                if data_type == "synthetic":  # If the dataset is synthetic
                    real_res_sub = np.divide(real_res_sub_sum, real_res_mask)
                    real_diff_matrix_sub = abs(real_res_sub - res_sub)

                    marg_diff = np.nansum(real_diff_matrix_sub, axis=1)  # assumes Nan as zero
                    marg_sum_list.append(np.sum(marg_diff))

                    real_diff_matrix_sub = np.nan_to_num(real_diff_matrix_sub)
                    fro_norm = np.linalg.norm(real_diff_matrix_sub, ord='fro')
                    fro_norm_list.append(fro_norm)

                ##########

                if data_type == "synthetic":  # If the dataset is synthetic
                    title_str = "matrix_real_similarity_pos_so_far_" + str(sub)
                    filename = global_dir + "matrix/" + str(title_str) + ".out"
                    np.savetxt(filename, real_res_sub)
                    # filename = dir_info + "/matrix/Matrix_" + str(title_str) + "_count.out"
                    # np.savetxt(filename, real_res_mask)

                    title_str = "matrix_infer_similarity_pos_so_far_" + str(sub)
                    filename = global_dir + "matrix/" + str(title_str) + ".out"
                    np.savetxt(filename, res_sub)
                    # filename = global_dir + "/matrix/Matrix_" + str(title_str) + "_count.out"
                    # np.savetxt(filename, res_mask)

                    title_str = "matrix_absolute_difference_pos_so_far_" + str(sub)
                    filename = global_dir + "matrix/" + str(title_str) + ".out"
                    np.savetxt(filename, real_diff_matrix_sub)

                    if False:
                        plt.figure(figsize=(20, 12))

                        plt.subplot(131)
                        plt.imshow(real_res_sub, interpolation='nearest', cmap='viridis', vmin=0, vmax=1)
                        plt.colorbar()
                        plt.xlabel("Cells")
                        plt.ylabel("Cells")
                        title_str = "matrix_real_similarity_pos_so_far_" + str(sub)
                        plt.title(title_str)

                        plt.subplot(132)
                        plt.imshow(res_sub, interpolation='nearest', cmap='viridis', vmin=0, vmax=1)
                        plt.colorbar()
                        plt.xlabel("Cells")
                        plt.ylabel("Cells")
                        title_str = "matrix_infer_similarity_pos_so_far_" + str(sub)
                        plt.title(title_str)

                        plt.subplot(133)
                        plt.imshow(real_diff_matrix_sub, interpolation='nearest', cmap='viridis', vmin=0, vmax=1)
                        plt.colorbar()
                        plt.xlabel("Cells")
                        plt.ylabel("Cells")
                        title_str = "matrix_absolute_difference_pos_so_far_" + str(sub)
                        plt.title(title_str)

                        title_str = "dbc_pos_so_far_" + str(sub)
                        filename = global_dir + "figures_comb/" + str(title_str) + ".png"
                        plt.savefig(filename)
                        plt.close()

                else:  # If the dataset is real-world
                    title_str = "matrix_infer_similarity_pos_so_far_" + str(sub)
                    filename = global_dir + "matrix/" + str(title_str) + ".out"
                    #np.savetxt(filename, res_sub)
                    # filename = global_dir + "/matrix/Matrix_" + str(title_str) + "_count.out"
                    # np.savetxt(filename, res_mask)

                    if False:
                        plt.figure(figsize=(20, 12))

                        plt.imshow(res_sub, interpolation='nearest', cmap='viridis', vmin=0, vmax=1)
                        plt.colorbar()
                        plt.xlabel("Cells")
                        plt.ylabel("Cells")
                        title_str = "matrix_infer_similarity_pos_so_far_" + str(sub)
                        plt.title(title_str)

                        title_str = "dbc_pos_so_far_" + str(sub)
                        filename = global_dir + "figures_comb/" + str(title_str) + ".png"
                        plt.savefig(filename)
                        plt.close()

                sub += 1
        except:
            num_invalid += 1
    print("\tNumber of invalid pos: \t", num_invalid)

    num_total = num_het + num_homoz
    print("\n*****\nAnalysis: ")
    print("\tTotal number: \t", num_total)
    print("\tHeterozygous: \t", num_het, ". \t%", (100 * num_het) / num_total)
    if num_het > 0:
        print("\tCorrect heterozygous: \t", num_correct_het, ". \t%", (100 * num_correct_het) / num_het)
    print("\tHomozygous: \t", num_homoz, ". \t%", (100 * num_homoz) / num_total)
    if num_homoz > 0:
        print("\t(Correct homozygous: \t", num_correct_homoz, ". \t%", (100 * num_correct_homoz) / num_homoz, ")")
    print("\t(Skipped positions: ", skipped_positions, ")")

    filename3 = global_dir + "commonZstatus/commonZanalysis.txt"
    with open(filename3, 'w') as out3:
        out3.write("inferredZ Analysis: ")
        out3.write("\nTotal number: %d" % num_total)
        out3.write("\nHeterozygous: %d. Perc: %.2f" % (num_het, (100 * num_het) / num_total))
        if num_het > 0:
            out3.write("\nCorrect heterozygous: %d. Perc: %.2f" % (num_correct_het, (100 * num_correct_het) / num_het))
        out3.write("\nHomozygous: %d. Perc: %.2f" % (num_homoz, (100 * num_homoz) / num_total))
        if num_homoz > 0:
            out3.write("\nCorrect homozygous: %d. Perc: %.2f"
                       % (num_correct_homoz, (100 * num_correct_homoz) / num_homoz))
        out3.write("\nSkipped positions: ")
        for sk_pos in skipped_positions:
            out3.write(" %s " % str(sk_pos))

    filename3 = global_dir + "commonZanalysis.txt"
    with open(filename3, 'w') as out3:
        out3.write("inferredZ Analysis: ")
        out3.write("\nTotal number: %d" % num_total)
        out3.write("\nHeterozygous: %d. Perc: %.2f" % (num_het, (100 * num_het) / num_total))
        if num_het > 0:
            out3.write("\nCorrect heterozygous: %d. Perc: %.2f" % (num_correct_het, (100 * num_correct_het) / num_het))
        out3.write("\nHomozygous: %d. Perc: %.2f" % (num_homoz, (100 * num_homoz) / num_total))
        if num_homoz > 0:
            out3.write("\nCorrect homozygous: %d. Perc: %.2f"
                       % (num_correct_homoz, (100 * num_correct_homoz) / num_homoz))
        out3.write("\nSkipped positions: ")
        for sk_pos in skipped_positions:
            out3.write(" %s " % str(sk_pos))

    # Save Final Matrices
    if data_type == "synthetic":  # If the dataset is synthetic
        filename = global_dir + "real_dif.csv"
        np.savetxt(filename, 1 - real_res_sub, delimiter=',', fmt="%.10f")
        filename = global_dir + "real_dif_count.out"
        np.savetxt(filename, real_res_mask)

    filename = global_dir + "infer_dif.csv"
    np.savetxt(filename, 1 - res_sub, delimiter=',')
    filename = global_dir + "infer_dif_count.out"
    np.savetxt(filename, res_mask)

    # Save the final figure
    if data_type == "synthetic":  # If the dataset is synthetic
        plt.figure(figsize=(20, 12))

        plt.subplot(131)
        plt.imshow(real_res_sub, interpolation='nearest', cmap='viridis', vmin=0, vmax=1)
        plt.colorbar()
        plt.xlabel("Cells")
        plt.ylabel("Cells")
        title_str = "matrix_real_similarity"
        plt.title(title_str)

        plt.subplot(132)
        plt.imshow(res_sub, interpolation='nearest', cmap='viridis', vmin=0, vmax=1)
        plt.colorbar()
        plt.xlabel("Cells")
        plt.ylabel("Cells")
        title_str = "matrix_infer_similarity"
        plt.title(title_str)

        plt.subplot(133)
        plt.imshow(real_diff_matrix_sub, interpolation='nearest', cmap='viridis', vmin=0, vmax=1)
        plt.colorbar()
        plt.xlabel("Cells")
        plt.ylabel("Cells")
        title_str = "matrix_absolute_difference"
        plt.title(title_str)

        title_str = "dbc"
        filename = global_dir + str(title_str) + ".png"
        plt.savefig(filename)
        plt.close()

    else:  # If the dataset is real-world
        plt.figure(figsize=(20, 12))

        plt.imshow(res_sub, interpolation='nearest', cmap='viridis', vmin=0, vmax=1)
        plt.colorbar()
        plt.xlabel("Cells")
        plt.ylabel("Cells")
        title_str = "matrix_similarity"
        plt.title(title_str)

        title_str = "dbc"
        filename = global_dir + str(title_str) + ".png"
        plt.savefig(filename)
        plt.close()

    if data_type == "synthetic":  # If the dataset is synthetic
        plt.figure(figsize=(20, 12))
        plt.plot(np.arange(len(marg_sum_list)) + 1, marg_sum_list)
        plt.xlabel("Number of positions so far")
        plt.ylabel("Sum of absolute difference matrix")
        plt.title("Marginal absolute difference vs. Num of positions so far")

        title_str = "dbc_marginals"
        filename = global_dir + str(title_str) + ".png"
        plt.savefig(filename)
        plt.close()

        plt.figure(figsize=(20, 12))
        plt.plot(np.arange(len(fro_norm_list)) + 1, fro_norm_list)
        plt.xlabel("Number of positions so far")
        plt.ylabel("Frobenius norm of absolute difference matrix")
        plt.title("Frobenius norm of absolute difference vs. Num of positions so far")

        title_str = "dbc_frobenius_norm"
        filename = global_dir + str(title_str) + ".png"
        plt.savefig(filename)
        plt.close()

    end_time = time.time()
    print("\nTotal time of combining multiple sites: ", end_time - start_time)


def main():
    dt = datetime.datetime.now()
    default_dir = "../results/%s_%s_%s/" % (dt.year, dt.month, dt.day)

    # code to process command line arguments
    parser = argparse.ArgumentParser(description='Distance between cells computation.')
    parser.add_argument('global_dir', help="Specify the data directory.", type=str)
    parser.add_argument('--data_type', help="Specify the data type. Default: real", type=str, default="real")
    parser.add_argument('--pos_range_min', help="Specify the position range (min value). Default: 0", type=int,
                        default=0)
    parser.add_argument('--pos_range_max', help="Specify the position range (max value). Default: ", type=int,
                        default=0)
    parser.add_argument('--scuphr_strategy',
                        help="Specify the strategy for Scuphr (paired, singleton, hybrid). Default: paired",
                        type=str, default="paired")
    parser.add_argument('--output_dir', help="Specify the output directory.", type=str, default=default_dir)
    args = parser.parse_args()

    proc_dict_dir = args.global_dir + "processed_data_dict/"
    dataset = load_json(proc_dict_dir + "data.txt")
    num_total_positions = len(dataset)
    num_cells = len(dataset['0']["cell_list"])

    fig_dir = args.output_dir + "figures_comb/"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        print("Combined figures directory is created")

    print("\n*****\nExtracting positions based on Scuphr strategy")
    if args.pos_range_max == 0:
        args.pos_range_max = num_total_positions
    print("\nPosition range: [", args.pos_range_min, ", ", args.pos_range_max, ")")
    print("Number of positions in range: ", args.pos_range_max - args.pos_range_min)
    # This part selects relevant sites to the Scuphr strategy.
    # hybrid accepts all sites & site-pairs,
    # paired accepts only the site-pairs, singleton accepts only the single-sites.
    print("Strategy: ", args.scuphr_strategy)
    positions_orig = np.arange(args.pos_range_min, args.pos_range_max)
    positions = list(positions_orig)
    if args.scuphr_strategy == "paired":
        for pos in positions_orig:
            cur_bulk = np.array(dataset[str(pos)]["bulk"])
            if cur_bulk.shape != (2, 2):
                positions.remove(pos)
    elif args.scuphr_strategy == "singleton":
        for pos in positions_orig:
            cur_bulk = np.array(dataset[str(pos)]["bulk"])
            if cur_bulk.shape != (2,):
                positions.remove(pos)
    positions = np.array(positions)
    print("Number of valid positions in range: ", len(positions))

    if len(positions) == 0:
        print("\nWARNING! There are no positions to combine.")
        sys.exit()

    print("\n*****\nLoading dictionaries")
    real_dist_matrix_dict, dist_matrix_dict = load_dicts(args.output_dir, positions, args.data_type)

    print("\n*****\nCombining multiple sites")
    start_time = time.time()
    combine_multiple_sites(args.output_dir, real_dist_matrix_dict, dist_matrix_dict, num_cells, positions,
                           args.data_type)
    end_time = time.time()
    print("\nTotal time of combining sites: ", end_time - start_time)


if __name__ == "__main__":
    main()
