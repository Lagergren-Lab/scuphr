import os
import json
import time
import pickle
import argparse
import numpy as np

import cell_class
from fragment_enumerator import enumerate_distance1_genotype


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


def convert_base_to_idx(read):
    read_idx = [0, 0]
    if read[0] == "C":
        read_idx[0] = 1
    elif read[0] == "G":
        read_idx[0] = 2
    elif read[0] == "T":
        read_idx[0] = 3
    if read[1] == "C":
        read_idx[1] = 1
    elif read[1] == "G":
        read_idx[1] = 2
    elif read[1] == "T":
        read_idx[1] = 3
    return read_idx


def generate_real_dataset(cell_name_list, bulk_pair_dict, all_cell_dict_pair, truth_dir, min_cell_count, min_read_count,
                          max_read_count, max_site_count, data_type, seed_val):
    np.random.seed(seed_val)

    phred_chars = ["!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4",
                   "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H",
                   "I", "J", "K"]

    if data_type == 'synthetic':

        try:
            filename = truth_dir + "bulk_genome.txt"
            bulk_genome = np.array(load_json(filename))

            filename = truth_dir + "mut_genome.txt"
            mut_genotype = np.array(load_json(filename))

            filename = truth_dir + "mut_locations.txt"
            mut_locations = np.array(load_json(filename))
        except:
            filename = truth_dir + "bulk_genome.txt"
            bulk_genome = np.loadtxt(filename)

            filename = truth_dir + "mut_genome.txt"
            mut_genotype = np.loadtxt(filename)

            filename = truth_dir + "mut_locations.txt"
            mut_locations = np.loadtxt(filename)

    position_to_idx = {}
    idx_to_position = {}
    idx = 0

    num_skipped_pos = 0

    data = {}

    for pos_pair in bulk_pair_dict.keys():
        if max_site_count != 0 and idx >= max_site_count:
            break

        cell_count = 0
        for cell_name in cell_name_list:
            if pos_pair in all_cell_dict_pair[cell_name].keys():
                cell_count += 1

        if cell_count >= min_cell_count:
            tot_mut_count = 0
            skipped_cell_count = 0

            # position_to_idx[pos_pair] = idx
            # idx_to_position[idx] = pos_pair
            # idx = idx + 1

            # cell_dict = {}
            cell_list_temp = []
            for cell_name in cell_name_list:
                if pos_pair in all_cell_dict_pair[cell_name].keys():
                    #cell_depth = all_cell_dict_pair[cell_name][pos_pair]["depth"]

                    cell_reads = []
                    for read in all_cell_dict_pair[cell_name][pos_pair]["reads"]:
                        read_idx = convert_base_to_idx(read)
                        cell_reads.append(read_idx)
                    cell_reads = np.array(cell_reads)

                    cell_depth = len(cell_reads)

                    # Convert phred score to error probability
                    q_error_chars = np.array(all_cell_dict_pair[cell_name][pos_pair]["quals"])
                    q_error = np.zeros_like(q_error_chars)
                    for i_ in range(q_error.shape[0]):
                        for j_ in range(q_error.shape[1]):
                            if q_error_chars[i_, j_] in phred_chars:
                                q_error[i_, j_] = phred_chars.index(q_error_chars[i_, j_])
                            else:
                                # chr conversion is added due to the differences in rust data simulation
                                #q_error[i_, j_] = phred_chars.index(chr(q_error_chars[i_, j_]))
                                q_error[i_, j_] = q_error_chars[i_, j_]

                    q_error = q_error.astype(int)
                    #print("q_error_chars", q_error_chars)
                    #print("q_err", q_error)

                    p_error = np.power((10 * np.ones(q_error.shape)), (-0.1 * q_error))
                    cell_quals = p_error

                    if cell_depth < min_read_count:
                        cell_reads = np.array([])
                        cell_quals = np.array([])
                        skipped_cell_count += 1

                    elif max_read_count != 0 and cell_depth > max_read_count:
                        # Most basic random selection
                        idx_list = np.arange(cell_depth)
                        np.random.shuffle(idx_list)
                        sample_idx = idx_list[0:max_read_count]

                        # Highest total ranked read selection
                        # cell_quals_avg = np.sum(cell_quals, axis=1)
                        # sorted_idx = np.argsort(cell_quals_avg)
                        # sample_idx = sorted_idx[0:max_read_count]

                        # Highest mult ranked read selection
                        # cell_quals_mult = np.multiply(cell_quals[:,0],cell_quals[:,1])
                        # sorted_idx = np.argsort(cell_quals_mult)
                        # sample_idx = sorted_idx[0:max_read_count]

                        cell_reads = cell_reads[sample_idx]
                        cell_quals = cell_quals[sample_idx]
                else:
                    cell_reads = np.array([])
                    cell_quals = np.array([])

                # Add real info
                if data_type == 'synthetic':

                    try:
                        filename = truth_dir + "cell_" + cell_name + "_genome.txt"
                        cell_genome = np.array(load_json(filename))

                        filename = truth_dir + "cell_" + cell_name + "_masked_genome.txt"
                        cell_masked_genome = np.array(load_json(filename))
                    except:
                        filename = truth_dir + "cell_" + cell_name + "_genome.txt"
                        cell_genome = np.loadtxt(filename)

                        filename = truth_dir + "cell_" + cell_name + "_masked_genome.txt"
                        cell_masked_genome = np.loadtxt(filename)

                    het_pos = int(pos_pair[0:pos_pair.index("_")])
                    hom_pos = int(pos_pair[pos_pair.index("_") + 1:])

                    if cell_genome[hom_pos - 1, 0] == bulk_genome[hom_pos - 1, 0] and cell_genome[hom_pos - 1, 1] == \
                            bulk_genome[hom_pos - 1, 1]:
                        cell_mut_status = 0
                    else:
                        cell_mut_status = 1
                        tot_mut_count += 1

                    if cell_masked_genome[hom_pos - 1, 0] != -1 and cell_masked_genome[hom_pos - 1, 1] != -1:
                        cell_ado_status = 0
                    elif cell_masked_genome[hom_pos - 1, 0] == -1 and cell_masked_genome[hom_pos - 1, 1] == -1:
                        cell_ado_status = 3
                    elif cell_masked_genome[hom_pos - 1, 0] == -1 and cell_masked_genome[hom_pos - 1, 1] != -1:
                        cell_ado_status = 1
                    else:
                        cell_ado_status = 2

                    b1 = [bulk_genome[het_pos - 1, 0], bulk_genome[hom_pos - 1, 0]]
                    b2 = [bulk_genome[het_pos - 1, 1], bulk_genome[hom_pos - 1, 1]]
                    bulk = np.array([b1, b2])

                else:
                    # Bulk_str = bulk_pair_dict[pos_pair]
                    # b1 = convertBaseToIdx(Bulk_str[0])
                    # b2 = convertBaseToIdx(Bulk_str[1])
                    # bulk = np.array([b1,b2])
                    bulk = np.array(bulk_pair_dict[pos_pair])

                    cell_ado_status = -1
                    cell_mut_status = -1

                cell_sample = cell_class.Cell(cell_reads, cell_quals, cell_ado_status, cell_mut_status)
                cell_list_temp.append(cell_sample)

            pos_dict = {}

            z_list_temp = enumerate_distance1_genotype(bulk)
            z_list = []
            for z in z_list_temp:
                z_list.append(z.astype(int).tolist())

            cell_list = []
            for c_idx in range(len(cell_list_temp)):
                cell_data = cell_list_temp[c_idx]
                c_dict = {"y": cell_data.Y, "x": cell_data.X, "lc": cell_data.lc,
                          "reads": cell_data.reads.tolist(), "p_error": cell_data.p_error.tolist()}
                cell_list.append(c_dict)

            pos_dict["bulk"] = bulk.astype(int).tolist()
            pos_dict["cell_list"] = cell_list

            if data_type == 'synthetic':
                pos_dict["mut_count"] = tot_mut_count

                if hom_pos - 1 not in mut_locations:
                    z = bulk
                else:
                    z = np.copy(bulk)
                    z[:, 1] = mut_genotype[hom_pos - 1]
            else:
                pos_dict["mut_count"] = -1
                z = -1

            #print(pos_pair, z)
            pos_dict['common_z'] = z.astype(int).tolist()  # -1
            pos_dict['z_list'] = z_list
            pos_dict['pos_pair'] = [het_pos, hom_pos]  # pos_pair

            if cell_count - skipped_cell_count >= 2:
                position_to_idx[pos_pair] = idx
                idx_to_position[idx] = pos_pair
                data[str(idx)] = pos_dict
                idx += 1
            else:
                num_skipped_pos += 1

    print("\tTotal number of positions: ", len(list(data.keys())))
    print("\tTotal number of skipped positions: ", num_skipped_pos)

    return data, position_to_idx, idx_to_position


def analyse_results(global_dir, truth_dir, read_length):
    # Load necessary files
    filename = global_dir + "data_position_to_idx_orig.txt"
    position_to_idx_dict = load_dictionary(filename)

    filename = truth_dir + "mut_locations.txt"
    mut_locations = load_dictionary(filename)

    filename = truth_dir + "gsnv_locations.txt"
    gsnv_locations = load_dictionary(filename)

    filename = truth_dir + "leaf_nodes.txt"
    leaf_nodes = load_dictionary(filename)

    filename = truth_dir + "mut_origin_nodes.txt"
    mut_origin_nodes = load_dictionary(filename)

    predict_correct_mut_count = 0
    predict_wrong_mut_count = 0
    num_mut_sites_leaf_det = 0
    num_mut_sites_inner_det = 0

    false_guesses = []
    guesses = []

    for pos_pair in position_to_idx_dict:
        hom_pos = int(pos_pair[pos_pair.index("_") + 1:])
        hom_pos_idx = hom_pos - 1

        guesses.append(hom_pos_idx)

        if hom_pos_idx in mut_locations:
            predict_correct_mut_count += 1

            mut_idx = mut_locations.index(hom_pos_idx)
            mut_origin_node = mut_origin_nodes[mut_idx]

            if mut_origin_node in leaf_nodes:
                num_mut_sites_leaf_det += 1
            else:
                num_mut_sites_inner_det += 1
        else:
            predict_wrong_mut_count += 1
            false_guesses.append(hom_pos_idx)

    undetected = []

    num_global_mut_sites = len(mut_locations)

    num_mut_sites = 0
    num_mut_sites_inner = 0
    num_mut_sites_leaf = 0
    for hom_pos in mut_locations:
        flag = False
        for het_pos in gsnv_locations:
            if abs(hom_pos - het_pos) < read_length:
                flag = True
                break
        if flag:
            num_mut_sites += 1

            mut_idx = mut_locations.index(hom_pos)
            mut_origin_node = mut_origin_nodes[mut_idx]

            if mut_origin_node in leaf_nodes:
                num_mut_sites_leaf += 1
            else:
                num_mut_sites_inner += 1

                if hom_pos not in guesses:
                    undetected.append(hom_pos)

    print("\tSite detection results: ")
    print("\tTotal number of (real) mutated sites: ", num_global_mut_sites)

    print("\n\tTotal number of (real) detectable mutated sites (close to gSNV locations): ", num_mut_sites)
    print("\t\tNumber of (real) detectable mutated sites (on internal edges): \t", num_mut_sites_inner)
    print("\t\tNumber of (real) detectable mutated sites (on leaf edges): \t", num_mut_sites_leaf)

    true_positive = predict_correct_mut_count
    false_positive = predict_wrong_mut_count
    false_negative = num_mut_sites - true_positive

    print("\n\tTotal number of sites correctly detected \t(True Positive): \t", true_positive)
    print("\t\tNumber of mutated sites (detected) (on internal edges): \t", num_mut_sites_inner_det, "\t out of ",
          num_mut_sites_inner, "\tperc: ", round(100 * num_mut_sites_inner_det / num_mut_sites_inner, 3))
    print("\t\tNumber of mutated sites (detected) (on leaf edges): \t\t", num_mut_sites_leaf_det, "\t out of ",
          num_mut_sites_leaf, "\tperc: ", round(100 * num_mut_sites_leaf_det / num_mut_sites_leaf, 3))
    print("\tTotal number of sites incorrectly detected \t(False Positive): \t", false_positive)
    print("\n\tTotal number of mutated sites undetected \t(False Negative): \t", false_negative)

    sensitivity = true_positive / num_mut_sites
    precision = true_positive / (true_positive + false_positive)
    miss_rate = false_negative / num_mut_sites
    false_discovery_rate = false_positive / (false_positive + true_positive)
    f1_score = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)

    print("\n\tSensitivity (True positive rate): \t", sensitivity)
    print("\tPrecision: \t\t\t\t", precision)
    print("\tMiss rate (False negative rate): \t", miss_rate)
    print("\tFalse discovery rate: \t\t\t", false_discovery_rate)
    print("\tF1 score: \t\t\t\t", f1_score)

    ###
    print("\n\tUndetected inner edge mutations: ", len(undetected))
    print(undetected)


def shuffle_partition_data(dir_name, seed_val, train_frac=2):
    print("\nShuffle and partition dataset")

    np.random.seed(seed_val)

    # 0. Load original data
    filename = dir_name + "data_orig.txt"
    data = load_json(filename)
    filename = dir_name + "data_idx_to_position_orig.txt"
    data_idx_to_pos = load_json(filename)

    # 1. Shuffle indices and decide training separation
    num_pos_pair = len(data.keys())
    idx_list = np.arange(num_pos_pair)
    np.random.shuffle(idx_list)
    num_train = int(np.ceil(num_pos_pair / train_frac))

    # 2. Partition
    data_new = {}
    data_idx_to_pos_new = {}
    data_pos_to_idx_new = {}

    data_train = {}
    data_idx_to_pos_train = {}
    data_pos_to_idx_train = {}

    data_test = {}
    data_idx_to_pos_test = {}
    data_pos_to_idx_test = {}

    for i in range(num_pos_pair):
        prev_i = idx_list[i]
        data_prev_i = data[str(prev_i)]
        pospair_prev_i = data_idx_to_pos[str(prev_i)]

        data_new[str(i)] = data_prev_i
        data_idx_to_pos_new[i] = pospair_prev_i
        data_pos_to_idx_new[pospair_prev_i] = i

        if i < num_train:
            data_train[str(i)] = data_prev_i
            data_idx_to_pos_train[i] = pospair_prev_i
            data_pos_to_idx_train[pospair_prev_i] = i
        else:
            new_i = i - num_train
            data_test[str(new_i)] = data_prev_i
            data_idx_to_pos_test[new_i] = pospair_prev_i
            data_pos_to_idx_test[pospair_prev_i] = new_i

    # 3. Save new data partitioning
    filename = dir_name + "data_shuffled.txt"
    save_json(filename, data_new)
    print("\nShuffled data (", len(data_new.keys()), ") is saved to: ", filename)
    filename = dir_name + "data_idx_to_position_shuffled.txt"
    save_json(filename, data_idx_to_pos_new)
    filename = dir_name + "data_position_to_idx_shuffled.txt"
    save_json(filename, data_pos_to_idx_new)

    filename = dir_name + "data_test.txt"
    save_json(filename, data_test)
    print("Saving test data (", len(data_test.keys()), ") is saved to: ", filename)
    filename = dir_name + "data_idx_to_position_test.txt"
    save_json(filename, data_idx_to_pos_test)
    filename = dir_name + "data_position_to_idx_test.txt"
    save_json(filename, data_pos_to_idx_test)

    filename = dir_name + "data_train.txt"
    save_json(filename, data_train)
    print("Saving train data (", len(data_train.keys()), ") is saved to: ", filename)
    filename = dir_name + "data_idx_to_position_train.txt"
    save_json(filename, data_idx_to_pos_train)
    filename = dir_name + "data_position_to_idx_train.txt"
    save_json(filename, data_pos_to_idx_train)

    filename = dir_name + "data.txt"
    save_json(filename, data_new)
    print("Saving shuffled data (", len(data_new.keys()), ") is saved to: ", filename)
    filename = dir_name + "data_idx_to_position.txt"
    save_json(filename, data_idx_to_pos_new)
    filename = dir_name + "data_position_to_idx.txt"
    save_json(filename, data_pos_to_idx_new)

    # filename = dir_name + "data.pickle"
    # saveDictionary(filename, data_train)
    # print("Saving train data (", len(data_train.keys()), ") is saved to: ", filename)
    # filename = dir_name + "data_idx_to_position.pickle"
    # saveDictionary(filename, data_idx_to_pos_train)
    # filename = dir_name + "data_position_to_idx.pickle"
    # saveDictionary(filename, data_pos_to_idx_train)


def main():
    # code to process command line arguments
    parser = argparse.ArgumentParser(description='Site detection.')
    parser.add_argument('global_dir', help="Specify the directory.", type=str)
    parser.add_argument('num_cells', help="Specify the number of cells.", type=int)
    parser.add_argument('--chr_id', help="Specify the chromosome number. Default: 1", type=int, default=1)
    parser.add_argument('--data_type', help="Specify the data type. Default: real", type=str, default="real")
    parser.add_argument('--min_read_count', help="Specify the minimum read count. Default: 0", type=int, default=0)
    parser.add_argument('--max_read_count', help="Specify the maximum read count (0 for all). Default: 0", type=int,
                        default=0)
    parser.add_argument('--max_site_count', help="Specify the maximum site count (0 for all). Default: 0", type=int,
                        default=0)
    parser.add_argument('--min_cell_count', help="Specify the minimum cell count. Default: 2", type=int, default=2)
    parser.add_argument('--output_dict_dir', help="Specify the output dictionary directory. ", type=str, default="")
    parser.add_argument('--read_length', help="Specify the read length. Default: 100", type=int, default=100)
    parser.add_argument('--seed_val', help="Specify the seed. Default: 123", type=int, default=123)
    args = parser.parse_args()

    start_time_global = time.time()

    print("Global directory: ", args.global_dir)
    truth_dir = args.global_dir + "truth/"
    proc_dir = args.global_dir + "processed_data/"
    proc_dict_dir = args.global_dir + "processed_data_dict/" + args.output_dict_dir
    print("Output directory: ", proc_dict_dir)
    if not os.path.exists(proc_dict_dir):
        os.makedirs(proc_dict_dir)
        print("Directory is created.\t", proc_dict_dir)

    # PART 1: Load bulk pairs dictionary
    start_time = time.time()
    print("\nPart 1: Load bulk pairs dictionary.")
    bulk_out_filename = proc_dir + "bulk_pairs.pickle"
    bulk_pair_dict = load_dictionary(bulk_out_filename)

    end_time = time.time()
    print("\nTotal time: ", end_time - start_time)
    print("Part 1 ends...")

    # PART 2: Load cells' pairs dictionaries
    start_time = time.time()
    print("\nPart 2: Load cells' pairs dictionaries.\n")

    all_cell_dict_pair = {}
    for cell_idx in range(args.num_cells):
        # cell_filename = global_dir + "cell_idx_" + str(cell_idx) + ".bam"

        cell_out_filename = proc_dir + "cell_" + str(cell_idx) + "_pairs.pickle"
        all_cell_dict_pair[str(cell_idx)] = load_dictionary(cell_out_filename)

        #cell_out_filename = proc_dir + "cell_" + str(cell_idx) + "_pairs.json"
        #all_cell_dict_pair[str(cell_idx)] = load_json(cell_out_filename)

    end_time = time.time()
    print("\nTotal time: ", end_time - start_time)
    print("Part 2 ends...")

    # PART 3: Generate the dataset
    start_time = time.time()
    print("\nPart 3: Generate dataset.\n")

    cell_name_list = []
    for cell_idx in range(args.num_cells):
        cell_name_list.append(str(cell_idx))

    data, position_to_idx, idx_to_position = generate_real_dataset(cell_name_list, bulk_pair_dict, all_cell_dict_pair,
                                                                   truth_dir,
                                                                   min_cell_count=args.min_cell_count,
                                                                   min_read_count=args.min_read_count,
                                                                   max_read_count=args.max_read_count,
                                                                   max_site_count=args.max_site_count,
                                                                   data_type=args.data_type, seed_val=args.seed_val)
    out_filename = proc_dict_dir + "data_position_to_idx_orig.txt"
    save_json(out_filename, position_to_idx)
    out_filename = proc_dict_dir + "data_idx_to_position_orig.txt"
    save_json(out_filename, idx_to_position)
    out_filename = proc_dict_dir + "data_orig.txt"
    save_json(out_filename, data)

    end_time = time.time()
    print("\nTotal time: ", end_time - start_time)
    print("Part 3 ends...")

    # PART 4: Shuffle and partition the dataset
    start_time = time.time()
    print("\nPart 4: Shuffle and partition the dataset.\n")
    shuffle_partition_data(proc_dict_dir, seed_val=args.seed_val)
    end_time = time.time()
    print("\nTotal time: ", end_time - start_time)
    print("Part 4 ends...")

    if args.data_type == 'synthetic':
        # PART 5: Analyse results
        start_time = time.time()
        print("\nPart 5: Analyse results.\n")
        #analyse_results(proc_dict_dir, truth_dir, read_length=args.read_length)

        end_time = time.time()
        print("\nTotal time: ", end_time - start_time)
        print("Part 5 ends...")

    ###
    print("\nAll done!")
    end_time_global = time.time()
    print("Total time: ", end_time_global - start_time_global)


if __name__ == "__main__":
    main()
