import sys
import time
import numpy as np

from fragment_enumerator import enumerate_fragments_possible_single_site


def get_lambdas(lambda_key):
    indices_1 = lambda_key.find("_")
    lambda_1 = lambda_key[0:indices_1]

    lambda_key = lambda_key[indices_1 + 1:]

    indices_2 = lambda_key.find("_")
    lambda_2 = lambda_key[0:indices_2]
    lambda_3 = lambda_key[indices_2 + 1:]
    return [int(lambda_1), int(lambda_2), int(lambda_3)]


def get_fragments(fragment_key):
    indices_1 = fragment_key.find("_")
    fragment_1 = fragment_key[0:indices_1]

    fragment_key = fragment_key[indices_1 + 1:]
    indices_2 = fragment_key.find("_")
    fragment_2 = fragment_key[0:indices_2]
    fragment_3 = fragment_key[indices_2 + 1:]

    return [fragment_1, fragment_2, fragment_3]


def convert_fragment_lambda_key_format(fragment_lambda_key):
    # Just to make sure no ',' in the  key.
    fragment_lambda_key = fragment_lambda_key.replace(",", "")

    fragment_key = fragment_lambda_key[0:5]
    lambdas_key = fragment_lambda_key[6:]

    fragments = get_fragments(fragment_key)
    lambdas = get_lambdas(lambdas_key)

    # idx = np.argsort(lambdas)[::-1]
    # lambdas = np.array(lambdas)[idx]
    # fragments = np.array(fragments)[idx]

    if fragments[0] == fragments[1] and fragments[0] == fragments[2]:
        lambdas[0] = lambdas[0] + lambdas[1] + lambdas[2]
        lambdas[1] = 0
        lambdas[2] = 0
    elif fragments[0] == fragments[1]:
        lambdas[0] += lambdas[1]
        lambdas[1] = 0
    elif fragments[0] == fragments[2]:
        lambdas[0] += lambdas[2]
        lambdas[2] = 0
    elif fragments[1] == fragments[2]:
        lambdas[1] += lambdas[2]
        lambdas[2] = 0

    # sort again
    idx = np.argsort(lambdas)[::-1]
    lambdas = np.array(lambdas)[idx]
    fragments = np.array(fragments)[idx]

    if 0 in lambdas:
        idx_zero = np.where(lambdas == 0)[0][0]
        lambdas = lambdas[0:idx_zero]
        fragments = fragments[0:idx_zero]

        # check same num of reads, sort if necessary
        num_unique_lambdas = len(list(set(lambdas)))
        if num_unique_lambdas != len(lambdas):  # the only case is 2_2 etc.
            fragments = np.sort(fragments)

        new_fragment_key = "_".join(str(aa) for aa in fragments)
        new_lambda_key = "_".join(str(aa) for aa in lambdas)
        new_key = new_fragment_key + "_" + new_lambda_key

    else:
        # check same num of reads etc
        num_unique_lambdas = len(list(set(lambdas)))

        if num_unique_lambdas == len(lambdas):
            new_fragment_key = "_".join(str(aa) for aa in fragments)
            new_lambda_key = "_".join(str(aa) for aa in lambdas)
            new_key = new_fragment_key + "_" + new_lambda_key

        else:
            new_fragments = []

            if lambdas[0] == lambdas[1] and lambdas[1] == lambdas[2]:
                new_fragments = np.sort(fragments)

            elif lambdas[0] == lambdas[1]:
                new_sorted = np.sort(fragments[0:2])
                new_fragments.append(new_sorted[0])
                new_fragments.append(new_sorted[1])
                new_fragments.append(fragments[2])

            else:  # lambdas[1] == lambdas[2]
                new_sorted = np.sort(fragments[1:])
                new_fragments.append(fragments[0])
                new_fragments.append(new_sorted[0])
                new_fragments.append(new_sorted[1])

            new_fragment_key = "_".join(str(aa) for aa in new_fragments)
            new_lambda_key = "_".join(str(aa) for aa in lambdas)
            new_key = new_fragment_key + "_" + new_lambda_key

    return new_key


def filter_read_dict(read_dicts):
    prob_diff_threshold = 1e-10

    read_dicts_filtered = {}
    counter = 0

    for cell_key in read_dicts:
        read_dicts_filtered[cell_key] = {}

        for fragment_lambda_key in read_dicts[cell_key]:

            new_key = convert_fragment_lambda_key_format(fragment_lambda_key)

            if new_key not in read_dicts_filtered[cell_key]:
                read_dicts_filtered[cell_key][new_key] = read_dicts[cell_key][fragment_lambda_key]
            else:
                res_1 = read_dicts[cell_key][fragment_lambda_key]
                res_2 = read_dicts_filtered[cell_key][new_key]
                # print("\tAlready exists: ", new_key)
                if res_1 != res_2:
                    if np.abs(np.exp(res_1) - np.exp(res_2)) > prob_diff_threshold:
                        print("\nERROR: Probabilities do not match.\t", res_1, res_2)
                        print("Diff: ", np.exp(res_1) - np.exp(res_2))
                        print("Keys: ", fragment_lambda_key, new_key)
                        print("Cell: ", cell_key)
                        sys.exit()
                    else:
                        counter += 1

    #print("\tNumber of mismatched probabilities: ", counter)
    return read_dicts_filtered


def genotype_prob(read, p_error, fragment):
    """
    This function returns the genotype probability based on given read, error probability and fragment genotype.
    Input: read: array (2,1), p_error: array (2,1), fragment: array (2,1),
    Output: double
    """
    if read == fragment:
        return 1 - p_error
    else:
        return p_error / 3


def precompute_reads(cell_list, z_list, bulk, print_results=False):
    """
    This function computes the log read probabilities of cells based on Bulk, Z_list and reads of cells.
    Input: cell_list: array of Cell objects (numCells,1), Z_list: array (12,2), Bulk: array (2,2), printResults: boolean
    Output: read_dicts: dict (numCells,...)
    """
    #print("\n*****\nComputing read probabilities...")
    start_time = time.time()

    num_cells = len(cell_list)
    read_dicts = {}

    for c in range(num_cells):
        read_dicts[str(c)] = {}

    for z_ind in range(len(z_list)):
        if print_results:
            print("\tComputing for z_ind: ", z_ind)
        z = z_list[z_ind]
        fragment_list = enumerate_fragments_possible_single_site(np.array(bulk).astype(int), np.array(z).astype(int))


        for c in range(num_cells):
            if print_results:
                print("\t\tComputing for cell: ", c)
            #print("ccccc", cell_list[c])
            reads = cell_list[c]['reads']
            p_error = cell_list[c]['p_error']
            lc = cell_list[c]['lc']

            #reads = cell_list[c].reads
            #p_error = cell_list[c].p_error
            #lc = cell_list[c].lc

            if lc == 0:
                read_dicts[str(c)] = {}
            else:
                for fragment in fragment_list:
                    temp_lambdas = [lc, 0, 0]
                    key_1 = "_".join(str(aa) for aa in fragment)
                    key_2 = "_".join(str(aa) for aa in temp_lambdas)
                    key = key_1 + "_" + key_2

                    if key not in read_dicts[str(c)]:
                        prob_dict, _ = read_probability(reads, p_error, fragment)
                        for mid_key in prob_dict.keys():
                            read_dicts[str(c)][mid_key] = prob_dict[mid_key]

    end_time = time.time()
    read_dicts_filtered = filter_read_dict(read_dicts)

    if print_results:
        print("Total time: ", end_time - start_time)
        print("Computing read probabilities is finished...\n*****")
        print("Filtering read probabilities is finished...\n*****")

    return read_dicts_filtered  # read_dicts


def read_probability(reads, p_error, fragments):
    """
    This function calculates the log probability of reads based on fragments for all possible lambda partitions.
    It fills the lookup table.
    Input: reads: array (numReads,2), p_error: array (numReads,2), fragments: array (3,2)
    Output: prob_dict_filtered: dict (...)
    """
    #print("Computing read probability...")
    num_reads = len(reads)
    key_1 = "_".join(str(aa) for aa in fragments)

    prob_dict = {}
    prob_dict_filtered = {}

    max_i = num_reads + 1
    max_j = num_reads + 1
    max_k = num_reads + 1

    fragments = np.array(fragments)
    #print("\tFragments: ", fragments)
    #print(type(fragments[0]), type(fragments[1]), type(fragments[2]))

    if fragments[0] == fragments[1] and fragments[0] == fragments[2]:
        #print("\tall same")
        max_j = 1
        max_k = 1
    elif fragments[0] == fragments[1]:
        #print("\tfirst and second same")
        max_j = 1
    elif fragments[0] == fragments[2]:
        #print("\tfirst and third same")
        max_k = 1
    elif fragments[1] == fragments[2]:
        #print("\tsecond and third same")
        max_k = 1

    for i in range(max_i):
        for j in range(max_j):
            for k in range(max_k):
                lambdas = [i, j, k]
                #print("\n***\nlambdas: ", lambdas)
                key_2 = "_".join(str(aa) for aa in lambdas)
                key = key_1 + "_" + key_2

                if i + j + k > num_reads:
                    break
                # elif i==0 and j==0 and k==num_reads: # We commented this case to allow amplification error on root.
                #    continue;
                elif i == 0 and j == 0 and k == 0:
                    prob_dict[key] = np.log(1)
                else:
                    cur_read = reads[i + j + k - 1]
                    cur_p_error = p_error[i + j + k - 1]

                    num_pos = 0
                    temp_prob_gen = np.NINF

                    temp_prob = np.NINF  # np.log(0)
                    if i > 0:
                        num_pos += 1
                        temp_lambdas = [i - 1, j, k]
                        # print("\ti part: temp lambdas: ", temp_lambdas)
                        key_2 = "_".join(str(aa) for aa in temp_lambdas)
                        temp_key = key_1 + "_" + key_2
                        gen_prob = genotype_prob(cur_read, cur_p_error, fragments[0])
                        log_partial = np.log(gen_prob) + prob_dict[temp_key]
                        temp_prob = log_partial
                        # print("\t\tprev_key: ", temp_key) print("\t\t", np.log(gen_prob), " + ", prob_dict[
                        # temp_key], " = ", log_partial) print("\t\t", gen_prob, " * ", np.exp(prob_dict[temp_key]),
                        # " = ", gen_prob*np.exp(prob_dict[temp_key])) print("\t\tlog_partial: ", temp_prob) print(
                        # "\t\tpartial: ", np.exp(temp_prob))

                    temp_prob_gen = np.logaddexp(temp_prob, temp_prob_gen)

                    temp_prob = np.NINF  # np.log(0)
                    if j > 0:
                        num_pos += 1
                        temp_lambdas = [i, j - 1, k]
                        # print("\tj part: temp lambdas: ", temp_lambdas)
                        key_2 = "_".join(str(aa) for aa in temp_lambdas)
                        temp_key = key_1 + "_" + key_2
                        gen_prob = genotype_prob(cur_read, cur_p_error, fragments[1])
                        log_partial = np.log(gen_prob) + prob_dict[temp_key]
                        temp_prob = log_partial
                        # print("\t\tprev_key: ", temp_key) print("\t\t", np.log(gen_prob), " + ", prob_dict[
                        # temp_key], " = ", log_partial) print("\t\t", gen_prob, " * ", np.exp(prob_dict[temp_key]),
                        # " = ", gen_prob*np.exp(prob_dict[temp_key])) print("\t\tlog_partial: ", temp_prob) print(
                        # "\t\tpartial: ", np.exp(temp_prob))

                    temp_prob_gen = np.logaddexp(temp_prob, temp_prob_gen)

                    temp_prob = np.NINF  # np.log(0)
                    if k > 0:
                        num_pos += 1
                        temp_lambdas = [i, j, k - 1]
                        # print("\tk part: temp lambdas: ", temp_lambdas)
                        key_2 = "_".join(str(aa) for aa in temp_lambdas)
                        temp_key = key_1 + "_" + key_2
                        gen_prob = genotype_prob(cur_read, cur_p_error, fragments[2])
                        log_partial = np.log(gen_prob) + prob_dict[temp_key]
                        temp_prob = log_partial
                        # print("\t\tprev_key: ", temp_key) print("\t\t", np.log(gen_prob), " + ", prob_dict[
                        # temp_key], " = ", log_partial) print("\t\t", gen_prob, " * ", np.exp(prob_dict[temp_key]),
                        # " = ", gen_prob*np.exp(prob_dict[temp_key])) print("\t\tlog_partial: ", temp_prob) print(
                        # "\t\tpartial: ", np.exp(temp_prob))

                    temp_prob_gen = np.logaddexp(temp_prob, temp_prob_gen)
                    prob_dict[key] = temp_prob_gen

                    # print("\ni,j,k: ",i,j,k)
                    # print("key: ", key)
                    # print("read_idx: ", i+j+k-1)
                    # print("temp_prob_log: ", temp_prob_gen)
                    # print("temp_prob: ", np.exp(temp_prob_gen))
                    # print(prob_dict)

                    if np.exp(temp_prob_gen) > 1 or np.exp(temp_prob_gen) < 0:
                        print("\nERROR: Probability is not in the range [0,1].")
                        sys.exit()

                    if i + j + k == num_reads:
                        # div = comb(num_reads,i) * comb(num_reads-i,j) * comb(num_reads-i-j,k)
                        # print("lambdas: ", lambdas, "\tdiv: ", div)
                        # prob_dict_filtered[key] = temp_prob_gen / div
                        prob_dict_filtered[key] = prob_dict[key]

    return prob_dict_filtered, prob_dict
