import time
import numpy as np
import scipy.special as sp


def compute_site_mutation_probabilities_log_dp(mut_prior, cell_list, z_list, alpha_list, log_zcy_dict, a_g, b_g, p_ado,
                                               print_results=False):
    """
    This function computes the log mutation type probabilities based on cells, Z_list, alpha_list and log_ZCY_dict.
    Input: cell_list: array of Cell objects (numCells,1), Z_list: array (12,2), alpha_list: array (12,1),
           log_ZCY_dict (...), printResults: boolean
    Output: normed_log_prob_z: array (12,1), highest_z: array (2,2), highest_z_prob: double, max_key: int,
            general_lookup_table: dict (12,...)
    """
    if print_results:
        print("\n*****\nComputing site mutation type probabilities...")
    start_time = time.time()

    num_cells = len(cell_list)

    # Uniform prior
    # a_g = 1
    # b_g = 1

    general_lookup_table = {}

    log_prob_z = []
    for z_ind in range(len(z_list)):
        # if printResults:
        # print("\tComputing for z_ind: ", z_ind)

        log_prob_cur_z = np.NINF
        lookup = {}
        for m in range(1, num_cells + 1):
            # print("\n\tm: ", m)
            # print("\tlookup before: ", lookup)
            log_prob_m = dp_mutation_probability(m, num_cells, cell_list, lookup, log_zcy_dict, z_ind, p_ado)
            log_prob_m = log_prob_m + np.log(sp.beta(a_g + m, b_g + num_cells - m) / sp.beta(a_g, b_g))
            log_prob_cur_z = np.logaddexp(log_prob_cur_z, log_prob_m)
            # print("\n\tlog_prob_m: ", log_prob_m)
            # print("\tlog_prob_cur_z: ", log_prob_cur_z)
            # print("\n\tlookup after: ", lookup)

        log_prob_cur_z = log_prob_cur_z + np.log(sp.beta(np.sum(alpha_list), 1)) - np.log(sp.beta(alpha_list[z_ind], 1))
        log_prob_z.append(log_prob_cur_z)

        # print("\tfinal current log prob: ", log_prob_cur_z)

        general_lookup_table[str(z_ind)] = lookup

        # print("\ngen_lookup: ", general_lookup_table)

    p_m_1 = np.array(log_prob_z) + np.log(mut_prior)

    m = 0
    z_ind = 0
    log_prob_nomut = dp_mutation_probability(m, num_cells, cell_list, lookup, log_zcy_dict, z_ind, p_ado)
    p_m_0 = log_prob_nomut + np.log(1 - mut_prior)

    # See this link:
    # https://stats.stackexchange.com/questions/66616/converting-normalizing-very-small-likelihood-values-to-probability?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    subs = log_prob_z - max(log_prob_z)

    eps = 1e-50
    threshold = np.log(eps) - np.log(len(log_prob_z))
    bad_idx = np.where(subs < threshold)[0]

    norm_scale = np.exp(subs)
    norm_scale[bad_idx] = 0
    normed_log_prob_z = norm_scale / sum(norm_scale)

    max_key = np.argmax(np.array(normed_log_prob_z))
    highest_z = z_list[max_key]
    highest_z_prob = normed_log_prob_z[max_key]
    ###

    end_time = time.time()
    print("Total time: ", end_time - start_time)

    if print_results:
        print("Computing mutation type probabilities is finished...\n*****")
        # print("\nNormalized probabilities of common mutation type Z: \n", prob_Z)
        print("\nMaximum probability: ", highest_z_prob, ". Dict key: ", max_key)
        print("Corresponding mutation type:")
        print(highest_z)

    return normed_log_prob_z, highest_z, highest_z_prob, max_key, general_lookup_table, log_prob_z, p_m_1, \
           log_prob_nomut, p_m_0


def compute_mutation_probabilities_log_dp(cell_list, z_list, alpha_list, log_zcy_dict, a_g, b_g, p_ado,
                                          print_results=False):
    """
    This function computes the log mutation type probabilities based on cells, Z_list, alpha_list and log_ZCY_dict.
    Input: cell_list: array of Cell objects (numCells,1), Z_list: array (12,2), alpha_list: array (12,1),
           log_ZCY_dict (...), printResults: boolean
    Output: normed_log_prob_z: array (12,1), highest_z: array (2,2), highest_z_prob: double, max_key: int,
            general_lookup_table: dict (12,...)
    """
    if print_results:
        print("\n*****\nComputing mutation type probabilities...")
    start_time = time.time()

    num_cells = len(cell_list)

    # Uniform prior
    # a_g = 1
    # b_g = 1

    general_lookup_table = {}

    log_prob_z = []
    for z_ind in range(len(z_list)):
        # if printResults:
        print("\tComputing for z_ind: ", z_ind)

        log_prob_cur_z = np.NINF
        lookup = {}
        for m in range(0, num_cells + 1):
            # print("\n\tm: ", m)
            # print("\tlookup before: ", lookup)
            log_prob_m = dp_mutation_probability(m, num_cells, cell_list, lookup, log_zcy_dict, z_ind, p_ado)
            log_prob_m = log_prob_m + np.log(sp.beta(a_g + m, b_g + num_cells - m) / sp.beta(a_g, b_g))
            log_prob_cur_z = np.logaddexp(log_prob_cur_z, log_prob_m)
            # print("\n\tlog_prob_m: ", log_prob_m)
            # print("\tlog_prob_cur_z: ", log_prob_cur_z)
            # print("\n\tlookup after: ", lookup)

        log_prob_cur_z = log_prob_cur_z + np.log(sp.beta(np.sum(alpha_list), 1)) - np.log(sp.beta(alpha_list[z_ind], 1))
        log_prob_z.append(log_prob_cur_z)

        # print("\tfinal current log prob: ", log_prob_cur_z)

        general_lookup_table[str(z_ind)] = lookup

        # print("\ngen_lookup: ", general_lookup_table)

    # See this link:
    # https://stats.stackexchange.com/questions/66616/converting-normalizing-very-small-likelihood-values-to-probability?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    subs = log_prob_z - max(log_prob_z)

    eps = 1e-50
    threshold = np.log(eps) - np.log(len(log_prob_z))
    bad_idx = np.where(subs < threshold)[0]

    norm_scale = np.exp(subs)
    norm_scale[bad_idx] = 0
    normed_log_prob_z = norm_scale / sum(norm_scale)

    max_key = np.argmax(np.array(normed_log_prob_z))
    highest_z = z_list[max_key]
    highest_z_prob = normed_log_prob_z[max_key]
    ###

    end_time = time.time()
    print("Total time: ", end_time - start_time)

    if print_results:
        print("Computing mutation type probabilities is finished...\n*****")
        # print("\nNormalized probabilities of common mutation type Z: \n", prob_Z)
        print("\nMaximum probability: ", highest_z_prob, ". Dict key: ", max_key)
        print("Corresponding mutation type:")
        print(highest_z)

    return normed_log_prob_z, highest_z, highest_z_prob, max_key, general_lookup_table, log_prob_z


def dp_mutation_probability(mut, cells_so_far, cell_list, lookup, log_zcy_dict, z_ind, p_ado):
    """
    This function is the dynamic programming part of log mutation type probability calculation.
    Based on the mutation count, cells so far count, cells, lookup table, log_ZCY_dict and mutation index,
    it calculates the probability.
    Input: mut: int, cellssofar: int, cell_list: array of Cell objects (numCells,1), lookup: dict (...),
           log_ZCY_dict (...), z_ind: int
    Output: result: double
    """
    key = "" + str(mut) + "_" + str(cells_so_far)
    if key in lookup.keys():
        return lookup[key]

    if mut > cells_so_far:
        return np.NINF
    if mut < 0:
        return np.NINF

    if cells_so_far == 0:  # Base case
        if mut == 0:
            lookup[key] = np.log(1)
            return np.log(1)
        else:
            return np.NINF

    current_cell_id = cells_so_far - 1
    if cell_list[current_cell_id].lc == 0:
        result = dp_mutation_probability(mut, cells_so_far - 1, cell_list, lookup, log_zcy_dict, z_ind, p_ado)
        # p_ado = 0.5

        # current is mutated
        # res1 = dpMutationProbability(mut-1, cellssofar-1, cell_list, lookup, log_ZCY_dict, z_ind, p_ado)
        # res1 = res1 + np.log(np.power(p_ado,2)) #log_ZCY_dict[zcy_key]

        # current is not mutated
        # res2 = dpMutationProbability(mut, cellssofar-1, cell_list, lookup, log_ZCY_dict, z_ind, p_ado)
        # res2 = res2 + np.log(np.power(p_ado,2)) #log_ZCY_dict[zcy_key]

        # result = np.logaddexp(res1, res2)

    else:
        # current is mutated
        res1 = dp_mutation_probability(mut - 1, cells_so_far - 1, cell_list, lookup, log_zcy_dict, z_ind, p_ado)

        zcy_key = str(z_ind) + "_" + str(current_cell_id) + "_" + str(1)

        cur_prob = np.NINF
        # No presence
        zcyp_key = zcy_key + "_" + str(0)
        cur_prob = np.logaddexp(cur_prob, log_zcy_dict[zcyp_key])
        # Presence
        zcyp_key = zcy_key + "_" + str(1)
        cur_prob = np.logaddexp(cur_prob, log_zcy_dict[zcyp_key])

        res1 = res1 + cur_prob

        # current is not mutated
        res2 = dp_mutation_probability(mut, cells_so_far - 1, cell_list, lookup, log_zcy_dict, z_ind, p_ado)

        zcy_key = str(z_ind) + "_" + str(current_cell_id) + "_" + str(0)

        cur_prob = np.NINF
        # No presence
        zcyp_key = zcy_key + "_" + str(0)
        cur_prob = np.logaddexp(cur_prob, log_zcy_dict[zcyp_key])
        # Presence
        zcyp_key = zcy_key + "_" + str(1)
        cur_prob = np.logaddexp(cur_prob, log_zcy_dict[zcyp_key])

        res2 = res2 + cur_prob

        result = np.logaddexp(res1, res2)

    lookup[key] = result
    return result
