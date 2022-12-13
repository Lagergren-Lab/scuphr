import time
import numpy as np
import scipy.special as sp


def distance_between_cells_all_configs_log_dp(constant_cell_ids, z_list, cell_list, alpha_list, log_zcy_dict,
                                              general_lookup_table, a_g, b_g, p_ado, print_results=False):
    """
    This function computes the log distance between cells probability.
    First, it computes the probability of all other cells.
    Then, for each genotype configuration of given cells, it calculates the log probability of distances.
    Input: constant_cell_ids: array (2,1), Z_list: array (12,2), cell_list: array of Cell objects (numCells,1),
           alpha_list: array (12,1), log_ZCY_dict (...), general_lookup_table: dict (...), printResults: boolean
    Output: normed_prob_dists: array (4,1), prob_dists: array (4,1), highest_config: binary array (2,1),
            highest_config_prob: double, max_key: int
    """
    if print_results:
        print("\n*****\nComputing distance between cells probabilities for cells ", constant_cell_ids, "...")

    start_time = time.time()

    configs = []
    log_prob_dists = []

    num_cells = len(cell_list)
    other_cell_ids = list(range(num_cells))
    other_cell_ids.pop(max(constant_cell_ids))
    other_cell_ids.pop(min(constant_cell_ids))
    # print("Other cell ids: ", other_cell_ids)
    others = distance_between_cells_others(cell_list, log_zcy_dict, z_list, constant_cell_ids, other_cell_ids,
                                           general_lookup_table, p_ado)
    # print("\n***\nOthers: ")
    # for key in sorted(others):
    #    print("\n\tz: ", key, "\t", others[key])

    constant_cells = {}
    for i in range(2):
        for j in range(2):
            constant_cells[constant_cell_ids[0]] = i
            constant_cells[constant_cell_ids[1]] = j

            constant_presences = {}
            for pi in range(2):
                for pj in range(2):
                    constant_presences[constant_cell_ids[0]] = pi
                    constant_presences[constant_cell_ids[1]] = pj

                    configs.append([i, j, pi, pj])

                    log_prob_dists.append(distance_between_cells_log_dp(constant_cells, constant_presences, z_list,
                                                                        cell_list, alpha_list, log_zcy_dict, others,
                                                                        a_g, b_g)[1])

    prob_dists = np.exp(log_prob_dists)

    # See this link:
    # https://stats.stackexchange.com/questions/66616/converting-normalizing-very-small-likelihood-values-to-probability?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    log_probs = log_prob_dists
    subs = log_probs - max(log_probs)

    eps = 1e-50
    threshold = np.log(eps) - np.log(len(log_probs))

    bad_idx = np.where(subs < threshold)[0]

    norm_scale = np.exp(subs)
    norm_scale[bad_idx] = 0

    normed_prob_dists = norm_scale / sum(norm_scale)
    ###

    max_key = np.argmax(np.array(normed_prob_dists))
    highest_config = configs[max_key]
    highest_config_prob = normed_prob_dists[max_key]

    end_time = time.time()

    if print_results:
        print("Total time: ", end_time - start_time)
        print("Computing distance between cells probabilities for cells ", constant_cell_ids, " is finished...\n*****")
        print("\nNormalized probabilities of distance between cells: %d and %d" % (
            constant_cell_ids[0], constant_cell_ids[1]))
        print(normed_prob_dists)
        print("\nMaximum probability: ", highest_config_prob, ". Index: ", max_key, ". Corresponding genotypes: ",
              highest_config, "\n*****")
        for c in constant_cell_ids:
            print("\nOriginal stats of cell: ", c)
            cell_list[c].printCell()

    return normed_prob_dists, prob_dists, highest_config, highest_config_prob, max_key


def distance_between_cells_log_dp(constant_cells, constant_presences, z_list, cell_list, alpha_list, log_zcy_dict,
                                  others, a_g, b_g):
    """
    This function computes the log probability of given cells by dynamic programming.
    Input: constant_cells: dict (2,1), Z_list: array (12,2), cell_list: array of Cell objects (numCells,1),
           alpha_list: array (12,1), log_ZCY_dict (...), others: dict (...), printResults: boolean
    Output: prob_dist: double, log_prob_dist: double
    """
    num_cells = len(cell_list)
    # a_g, b_g = getBetaPriorParameters(num_cells, p_se_a, p_se_b)

    # Uniform prior
    # a_g = 1
    # b_g = 1

    const_gen_sum = 0
    for gen in constant_cells.values():
        const_gen_sum = const_gen_sum + gen

    log_z = np.NINF
    for z_ind in range(len(z_list)):

        log_prob_cur_z = np.NINF
        for m in range(const_gen_sum, num_cells - 1 + const_gen_sum):
            log_prob_m = dp_distance_between_cells(m, len(cell_list), cell_list, log_zcy_dict, z_ind, others,
                                                   constant_cells, constant_presences)
            log_prob_m = log_prob_m + np.log(sp.beta(a_g + m, b_g + num_cells - m) / sp.beta(a_g, b_g))
            log_prob_cur_z = np.logaddexp(log_prob_cur_z, log_prob_m)

        log_prob_cur_z = log_prob_cur_z + np.log(sp.beta(np.sum(alpha_list), 1)) - np.log(sp.beta(alpha_list[z_ind], 1))
        log_z = np.logaddexp(log_z, log_prob_cur_z)

    log_prob_dist = log_z
    prob_dist = np.exp(log_prob_dist)

    return prob_dist, log_prob_dist


def dp_distance_between_cells(mut, cells_so_far, cell_list, log_zcy_dict, z_ind, others, constant_cells,
                              constant_presences):
    """
    This function is the dynamic programming part of log probability calculation of cells.
    Input: mut: int, cellssofar: int, cell_list: array of Cell objects (numCells,1), log_ZCY_dict (...),
           z_ind: int, others: dict (...), constant_cells: dict (2,1)
    Output: result: double
    """
    const_gen_sum = 0
    for gen in constant_cells.values():
        const_gen_sum = const_gen_sum + gen

    mut_others = mut - const_gen_sum
    cells_so_far_others = cells_so_far - 2

    key = "" + str(mut_others) + "_" + str(cells_so_far_others)
    result = others[str(z_ind)][key]

    for current_cell_id in constant_cells:
        gen = constant_cells[current_cell_id]
        presence = constant_presences[current_cell_id]

        if cell_list[current_cell_id].lc > 0:
            zcy_key = str(z_ind) + "_" + str(current_cell_id) + "_" + str(gen) + "_" + str(presence)
            result = result + log_zcy_dict[zcy_key]
        # else:
        #    result = result + np.log(np.power(p_ado,2))

    return result


def dp_distance_between_cells_others(mut, cells_so_far, cell_list, lookup, log_zcy_dict, z_ind, constant_cell_ids,
                                     other_cell_ids, general_lookup_table, p_ado):
    """
    This function is the dynamic programming part of log probability calculation of non-constant cells.
    Input: mut: int, cellssofar: int, cell_list: array of Cell objects (numCells,1), lookup: dict (...),
           log_ZCY_dict (...), z_ind: int, constant_cell_ids: array (2,1), other_cell_ids: array (numCells-2,1),
           general_lookup_table: dict (...)
    Output: result: double
    """
    key = "" + str(mut) + "_" + str(cells_so_far)
    if key in lookup.keys():
        # print("Fetching from lookup. key: ", key, "\tvalue: ", lookup[key])
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

    current_cell_id = other_cell_ids[cells_so_far - 1]
    # print("Other cell ids: ", other_cell_ids)
    # print("csf-1: ", cellssofar - 1)
    # print("\tcurrent cell_id: ", current_cell_id)

    if current_cell_id < min(constant_cell_ids):
        # print("1")
        key_gen = "" + str(mut) + "_" + str(current_cell_id + 1)
        result = general_lookup_table[str(z_ind)][key_gen]

    elif cell_list[current_cell_id].lc == 0:
        # print("2")
        result = dp_distance_between_cells_others(mut, cells_so_far - 1, cell_list, lookup, log_zcy_dict, z_ind,
                                                  constant_cell_ids, other_cell_ids, general_lookup_table, p_ado)

        # current is mutated zcy_key = str(z_ind) + "_" + str(current_cell_id) + "_" + str(1) res1 =
        # dpDistanceBetweenCellsOthers(mut-1, cellssofar-1, cell_list, lookup, log_ZCY_dict, z_ind,
        # constant_cell_ids, other_cell_ids, general_lookup_table, p_ado) res1 = res1 + np.log(np.power(p_ado,2))

        # current is not mutated zcy_key = str(z_ind) + "_" + str(current_cell_id) + "_" + str(0) res2 =
        # dpDistanceBetweenCellsOthers(mut, cellssofar-1, cell_list, lookup, log_ZCY_dict, z_ind, constant_cell_ids,
        # other_cell_ids, general_lookup_table, p_ado) res2 = res2 + np.log(np.power(p_ado,2))

        # result = np.logaddexp(res1, res2)

    else:
        # print("3")
        # current is mutated
        res1 = dp_distance_between_cells_others(mut - 1, cells_so_far - 1, cell_list, lookup, log_zcy_dict, z_ind,
                                                constant_cell_ids, other_cell_ids, general_lookup_table, p_ado)
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
        res2 = dp_distance_between_cells_others(mut, cells_so_far - 1, cell_list, lookup, log_zcy_dict, z_ind,
                                                constant_cell_ids, other_cell_ids, general_lookup_table, p_ado)
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


def distance_between_cells_others(cell_list, log_zcy_dict, z_list, constant_cell_ids, other_cell_ids,
                                  general_lookup_table, p_ado):
    """
    This function is the log probability calculation of non-constant cells.
    Input: cell_list: array of Cell objects (numCells,1), log_ZCY_dict (...), Z_list: array (12,2),
           constant_cell_ids: array (2,1), other_cell_ids: array (numCells-2,1), general_lookup_table: dict (...)
    Output: others: dict (...)
    """
    num_cells = len(cell_list)
    cells_so_far = num_cells - 2

    others = {}
    for z_ind in range(len(z_list)):
        lookup = {}
        for mut in range(num_cells - 1):
            # print("\n\nBeginning")
            # print("\n***\nz_ind: ", z_ind, "\tmut: ", mut, "\tlookup: ")
            # print(lookup)
            # print("\nothers:")
            # print(others)

            _ = dp_distance_between_cells_others(mut, cells_so_far, cell_list, lookup, log_zcy_dict, z_ind,
                                                 constant_cell_ids, other_cell_ids, general_lookup_table, p_ado)
            if mut == 0:
                others[str(z_ind)] = lookup
            else:
                others[str(z_ind)].update(lookup)

            # print("\n***\nz_ind: ", z_ind, "lookup: ")
            # print(lookup)
            # print("\nothers:")
            # print(others)
            # print("\n\nEnd")

    return others
