import sys
import time
import numpy as np

from lambda_iter import partition_lambda
from fragment_probability_single_site_without_pae import fragment_log_probability
from read_probability_single_site import convert_fragment_lambda_key_format


def compute_zcy_log_dict_pos_original(dataset, read_dicts, p_ado, p_ae, print_results):
    """ Original log_ZCY calculation. Returns a dictionary with zcy key. """
    if print_results:
        print("\n*****\nComputing Log Z_c_y dictionary...")

    start_time = time.time()

    cell_list = dataset['cell_list']
    bulk = dataset['bulk']
    z_list = dataset['z_list']

    log_zcy_dict = {}
    for z_ind in range(len(z_list)):
        if print_results:
            print("\tComputing for z_ind: ", z_ind)

        z = z_list[z_ind]

        for c in range(len(cell_list)):
            if print_results:
                print("\t\tComputing for cell: ", c)
            for y in range(2):
                zcy_key = str(z_ind) + "_" + str(c) + "_" + str(y)

                if y == 0:
                    f1f2 = np.array(bulk).astype(int)
                else:
                    f1f2 = np.array(z).astype(int)

                if y == 0 and z_ind > 0:  # for all z, probability of no mutation is the same (bulk)
                    prev_zcy_key = str(0) + "_" + str(c) + "_" + str(y)
                    log_sum_zcy = log_zcy_dict[prev_zcy_key]

                else:
                    log_sum_x1 = np.NINF
                    for x1 in range(2):

                        log_sum_x2 = np.NINF
                        for x2 in range(2):
                            lc = cell_list[c]['lc']

                            if x1 == 1 and x2 == 1:
                                if lc != 0:
                                    log_sum_frags = np.NINF
                                else:
                                    log_sum_frags = np.log(1)

                            else:
                                ado_stats = np.array([x1, x2])
                                lambda_list = partition_lambda(lc, ado_stats)

                                fragment_list = []
                                if y == 0:
                                    for i in range(3):
                                        fragment_list.append([f1f2[0], f1f2[1], (f1f2[0] + i + 1) % 4])
                                else:
                                    for i in range(4):
                                        fragment_list.append([int(z[0]), int(z[1]), int((z[0] + i) % 4)])

                                log_sum_frags = np.NINF
                                log_frag_prob_dict = {}

                                for fragments in fragment_list:
                                    for lambdas in lambda_list:
                                        key_1 = "_".join(str(aa) for aa in fragments)
                                        key_2 = "_".join(str(aa) for aa in lambdas)
                                        key = key_1 + "_" + key_2
                                        new_read_key = convert_fragment_lambda_key_format(key)
                                        log_prob_read = read_dicts[str(c)][new_read_key]

                                        new_frag_key = key
                                        if lambdas[2] == 0:
                                            new_frag_key = key[0:13] + 'N N' + key[16:]

                                        if new_frag_key not in log_frag_prob_dict:
                                            log_prob_frag = fragment_log_probability(fragments, lambdas, f1f2,
                                                                                     ado_stats, p_ae)

                                            if log_prob_frag > np.log(1) or log_prob_frag < np.NINF:
                                                print("\nError: Probability out of range!")
                                                print(zcy_key, log_prob_frag, np.exp(log_prob_frag), key, new_frag_key)
                                                sys.exit()

                                            log_sum_frags = np.logaddexp(log_sum_frags, (log_prob_read + log_prob_frag))
                                            log_frag_prob_dict[new_frag_key] = log_prob_frag

                            log_temp_x2 = log_sum_frags + np.log(np.power(p_ado, x2) * np.power(1 - p_ado, 1 - x2))
                            log_sum_x2 = np.logaddexp(log_sum_x2, log_temp_x2)

                        log_temp_x1 = log_sum_x2 + np.log(np.power(p_ado, x1) * np.power(1 - p_ado, 1 - x1))
                        log_sum_x1 = np.logaddexp(log_sum_x1, log_temp_x1)

                    log_sum_zcy = log_sum_x1

                log_zcy_dict[zcy_key] = log_sum_zcy

    end_time = time.time()
    if print_results:
        print("Computing Log Z_c_y dictionary is finished...\n*****")
        print("Total time: ", end_time - start_time)
    sys.stdout.flush()

    return log_zcy_dict


def compute_zcydd_log_dict_pos(dataset, read_dicts, p_ae, print_results, log_zcydda_dict=None):
    """ Original log_ZCY calculation. Returns a dictionary with zcy key. """
    if print_results:
        print("\n*****\nComputing Log Z_c_y dictionary...")

    start_time = time.time()

    cell_list = dataset['cell_list']
    bulk = dataset['bulk']
    z_list = dataset['z_list']

    log_zcydd_dict = {}

    log_zcydda_dict_loaded = True
    if log_zcydda_dict is None:
        log_zcydda_dict = {}
        log_zcydda_dict_loaded = False

    for z_ind in range(len(z_list)):
        if print_results:
            print("\tComputing for z_ind: ", z_ind)

        z = z_list[z_ind]

        for c in range(len(cell_list)):
            if print_results:
                print("\t\tComputing for cell: ", c)
            for y in range(2):
                if y == 0:
                    f1f2 = np.array(bulk).astype(int)
                else:
                    f1f2 = np.array(z).astype(int)

                for x1 in range(2):
                    for x2 in range(2):
                        zcydd_key = str(z_ind) + "_" + str(c) + "_" + str(y) + "_" + str(x1) + "_" + str(x2)
                        log_sum_a = np.NINF

                        for a in range(2):
                            zcydda_key = str(z_ind) + "_" + str(c) + "_" + str(y) + "_" + str(x1) + "_" + str(x2) + "_" + str(a)
                            lc = cell_list[c]['lc']

                            if log_zcydda_dict_loaded:
                                log_sum_frags = log_zcydda_dict[zcydda_key]
                            else:
                                if y == 0 and z_ind > 0:  # for all z, probability of no mutation is the same (bulk)
                                    prev_zcydda_key = str(0) + "_" + str(c) + "_" + str(y) + "_" + str(x1) + "_" + str(x2) + "_" + str(a)
                                    log_sum_frags = log_zcydda_dict[prev_zcydda_key]

                                else:
                                    log_sum_frags = np.NINF

                                    # TODO not sure about this if part
                                    if x1 == 1 and x2 == 1:
                                        if lc == 0:
                                            log_sum_frags = np.log(1)

                                    else:
                                        ado_stats = np.array([x1, x2])
                                        lambda_list = partition_lambda(lc, ado_stats)

                                        fragment_list = []
                                        if y == 0:
                                            for i in range(3):
                                                fragment_list.append([f1f2[0], f1f2[1], (f1f2[0] + i + 1) % 4])
                                        else:
                                            for i in range(4):
                                                fragment_list.append([int(z[0]), int(z[1]), int((z[0] + i) % 4)])

                                        log_frag_prob_dict = {}

                                        for fragments in fragment_list:
                                            for lambdas in lambda_list:
                                                if (a == 0 and lambdas[-1] == 0) or (a == 1 and lambdas[-1] > 0):
                                                    key_1 = "_".join(str(aa) for aa in fragments)
                                                    key_2 = "_".join(str(aa) for aa in lambdas)
                                                    key = key_1 + "_" + key_2
                                                    new_read_key = convert_fragment_lambda_key_format(key)
                                                    log_prob_read = read_dicts[str(c)][new_read_key]

                                                    new_frag_key = key
                                                    if lambdas[2] == 0:
                                                        new_frag_key = key[0:13] + 'N N' + key[16:]

                                                    if new_frag_key not in log_frag_prob_dict:
                                                        log_prob_frag = fragment_log_probability(fragments, lambdas, f1f2, ado_stats)
                                                        log_sum_frags = np.logaddexp(log_sum_frags, (log_prob_read + log_prob_frag))
                                                        log_frag_prob_dict[new_frag_key] = log_prob_frag

                            if not log_zcydda_dict_loaded:
                                log_zcydda_dict[zcydda_key] = log_sum_frags

                            num_edges = 2 * lc - 1
                            if x1 == 0 and x2 == 0:
                                num_edges = 2 * lc - 2

                            if a == 0:
                                log_sum_frags += num_edges * np.log(1 - p_ae)
                            else:
                                log_sum_frags += np.log(p_ae) + (num_edges - 1) * np.log(1 - p_ae)

                            log_sum_a = np.logaddexp(log_sum_a, log_sum_frags)

                        log_zcydd_dict[zcydd_key] = log_sum_a

    end_time = time.time()
    if print_results:
        print("Computing Log Z_c_y_d_d dictionary is finished...\n*****")
        print("Total time: ", end_time - start_time)
    sys.stdout.flush()

    return log_zcydd_dict, log_zcydda_dict


def compute_zcy_log_dict_pos(dataset, read_dicts, p_ado, p_ae, print_results, log_zcydda_dict=None):
#def compute_zcy_log_dict_pos_via_dd(dataset, read_dicts, p_ado, p_ae, print_results, log_zcydda_dict=None):
    """ Original log_ZCY calculation. Returns a dictionary with zcy key. """
    if print_results:
        print("\n*****\nComputing Log Z_c_y dictionary...")

    start_time = time.time()

    cell_list = dataset['cell_list']
    z_list = dataset['z_list']

    log_zcydd_dict, log_zcydda_dict = compute_zcydd_log_dict_pos(dataset, read_dicts, p_ae, print_results, log_zcydda_dict)

    log_zcy_dict = {}
    for z_ind in range(len(z_list)):
        if print_results:
            print("\tComputing for z_ind: ", z_ind)

        for c in range(len(cell_list)):
            if print_results:
                print("\t\tComputing for cell: ", c)
            for y in range(2):
                zcy_key = str(z_ind) + "_" + str(c) + "_" + str(y)

                if y == 0 and z_ind > 0:  # for all z, probability of no mutation is the same (bulk)
                    prev_zcy_key = str(0) + "_" + str(c) + "_" + str(y)
                    log_sum_zcy = log_zcy_dict[prev_zcy_key]

                else:
                    log_sum_x1 = np.NINF
                    for x1 in range(2):

                        log_sum_x2 = np.NINF
                        for x2 in range(2):
                            zcydd_key = zcy_key + "_" + str(x1) + "_" + str(x2)
                            log_sum_frags = log_zcydd_dict[zcydd_key]

                            log_temp_x2 = log_sum_frags + np.log(np.power(p_ado, x2) * np.power(1 - p_ado, 1 - x2))
                            log_sum_x2 = np.logaddexp(log_sum_x2, log_temp_x2)

                        log_temp_x1 = log_sum_x2 + np.log(np.power(p_ado, x1) * np.power(1 - p_ado, 1 - x1))
                        log_sum_x1 = np.logaddexp(log_sum_x1, log_temp_x1)

                    log_sum_zcy = log_sum_x1

                log_zcy_dict[zcy_key] = log_sum_zcy

    end_time = time.time()
    if print_results:
        print("Computing Log Z_c_y dictionary is finished...\n*****")
        print("Total time: ", end_time - start_time)
    sys.stdout.flush()

    return log_zcy_dict, log_zcydda_dict

