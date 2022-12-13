import sys
import time
import numpy as np

from lambda_iter import partition_lambda
from fragment_enumerator import enumerate_fragment3
from fragment_probability import fragment_probability
from fragment_probability import fragment_log_probability
from read_probability import convert_fragment_lambda_key_format


def compute_zcyp_log_dict_pos(dataset, read_dicts, p_ado, p_ae, print_results):
    if print_results:
        print("\n*****\nComputing Log Z_c_y dictionary...")

    start_time = time.time()

    cell_list = dataset['cell_list']
    bulk = dataset['bulk']
    z_list = dataset['z_list']

    log_zcy_dict = {}
    for z_ind in range(len(z_list)):
        # for z_ind in range(1,2):
        if print_results:
            print("\tComputing for z_ind: ", z_ind)

        z = z_list[z_ind]

        if np.array_equal(bulk[1, :], z[1, :]):
            inf_allele = 0
        else:
            inf_allele = 1

        for c in range(len(cell_list)):
            # for c in range(0,3):
            if print_results:
                print("\t\tComputing for cell: ", c)
            for y in range(2):

                if y == 0:
                    f1f2 = bulk
                else:
                    f1f2 = z

                # Presence
                for pc in range(2):

                    zcy_key = str(z_ind) + "_" + str(c) + "_" + str(y) + "_" + str(pc)
                    # print("ZCY key: ", zcy_key)

                    # TODO We need to add a check like below. Otherwise, we make unnecessaty repetitive calculations.
                    #  For a cell c and non-mutation status (y=0), some calculations are unncessary. If we previously
                    #  calculated a mutation z with the same informative allele, we can just read its result and put it.
                    #if y == 0 and z_ind > 0:  # for all z, probability of no mutation is the same (bulk)
                    #    prev_zcy_key = str(0) + "_" + str(c) + "_" + str(y)
                    #    log_sum_zcy = log_zcy_dict[prev_zcy_key]

                    #else:
                    if inf_allele == 0:
                        log_sum_x1 = np.NINF

                        x1 = 1 - pc

                        log_sum_x2 = np.NINF
                        for x2 in range(2):
                            lc = cell_list[c].lc

                            if x1 == 1 and x2 == 1:
                                if lc != 0:
                                    log_sum_frags = np.NINF
                                else:
                                    # print("not skip locus dropout for ZCY: ", zcy_key)
                                    log_sum_frags = np.log(1)

                            else:
                                # print("\tAdo stats: ", x1, x2)

                                ado_stats = np.array([x1, x2])
                                lambda_list = partition_lambda(lc, ado_stats)
                                fragment_list = enumerate_fragment3(f1f2)

                                log_sum_frags = np.NINF
                                log_frag_prob_dict = {}

                                for fragments in fragment_list:
                                    for lambdas in lambda_list:
                                        key_1 = "_".join(str(aa) for aa in fragments)
                                        key_2 = "_".join(str(aa) for aa in lambdas)
                                        key = key_1 + "_" + key_2
                                        # print("\t\t\tCell: ", c)
                                        # print("\t\t\tKey: ", key)

                                        new_read_key = convert_fragment_lambda_key_format(key)
                                        # print("\t\t\tNew Key: ", new_read_key)
                                        log_prob_read = read_dicts[c][new_read_key]
                                        # print("\t\t\tRead prob: ", new_read_key, np.exp(log_prob_read))

                                        new_frag_key = key
                                        if lambdas[2] == 0:
                                            new_frag_key = key[0:13] + 'N N' + key[16:]

                                        if new_frag_key not in log_frag_prob_dict:
                                            log_prob_frag = fragment_log_probability(fragments, lambdas, f1f2,
                                                                                     ado_stats, p_ae)
                                            # prob_frag = fragment_probability(fragments, lambdas, f1f2, ado_stats,
                                            # p_ae)

                                            # print("\t\t\tFrag: ", key, "\tAdo: ", ado_stats, "\tProb: ", prob_frag)

                                            if log_prob_frag > np.log(1) or log_prob_frag < np.NINF:
                                                print("\nError: Probability out of range!")
                                                print(zcy_key, log_prob_frag, np.exp(log_prob_frag), key, new_frag_key)
                                                sys.exit()

                                            # if prob_frag > 1 or prob_frag < 0:
                                            #    print("\nError: Probability out of range!")
                                            #    print(zcy_key, prob_frag, key, new_frag_key)
                                            #    sys.exit()

                                            # if prob_frag == 0:
                                            #    log_prob_frag = np.NINF
                                            # else:
                                            #    log_prob_frag = np.log(prob_frag)

                                            # sum_frags = sum_frags + np.exp(log_prob_read) * prob_frag
                                            # frag_prob_dict[new_frag_key] = prob_frag
                                            log_sum_frags = np.logaddexp(log_sum_frags, (log_prob_read + log_prob_frag))
                                            log_frag_prob_dict[new_frag_key] = log_prob_frag

                                # print("\t\tFrag sum: \t", sum_frags)

                            # temp_x2 = sum_frags * (np.power(p_ado,x2)) * np.power(1-p_ado,1-x2)
                            log_temp_x2 = log_sum_frags + np.log(np.power(p_ado, x2) * np.power(1 - p_ado, 1 - x2))

                            # if x1==1 and x2==1:
                            #    print("\t\tTemp_x2: \t", temp_x2)

                            # sum_x2 = sum_x2 + temp_x2
                            log_sum_x2 = np.logaddexp(log_sum_x2, log_temp_x2)

                            # print("\tSum_x2: \t", sum_x2)

                        # temp_x1 = sum_x2 * (np.power(p_ado,x1)) * np.power(1-p_ado,1-x1)
                        log_temp_x1 = log_sum_x2 + np.log(np.power(p_ado, x1) * np.power(1 - p_ado, 1 - x1))
                        # print("\tTemp_x1: \t", temp_x1)

                        # sum_x1 = sum_x1 + temp_x1
                        log_sum_x1 = np.logaddexp(log_sum_x1, log_temp_x1)

                        # print("\tSum_x1: \t", sum_x1)

                    # inf_allele == 1
                    else:
                        log_sum_x1 = np.NINF
                        for x1 in range(2):

                            log_sum_x2 = np.NINF
                            x2 = 1 - pc
                            lc = cell_list[c].lc

                            if x1 == 1 and x2 == 1:
                                if lc != 0:
                                    log_sum_frags = np.NINF
                                else:
                                    # print("not skip locus dropout for ZCY: ", zcy_key)
                                    log_sum_frags = np.log(1)

                            else:
                                # print("\tAdo stats: ", x1, x2)

                                ado_stats = np.array([x1, x2])
                                lambda_list = partition_lambda(lc, ado_stats)
                                fragment_list = enumerate_fragment3(f1f2)

                                log_sum_frags = np.NINF
                                log_frag_prob_dict = {}

                                for fragments in fragment_list:
                                    for lambdas in lambda_list:
                                        key_1 = "_".join(str(aa) for aa in fragments)
                                        key_2 = "_".join(str(aa) for aa in lambdas)
                                        key = key_1 + "_" + key_2
                                        # print("\t\t\tCell: ", c)
                                        # print("\t\t\tKey: ", key)

                                        new_read_key = convert_fragment_lambda_key_format(key)
                                        # print("\t\t\tNew Key: ", new_read_key)
                                        log_prob_read = read_dicts[c][new_read_key]
                                        # print("\t\t\tRead prob: ", new_read_key, np.exp(log_prob_read))

                                        new_frag_key = key
                                        if lambdas[2] == 0:
                                            new_frag_key = key[0:13] + 'N N' + key[16:]

                                        if new_frag_key not in log_frag_prob_dict:
                                            log_prob_frag = fragment_log_probability(fragments, lambdas, f1f2,
                                                                                     ado_stats, p_ae)
                                            # prob_frag = fragment_probability(fragments, lambdas, f1f2, ado_stats,
                                            # p_ae)

                                            # print("\t\t\tFrag: ", key, "\tAdo: ", ado_stats, "\tProb: ", prob_frag)

                                            if log_prob_frag > np.log(1) or log_prob_frag < np.NINF:
                                                print("\nError: Probability out of range!")
                                                print(zcy_key, log_prob_frag, np.exp(log_prob_frag), key, new_frag_key)
                                                sys.exit()

                                            # if prob_frag > 1 or prob_frag < 0:
                                            #    print("\nError: Probability out of range!")
                                            #    print(zcy_key, prob_frag, key, new_frag_key)
                                            #    sys.exit()

                                            # if prob_frag == 0:
                                            #    log_prob_frag = np.NINF
                                            # else:
                                            #    log_prob_frag = np.log(prob_frag)

                                            # sum_frags = sum_frags + np.exp(log_prob_read) * prob_frag
                                            # frag_prob_dict[new_frag_key] = prob_frag
                                            log_sum_frags = np.logaddexp(log_sum_frags, (log_prob_read + log_prob_frag))
                                            log_frag_prob_dict[new_frag_key] = log_prob_frag

                                # print("\t\tFrag sum: \t", sum_frags)

                            # temp_x2 = sum_frags * (np.power(p_ado,x2)) * np.power(1-p_ado,1-x2)
                            log_temp_x2 = log_sum_frags + np.log(np.power(p_ado, x2) * np.power(1 - p_ado, 1 - x2))

                            # if x1==1 and x2==1:
                            #    print("\t\tTemp_x2: \t", temp_x2)

                            # sum_x2 = sum_x2 + temp_x2
                            log_sum_x2 = np.logaddexp(log_sum_x2, log_temp_x2)

                            # print("\tSum_x2: \t", sum_x2)

                            # temp_x1 = sum_x2 * (np.power(p_ado,x1)) * np.power(1-p_ado,1-x1)
                            log_temp_x1 = log_sum_x2 + np.log(np.power(p_ado, x1) * np.power(1 - p_ado, 1 - x1))
                            # print("\tTemp_x1: \t", temp_x1)

                            # sum_x1 = sum_x1 + temp_x1
                            log_sum_x1 = np.logaddexp(log_sum_x1, log_temp_x1)

                            # print("\tSum_x1: \t", sum_x1)

                    # if printResults:
                    #    print("\tZCY: ", zcy_key, sum_zcy)

                    log_zcy_dict[zcy_key] = log_sum_x1

    if print_results:
        print("Computing Log Z_c_y dictionary is finished...\n*****")

    end_time = time.time()
    print("Total time: ", end_time - start_time)
    sys.stdout.flush()

    return log_zcy_dict