import os
import sys
import time
import json
import pickle
import datetime
import argparse
import matplotlib
import numpy as np
import scipy.special as sp
import multiprocessing as mp
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from mutation_type_probability import compute_mutation_probabilities_log_dp
from read_probability import precompute_reads as precompute_reads_paired
from read_probability_single_site import precompute_reads as precompute_reads_singleton
from compute_zcy import compute_zcy_log_dict_pos as compute_zcy_log_dict_pos_paired
from compute_zcy_single_site import compute_zcy_log_dict_pos as compute_zcy_log_dict_pos_singleton


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


def plot_chain(output_dir, chain_idx, p_ado_samples, p_ae_samples, acceptance_samples, log_posterior_samples, lag_list):
    num_iter = p_ado_samples.shape[0]
    iter_array = np.arange(num_iter)

    print("\tAcceptance rate: ", np.divide(np.cumsum(acceptance_samples), iter_array+1)[-1])

    plt.figure(figsize=(20, 3))
    plt.plot(iter_array, log_posterior_samples)
    plt.xlim([0,len(iter_array)])
    plt.xlabel("Iterations")
    plt.ylabel("Log posterior")
    title_str = "chain_" + str(chain_idx) + "_logposterior"
    plt.title(title_str)
    filename = output_dir + str(title_str) + ".png"
    plt.savefig(filename)
    plt.close()

    plt.figure(figsize=(20, 3))
    plt.plot(iter_array, np.divide(np.cumsum(acceptance_samples),iter_array+1))
    plt.xlim([0,len(iter_array)])
    plt.ylim([0,1.1])
    plt.xlabel("Iterations")
    plt.ylabel("Acceptance rate")
    title_str = "chain_" + str(chain_idx) + "_acceptance_rate"
    plt.title(title_str)
    filename = output_dir + str(title_str) + ".png"
    plt.savefig(filename)
    plt.close()

    plt.figure(figsize=(20, 3))
    plt.subplot(1, 2, 1)
    plt.hist(p_ado_samples,bins=50)
    #plt.scatter(start_p_ado, 0, color='g')
    plt.xlabel("p_ado values")
    plt.xlim([0,1])
    plt.title("Posterior of p_ado")

    plt.subplot(1,2,2)
    plt.plot(p_ado_samples)
    plt.title("Samples of p_ado")
    plt.xlim([0,len(iter_array)])
    plt.xlabel("Iterations")

    title_str = "chain_" + str(chain_idx) + "_p_ado"
    plt.title(title_str)
    filename = output_dir + str(title_str) + ".png"
    plt.savefig(filename)
    plt.close()

    plt.figure(figsize=(20, 3))
    plt.subplot(1,2,1)
    plt.hist(p_ae_samples,bins=50)
    #plt.scatter(start_p_ado, 0, color='g')
    plt.xlabel("p_ae values")
    plt.xlim([0,1])
    plt.title("Posterior of p_ae")

    plt.subplot(1, 2, 2)
    plt.plot(p_ae_samples)
    plt.title("Samples of p_ae")
    plt.xlim([0,len(iter_array)])
    plt.xlabel("Iterations")

    title_str = "chain_" + str(chain_idx) + "_p_ae"
    plt.title(title_str)
    filename = output_dir + str(title_str) + ".png"
    plt.savefig(filename)
    plt.close()

    plt.figure(figsize=(20, 3))
    plt.plot(p_ado_samples, p_ae_samples)
    #plt.scatter(trueA, trueSd, color='r')
    plt.xlabel("p_ado")
    plt.ylabel("p_ae")
    title_str = "chain_" + str(chain_idx) + "_parameter_samples"
    plt.title(title_str)
    filename = output_dir + str(title_str) + ".png"
    plt.savefig(filename)
    plt.close()

    #auto-correlation
    rho_list = np.zeros((len(lag_list), 2))

    for idx in range(len(lag_list)):
        lag = lag_list[idx]
        rho_list[idx,:] = calculate_autocorrelation(p_ado_samples, p_ae_samples, lag)

    plt.figure(figsize=(20,5))
    plt.scatter(lag_list, rho_list[:, 0], color='b')
    plt.scatter(lag_list, rho_list[:, 1], color='r')
    plt.plot(lag_list, rho_list[:, 0], color='b', label='p_ado')
    plt.plot(lag_list, rho_list[:, 1], color='r', label='p_ae')
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    title_str = "chain_" + str(chain_idx) + "_autocorrelation"
    plt.title(title_str)
    plt.legend()
    plt.xlim([0,max(lag_list)])
    filename = output_dir + str(title_str) + ".png"
    plt.savefig(filename)
    plt.close()


def calculate_autocorrelation(p_ado_samples, p_ae_samples, lag):
    chain = np.zeros((p_ado_samples.shape[0], 2))
    chain[:, 0] = p_ado_samples
    chain[:, 1] = p_ae_samples

    chain_mean = np.mean(chain, axis=0)

    denom = np.sum(np.power(chain - chain_mean, 2), axis=0)

    up = np.zeros(chain.shape[1])
    for i in range(chain.shape[0] - lag):
        up = up + np.multiply((chain[i, :] - chain_mean), (chain[i + lag, :] - chain_mean))

    rho = np.divide(up, denom)
    return rho


def calculate_log_prior(p_ado, p_ae, a_ado=1, b_ado=1, a_ae=1, b_ae=1):
    log_prior_ado = (a_ado - 1) * np.log(p_ado) + (b_ado - 1) * np.log(1 - p_ado) - np.log(sp.beta(a_ado, b_ado))
    log_prior_ae = (a_ae - 1) * np.log(p_ae) + (b_ae - 1) * np.log(1 - p_ae) - np.log(sp.beta(a_ae, b_ae))

    return log_prior_ado + log_prior_ae


def proposal_function(p_ado, p_ae):
    sd_ado = 0.01
    sd_ae = 0.01
    prop_p_ado = np.random.normal(p_ado, sd_ado)
    prop_p_ae = np.random.normal(p_ae, sd_ae)

    # Set limits

    while prop_p_ado <= 0 or prop_p_ado > 1:
        prop_p_ado = np.random.normal(p_ado, sd_ado)
    while prop_p_ae <= 0 or prop_p_ae > 1:
        prop_p_ae = np.random.normal(prop_p_ae, sd_ae)

    return prop_p_ado, prop_p_ae


# TODO FIX HERE! Pickling error, cannot to multiprocessing
def calculate_loglikelihood(positions, p_ado, p_ae, dataset, read_prob_dir, a_g, b_g, print_results):
    #'''
    pool = mp.Pool() # mp.Pool(processes=num_processes)
    infer_results = pool.starmap(analyse_mut_probs_onepos_pool, [(int(pos), dataset[str(pos)], p_ae, p_ado, a_g, b_g,
                                                                  read_prob_dir, False) for pos in positions])
    pool.close()
    pool.join()

    log_likelihood = 0
    for pos in range(len(infer_results)):
        log_likelihood += np.log(infer_results[pos][1])
    '''

    log_likelihood = 0
    for pos in positions:
        _, res = analyse_mut_probs_onepos_pool(int(pos), dataset[str(pos)], p_ae, p_ado, a_g, b_g,
                                               read_prob_dir, print_results=False)
        log_likelihood += np.log(res)

    '''
    return log_likelihood


def analyse_mut_probs_onepos_pool(pos, dataset, p_ae, p_ado, a_g, b_g, read_prob_dir, print_results=False):
    start_time = time.time()

    if print_results:
        print("\n******\nAnalysing mutations for parameters: ", p_ae, p_ado, a_g, b_g)
        print("Pos idx of process is: ", pos)
        print("Process id: ", os.getpid(), ". Uname: ", os.uname())
        print("\n***********\nPosition: ", pos)

    cell_list = dataset['cell_list']
    bulk = dataset['bulk']
    z_list = dataset['z_list']
    alpha_list = np.ones(len(z_list))

    if len(z_list) == 12:
        pos_is_paired = True
    else:  # elif bulk.shape == (2,):
        pos_is_paired = False

    filename = read_prob_dir + "read_dict.pickle"

    if os.path.exists(filename):
        if print_results:
            print("\nLoading read dictionary...")
        read_dicts = load_json(filename)
        read_dicts = read_dicts[int(pos)]
    else:
        filename = read_prob_dir + "read_dict_" + str(pos) + ".pickle"
        if os.path.exists(filename):
            if print_results:
                print("\nLoading read position dictionary...")
            read_dicts = load_json(filename)
        else:
            #print("\nError. No read dictionary for pos ", str(pos))
            #print("Calculating read probability...")
            if pos_is_paired:
                read_dicts = precompute_reads_paired(cell_list, z_list, bulk, print_results=False)
            else:
                read_dicts = precompute_reads_singleton(cell_list, z_list, bulk, print_results=False)
            save_json(filename, read_dicts)

    log_zcy_filename = read_prob_dir + "log_zcy_" + str(p_ado) + "_" + str(p_ae) + ".pickle"
    if os.path.exists(log_zcy_filename):
        if print_results:
            print("\nLoading log_ZCY dictionary...")
        log_ZCY_dict_prior = load_json(log_zcy_filename)
        log_ZCY_dict_prior = log_ZCY_dict_prior[int(pos)]
    else:
        log_zcy_pos_filename = read_prob_dir + "log_zcy_" + str(p_ado) + "_" + str(p_ae) + "_" + str(pos) + ".pickle"
        if os.path.exists(log_zcy_pos_filename):
            if print_results:
                print("\nLoading log_ZCY position dictionary...")
            log_ZCY_dict_prior = load_json(log_zcy_pos_filename)
        else:
            if pos_is_paired:
                log_zcydda_dict = None

                # Check if ZCYDDA file exists
                log_zcydda_pos_filename = read_prob_dir + "log_zcydda_" + str(pos) + ".pickle"
                if os.path.exists(log_zcydda_pos_filename) and os.path.getsize(log_zcydda_pos_filename) > 0:
                    #print("\nLoading log_ZCYDDA position dictionary...")
                    log_zcydda_dict = load_json(log_zcydda_pos_filename)
                    log_ZCY_dict_prior, _ = compute_zcy_log_dict_pos_paired(dataset, read_dicts,
                                                                            p_ado, p_ae, False, log_zcydda_dict)
                else:
                    log_ZCY_dict_prior, log_zcydda_dict = compute_zcy_log_dict_pos_paired(dataset, read_dicts,
                                                                                          p_ado, p_ae,
                                                                                          False, log_zcydda_dict)
                    save_json(log_zcydda_pos_filename, log_zcydda_dict)

            else:
                log_zcydda_dict = None

                # Check if ZCYDDA file exists
                log_zcydda_pos_filename = read_prob_dir + "log_zcydda_" + str(pos) + ".pickle"
                if os.path.exists(log_zcydda_pos_filename) and os.path.getsize(log_zcydda_pos_filename) > 0:
                    # print("\nLoading log_ZCYDDA position dictionary...")
                    log_zcydda_dict = load_json(log_zcydda_pos_filename)
                    log_ZCY_dict_prior, _ = compute_zcy_log_dict_pos_singleton(dataset, read_dicts,
                                                                               p_ado, p_ae, False, log_zcydda_dict)
                else:
                    log_ZCY_dict_prior, log_zcydda_dict = compute_zcy_log_dict_pos_singleton(dataset, read_dicts,
                                                                                             p_ado, p_ae,
                                                                                             False, log_zcydda_dict)
                    save_json(log_zcydda_pos_filename, log_zcydda_dict)

            #save_json(log_zcy_pos_filename, log_ZCY_dict_prior) # TODO remove later

    normed_prob_Z, highestZ, highestZ_prob, max_key, general_lookup_table, log_prob_Z = \
        compute_mutation_probabilities_log_dp(cell_list, z_list, alpha_list, log_ZCY_dict_prior,
                                              a_g, b_g, p_ado, print_results=False)

    if False:
        print("\n***Common Z results:\n")
        print("\nNormalized probabilities of common mutation type Z: \n", normed_prob_Z)
        print("\nMaximum probability: ", highestZ_prob, ". Dict key: ", max_key)
        print("\nCorresponding mutation type: ", highestZ[0], highestZ[1])
        print("\nLog probabilities of common mutation type Z: \n", log_prob_Z)

    marg_Z = sum(np.exp(log_prob_Z))
    end_time = time.time()

    if print_results:
        print("\nsum Z: \n", marg_Z, "\tpos: ", pos, "\tp_ado: ", p_ado, "\tp_ae", p_ae)
        print("\nTotal time of position ", pos, " is: ", end_time - start_time)
    sys.stdout.flush()
    return pos, marg_Z


def sample_parameters(chain_results, max_iter):
    p_ado_samples = []
    p_ae_samples = []

    for chain_idx in range(len(chain_results)):
        _, chain_p_ado_samples, chain_p_ae_samples, _, _ = chain_results[chain_idx]
        p_ado_samples += list(chain_p_ado_samples)
        p_ae_samples += list(chain_p_ae_samples)

    mean_p_ado = np.mean(p_ado_samples)
    mean_p_ae = np.mean(p_ae_samples)

    std_p_ado = np.std(p_ado_samples)
    std_p_ae = np.std(p_ae_samples)

    return mean_p_ado, mean_p_ae, std_p_ado, std_p_ae
