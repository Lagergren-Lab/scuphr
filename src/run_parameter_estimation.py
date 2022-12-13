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

from parameter_estimation import calculate_loglikelihood, calculate_log_prior, proposal_function, plot_chain, sample_parameters

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


def run_metropolis(chain, dataset, read_prob_dir, output_dir, max_iter, burnin, start_p_ae,
                   start_p_ado, a_g, b_g, seed_val, printResults, positions):
    start_time = time.time()

    print("\tRunning Metropolis-Hastings for chain: ", chain)
    # print("Process id: ", os.getpid(), ". Uname: ", os.uname())

    np.random.seed(seed_val + chain)

    if start_p_ae == -1:
        start_p_ae = np.random.rand() / 10
    if start_p_ado == -1:
        start_p_ado = np.random.rand()

    # print("Start p_ae: ", start_p_ae)
    # print("Start p_ado: ", start_p_ado)

    p_ado_samples = np.zeros(max_iter)
    p_ae_samples = np.zeros(max_iter)
    acceptance_samples = np.zeros(max_iter)
    log_posterior_samples = np.zeros(max_iter)
    log_prior_samples = np.zeros(max_iter)
    log_likelihood_samples = np.zeros(max_iter)
    ratio_samples = np.zeros(max_iter)

    # also save the rejected ones
    p_ado_samples_all = []
    p_ae_samples_all = []
    acceptance_samples_all = []
    log_posterior_samples_all = []
    log_prior_samples_all = []
    log_likelihood_samples_all = []
    ratio_samples_all = []
    #

    p_ado_samples[0] = start_p_ado
    p_ae_samples[0] = start_p_ae
    acceptance_samples[0] = 1
    start_log_lik = calculate_loglikelihood(positions, start_p_ado, start_p_ae, dataset, read_prob_dir,
                                            a_g, b_g, print_results=False)
    start_log_pri = calculate_log_prior(start_p_ado, start_p_ae)
    log_prior_samples[0] = start_log_pri
    log_likelihood_samples[0] = start_log_lik
    log_posterior_samples[0] = start_log_lik + start_log_pri

    p_ado_samples_all.append(start_p_ado)
    p_ae_samples_all.append(start_p_ae)
    acceptance_samples_all.append(1)
    log_posterior_samples_all.append(log_posterior_samples[0])
    log_prior_samples_all.append(start_log_pri)
    log_likelihood_samples_all.append(start_log_lik)
    ratio_samples_all.append(-1)

    for iter_ in range(1, max_iter):
        if iter_ % 200 == 0:
            print("\tIteration: ", iter_, "\tp_ado: ", p_ado_samples[iter_ - 1], "\tp_ae: ", p_ae_samples[iter_ - 1])

        # Get current values
        current_p_ado = p_ado_samples[iter_ - 1]
        current_p_ae = p_ae_samples[iter_ - 1]
        current_logposterior = log_posterior_samples[iter_ - 1]
        current_log_pri = log_prior_samples[iter_ - 1]
        current_log_lik = log_likelihood_samples[iter_ - 1]

        # Propose values
        prop_p_ado, prop_p_ae = proposal_function(current_p_ado, current_p_ae)
        prop_log_lik = calculate_loglikelihood(positions, prop_p_ado, prop_p_ae, dataset, read_prob_dir,
                                               a_g, b_g, print_results=False)
        prop_log_pri = calculate_log_prior(prop_p_ado, prop_p_ae)
        prop_logposterior = prop_log_lik + prop_log_pri

        # Calculate ratios (symmetric proposal)
        ratio_posterior = min(1, np.exp(prop_logposterior - current_logposterior))

        p_ado_samples_all.append(prop_p_ado)
        p_ae_samples_all.append(prop_p_ae)
        log_posterior_samples_all.append(prop_logposterior)
        log_prior_samples_all.append(prop_log_pri)
        log_likelihood_samples_all.append(prop_log_lik)
        ratio_samples_all.append(ratio_posterior)

        # Accept/reject
        u = np.random.rand()
        if u <= ratio_posterior:
            p_ado_samples[iter_] = prop_p_ado
            p_ae_samples[iter_] = prop_p_ae
            log_posterior_samples[iter_] = prop_logposterior
            acceptance_samples[iter_] = 1
            log_prior_samples[iter_] = prop_log_pri
            log_likelihood_samples[iter_] = prop_log_lik

            acceptance_samples_all.append(1)
        else:
            p_ado_samples[iter_] = current_p_ado
            p_ae_samples[iter_] = current_p_ae
            log_posterior_samples[iter_] = current_logposterior
            log_prior_samples[iter_] = current_log_pri
            log_likelihood_samples[iter_] = current_log_lik

            acceptance_samples_all.append(0)

        if printResults:
            sys.stdout.flush()
            print("\nIter: ", iter_)
            print("\tCurrent p_ado: ", current_p_ado, "\tProposed p_ado: ", prop_p_ado)
            print("\tCurrent p_ae: ", current_p_ae, "\tProposed p_ae: ", prop_p_ae)
            print("\tCurrent log_posterior: ", current_logposterior, "\tProposed log_posterior: ", prop_logposterior, )
            print("\tRatio: ", ratio_posterior, "\tu-val: ", u, "\tAcceptance: ", acceptance_samples[iter_])
            sys.stdout.flush()

        if iter_ % 1000 == 0 or iter_ == max_iter - 1:
            filename = output_dir + "chain_" + str(chain) + "_iter" + str(iter_) + "_p_ado_samples"
            np.save(filename, p_ado_samples)
            filename = output_dir + "chain_" + str(chain) + "_iter" + str(iter_) + "_p_ae_samples"
            np.save(filename, p_ae_samples)
            filename = output_dir + "chain_" + str(chain) + "_iter" + str(iter_) + "_acceptance_samples"
            np.save(filename, acceptance_samples)
            filename = output_dir + "chain_" + str(chain) + "_iter" + str(iter_) + "_logposterior_samples"
            np.save(filename, log_posterior_samples)
            filename = output_dir + "chain_" + str(chain) + "_iter" + str(iter_) + "_loglikelihood_samples"
            np.save(filename, log_likelihood_samples)
            filename = output_dir + "chain_" + str(chain) + "_iter" + str(iter_) + "_logprior_samples"
            np.save(filename, log_prior_samples)

            filename = output_dir + "chain_" + str(chain) + "_iter" + str(iter_) + "_p_ado_samples_all"
            np.save(filename, np.array(p_ado_samples_all))
            filename = output_dir + "chain_" + str(chain) + "_iter" + str(iter_) + "_p_ae_samples_all"
            np.save(filename, np.array(p_ae_samples_all))
            filename = output_dir + "chain_" + str(chain) + "_iter" + str(iter_) + "_acceptance_samples_all"
            np.save(filename, np.array(acceptance_samples_all))
            filename = output_dir + "chain_" + str(chain) + "_iter" + str(iter_) + "_logposterior_samples_all"
            np.save(filename, np.array(log_posterior_samples_all))
            filename = output_dir + "chain_" + str(chain) + "_iter" + str(iter_) + "_loglikelihood_samples_all"
            np.save(filename, np.array(log_likelihood_samples_all))
            filename = output_dir + "chain_" + str(chain) + "_iter" + str(iter_) + "_logprior_samples_all"
            np.save(filename, np.array(log_prior_samples_all))
            filename = output_dir + "chain_" + str(chain) + "_iter" + str(iter_) + "_ratio_samples_all"
            np.save(filename, np.array(ratio_samples_all))

    end_time = time.time()
    print("Total time of chain ", chain, " is: ", end_time - start_time)
    sys.stdout.flush()
    #output.put((chain, p_ado_samples[burnin:], p_ae_samples[burnin:], acceptance_samples[burnin:],
    #            log_posterior_samples[burnin:]))
    return chain, p_ado_samples[burnin:], p_ae_samples[burnin:], acceptance_samples[burnin:], log_posterior_samples[burnin:]


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
    parser.add_argument('--burnin', help="Specify the percentage of burn-in iterations. Default: 20",
                        type=int, default=10)
    parser.add_argument('--data_type', help="Specify the data type. Default: real", type=str, default="real")
    parser.add_argument('--output_dir', help="Specify the output directory.", type=str, default=default_dir)
    parser.add_argument('--max_iter', help="Specify the number of Metropolis-Hastings iterations. Default: 1000",
                        type=int, default=100)
    parser.add_argument('--num_chains', help="Specify the number of Metropolis-Hastings chains. Default: 3",
                        type=int, default=3)
    parser.add_argument('--p_ado', help="Specify the initial allelic dropout probability of a base. Default: -1",
                        type=float, default=0.1)
    parser.add_argument('--p_ae', help="Specify the initial amplification error probability of a base. Default: -1",
                        type=float, default=0.001)
    parser.add_argument('--pos_range_min', help="Specify the position range (min value). Default: 0", type=int,
                        default=0)
    parser.add_argument('--pos_range_max', help="Specify the position range (max value). Default: 0", type=int,
                        default=0)
    parser.add_argument('--print_status', help="Specify the print (0 for do not print, 1 for print). Default: 0",
                        type=int, default=0)
    parser.add_argument('--scuphr_strategy',
                        help="Specify the strategy for Scuphr (paired, singleton, hybrid). Default: paired",
                        type=str, default="paired")
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

    chains_dir = args.output_dir + "chains/"
    if not os.path.exists(chains_dir):
        os.makedirs(chains_dir)
        print("Chains directory is created")

    print("\nLoading the dataset...")

    dataset = load_json(proc_dict_dir + "data.txt")
    num_cells = len(dataset['0']['cell_list'])
    num_total_positions = len(dataset)
    print("The dataset is loaded. Number of cells: %d, number of position pairs: %d" % (num_cells, num_total_positions))

    if args.pos_range_max == 0:
        args.pos_range_max = num_total_positions

    print("\nPosition range: [", args.pos_range_min, ", ", args.pos_range_max, ")")
    print("Number of positions in range: ", args.pos_range_max - args.pos_range_min)

    # This part selects relevant sites to the Scuphr strategy.
    # hybrid accepts all sites & site-pairs,
    # paired accepts only the site-pairs, singleton accepts only the single-sites.
    print("Strategy: ", args.scuphr_strategy)
    positions_orig = np.arange(args.pos_range_min, args.pos_range_max)
    positions_paired = []
    positions_singleton = []
    for pos in positions_orig:
        cur_bulk = np.array(dataset[str(pos)]["bulk"])
        if cur_bulk.shape == (2, 2):
            positions_paired.append(pos)
        elif cur_bulk.shape == (2, ):
            positions_singleton.append(pos)
    print("\tTotal number of paired positions: \t", len(positions_paired))
    print("\tTotal number of singleton positions: \t", len(positions_singleton))

    if args.scuphr_strategy == "paired":
        positions = positions_paired
    elif args.scuphr_strategy == "singleton":
        positions = positions_singleton
    else:
        positions = positions_orig
    positions = np.array(positions)
    print("Number of valid positions in range: ", len(positions))
    print(positions)

    if len(positions) == 0:
        print("\nWARNING! There are no positions to estimate the parameters.")
        sys.exit()

    print("\n*****\nEstimating parameters...")
    '''
    output = mp.Queue()
    num_processes = max(1, int(mp.cpu_count() / args.num_chains))
    print("\tNumber of chains: \t", args.num_chains, "\tNumber of processes per chain: \t", num_processes)
    processes = [mp.Process(target=run_metropolis,
                            args=(num_processes, int(chain), dataset, output, read_prob_dir, chains_dir,
                                  args.max_iter, args.burnin, args.p_ae, args.p_ado, args.a_g, args.b_g, args.seed_val,
                                  args.print_status, positions)) for chain in range(args.num_chains)]

    for p in processes:
        p.daemon = False
        p.start()

    sys.stdout.flush()

    chain_results = []
    while 1:
        running = any(p.is_alive() for p in processes)
        while not output.empty():
            chain_results.append(output.get())
            sys.stdout.flush()
        if not running:
            break

    '''
    print("\tNumber of chains: \t", args.num_chains)
    chain_results = []
    for chain in range(args.num_chains):
        chain_results.append(run_metropolis(int(chain), dataset, read_prob_dir, chains_dir, args.max_iter,
                                            args.burnin, args.p_ae, args.p_ado, args.a_g, args.b_g, args.seed_val,
                                            args.print_status, positions))

    sys.stdout.flush()
    print("\nMetropolis-Hastings is done.")

    # Plot results
    for chain_idx in range(len(chain_results)):
        _, p_ado_samples, p_ae_samples, acceptance_samples, log_posterior_samples = chain_results[chain_idx]

        print("\nPlotting results for chain ", chain_idx)
        lag_list = [0, 2, 5, 10, 50, 100]
        plot_chain(chains_dir, chain_idx, p_ado_samples, p_ae_samples, acceptance_samples, log_posterior_samples,
                   lag_list)

        print("\tSaving results for chain ", chain_idx)
        filename = chains_dir + "chain_" + str(chain_idx) + "_p_ado_samples"
        np.save(filename, p_ado_samples)
        filename = chains_dir + "chain_" + str(chain_idx) + "_p_ae_samples"
        np.save(filename, p_ae_samples)
        filename = chains_dir + "chain_" + str(chain_idx) + "_acceptance_samples"
        np.save(filename, acceptance_samples)
        filename = chains_dir + "chain_" + str(chain_idx) + "_logposterior_samples"
        np.save(filename, log_posterior_samples)

    sys.stdout.flush()
    print("\nResult plotting and saving are done.")

    print("\nSampling paramters from Metropolis-Hastings chains...")
    mean_p_ado, mean_p_ae, std_p_ado, std_p_ae = sample_parameters(chain_results, args.max_iter)
    print("\tp_ado: \tmean: ", mean_p_ado, "\tstd: ", std_p_ado)
    print("\tp_ae: \tmean: ", mean_p_ae, "\tstd: ", std_p_ae)


if __name__ == "__main__":
    main()
