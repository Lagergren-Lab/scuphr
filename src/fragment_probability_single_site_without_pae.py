import sys
import math
import numpy as np


def get_edge_stats(t, d):
    if d == t:
        return 1
    elif t > d > 0:
        return (2 * t) / (d * (d + 1))
    else:
        print("\nError: Invalid d, t.", d, t)
        sys.exit()


def get_log_factorial(t):
    """
    This function calculates the logarithm of factorial, log(t!).
    Input: t: int
    Output: val: double
    """
    val = np.log(1)
    for n in range(1, t + 1):
        val = val + np.log(n)
    return val


def tree_stats(t):
    num_unique_trees = math.factorial(t - 1)
    num_edges_per_tree = 2 * (t - 1) + 1
    num_total_edges = num_unique_trees * num_edges_per_tree
    return num_unique_trees, num_edges_per_tree, num_total_edges


def get_edge_stats_prev(t, d):
    num_unique_trees, num_edges_per_tree, num_total_edges = tree_stats(t)

    if t > d > 0:
        num_pair = (2 * math.factorial(t)) / (d * (d + 1))
    elif d == t:
        num_pair = num_unique_trees

    rho_pair = num_pair / num_total_edges
    rho_per_tree = num_pair / num_unique_trees
    return num_pair, rho_pair, rho_per_tree


def get_edge_stats_simple(t, d):
    # returns num_pair, rho_pair, rho_per_tree
    if d == t:
        return 1, 1, 1
    elif t > d > 0:
        return 1, 1, ((2 * t) / (d * (d + 1)))
    else:
        print("\nError: Invalid d, t.", d, t)
        sys.exit()


def dist(fragment_0, fragment_1):
    return fragment_0.size - (fragment_0 == fragment_1).sum()


def fragment_log_probability(fragments, lambdas, cell_genotype, ado_stats):
    log_prob = np.NINF

    # check fragment 1 fragment 2 probabilities first!
    fragments = np.array(fragments)
    #print("frags: ", fragments, " cell_gen: ", cell_genotype)
    if fragments[0] != cell_genotype[0] or fragments[1] != cell_genotype[1]:
        return log_prob

    # then, direct to corresponding function
    # if d0 == 0 and d1 == 1:
    if np.array_equal(ado_stats, np.array([0, 1])):
        log_prob = get_fragment_log_probability_ado_second(fragments, lambdas)
    # elif d0 == 1 and d1 == 0:
    elif np.array_equal(ado_stats, np.array([1, 0])):
        log_prob = get_fragment_log_probability_ado_first(fragments, lambdas)
    elif np.array_equal(ado_stats, np.array([0, 0])):
        # elif d0 == 0 and d1 == 0:
        log_prob = get_fragment_log_probability_no_ado(fragments, lambdas)

    return log_prob


def get_fragment_log_probability_ado_second(fragments, lambdas):
    mask_lambdas = np.copy(lambdas)
    mask_lambdas[mask_lambdas > 0] = 1

    lc = sum(lambdas)
    log_prob = np.NINF

    if np.array_equal(mask_lambdas, np.array([1, 0, 0])):  # no ae: (lambda1,0,0)
        log_prob = np.log(1)
    elif np.array_equal(mask_lambdas, np.array([0, 0, 1])):  # ae on root of first: (0,0,lambda3)
        if dist(fragments[0], fragments[2]) == 1:
            rho = get_edge_stats(lambdas[2], lambdas[2])
            log_prob = np.log(1 / 3) + np.log(rho)
    elif np.array_equal(mask_lambdas, np.array([1, 0, 1])):  # ae on inner of first: (lambda1,0,lambda3)
        if dist(fragments[0], fragments[2]) == 1:
            rho = get_edge_stats(lc, lambdas[2])
            log_prob = np.log(1 / 3) + np.log(rho)

    return log_prob


def get_fragment_log_probability_ado_first(fragments, lambdas):
    mask_lambdas = np.copy(lambdas)
    mask_lambdas[mask_lambdas > 0] = 1

    lc = sum(lambdas)
    log_prob = np.NINF

    if np.array_equal(mask_lambdas, np.array([0, 1, 0])):  # no ae: (0,lambda2,0)
        log_prob = np.log(1)
    elif np.array_equal(mask_lambdas, np.array([0, 0, 1])):  # ae on root of second: (0,0,lambda3)
        if dist(fragments[1], fragments[2]) == 1:
            rho = get_edge_stats(lambdas[2], lambdas[2])
            log_prob = np.log(1 / 3) + np.log(rho)
    elif np.array_equal(mask_lambdas, np.array([0, 1, 1])):  # ae on inner of second: (0,lambda2,lambda3)
        if dist(fragments[1], fragments[2]) == 1:
            rho = get_edge_stats(lc, lambdas[2])
            log_prob = np.log(1 / 3) + np.log(rho)

    return log_prob


def get_fragment_log_probability_no_ado(fragments, lambdas):
    mask_lambdas = np.copy(lambdas)
    mask_lambdas[mask_lambdas > 0] = 1

    lc = sum(lambdas)
    log_prob = np.NINF

    if np.array_equal(mask_lambdas, np.array([1, 1, 0])):  # no ae: (lambda1,lambda2,0)
        log_prob = np.log(1 / (lc - 1))

    elif np.array_equal(mask_lambdas, np.array([0, 1, 1])):  # ae on root of first: (0,lambda2,lambda3)
        if dist(fragments[0], fragments[2]) == 1:
            rho = get_edge_stats(lambdas[2], lambdas[2])
            log_prob = np.log(1 / (lc - 1)) + np.log(1 / 3) + np.log(rho)

    elif np.array_equal(mask_lambdas, np.array([1, 0, 1])):  # ae on root of second: (lambda1,0,lambda3)
        if dist(fragments[1], fragments[2]) == 1:
            rho = get_edge_stats(lambdas[2], lambdas[2])
            log_prob = np.log(1 / (lc - 1)) + np.log(1 / 3) + np.log(rho)

    elif np.array_equal(mask_lambdas, np.array([1, 1, 1])):  # ae on inner of one of the trees: (lambda1,lambda2,lambda3)

        if dist(fragments[0], fragments[2]) == 1 and dist(fragments[1], fragments[2]) != 1:  # ae on first
            rho = get_edge_stats(lambdas[0] + lambdas[2], lambdas[2])
            log_prob = np.log(1 / (lc - 1)) + np.log(1 / 3) + np.log(rho)

        elif dist(fragments[0], fragments[2]) != 1 and dist(fragments[1], fragments[2]) == 1:  # ae on second
            rho = get_edge_stats(lambdas[1] + lambdas[2], lambdas[2])
            log_prob = np.log(1 / (lc - 1)) + np.log(1 / 3) + np.log(rho)

        elif dist(fragments[0], fragments[2]) == 1 and dist(fragments[1], fragments[2]) == 1:  # ae on either first or second
            rho_0 = get_edge_stats(lambdas[0] + lambdas[2], lambdas[2])
            rho_1 = get_edge_stats(lambdas[1] + lambdas[2], lambdas[2])
            rho = rho_0 + rho_1
            log_prob = np.log(1 / (lc - 1)) + np.log(1 / 3) + np.log(rho)  # + np.log(2)

    return log_prob
