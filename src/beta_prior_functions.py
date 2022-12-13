import math


def get_beta_prior_parameters(t, a, b):
    """
    The function returns beta prior parameters based on the total number of leaves.
    First, it calculates the expected number of leaves affected if there is a mutation on the tree.
    Then, based on the probability, it returns the parameters.
    Input: t: int, b: double
    Output: a: double, b: double
    """
    if a <= 0 or b <= 0:
        b = 5
        _, prob = get_expected_number(t)
        a = max(0.05, (prob * b) / (1 - prob))

    return a, b


def get_expected_number(t):
    """
    This function returns the expected number of mutated leaves, if there is one mutation in the tree.
    Input: t: int
    Output: exp_leaves: double, prob: double
    """
    edge_count = 2 * (t - 1)
    divisor = edge_count * get_tree_count(t)

    exp_leaves = 0
    for d in range(1, t):
        te_count = get_tree_edge_count(t, d)
        exp_leaves = exp_leaves + (d * te_count) / divisor

    prob = exp_leaves / t

    return exp_leaves, prob


def get_tree_count(t):
    """
    This function returns the total tree count based on the number of leaves.
    C(t) = (t-1)!
    Input: t: int
    Output: result: int
    """
    result = math.factorial(t - 1)
    return result


def get_tree_edge_count(t, d):
    """
    This function returns total number of tree-edge pairs.
    C(t,d) = 2(t)! / (d(d+1))
    Input: t: int, d: int
    Output: result: double
    """
    result = (2 * math.factorial(t)) / (d * (d + 1))
    return result
