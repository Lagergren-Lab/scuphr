import numpy as np


def get_lambda_mask(ado_stats, ae_stat=0):
    """
    This function generates a binary mask for lambdas based on dropout and amplification error statuses.
    Input: ado_stats: binary array (2,1), ae_stat: binary
    Output: lambda_mask: binary array (3,1)
    """
    lambda_mask = np.zeros(3)

    if ado_stats[0] == 1 and ado_stats[1] == 1:  # Both alleles have dropout
        lambda_mask = lambda_mask.astype(int)
        return lambda_mask

    if ado_stats[0] == 0:  # No dropout on first allele
        lambda_mask[0] = 1
    if ado_stats[1] == 0:  # No dropout on second allele
        lambda_mask[1] = 1

    if ae_stat == 1:  # Amplification error on one of the alleles
        lambda_mask[2] = 1

    lambda_mask = lambda_mask.astype(int)
    return lambda_mask


def partition_lambda(l, ado_stats, ae_stat=None):
    """
    This function partitions L into lambdas based on dropout and amplification error statuses.
    Note that it allows amplification error at root.
    Input: l: int, ado_stats: binary array (2,1), ae_stat: binary
    Output: lambda_list: binary array (numPossibleLambdas,3)
    """
    if ae_stat is not None:
        lambda_list = partition_lambda_default(l, ado_stats, ae_stat)

        # This part is for allowing the amplification error at root. Add extra cases to the general solution.
        if ae_stat == 1:
            if sum(ado_stats) == 0:
                new_lambdas = partition_lambda_with_mask(np.array([0, 1, 1]), l)
                for new_lambda in new_lambdas:
                    lambda_list.append(new_lambda)
                new_lambdas = partition_lambda_with_mask(np.array([1, 0, 1]), l)
                for new_lambda in new_lambdas:
                    lambda_list.append(new_lambda)
            elif sum(ado_stats) == 1:
                new_lambdas = partition_lambda_with_mask(np.array([0, 0, 1]), l)
                for new_lambda in new_lambdas:
                    lambda_list.append(new_lambda)
    else:
        lambda_list = partition_lambda(l, ado_stats, ae_stat=0)

        if l > 0:
            new_lambdas = partition_lambda(l, ado_stats, ae_stat=1)
            for new_lambda in new_lambdas:
                lambda_list.append(new_lambda)

    return lambda_list


def partition_lambda_default(l, ado_stats, ae_stat=0):
    """
    This function partitions L into lambdas based on dropout and amplification error statuses.
    This function is the default version.
    Input: l: int, ado_stats: binary array (2,1), ae_stat: binary
    Output: lambda_list: binary array (numPossibleLambdas,3)
    """
    lambda_mask = get_lambda_mask(ado_stats, ae_stat)
    lambda_list = partition_lambda_with_mask(lambda_mask, l)

    return lambda_list


def partition_lambda_with_mask(lambda_mask, l):
    """
    This function partitions L into lambdas based on given mask array.
    Input: lambda_mask: binary array (3,1), l: int
    Output: lambda_list: binary array (numPossibleLambdas,3)
    """
    group_idx = np.where(lambda_mask == 1)[0]
    lambda_list = []

    num_groups = len(group_idx)
    if num_groups > l:
        # print("Invalid. Number of groups is bigger than read count.")
        return lambda_list

    if num_groups == 1:  # L is assigned to the only active position. There is only one possible scenario. [(L,0,0)]
        lambda_ = np.zeros(3)
        lambda_[group_idx[0]] = l
        lambda_ = lambda_.astype(int)
        lambda_list.append(lambda_)

    # L is partitioned to two positions. There are L-1 possible scenarios. [(1,L-1,0),(2,L-2,0),...,(L-1,1,0)]
    elif num_groups == 2:
        for i in range(l - 1):
            lambda_ = np.zeros(3)
            lambda_[group_idx[0]] = i + 1
            lambda_[group_idx[1]] = l - 1 - i
            lambda_ = lambda_.astype(int)
            lambda_list.append(lambda_)

    # L is partitioned to three positions. [(1,1,L-2),(1,2,L-3),...,(1,L-2,1),(2,1,L-3),...,(2,L-3,1),...,(L-2,1,1)]
    elif num_groups == 3:
        for i in range(l - 1):
            for j in range(l - 1 - i - 1):
                lambda_ = np.zeros(3)
                lambda_[group_idx[0]] = i + 1
                lambda_[group_idx[1]] = j + 1
                lambda_[group_idx[2]] = l - (i + j + 2)
                lambda_ = lambda_.astype(int)
                lambda_list.append(lambda_)
    # elif num_groups == 0 and L==0:
    #    lambda_ = np.zeros(3)
    #    lambda_ = lambda_.astype(int)
    #    lambda_list.append(lambda_)

    return lambda_list


def partition_lambda_all(l):
    """
    This function partitions L into lambdas in every possible way.
    Input: l: int
    Output: lambda_list: binary array (numPossibleLambdas,3)
    """
    lambdas = []

    ado_stats_list = [[0, 0], [0, 1], [1, 0]]
    ae_stat_list = [0, 1]

    for i in range(len(ado_stats_list)):
        ado_stats = ado_stats_list[i]
        for j in range(len(ae_stat_list)):
            ae_stat = ae_stat_list[j]
            lambdas.append(partition_lambda(l, ado_stats, ae_stat))

    lambda_list = [item for sublist in lambdas for item in sublist]
    return lambda_list
