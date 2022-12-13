import numpy as np


def enumerate_fragment3(f1f2, dup=False):
    """
    This function enumerates all possible fragment3s based on fragment1 and fragment2.
    Input: f1f2: array (2,2), dup: boolean
    Output: fragment_list: array (numFragments,2)
    """
    f1 = f1f2[0]
    f2 = f1f2[1]

    f3_dict = {}
    for i in range(len(f1)):
        for k in range(3):
            f3_1 = np.copy(f1)
            f3_1[i] = (f3_1[i] + k + 1) % 4
            f3_dict["".join(str(aa) for aa in f3_1)] = f3_1

            f3_2 = np.copy(f2)
            f3_2[i] = (f3_2[i] + k + 1) % 4
            f3_dict["".join(str(aa) for aa in f3_2)] = f3_2

    if dup:
        f3_dict["".join(str(aa) for aa in f1)] = f1
        f3_dict["".join(str(aa) for aa in f2)] = f2

    f3_list = list(f3_dict.values())

    fragment_list = []
    for i in range(len(f3_list)):
        fragment_list.append([f1, f2, f3_list[i]])

    return fragment_list


def enumerate_distance1_genotype(ref_genotype):
    """
    This function enumerates all possible distance 1 genotypes based on given genotype.
    Input: ref_genotype: array (2,1)
    Output: genotype_list: array (numGenotypes,2)
    """
    genotype_list = []

    for i in range(ref_genotype.shape[0]):
        for j in range(ref_genotype.shape[1]):
            for k in range(3):
                temp = np.copy(ref_genotype)
                temp[i, j] = (temp[i, j] + k + 1) % 4
                genotype_list.append(temp)

    return genotype_list


def enumerate_fragments_possible(bulk, z):
    """
    This function enumerates all possible fragments based on bulk and mutation type genotypes.
    Input: Bulk: array (2,2), Z: array (2,2)
    Output: fragment_list_possible: array (numFragments,2)
    """
    fragment_list_possible = []

    fragment_list = enumerate_fragment3(bulk)
    for fragment in fragment_list:
        fragment_list_possible.append(fragment)

    fragment_list = enumerate_fragment3(z)
    for fragment in fragment_list:
        fragment_list_possible.append(fragment)

    return fragment_list_possible


def enumerate_fragments_possible_single_site(bulk, z):
    fragment_list = []

    for i in range(3):
        fragment_list.append([bulk[0], bulk[1], (bulk[0] + i + 1) % 4])

    for i in range(4):
        fragment_list.append([z[0], z[1], (z[0] + i) % 4])

    return fragment_list