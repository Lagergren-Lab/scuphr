import csv
import argparse
import dendropy
import datetime
import numpy as np


def compare_trees(tree_sample1, tree_sample2):
    """
    This function compares two trees based on different distance criteria.
    Input: tree_sample1: dendropy.Tree object, tree_sample2: dendropy.Tree object
    Output: rf_dist: double, n_rf_dist: double, weighted_rf_dist: double, n_weighted_rf_dist: double, euc_dist: double
    """
    num_leaf = len(tree_sample1.leaf_nodes())
    num_edges = 2 * (num_leaf - 1)

    # Robinson-Foulds Distance
    rf_dist = dendropy.calculate.treecompare.unweighted_robinson_foulds_distance(tree_sample1, tree_sample2)
    print("R-F: ", rf_dist)

    # Normalised Robinson-Foulds Distance
    n_rf_dist = rf_dist / (2 * num_edges)
    print("Normalised R-F: ", n_rf_dist)

    # Weighted Robinson-Foulds Distance
    weighted_rf_dist = dendropy.calculate.treecompare.weighted_robinson_foulds_distance(tree_sample1, tree_sample2)
    print("Weighted R-F: ", weighted_rf_dist)

    # Normalised Weighted Robinson-Foulds Distance
    n_weighted_rf_dist = weighted_rf_dist / (2 * num_edges)
    print("Normalised Weighted R-F: ", n_weighted_rf_dist)

    # Euclidean Distance
    euc_dist = dendropy.calculate.treecompare.euclidean_distance(tree_sample1, tree_sample2)
    print("Euc: ", euc_dist)

    return rf_dist, n_rf_dist, weighted_rf_dist, n_weighted_rf_dist, euc_dist


def write_trees(output_filename, real_newick_filename, sciphi_newick_filename,  scuphr_newick_filename):
    file = open(output_filename, "w")

    tns = dendropy.TaxonNamespace()
    sciphi_tree = dendropy.Tree.get(file=open(sciphi_newick_filename), schema="newick", taxon_namespace=tns,
                                    rooting="default-rooted")
    file.write("\nSciphi tree: \n")
    file.write(sciphi_tree.as_ascii_plot())
    file.write(sciphi_tree.as_string(schema="newick"))

    if real_newick_filename != "":
        real_tree = dendropy.Tree.get(file=open(real_newick_filename), schema="newick", taxon_namespace=tns,
                                      rooting="default-rooted")
        file.write("\nReal tree: \n")
        file.write(real_tree.as_ascii_plot())
        file.write(real_tree.as_string(schema="newick"))

    if scuphr_newick_filename != "":
        scuphr_tree = dendropy.Tree.get(file=open(scuphr_newick_filename), schema="newick", taxon_namespace=tns,
                                        rooting="default-rooted")
        file.write("\nScuphr tree: \n")
        file.write(scuphr_tree.as_ascii_plot())
        file.write(scuphr_tree.as_string(schema="newick"))

    file.write("\nTree Comparisons\n")

    if real_newick_filename != "":
        real_tree = dendropy.Tree.get(file=open(real_newick_filename), schema="newick", taxon_namespace=tns,
                                      rooting="default-rooted")

        rf_dist, n_rf_dist, weighted_rf_dist, n_weighted_rf_dist, euc_dist = compare_trees(real_tree, sciphi_tree)
        file.write("\nReal tree vs Sciphi tree: \n")
        file.write("\nR-F:\n")
        file.write(str(rf_dist))
        file.write("\nNormalised R-F:\n")
        file.write(str(n_rf_dist))
        file.write("\nWeighted R-F:\n")
        file.write(str(weighted_rf_dist))
        file.write("\nNormalised Weighted R-F:\n")
        file.write(str(n_weighted_rf_dist))
        file.write("\nEuc:\n")
        file.write(str(euc_dist))

    if scuphr_newick_filename != "":
        scuphr_tree = dendropy.Tree.get(file=open(scuphr_newick_filename), schema="newick", taxon_namespace=tns,
                                        rooting="default-rooted")

        if real_newick_filename != "":
            rf_dist, n_rf_dist, weighted_rf_dist, n_weighted_rf_dist, euc_dist = compare_trees(real_tree, scuphr_tree)
            file.write("\nReal tree vs Scuphr tree: \n")
            file.write("\nR-F:\n")
            file.write(str(rf_dist))
            file.write("\nNormalised R-F:\n")
            file.write(str(n_rf_dist))
            file.write("\nWeighted R-F:\n")
            file.write(str(weighted_rf_dist))
            file.write("\nNormalised Weighted R-F:\n")
            file.write(str(n_weighted_rf_dist))
            file.write("\nEuc:\n")
            file.write(str(euc_dist))

        rf_dist, n_rf_dist, weighted_rf_dist, n_weighted_rf_dist, euc_dist = compare_trees(sciphi_tree, scuphr_tree)
        file.write("\nSciphi tree vs Scuphr tree: \n")
        file.write("\nR-F:\n")
        file.write(str(rf_dist))
        file.write("\nNormalised R-F:\n")
        file.write(str(n_rf_dist))
        file.write("\nWeighted R-F:\n")
        file.write(str(weighted_rf_dist))
        file.write("\nNormalised Weighted R-F:\n")
        file.write(str(n_weighted_rf_dist))
        file.write("\nEuc:\n")
        file.write(str(euc_dist))

    file.close()


def check_mutation_statistics(sciphi_tsv, data_dir):
    real_mut_locations = np.loadtxt(data_dir + "truth/mut_locations.txt", dtype=int)

    tsv_file = open(sciphi_tsv)
    read_tsv = csv.reader(tsv_file, delimiter="\t")

    correct_muts = []
    incorrect_muts = []

    row_idx = 0
    for row in read_tsv:
        if row_idx > 0:
            mut_array_idx = int(row[1])-1
            if mut_array_idx in real_mut_locations:
                correct_muts.append(mut_array_idx)
            else:
                incorrect_muts.append(mut_array_idx)
        row_idx += 1
    tsv_file.close()

    print("\tTotal number of mutations: ", row_idx-1)
    print("\tTotal number of correct mutations: ", len(correct_muts))
    print("\t", sorted(correct_muts))
    print("\tTotal number of incorrect mutations: ", len(incorrect_muts))
    print("\t", sorted(incorrect_muts))


def main():
    dt = datetime.datetime.now()
    default_dir = "../../data/%s_%s_%s/" % (dt.year, dt.month, dt.day)

    # code to process command line arguments
    parser = argparse.ArgumentParser(description='Sciphi tree RF calculation.')
    parser.add_argument('sciphi_newick', help="Specify the Newick file of Sciphi tree.", type=str)
    parser.add_argument('result_filename', help="Specify the output file.", type=str)
    parser.add_argument('--real_newick', help="Specify the Newick file of true tree.", type=str, default="")
    parser.add_argument('--scuphr_newick', help="Specify the Newick file of Sciphi tree.", type=str, default="")
    parser.add_argument('--sciphi_tsv', help="Specify the mut2Sample.tsv file of Sciphi.", type=str, default="")
    parser.add_argument('--data_dir', help="Specify the data directory", type=str, default=default_dir)
    args = parser.parse_args()

    print("\nSTEP 1: Comparing trees")
    tns = dendropy.TaxonNamespace()

    if args.real_newick != "":
        print("Comparing Sciphi tree and real tree: ")
        tree_sample1 = dendropy.Tree.get(file=open(args.sciphi_newick), schema="newick", taxon_namespace=tns,
                                         rooting="default-rooted")
        tree_sample2 = dendropy.Tree.get(file=open(args.real_newick), schema="newick", taxon_namespace=tns,
                                         rooting="default-rooted")
        rf_dist, n_rf_dist, weighted_rf_dist, n_weighted_rf_dist, euc_dist = compare_trees(tree_sample1, tree_sample2)

    if args.scuphr_newick != "":
        print("Comparing Sciphi tree and Scuphr tree: ")
        tree_sample2 = dendropy.Tree.get(file=open(args.scuphr_newick), schema="newick", taxon_namespace=tns,
                                         rooting="default-rooted")
        rf_dist, n_rf_dist, weighted_rf_dist, n_weighted_rf_dist, euc_dist = compare_trees(tree_sample1, tree_sample2)

        print("Comparing real tree and Scuphr tree: ")
        tree_sample1 = dendropy.Tree.get(file=open(args.real_newick), schema="newick", taxon_namespace=tns,
                                         rooting="default-rooted")
        rf_dist, n_rf_dist, weighted_rf_dist, n_weighted_rf_dist, euc_dist = compare_trees(tree_sample1, tree_sample2)

    print("\nSTEP 2: Saving results")
    write_trees(args.result_filename, args.real_newick, args.sciphi_newick, args.scuphr_newick)

    if args.sciphi_tsv != "":
        print("\nSTEP 3: Checking Sciphi's mutation statistics")
        check_mutation_statistics(args.sciphi_tsv, args.data_dir)


if __name__ == "__main__":
    main()
