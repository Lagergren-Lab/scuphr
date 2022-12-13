import json
import argparse
import numpy as np
from ete3 import Tree


def load_json(filename):
    with open(filename) as fp:
        cell_dict = json.load(fp)
    return cell_dict


def create_branch_files(truth_dir):
    """ Creates the parent-child branch file in tab delimited format.
    It also saves a normalised branch length version as well. """

    try:
        parent_nodes = np.array(load_json(truth_dir + "parent_nodes.txt"))
        leaf_nodes = np.array(load_json(truth_dir + "leaf_nodes.txt"))
        mut_origin_nodes = np.array(load_json(truth_dir + "mut_origin_nodes.txt"))
    except:
        parent_nodes = np.loadtxt(truth_dir + "parent_nodes.txt",)
        leaf_nodes = np.loadtxt(truth_dir + "leaf_nodes.txt")
        mut_origin_nodes = np.loadtxt(truth_dir + "mut_origin_nodes.txt")

    mid_filename = truth_dir + "real_mid.txt"
    mid_filename_normed = truth_dir + "real_mid_normed.txt"

    mid_file = open(mid_filename, 'w')
    mid_file_normed = open(mid_filename_normed, 'w')

    total_muts = len(mut_origin_nodes)

    for parent in np.unique(parent_nodes).astype(int):
        children = np.where(parent_nodes == parent)[0]

        for child in children:
            if child != parent:
                child_muts = len(np.where(mut_origin_nodes == child)[0])

                if child not in leaf_nodes:
                    stri = str(parent) + "\t" + str(child) + "\t" + str(child_muts) + "\n"
                    normed_stri = str(parent) + "\t" + str(child) + "\t" + str(child_muts / total_muts) + "\n"
                else:
                    stri = str(parent) + "\t" + "V" + str(np.where(leaf_nodes == child)[0][0]) + "\t" + str(
                        child_muts) + "\n"
                    normed_stri = str(parent) + "\t" + "V" + str(np.where(leaf_nodes == child)[0][0]) + "\t" + str(
                        child_muts / total_muts) + "\n"
                # print(stri)

                mid_file.write(stri)
                mid_file_normed.write(normed_stri)

    mid_file.close()
    mid_file_normed.close()
    return mid_filename, mid_filename_normed


def create_newick_files(out_filename, out_filename_normed, mid_filename, mid_filename_normed):
    """ Reads the parent-child branch files and saves the trees in Newick format. """

    t = Tree.from_parent_child_table([line.split() for line in open(mid_filename)])
    t.write(format=5, outfile=out_filename)

    t = Tree.from_parent_child_table([line.split() for line in open(mid_filename_normed)])
    t.write(format=5, outfile=out_filename_normed)


def main():
    # code to process command line arguments
    parser = argparse.ArgumentParser(description='Scuphr real tree Newick generator.')
    parser.add_argument('data_dir', help="Specify the data directory.", type=str)
    parser.add_argument('results_dir', help="Specify the results directory.", type=str)
    args = parser.parse_args()

    truth_dir = args.data_dir + "truth/"

    print("\nSTEP 1. Creating the parent-child files...")
    mid_filename, mid_filename_normed = create_branch_files(truth_dir)

    print("\nSTEP 2. Creating the ete3 trees and saving Newick files...")
    create_newick_files(truth_dir + "real.tre", truth_dir + "real_normed.tre", mid_filename, mid_filename_normed)
    create_newick_files(args.results_dir + "real.tre", args.results_dir + "real_normed.tre", mid_filename,
                        mid_filename_normed)


if __name__ == "__main__":
    main()
