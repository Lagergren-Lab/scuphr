import os
import argparse
from ete3 import Tree


def create_taxa_list(num_cells):
    """ Creates the taxa lists used by SciPhi. TODO Read from cell names file"""

    taxa_list = []
    for i in range(num_cells):
        taxa_list.append("cell_idx_" + str(i))
    return taxa_list


def create_information_map(filename, taxa_list):
    """ Creates an information map dictionary,
    which maps the node ids to their taxa and the number of mutations on incoming edges. """

    info_map = {}
    total_muts = 0

    with open(filename) as fp:
        for line in fp:
            if "[" in line:
                node_id, temp_contents = line.rsplit("[")
                a = temp_contents.rsplit('"')
                branch_contents = a[1].rsplit('\\n')

                num_n = a[1].count('\\n')
                label = ""
                if branch_contents[0] in taxa_list:
                    num_n -= 1
                    label = branch_contents[0]

                info_map[node_id] = {"label": label, "incoming_muts": num_n}
                total_muts += num_n

    for key in info_map.keys():
        info_map[key]["normed_incoming_muts"] = float(info_map[key]["incoming_muts"]) / total_muts

    print("\tTotal number of mutations: \t", total_muts)
    return info_map


def create_branch_files(filename, info_map, mid_filename, mid_filename_normed):
    """ Creates the parent-child branch file in tab delimited format.
    It also saves a normalised branch length version as well. """

    mid_file = open(mid_filename, 'w')
    mid_file_normed = open(mid_filename_normed, 'w')

    with open(filename) as fp:
        for line in fp:
            if "->" in line:
                # print("Line: ", line)
                temp_line = line.rsplit(" ")
                parent, child = temp_line[0].rsplit("->")

                incoming_muts = info_map[child]["incoming_muts"]
                normed_incoming_muts = info_map[child]["normed_incoming_muts"]

                if info_map[child]["label"] != "":
                    # child = info_map[child]["label"] # for labels such as cell_idx_0, cell_idx_1 etc

                    child = "V" + info_map[child]["label"].rsplit("_")[-1]

                mid_file.write(parent + "\t" + child + "\t" + str(incoming_muts) + "\n")
                mid_file_normed.write(parent + "\t" + child + "\t" + str(normed_incoming_muts) + "\n")

    mid_file.close()
    mid_file_normed.close()


def create_newick_files(out_filename, out_filename_normed, mid_filename, mid_filename_normed):
    """ Reads the parent-child branch files and saves the trees in Newick format. """

    t = Tree.from_parent_child_table([line.split() for line in open(mid_filename)])
    t.write(format=5, outfile=out_filename)

    t = Tree.from_parent_child_table([line.split() for line in open(mid_filename_normed)])
    t.write(format=5, outfile=out_filename_normed)


def main():
    # code to process command line arguments
    parser = argparse.ArgumentParser(description='SciPhi output parser.')
    parser.add_argument('input_filepath', help="Specify the input file path (.gv file).", type=str)
    parser.add_argument('num_cells', help="Specify the number of cells.", type=int)
    parser.add_argument('--output_filepath', help="Specify the output file path. Default: input_filepath_processed.tre",
                        type=str, default="")
    parser.add_argument('--output_filepath_normed',
                        help="Specify the output file path for normalised tree. "
                             "Default: input_filepath_processed_normed.tre", type=str, default="")
    args = parser.parse_args()

    if args.output_filepath == "":
        args.output_filepath = args.input_filepath + "_processed.tre"
    if args.output_filepath_normed == "":
        args.output_filepath_normed = args.input_filepath + "_processed_normed.tre"

    print("\nSTEP 1. Creating taxa list...")
    taxa_list = create_taxa_list(args.num_cells)

    print("\nSTEP 2. Creating the information mapping...")
    info_map = create_information_map(args.input_filepath, taxa_list)

    print("\nSTEP 3. Creating the parent-child files...")
    mid_filename = args.input_filepath + "mid.txt"
    mid_filename_normed = args.input_filepath + "mid_normed.txt"
    create_branch_files(args.input_filepath, info_map, mid_filename, mid_filename_normed)

    print("\nSTEP 4. Creating the ete3 trees and saving Newick files...")
    create_newick_files(args.output_filepath, args.output_filepath_normed, mid_filename, mid_filename_normed)

    print("\nSTEP 5. Cleaning intermediate files...")
    cmd = "rm " + mid_filename
    os.system(cmd)
    cmd = "rm " + mid_filename_normed
    os.system(cmd)


if __name__ == "__main__":
    main()
