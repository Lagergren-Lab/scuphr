import argparse
import dendropy
import numpy as np


def generate_lineage_tree(filename, tns):
    """
    This function generates the cell lineage tree given a taxon namespace. We need this namespace to have the same id
    for cells.
    Input: tns: dendropy.TaxonNamespace object
    Output: tree_sample: dendropy.Tree object, nj_tree_sample: dendropy.Tree object, nj_tree_sample_str: string
    """
    pdm = dendropy.PhylogeneticDistanceMatrix.from_csv(src=open(filename), delimiter=",",
                                                       is_first_row_column_names=False, is_first_column_row_names=False)

    nj_tree_sample = pdm.nj_tree()
    nj_tree_sample_str = nj_tree_sample.as_string("newick")

    # Generate tree based on distance matrices
    tree_sample = dendropy.Tree.get(data=nj_tree_sample_str, schema="newick", taxon_namespace=tns,
                                    rooting="force-rooted")
    return tree_sample, nj_tree_sample, nj_tree_sample_str


def compare_trees(dir_info, case_name1, case_name2, condition_name, tns):
    """
    This function compares two trees based on different distance criteria.
    Input: tree_sample1: dendropy.Tree object, tree_sample2: dendropy.Tree object
    Output: rf_dist: double, n_rf_dist: double, weighted_rf_dist: double, n_weighted_rf_dist: double, euc_dist: double
    """
    filename = dir_info + case_name1 + condition_name
    tree_sample1 = dendropy.Tree.get(file=open(filename), schema="newick", taxon_namespace=tns)

    if case_name2 != "real":
        filename = dir_info + case_name2 + condition_name
        tree_sample2 = dendropy.Tree.get(file=open(filename), schema="newick", taxon_namespace=tns)
    else:
        filename = dir_info + case_name2 + ".tre"
    tree_sample2 = dendropy.Tree.get(file=open(filename), schema="newick", taxon_namespace=tns, rooting="default-rooted")

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


def save_different_trees(dir_info, csv_filename, case_name, tns):  # case_name = "global", "selected", "inferred"
    filename = dir_info + csv_filename

    tree_w_bulk_node, _, _ = generate_lineage_tree(filename, tns)
    new_filename = dir_info + case_name + "_w_bulk_node.tre"
    tree_w_bulk_node.write(file=open(new_filename, 'w'), schema="newick")

    tree_w_bulk_root = set_bulk_to_root(tree_w_bulk_node)
    new_filename = dir_info + case_name + "_w_bulk_root.tre"
    tree_w_bulk_root.write(file=open(new_filename, 'w'), schema="newick")

    # tree_w_bulk_midpoint = tree_w_bulk_root.reroot_at_midpoint(update_bipartitions=False)
    tree_w_bulk_root.reroot_at_midpoint(update_bipartitions=False)
    new_filename = dir_info + case_name + "_w_bulk_root_midpoint.tre"
    tree_w_bulk_root.write(file=open(new_filename, 'w'), schema="newick")

    dist_mat_temp = np.loadtxt(open(filename, "rb"), delimiter=",")
    dist_mat_temp = dist_mat_temp[0:dist_mat_temp.shape[0] - 1, 0:dist_mat_temp.shape[0] - 1]
    filename = dir_info + csv_filename + "_wo_bulk"
    np.savetxt(filename, dist_mat_temp, delimiter=',', fmt="%.10f")

    tree_wo_bulk, _, _ = generate_lineage_tree(filename, tns)
    new_filename = dir_info + case_name + "_wo_bulk.tre"
    tree_wo_bulk.write(file=open(new_filename, 'w'), schema="newick")

    # tree_wo_bulk_midpoint = tree_wo_bulk.reroot_at_midpoint(update_bipartitions=False)
    tree_wo_bulk.reroot_at_midpoint(update_bipartitions=False)
    new_filename = dir_info + case_name + "_wo_bulk_midpoint.tre"
    tree_wo_bulk.write(file=open(new_filename, 'w'), schema="newick")


def print_trees(dir_info, case_name):
    print("\nTrees of case: ", case_name, "\n")

    filename = dir_info + case_name + "_w_bulk_node.tre"
    tree = dendropy.Tree.get(file=open(filename), schema="newick")  # ,tree_offset=0)
    print("\n\tCase: ", case_name, ".\tTree with bulk as a node:\n")
    print(tree.as_ascii_plot())
    print(tree.as_string(schema="newick"))

    filename = dir_info + case_name + "_w_bulk_root.tre"
    tree = dendropy.Tree.get(file=open(filename), schema="newick")
    print("\n\tCase: ", case_name, ".\tTree with bulk as a root:\n")
    print(tree.as_ascii_plot())
    print(tree.as_string(schema="newick"))

    filename = dir_info + case_name + "_w_bulk_root_midpoint.tre"
    tree = dendropy.Tree.get(file=open(filename), schema="newick")
    print("\n\tCase: ", case_name, ".\tTree with bulk, midpoint rooted:\n")
    print(tree.as_ascii_plot())
    print(tree.as_string(schema="newick"))

    filename = dir_info + case_name + "_wo_bulk.tre"
    tree = dendropy.Tree.get(file=open(filename), schema="newick")
    print("\n\tCase: ", case_name, ".\tTree without bulk:\n")
    print(tree.as_ascii_plot())
    print(tree.as_string(schema="newick"))

    filename = dir_info + case_name + "_wo_bulk_midpoint.tre"
    tree = dendropy.Tree.get(file=open(filename), schema="newick")
    print("\n\tCase: ", case_name, ".\tTree without bulk, midpoint rooted:\n")
    print(tree.as_ascii_plot())
    print(tree.as_string(schema="newick"))


def write_trees(dir_info, case_name, file):
    file.write("\nTrees of case: %s \n" % case_name)

    filename = dir_info + case_name + "_w_bulk_node.tre"
    tree = dendropy.Tree.get(file=open(filename), schema="newick")  # ,tree_offset=0)
    file.write("\n\tCase: %s.\tTree with bulk as a node:\n" % case_name)
    file.write(tree.as_ascii_plot())
    file.write(tree.as_string(schema="newick"))

    filename = dir_info + case_name + "_w_bulk_root.tre"
    tree = dendropy.Tree.get(file=open(filename), schema="newick")
    file.write("\n\tCase: %s.\tTree with bulk as a root:\n" % case_name)
    file.write(tree.as_ascii_plot())
    file.write(tree.as_string(schema="newick"))

    filename = dir_info + case_name + "_w_bulk_root_midpoint.tre"
    tree = dendropy.Tree.get(file=open(filename), schema="newick")
    file.write("\n\tCase: %s.\tTree with bulk, midpoint rooted:\n" % case_name)
    file.write(tree.as_ascii_plot())
    file.write(tree.as_string(schema="newick"))

    filename = dir_info + case_name + "_wo_bulk.tre"
    tree = dendropy.Tree.get(file=open(filename), schema="newick")
    file.write("\n\tCase: %s.\tTree without bulk:\n" % case_name)
    file.write(tree.as_ascii_plot())
    file.write(tree.as_string(schema="newick"))

    filename = dir_info + case_name + "_wo_bulk_midpoint.tre"
    tree = dendropy.Tree.get(file=open(filename), schema="newick")
    file.write("\n\tCase: %s.\tTree without bulk, midpoint rooted:\n" % case_name)
    file.write(tree.as_ascii_plot())
    file.write(tree.as_string(schema="newick"))


def generate_and_analyse_lineage_tree(output_dir, data_type):
    """
    This function generates real and inferred lineage trees and compares them.
    Input: dataset_key: string
    """
    tns = dendropy.TaxonNamespace()

    fname = output_dir + "LineageTrees.txt"
    file = open(fname, "w")

    if data_type == "synthetic":  # If the dataset is synthetic
        # Save real trees
        if False:
            csv_filename = "global_dif.csv"
            case_name = "global"
            save_different_trees(output_dir, csv_filename, case_name, tns)
            print_trees(output_dir, case_name)
            write_trees(output_dir, case_name, file)

        case_name2 = "real"
        filename = output_dir + case_name2 + ".tre"
        tree_sample2 = dendropy.Tree.get(file=open(filename), schema="newick", taxon_namespace=tns, rooting="default-rooted")
        print("\n\tCase: ", case_name2, "\n")
        print(tree_sample2.as_ascii_plot())
        print(tree_sample2.as_string(schema="newick"))

        file.write("\n\tCase: %s.\t\n" % case_name2)
        file.write(tree_sample2.as_ascii_plot())
        file.write(tree_sample2.as_string(schema="newick"))

        csv_filename = "real_dif.csv"
        case_name = "selected"
        save_different_trees(output_dir, csv_filename, case_name, tns)
        print_trees(output_dir, case_name)
        write_trees(output_dir, case_name, file)

    # Save inferred trees
    csv_filename = "infer_dif.csv"
    case_name = "inferred"
    save_different_trees(output_dir, csv_filename, case_name, tns)
    print_trees(output_dir, case_name)
    write_trees(output_dir, case_name, file)

    try:
        # Compare with the true tree
        case_name2 = "real"
        condition_name = "_w_bulk_root.tre"
        title = "\n\nComparison Results for " + case_name + " and " + case_name2 + " for " + condition_name
        print(title)
        rf_dist, n_rf_dist, weighted_rf_dist, n_weighted_rf_dist, euc_dist = compare_trees(
            output_dir, case_name, case_name2, condition_name, tns)

        file.write(title)
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
    except:
        print("Cannot find real.tre")

    if data_type == "synthetic":  # If the dataset is synthetic
        case_name1 = "inferred"

        # case_name2_list = ["global", "selected"]
        case_name2_list = ["selected"]
        condition_name_list = ["_w_bulk_root.tre", "_w_bulk_root_midpoint.tre", "_wo_bulk.tre", "_wo_bulk_midpoint.tre"]

        for case_name2 in case_name2_list:
            for condition_name in condition_name_list:
                title = "\n\nComparison Results for " + case_name1 + " and " + case_name2 + " for " + condition_name
                print(title)
                rf_dist, n_rf_dist, weighted_rf_dist, n_weighted_rf_dist, euc_dist = compare_trees(output_dir,
                                                                                                   case_name1,
                                                                                                   case_name2,
                                                                                                   condition_name, tns)

                file.write(title)
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

        # Compare selected NJ with the true tree
        case_name2 = "real"
        temp_case_name1 = "selected"
        condition_name = "_w_bulk_root.tre"
        title = "\n\nComparison Results for " + temp_case_name1 + " and " + case_name2 + " for " + condition_name
        print(title)
        rf_dist, n_rf_dist, weighted_rf_dist, n_weighted_rf_dist, euc_dist = compare_trees(output_dir,
                                                                                           temp_case_name1,
                                                                                           case_name2,
                                                                                           condition_name, tns)

        file.write(title)
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

        # Compare with the true tree
        case_name2 = "real"
        condition_name = "_w_bulk_root.tre"
        title = "\n\nComparison Results for " + case_name1 + " and " + case_name2 + " for " + condition_name
        print(title)
        rf_dist, n_rf_dist, weighted_rf_dist, n_weighted_rf_dist, euc_dist = compare_trees(output_dir,
                                                                                           case_name1,
                                                                                           case_name2,
                                                                                           condition_name, tns)

        file.write(title)
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


def set_bulk_to_root(tree_sample):
    """
    This function changes the root of the given tree. It finds the bulk taxa and moves it to the root of the tree.
    Input: tree_sample: dendropy.Tree object
    Output: tree_sample: dendropy.Tree object
    """
    # Create taxon of the bulk
    num_leaves = len(tree_sample.leaf_nodes())
    bulk_taxon = "V" + str(num_leaves - 1)
    # Find bulk
    new_root = tree_sample.find_node_with_taxon_label(label=bulk_taxon)
    # Change the tree
    tree_sample.reroot_at_node(new_root)
    return tree_sample


def main():
    # code to process command line arguments
    parser = argparse.ArgumentParser(description='Distance between cells computation.')
    parser.add_argument('output_dir', help="Specify the results directory.", type=str)
    parser.add_argument('--data_type', help="Specify the data type. Default: real", type=str, default="real")
    args = parser.parse_args()

    generate_and_analyse_lineage_tree(args.output_dir, args.data_type)


if __name__ == "__main__":
    main()
