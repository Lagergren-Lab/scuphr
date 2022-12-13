import argparse
import dendropy
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import pickle
import os

def save_dictionary(filename, cell_dict):
    with open(filename, 'wb') as fp:
        pickle.dump(cell_dict, fp)

def load_dictionary(filename):
    with open(filename, 'rb') as fp:
        cell_dict = pickle.load(fp)
    return cell_dict


def displayTree(filename):
    # i.e: displayTree("sampled_trees/result_10.tre")
    bootstrap_tree = dendropy.Tree.get(file=open(filename, 'r'), schema="nexus")
    print(bootstrap_tree.as_ascii_plot())
    print(bootstrap_tree.as_string(schema="nexus"))


def generateCommand(numTrees, out_filename="result.tre", target=""):
    # i.e: generateCommand(3, out_filename="result.tre", target="t_0.tre")

    stri = "sumtrees.py --decimals=0 --percentages"
    for tree in range(numTrees):
        stri = stri + " t_" + str(tree) + ".tre"  # + "_midpoint.tre"

    stri = stri + " --output=" + out_filename
    if target is not "":
        stri = stri + " --target=" + target

    return stri


def generateCommandWoBulk(numTrees, out_filename="result_wo_bulk.tre", target=""):
    stri = "sumtrees.py --decimals=0 --percentages"
    for tree in range(numTrees):
        stri = stri + " t_" + str(tree) + "_wo_bulk.tre"

    stri = stri + " --output=" + out_filename
    if target is not "":
        stri = stri + " --target=" + target

    return stri


def generateCommandClade(numTrees, freq=0.90, out_filename="result_minclade.tre"):
    stri = "sumtrees.py  --percentages "
    for tree in range(numTrees):
        stri = stri + " t_" + str(tree) + ".tre"

    stri = stri + " --output=" + out_filename + " --min-clade-freq=" + str(freq)

    return stri


def generateCommandCladeWoBulk(numTrees, freq=0.90, out_filename="result_minclade_wo_bulk.tre"):
    stri = "sumtrees.py  --percentages "
    for tree in range(numTrees):
        stri = stri + " t_" + str(tree) + "_wo_bulk.tre"

    stri = stri + " --output=" + out_filename + " --min-clade-freq=" + str(freq)

    return stri


def generateLineageTree(filename, tns):
    pdm = dendropy.PhylogeneticDistanceMatrix.from_csv(src=open(filename), delimiter=",",
                                                       is_first_row_column_names=False, is_first_column_row_names=False)

    nj_tree_sample = pdm.nj_tree()
    nj_tree_sample_str = nj_tree_sample.as_string("newick")

    # Generate tree based on distance matrices
    tree_sample = dendropy.Tree.get(data=nj_tree_sample_str, schema="newick", taxon_namespace=tns,
                                    rooting="force-rooted")
    return tree_sample, nj_tree_sample, nj_tree_sample_str


def setBulkToRoot(tree_sample):
    # Create taxon of the bulk
    num_leaves = len(tree_sample.leaf_nodes())
    bulk_taxon = "V" + str(num_leaves - 1)
    # Find bulk
    new_root = tree_sample.find_node_with_taxon_label(label=bulk_taxon)
    # Change the tree
    tree_sample.reroot_at_node(new_root)
    return (tree_sample)


def combineMultipleSites(C, numPosPerChromosome, chromosome_ids, global_dir_info, dir_info, numSkip, epochId=0,
                         scuphr_strategy='hybrid'):
    np.random.seed(epochId)

    start_time = time.time()
    print("Combining multiple sites for epoch: ", epochId)

    numPos = np.sum(numPosPerChromosome)
    numPosCumulative = np.cumsum(numPosPerChromosome)
    print("Total number of positions: ", numPos)

    res_sub_sum = np.zeros((C + 1, C + 1))
    res_mask = np.zeros((C + 1, C + 1))

    sub = 1

    num_invalid = 0

    numHet = 0
    numHomoz = 0
    numCorrectHet = 0
    numCorrectHomoz = 0
    skippedPositions = []
    skippedPositionsRuns = []
    skippedChromosomes = []

    chr_dist = np.random.multinomial(numPos, pvals=numPosPerChromosome / numPos)
    # print("chr_dist: ", chr_dist)

    for i in range(len(chr_dist)):
        chr_idx = chromosome_ids[i]
        num_pos_current_chr = chr_dist[i]
        pos_list = np.random.choice(numPosPerChromosome[i], size=num_pos_current_chr, replace=True)
        # print("pos_list: ", pos_list)

        filename2 = dir_info + "chr" + str(chr_idx) + "_commonZstatus_combined.pkl"
        commonZdict = load_dictionary(filename2)

        filename = dir_info + "chr" + str(chr_idx) + "_matrix_infer_similarity_combined.pkl"
        dist_matrix_dict = load_dictionary(filename)

        for pos in pos_list:
            try:
                status = commonZdict[str(pos)]

                # He site is homozygous
                if status == 0 or status == 2:
                    numHomoz = numHomoz + 1
                    skippedPositions.append(pos)
                    skippedChromosomes.append(chr_idx)
                    # print("Skipping position ", pos, " (He is homozygous in inferredZ)")

                # He site is heterozygous
                else:
                    # print("\nUsing ", sub, " positions")
                    numHet = numHet + 1

                    dist_matrix_pos = dist_matrix_dict[str(pos)]

                    inferred_dist_matrix = np.copy(dist_matrix_pos)
                    # inferred_temp = np.ma.masked_less(inferred_dist_matrix,0)
                    inferred_temp = np.ma.masked_invalid(inferred_dist_matrix)
                    inferred_mask = (~inferred_temp.mask) * 1
                    inferred_temp = inferred_temp.filled(fill_value=0)

                    res_sub_sum = res_sub_sum + inferred_temp
                    res_mask = res_mask + inferred_mask
                    res_sub = np.divide(res_sub_sum, res_mask)
                    sub = sub + 1
            except:
                num_invalid += 1

    ### Save CommonZStatus Files
    filename3 = global_dir_info + "bootstrap_commonZanalysis/commonZanalysis_" + str(epochId) + ".txt"
    with open(filename3, 'w') as out3:
        numTotal = numHet + numHomoz
        out3.write("Epoch: %d" % epochId)
        out3.write("\nInferredZ Analysis: ")
        out3.write("\nTotal number: %d" % numTotal)
        out3.write("\nHeterozygous: %d. Perc: %.2f" % (numHet, (100 * numHet) / numTotal))
        out3.write("\nHomozygous: %d. Perc: %.2f" % (numHomoz, (100 * numHomoz) / numTotal))
        out3.write("\nSkipped positions: ")
        for sk_pos in skippedPositions:
            out3.write(" %s " % str(sk_pos))
        out3.write("\nSkipped chromosomes: ")
        for sk_chr in skippedChromosomes:
            out3.write(" %s " % str(sk_chr))
        out3.write("\nSkipped run seeds: ")
        for sk_run in skippedPositionsRuns:
            out3.write(" %s " % str(sk_run))
    print("Num_invalid: ", num_invalid)

    ### Save Distance Between Cells Matrix Plot
    plt.figure(figsize=(20, 12))
    plt.imshow(res_sub, interpolation='nearest', cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    plt.xlabel("Cells")
    plt.ylabel("Cells")
    title_str = "Similarity Matrix. Final (Epoch: " + str(epochId) + ")"
    plt.title(title_str)
    filename = global_dir_info + "bootstrap_dbc/DBC_" + str(epochId) + ".png"
    plt.savefig(filename)
    plt.close()

    ### Save Final Matrices
    filename = global_dir_info + "bootstrap_matrix/Matrix_Final_InferDif_count_" + str(epochId) + ".out"
    np.savetxt(filename, res_mask)
    filename = global_dir_info + "bootstrap_matrix/Matrix_Final_InferDif_" + str(epochId) + ".csv"
    np.savetxt(filename, 1 - res_sub, delimiter=',', fmt="%.10f")

    '''
    ### Generate and Save Tree
    tns = dendropy.TaxonNamespace()

    ## without bulk node
    filename = global_dir_info + "bootstrap_matrix/Matrix_Final_InferDif_" + str(epochId) + ".csv"
    #dist_mat_temp = np.loadtxt(open(filename, "rb"), delimiter=",")
    #dist_mat_temp = dist_mat_temp[0:dist_mat_temp.shape[0]-1,0:dist_mat_temp.shape[0]-1]
    #np.savetxt(filename, dist_mat_temp, delimiter=',', fmt="%.10f")

    sampled_tree, _, _ = generateLineageTree(filename, tns)
    sampled_tree = setBulkToRoot(sampled_tree)
    filename = global_dir_info + "sampled_trees/t_" + str(epochId) + ".tre"
    sampled_tree.write(file=open(filename, 'w'), schema="newick")

    filename = global_dir_info + "bootstrap_lineage_trees/Lineage_Tree_" + str(epochId) + ".txt"
    file = open(filename, "w")
    file.write("Epoch: %d" % epochId)
    file.write("\n\nLineage Tree\n")
    file.write(sampled_tree.as_ascii_plot())
    file.write("\nLineage Tree String\n")
    file.write(sampled_tree.as_string(schema="newick"))
    file.close()
    '''

    ## with bulk node
    filename = global_dir_info + "bootstrap_matrix/Matrix_Final_InferDif_" + str(epochId) + ".csv"
    tns = dendropy.TaxonNamespace()
    sampled_tree, _, _ = generateLineageTree(filename, tns)
    sampled_tree = setBulkToRoot(sampled_tree)
    filename = global_dir_info + "sampled_trees/t_" + str(epochId) + ".tre"
    sampled_tree.write(file=open(filename, 'w'), schema="newick")

    filename = global_dir_info + "bootstrap_lineage_trees/Lineage_Tree_" + str(epochId) + ".txt"
    file = open(filename, "w")
    file.write("Epoch: %d" % epochId)
    file.write("\n\nLineage Tree\n")
    file.write(sampled_tree.as_ascii_plot())
    file.write("\nLineage Tree String\n")
    file.write(sampled_tree.as_string(schema="newick"))

    ## without bulk node
    filename = global_dir_info + "bootstrap_matrix/Matrix_Final_InferDif_" + str(epochId) + ".csv"
    dist_mat_temp = np.loadtxt(open(filename, "rb"), delimiter=",")
    dist_mat_temp = dist_mat_temp[0:dist_mat_temp.shape[0] - 1, 0:dist_mat_temp.shape[0] - 1]
    np.savetxt(filename + "_wo_bulk", dist_mat_temp, delimiter=',', fmt="%.10f")

    tns = dendropy.TaxonNamespace()
    sampled_tree, _, _ = generateLineageTree(filename + "_wo_bulk", tns)

    file.write("\nEpoch: %d" % epochId)
    file.write("\n\nLineage Tree wo Bulk\n")
    file.write(sampled_tree.as_ascii_plot())
    file.write("\nLineage Tree String wo Bulk\n")
    file.write(sampled_tree.as_string(schema="newick"))
    file.close()

    filename = global_dir_info + "sampled_trees/t_" + str(epochId) + "_wo_bulk.tre"
    sampled_tree.write(file=open(filename, 'w'), schema="newick")

    end_time = time.time()
    print("\tTotal time of combining multiple sites: ", end_time - start_time)


def runBootstrap(C, numPosPerChromosome, chromosome_ids, numEpoch, global_dir_info, dir_info, numSkip,
                 scuphr_strategy='hybrid'):
    # np.random.seed(123)
    if not os.path.exists(global_dir_info):
        os.makedirs(global_dir_info)

    temp_name_list = ["bootstrap_commonZanalysis/", "bootstrap_dbc/", "bootstrap_matrix/", "sampled_trees/",
                      "bootstrap_lineage_trees/"]
    for sub_name in temp_name_list:
        temp_name = global_dir_info + sub_name
        if not os.path.exists(temp_name):
            os.makedirs(temp_name)

    print("***Running bootstrap for ", numEpoch, " times***\n")
    print("Total number of sites: ", np.sum(numPosPerChromosome))

    for epochId in range(numEpoch):
        combineMultipleSites(C, numPosPerChromosome, chromosome_ids, global_dir_info, dir_info, numSkip, epochId,
                             scuphr_strategy=scuphr_strategy)

    out_filename = "../result_" + str(numEpoch) + ".tre"
    command_stri = generateCommand(numEpoch, out_filename)
    print("\nCommand to combine trees: \n")
    print(command_stri)

    # out_filename="../result_target_" + str(numEpoch) + ".tre"
    # command_stri = generateCommand(numEpoch, out_filename, target="t_0.tre")
    # print("\nCommand to combine trees: \n")
    # print(command_stri)

    out_filename = "../result_minclade_" + str(numEpoch) + "_" + str(C) + ".tre"
    command_stri = generateCommandClade(numEpoch, 0.10, out_filename)
    print("\nCommand to combine trees: \n")
    print(command_stri)

    ### WO BULK
    out_filename = "../result_" + str(numEpoch) + "_wo_bulk.tre"
    command_stri = generateCommandWoBulk(numEpoch, out_filename)
    print("\nCommand to combine trees wo bulk: \n")
    print(command_stri)

    out_filename = "../result_minclade_" + str(numEpoch) + "_" + str(C) + "_wo_bulk.tre"
    command_stri = generateCommandCladeWoBulk(numEpoch, 0.10, out_filename)
    print("\nCommand to combine trees wo bulk: \n")
    print(command_stri)


def main():
    # code to process command line arguments
    parser = argparse.ArgumentParser(description='Bootstrapping.')
    parser.add_argument('dict_dir', help="Specify the dictionary directory.", type=str)
    parser.add_argument('num_cells', help="Specify the number of cells.", type=int)
    parser.add_argument('--num_epoch', help="Specify the number of bootstrap trees. Default: 100", type=int, default=100)
    parser.add_argument('--seed_val', help="Specify the seed. Default: 123", type=int, default=123)
    args = parser.parse_args()

    np.random.seed(args.seed_val)

    output_dir = args.dict_dir + "bootstrap/"

    # chromosome_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    chromosome_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    # chromosome_ids = [19, 20, 21, 22]

    print("Determining number of sites in each chromosome.")
    numPosPerChromosome = []
    for chr_idx in chromosome_ids:
        filename2 = args.dict_dir + "chr" + str(chr_idx) + "_commonZstatus_combined.pkl"
        commonZdict = load_dictionary(filename2)
        num_sites = len(commonZdict.keys())
        numPosPerChromosome.append(num_sites)
        print("\tChromosome ", chr_idx, ": ", num_sites, "sites")
    print("Total number of chromosomes: ", len(chromosome_ids))
    print("Total number of chromosomes: ", np.sum(numPosPerChromosome))

    print("Running bootstrapping...")
    runBootstrap(args.num_cells, numPosPerChromosome, chromosome_ids, args.num_epoch, output_dir, args.dict_dir,
                 numSkip=0, scuphr_strategy='hybrid')


if __name__ == "__main__":
    main()
