import os
import time
import json
import pysam
import pickle
import argparse
import numpy as np
# Disable warning messages which do not effect the overall performance.
np.seterr(invalid='ignore')


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


def get_all_gsnv_list(regions_filename):
    """
    This function reads the gSNV regions file and returns the list of gSNV positions and the corresponding chromosomes.
    :param regions_filename: String. Path of the gSNV regions BED file.
    :return: all_gsnv_list: Python list of integers. Contains the gSNV positions.
    :return: all_chr_list: Python list. Same length as all_gsnv_list. Contains the corresponding chromosomes.
    """
    all_gsnv_list = []
    all_chr_list = []
    with open(regions_filename) as regions_file:
        for line in regions_file:
            contents = line.strip().split("\t")
            chr_id = int(contents[0])
            het_pos = int((int(contents[1]) + int(contents[2])) / 2)
            all_chr_list.append(chr_id)
            all_gsnv_list.append(het_pos)
    return all_gsnv_list, all_chr_list


def retrieve_pair_reads_qualities(bam_filename, chr_list, gsnv_list, hom_list):
    """
    Given a site-pair, this function retrieves the reads covering the sites and their Phred scores.
    :param bam_filename: String. Path of the BAM file.
    :param cur_chr: String or integer. Chromosome name.
    :param cur_gsnv: Integer. gSNV position.
    :param cur_hom: Integer. Homozygous position.
    :param read_length: Integer. The length of reads. Each sequencing technology has different read length; i.e 100.
    :return: reads: Numpy (Lc, 2) array of integers. It contains the reads at site-pair.
    :return: quals: Numpy (Lc, 2) array. It contains the quality scores at site-pair.
    """
    nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    num_invalid_nucleotide = 0

    all_reads = []
    all_quals = []

    bam_file = pysam.AlignmentFile(bam_filename, "rb")
    for idx in range(len(chr_list)):
        cur_chr = chr_list[idx]
        cur_gsnv = gsnv_list[idx]
        cur_hom = hom_list[idx]

        pair_reads = []
        pair_quals = []

        #print(min(cur_gsnv, cur_hom), cur_gsnv, cur_hom)
        for read in bam_file.fetch(str(cur_chr), min(cur_gsnv, cur_hom) - 1, max(cur_gsnv, cur_hom)):
            cigar_tup = read.cigartuples
            if len(cigar_tup) == 1 and cigar_tup[0][0] == 0:
                seq_len = read.infer_read_length()
                seq = read.query_alignment_sequence
                qual = read.query_qualities

                start_pos = read.pos
                end_pos = start_pos + seq_len

                if start_pos < cur_gsnv < end_pos and start_pos < cur_hom < end_pos:
                    het_idx = cur_gsnv - start_pos - 1  # because of indexing differences between pysam and igv
                    nucleotide_het = seq[het_idx]
                    qual_het = qual[het_idx]

                    hom_idx = cur_hom - start_pos - 1  # because of indexing differences between pysam and igv
                    nucleotide_hom = seq[hom_idx]
                    qual_hom = qual[hom_idx]

                    try:
                        het_map_idx = nucleotide_map[nucleotide_het]
                        hom_map_idx = nucleotide_map[nucleotide_hom]

                        pair_reads.append([nucleotide_het, nucleotide_hom])
                        pair_quals.append([qual_het, qual_hom])
                    except:  # The except part is needed. Sometimes there are nucleotides other than ACGT, i.e. N.
                        num_invalid_nucleotide += 1

        all_reads.append(pair_reads)
        all_quals.append(pair_quals)

    bam_file.close()
    return all_reads, all_quals


def generate_final_dicts(proc_dir, chr_list, gsnv_list, hom_list, bulk_genotypes, num_cells, bam_dir):
    nucleotide_map = {0: [0, 0], 1: [0, 1], 2: [0, 2], 3: [0, 3], 4: [1, 0], 5: [1, 1], 6: [1, 2], 7: [1, 3],
                      8: [2, 0], 9: [2, 1], 10: [2, 2], 11: [2, 3], 12: [3, 0], 13: [3, 1], 14: [3, 2], 15: [3, 3]}

    bulk_pairs_dict = {}
    for pos_idx in range(len(chr_list)):
        pos_pair = str(gsnv_list[pos_idx]) + "_" + str(hom_list[pos_idx])
        bulk_pairs_dict[pos_pair] = np.array(
            [nucleotide_map[bulk_genotypes[pos_idx][0]], nucleotide_map[bulk_genotypes[pos_idx][1]]])

    bulk_out_filename = proc_dir + "bulk_pairs.pickle"
    save_dictionary(bulk_out_filename, bulk_pairs_dict)

    print("\tBulk pairs dictionary is saved to: \t", bulk_out_filename)

    all_cell = {}
    for cell_idx in range(num_cells):
        all_cell[str(cell_idx)] = {}

    for cell_idx in range(num_cells):
        cell_bam_filename = bam_dir + "cell_idx_" + str(cell_idx) + ".bam"
        all_reads, all_quals = retrieve_pair_reads_qualities(bam_filename=cell_bam_filename, chr_list=chr_list,
                                                             gsnv_list=gsnv_list, hom_list=hom_list)

        for pos_idx in range(len(chr_list)):
            pos_pair = str(gsnv_list[pos_idx]) + "_" + str(hom_list[pos_idx])
            pos_reads = all_reads[pos_idx]
            pos_quals = all_quals[pos_idx]
            depth = len(pos_reads)

            if depth > 0:
                all_cell[str(cell_idx)][pos_pair] = {}
                all_cell[str(cell_idx)][pos_pair]['depth'] = depth
                all_cell[str(cell_idx)][pos_pair]['reads'] = pos_reads
                all_cell[str(cell_idx)][pos_pair]['quals'] = pos_quals
            else:
                all_cell[str(cell_idx)][pos_pair] = {}
                all_cell[str(cell_idx)][pos_pair]['depth'] = 0
                all_cell[str(cell_idx)][pos_pair]['reads'] = []
                all_cell[str(cell_idx)][pos_pair]['quals'] = []

    for cell_idx in range(num_cells):
        cell_filename = proc_dir + "cell_" + str(cell_idx) + "_pairs.pickle"
        save_dictionary(cell_filename, all_cell[str(cell_idx)])

        print("\tCell pairs dictionary is saved to: \t", cell_filename)

    temp_filename = proc_dir + "all_cell_dict_pair.pickle"
    save_dictionary(temp_filename, all_cell)
    print("\tAll cell pairs dictionary is saved to: \t", temp_filename)


def find_closeby_pairs(truth_dir, gsnv_list, chr_list, read_length):
    mut_locations = np.array(load_json(truth_dir + "mut_locations.txt"))

    new_mut_list = []
    new_gsnv_list = []
    new_chr_list = []

    for mut_ in mut_locations:
        mut = mut_ + 1
        diff = np.abs(gsnv_list - mut)
        min_diff = np.min(diff)

        if min_diff < read_length:
            pos_idx = np.argmin(diff)

            new_chr_list.append(chr_list[pos_idx])
            new_gsnv_list.append(gsnv_list[pos_idx])
            new_mut_list.append(mut)

    return new_mut_list, new_gsnv_list, new_chr_list


def get_bulk_genotypes(truth_dir, chr_list, gsnv_list, mut_list):
    nucleotide_map = {0: [0, 0], 1: [0, 1], 2: [0, 2], 3: [0, 3], 4: [1, 0], 5: [1, 1], 6: [1, 2], 7: [1, 3],
                      8: [2, 0], 9: [2, 1], 10: [2, 2], 11: [2, 3], 12: [3, 0], 13: [3, 1], 14: [3, 2], 15: [3, 3]}

    filename = truth_dir + "bulk_genome.txt"
    bulk_genome = np.array(load_json(filename))

    bulk_genotype_list = []

    for idx in range(len(chr_list)):
        het_pos = gsnv_list[idx]
        hom_pos = mut_list[idx]

        b1 = [bulk_genome[het_pos - 1, 0], bulk_genome[hom_pos - 1, 0]]
        b2 = [bulk_genome[het_pos - 1, 1], bulk_genome[hom_pos - 1, 1]]

        for key, value in nucleotide_map.items():
            if value == b1:
                key1 = key
            if value == b2:
                key2 = key

        bulk_genotype_list.append([key1, key2])

    return bulk_genotype_list


def main():
    # code to process command line arguments
    parser = argparse.ArgumentParser(description='Site detection.')
    parser.add_argument('global_dir', help="Specify the directory.", type=str)
    parser.add_argument('num_cells', help="Specify the number of cells.", type=int)
    parser.add_argument('--bulk_depth_threshold', help="Specify the bulk depth threshold. Default: 20", type=int,
                        default=20)
    parser.add_argument('--cell_depth_threshold', help="Specify the cell depth threshold. Default: 0", type=int,
                        default=0)
    parser.add_argument('--chr_id', help="Specify the chromosome number. Default: 1", type=int, default=1)
    parser.add_argument('--het_ratio_threshold', help="Specify the bulk heterozygous ratio threshold. Default: 0.2",
                        type=float, default=0.2)
    parser.add_argument('--min_line', help="Specify the line number of min het position. Default: 0", type=int,
                        default=0)
    parser.add_argument('--max_line', help="Specify the line number of max het position. Default: 0", type=int,
                        default=0)
    parser.add_argument('--nuc_depth_threshold', help="Specify the minimum number of valid reads. Default: 2",
                        type=int, default=2)
    parser.add_argument('--read_length', help="Specify the read length. Default: 100", type=int, default=100)
    parser.add_argument('--seed_val', help="Specify the seed. Default: 123", type=int, default=123)
    args = parser.parse_args()

    start_time_global = time.time()

    print("Global directory: ", args.global_dir)
    bam_dir = args.global_dir + "bam/"
    proc_dir = args.global_dir + "processed_data/"
    if not os.path.exists(proc_dir):
        os.makedirs(proc_dir)
        print("Directory is created")

    # STEP 1. Extract gSNV sites from regions file
    start_time = time.time()
    print("\n*****STEP 1\n")
    regions_filename = bam_dir + "gsnv_vars_regions.bed"
    all_gsnv_list, all_chr_list = get_all_gsnv_list(regions_filename=regions_filename)
    num_total_gsnv = len(all_gsnv_list)

    print("\tTotal number of gSNV sites: ", num_total_gsnv)
    end_time = time.time()
    print("\n\tTotal time: ", end_time - start_time, "\n*****")

    # STEP 1.1. Limit the gSNV sites based on the given arguments.
    print("\n***** STEP 1.1\n")
    line_range = [args.min_line, num_total_gsnv]
    if args.max_line != 0:
        line_range[1] = args.max_line

    all_gsnv_list = all_gsnv_list[line_range[0]: line_range[1]]
    all_chr_list = all_chr_list[line_range[0]: line_range[1]]
    print("\tLimiting the number of analysed gSNVs to: ", len(all_gsnv_list))

    # STEP 2. Get site pairs
    start_time = time.time()
    print("\n***** STEP 2\n")
    truth_dir = args.global_dir + "truth/"
    valid_hom_positions, valid_gsnv_positions, valid_chr_list = find_closeby_pairs(truth_dir,
                                                                                   all_gsnv_list, all_chr_list,
                                                                                   read_length=args.read_length)

    end_time = time.time()
    print("\tTotal number of valid site pairs: ", len(valid_hom_positions))
    print("\n\tTotal time: ", end_time - start_time, "\n*****")

    # STEP 3. Get bulk genotypes at site pairs
    start_time = time.time()
    print("\n***** STEP 3\n")
    bulk_genotype_list = get_bulk_genotypes(truth_dir, valid_chr_list, valid_gsnv_positions, valid_hom_positions)

    # STEP 4. Save valid sites
    generate_final_dicts(proc_dir=proc_dir, chr_list=valid_chr_list, gsnv_list=valid_gsnv_positions,
                         hom_list=valid_hom_positions, bulk_genotypes=bulk_genotype_list, num_cells=args.num_cells,
                         bam_dir=bam_dir)

    end_time = time.time()
    print("\n\tTotal time: ", end_time - start_time, "\n*****")

    ###
    print("\n***** DONE!")
    end_time_global = time.time()
    print("\tTotal global time: ", end_time_global - start_time_global, "\n*****")


if __name__ == "__main__":
    main()
