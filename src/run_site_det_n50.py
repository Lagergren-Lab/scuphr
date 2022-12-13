import os
import time
import pysam
import pickle
import argparse
import numpy as np
# Disable warning messages which do not effect the overall performance.
np.seterr(invalid='ignore')


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


def retrieve_single_pos_coverage(bam_filename, cur_chr, cur_gsnv):
    """
    This function retrieves a single position's coverage from a BAM file and stores it in a (4,) array.
    :param bam_filename: String. Path of the BAM file.
    :param cur_chr: String or integer. Chromosome name.
    :param cur_gsnv: Integer. gSNV position.
    :return: site_coverage: Numpy (4, ) array of integers. It contains the coverage of gSNV site.
    """
    nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    num_invalid_nucleotide = 0

    site_coverage = np.zeros(4)  # Stores the number of A, C, G, T appears in reads

    bam_file = pysam.AlignmentFile(bam_filename, "rb")
    for read in bam_file.fetch(str(cur_chr), cur_gsnv - 1, cur_gsnv):
        cigar_tup = read.cigartuples
        if len(cigar_tup) == 1 and cigar_tup[0][0] == 0:
            seq_len = read.infer_read_length()
            seq = read.query_alignment_sequence
            start_pos = read.pos
            end_pos = start_pos + seq_len

            if start_pos < cur_gsnv <= end_pos:
                het_idx = cur_gsnv - start_pos - 1  # because of indexing differences between pysam and igv
                nucleotide = seq[het_idx]

                try:
                    map_idx = nucleotide_map[nucleotide]
                    site_coverage[map_idx] += 1
                except:  # The except part is needed. Sometimes there are nucleotides other than ACGT, i.e. N.
                    num_invalid_nucleotide += 1
    bam_file.close()
    return site_coverage


def retrieve_double_pos_coverage(bam_filename, cur_chr, cur_gsnv, read_length):
    """
    Given a gSNV position, this function looks the surrounding positions and retrieves each site-pair's coverage from
    a BAM file and stores it in an array. It also returns a list which contains the surrounding positions.
    :param bam_filename: String. Path of the BAM file.
    :param cur_chr: String or integer. Chromosome name.
    :param cur_gsnv: Integer. gSNV position.
    :param read_length: Integer. The length of reads. Each sequencing technology has different read length; i.e 100.
    :return: site_coverage: Numpy (2*(read_length-1), 16) array of integers. It contains the coverage of site-pair.
    :return: hom_pos_list: Numpy (2*(read_length-1), ) array of integers.
                           It contains the corresponding homozygous positions.
    """
    nucleotide_map = {'AA': 0, 'AC': 1, 'AG': 2, 'AT': 3, 'CA': 4, 'CC': 5, 'CG': 6, 'CT': 7,
                      'GA': 8, 'GC': 9, 'GG': 10, 'GT': 11, 'TA': 12, 'TC': 13, 'TG': 14, 'TT': 15}
    num_invalid_nucleotide = 0

    site_coverage = np.zeros((2 * (read_length - 1), 16))  # Stores the number of nucleotide pairs appears in reads
    hom_pos_list = np.zeros(2 * (read_length - 1))

    bam_file = pysam.AlignmentFile(bam_filename, "rb")

    site_idx = 0
    for hom_pos in range(cur_gsnv - read_length + 1, cur_gsnv + read_length):
        # TODO we can also limit the maximum range to be the genome length
        if hom_pos >= 1 and hom_pos != cur_gsnv:
            hom_pos_list[site_idx] = int(hom_pos)

            for read in bam_file.fetch(str(cur_chr), min(cur_gsnv, hom_pos) - 1, max(cur_gsnv, hom_pos)):
                cigar_tup = read.cigartuples
                if len(cigar_tup) == 1 and cigar_tup[0][0] == 0:
                    seq_len = read.infer_read_length()
                    seq = read.query_alignment_sequence

                    start_pos = read.pos
                    end_pos = start_pos + seq_len

                    if start_pos < cur_gsnv < end_pos and start_pos < hom_pos < end_pos:
                        het_idx = cur_gsnv - start_pos - 1  # because of indexing differences between pysam and igv
                        nucleotide_het = seq[het_idx]

                        hom_idx = hom_pos - start_pos - 1  # because of indexing differences between pysam and igv
                        nucleotide_hom = seq[hom_idx]

                        try:
                            map_idx = nucleotide_map[nucleotide_het+nucleotide_hom]
                            site_coverage[site_idx, map_idx] += 1
                        except:  # The except part is needed. Sometimes there are nucleotides other than ACGT, i.e. N.
                            num_invalid_nucleotide += 1
            site_idx += 1

    bam_file.close()
    return site_coverage, hom_pos_list


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


def get_valid_gsnv(chr_list, gsnv_list, bulk_bam_filename, num_cells, bam_dir):
    """
    Given a list of gSNVs, this function checks whether the cells are suitable for phasing or not
    and returns the list of valid gSNVs and the corresponding chromosomes.
    :param chr_list: Python list.
    :param gsnv_list: Python list of integers. Same length as chr_list.
    :param bulk_bam_filename: String. Path of bulk BAM file.
    :param num_cells: Integer. Number of cells.
    :param bam_dir: String. Path to cell BAM files.
    :return: valid_gsnv_list: Python list of integers. Contains the valid gSNV positions.
    :return: valid_chr_list: Python list. Same length as valid_gsnv_list. Contains the corresponding chromosomes.
    """
    num_all_gsnv = len(gsnv_list)

    valid_gsnv_list = []
    valid_chr_list = []

    for gsnv_idx in range(num_all_gsnv):
        cur_gsnv = gsnv_list[gsnv_idx]
        cur_chr = chr_list[gsnv_idx]

        # Get bulk's coverage matrix
        bulk_het_coverage = retrieve_single_pos_coverage(bam_filename=bulk_bam_filename,
                                                         cur_chr=cur_chr, cur_gsnv=cur_gsnv)

        # Get single-cells' coverage matrices
        cell_het_coverage = np.zeros((num_cells, 4), dtype=int)
        for cell_idx in range(num_cells):
            cell_bam_filename = bam_dir + "cell_idx_" + str(cell_idx) + ".bam"
            cell_het_coverage[cell_idx, :] = retrieve_single_pos_coverage(bam_filename=cell_bam_filename,
                                                                          cur_chr=cur_chr, cur_gsnv=cur_gsnv)
        # Check validity
        if check_gsnv_validity(bulk_het_coverage, cell_het_coverage):
            valid_gsnv_list.append(cur_gsnv)
            valid_chr_list.append(cur_chr)

    return valid_gsnv_list, valid_chr_list


def check_gsnv_validity(bulk_het_coverage, cell_het_coverage):
    """
    Given bulk and single-cell coverage matrices of a gSNV site,
    it checks whether the sites are suitable for read phasing or not.
    :param bulk_het_coverage: Numpy (4, ) array of integers. Contains how many ACGT are observed in reads.
    :param cell_het_coverage: Numpy (num_cells, 4) array of integers.
                              Contains how many ACGT are observed in each single-cell reads.
    :return: Boolean. If True, site is eligible for read-phasing.
    """
    # Get the total number of reads at the site.
    marg_bulk_het_coverage = np.sum(bulk_het_coverage)
    marg_cell_het_coverage = np.sum(cell_het_coverage, axis=1, keepdims=True)

    # Get the frequency of reads at the site.
    # Note: Shows warning due to division by zero, as expected. It doesn't effect the performance.
    bulk_het_freq = np.divide(bulk_het_coverage, marg_bulk_het_coverage)
    cell_het_freq = np.divide(cell_het_coverage, marg_cell_het_coverage)

    # Filter the matrices.
    bulk_het_coverage[bulk_het_coverage < 10] = 0
    bulk_het_coverage[bulk_het_coverage >= 10] = 1
    bulk_het_freq[bulk_het_freq < 0.2] = 0
    bulk_het_freq[bulk_het_freq >= 0.2] = 1
    temp_bulk = np.multiply(bulk_het_coverage, bulk_het_freq)

    cell_het_coverage[cell_het_coverage < 2] = 0
    cell_het_coverage[cell_het_coverage >= 2] = 1
    cell_het_freq[cell_het_freq < 0.2] = 0
    cell_het_freq[cell_het_freq >= 0.2] = 1
    temp = np.multiply(cell_het_coverage, cell_het_freq)

    # Check visibility. At least two single-cells should be visible in order to allow read-phasing.
    temp_vis = np.multiply(temp_bulk, temp)
    marg_temp_vis = np.sum(temp_vis, axis=1)
    if len(np.where(marg_temp_vis >= 2)[0]) >= 2:
        return True
    else:
        return False


def get_valid_site_pair(chr_list, gsnv_list, bulk_bam_filename, num_cells, bam_dir, read_length):
    """
    Given a list of gSNVs, this function checks the surrounding positions
    and returns returns the list of site-pairs worth for dataset generation.
    :param chr_list: Python list.
    :param gsnv_list: Python list of integers. Same length as chr_list.
    :param bulk_bam_filename: String. Path of bulk BAM file.
    :param num_cells: Integer. Number of cells.
    :param bam_dir: String. Path to cell BAM files.
    :param read_length: Integer. The length of reads. Each sequencing technology has different read length; i.e 100.
    :return: valid_hom_positions: Python list of integers.
                                  Contains the valid homozygous positions (potentially mutated sites).
    :return: valid_gsnv_positions: Python list of integers. Same length as valid_hom_positions.
                                   Contains the corresponding gSNV positions.
    :return: valid_chr_list: Python list. Same length as valid_hom_positions. Contains the corresponding chromosomes.
    :return: valid_bulk_genotype_list: Python list. Same length as valid_hom_positions.
                                       Contains the corresponding bulk genotypes.
    """
    num_all_gsnv = len(gsnv_list)

    valid_hom_positions = []
    valid_gsnv_positions = []
    valid_chr_list = []
    bulk_genotype_list = []

    for gsnv_idx in range(num_all_gsnv):
        cur_gsnv = gsnv_list[gsnv_idx]
        cur_chr = chr_list[gsnv_idx]

        # Get bulk's coverage matrix
        bulk_pair_coverage, hom_pos_list = retrieve_double_pos_coverage(bam_filename=bulk_bam_filename, cur_chr=cur_chr,
                                                                        cur_gsnv=cur_gsnv, read_length=read_length)

        # Get single-cells' coverage matrices
        cell_pair_coverage = np.zeros((num_cells, 2 * (read_length - 1), 16))
        for cell_idx in range(num_cells):
            cell_bam_filename = bam_dir + "cell_idx_" + str(cell_idx) + ".bam"
            cell_pair_coverage[cell_idx, :, :], _ = retrieve_double_pos_coverage(bam_filename=cell_bam_filename,
                                                                                 cur_chr=cur_chr, cur_gsnv=cur_gsnv,
                                                                                 read_length=read_length)

        # First, check gSNV validity for site pairs.
        # Marginalize out the homozygous sites
        bulk_het_coverage = np.zeros((2 * (read_length - 1), 4))
        bulk_het_coverage[:, 0] = np.sum(bulk_pair_coverage[:, :4], axis=1)
        bulk_het_coverage[:, 1] = np.sum(bulk_pair_coverage[:, 4:8], axis=1)
        bulk_het_coverage[:, 2] = np.sum(bulk_pair_coverage[:, 8:12], axis=1)
        bulk_het_coverage[:, 3] = np.sum(bulk_pair_coverage[:, 12:], axis=1)

        cell_het_coverage = np.zeros((num_cells, 2 * (read_length - 1), 4))
        cell_het_coverage[:, :, 0] = np.sum(cell_pair_coverage[:, :, :4], axis=2)
        cell_het_coverage[:, :, 1] = np.sum(cell_pair_coverage[:, :, 4:8], axis=2)
        cell_het_coverage[:, :, 2] = np.sum(cell_pair_coverage[:, :, 8:12], axis=2)
        cell_het_coverage[:, :, 3] = np.sum(cell_pair_coverage[:, :, 12:], axis=2)

        # Check validity of gSNV sites.
        valid_gsnv_pair_indices = []
        hom_positions = []
        for hom_pos_idx in range(len(hom_pos_list)):
            if check_gsnv_validity(bulk_het_coverage[hom_pos_idx, :], cell_het_coverage[:, hom_pos_idx, :]):
                valid_gsnv_pair_indices.append(hom_pos_idx)
                hom_positions.append(int(hom_pos_list[hom_pos_idx]))
        # print("\tTotal number of valid gSNV sites for pairs: ", len(valid_gsnv_pair_indices))

        # Get the valid subset of the coverage matrices.
        bulk_pair_coverage = bulk_pair_coverage[valid_gsnv_pair_indices, :]
        cell_pair_coverage = cell_pair_coverage[:, valid_gsnv_pair_indices, :]

        # Then, check homozygous sites and find the single-cells' disagreement with bulk.
        # Get the total number of reads at the site-pair.
        marg_bulk_pair_coverage = np.sum(bulk_pair_coverage, axis=1, keepdims=True)
        marg_cell_pair_coverage = np.sum(cell_pair_coverage, axis=2, keepdims=True)

        # Get the frequency of reads at the site-pair.
        # Note: Shows warning due to division by zero, as expected. It doesn't effect the performance.
        bulk_pair_freq = np.divide(bulk_pair_coverage, marg_bulk_pair_coverage)
        cell_pair_freq = np.divide(cell_pair_coverage, marg_cell_pair_coverage)

        # Filter the bulk matrix.
        bulk_pair_coverage[bulk_pair_coverage < 10] = 0
        bulk_pair_coverage[bulk_pair_coverage >= 10] = 1
        bulk_pair_freq[bulk_pair_freq < 0.2] = 0
        bulk_pair_freq[bulk_pair_freq >= 0.2] = 1
        temp_bulk = np.multiply(bulk_pair_coverage, bulk_pair_freq)

        # Here, we put 0 to bulk genotype and 1 to any other genotype.
        temp_bulk_inv = np.copy(temp_bulk)
        temp_bulk_inv[temp_bulk_inv == 0] = -1
        temp_bulk_inv[temp_bulk_inv == 1] = 0
        temp_bulk_inv[temp_bulk_inv == -1] = 1

        # Filter the single-cell matrix.
        cell_pair_coverage[cell_pair_coverage < 2] = 0
        cell_pair_coverage[cell_pair_coverage >= 2] = 1
        cell_pair_freq[cell_pair_freq < 0.2] = 0
        cell_pair_freq[cell_pair_freq >= 0.2] = 1
        temp = np.multiply(cell_pair_coverage, cell_pair_freq)

        # Check visibility. At least two, at most num_cells-1 single-cells should be visible.
        temp_vis = np.multiply(temp_bulk_inv, temp)
        marg_temp_vis = np.sum(np.nan_to_num(temp_vis), axis=0)
        temp_indices, votes = np.where(marg_temp_vis >= 2)
        temp_indices_2 = np.where(marg_temp_vis < num_cells)[0]
        intersect_indices = list(set(temp_indices) & set(temp_indices_2))

        for ind in intersect_indices:
            # Additional filter to avoid mutations in gSNV sites.
            # TODO This part might need more filters. For instance,
            #  if a subset of cells vote for mutation at gSNV site, others vote for mutation in homozygous site.
            if (np.where(temp_bulk[ind, :] == 1)[0][0] % 4) != (votes[list(temp_indices).index(ind)] % 4):
                valid_hom_positions.append(hom_positions[ind])
                valid_gsnv_positions.append(cur_gsnv)
                valid_chr_list.append(cur_chr)
                bulk_genotype_list.append(np.where(temp_bulk[ind, :] == 1)[0])
        # print("\tTotal number of valid hom sites around gsnv ", cur_gsnv, ": ", len(valid_hom_indices))

    valid_hom_positions, valid_gsnv_positions, valid_chr_list, bulk_genotype_list = remove_multiple_homozygous(
        hom_positions=valid_hom_positions, gsnv_positions=valid_gsnv_positions, chr_list=valid_chr_list,
        bulk_genotypes=bulk_genotype_list)

    print("\tTotal number of valid hom sites: ", len(valid_hom_positions))
    return valid_hom_positions, valid_gsnv_positions, valid_chr_list, bulk_genotype_list


def remove_multiple_homozygous(hom_positions, gsnv_positions, chr_list, bulk_genotypes):
    """
    Given the lists of site-pairs,
    this function checks how many gSNVs are mapped to each homozygous sites and decrease the number to 1.
    :param hom_positions: Python list of integers.
    :param gsnv_positions: Python list of integers. Same length as hom_positions.
    :param chr_list: Python list. Same length as gsnv_positions.
    :param bulk_genotypes: Python list. Same length as gsnv_positions.
    :return: valid_hom_positions: Python list of integers.
                                  Contains the valid homozygous positions (potentially mutated sites).
    :return: valid_gsnv_positions: Python list of integers. Same length as valid_hom_positions.
                                   Contains the corresponding gSNV positions.
    :return: valid_chr_list: Python list. Same length as valid_hom_positions. Contains the corresponding chromosomes.
    :return: valid_bulk_genotype_list: Python list. Same length as valid_hom_positions.
                                       Contains the corresponding bulk genotypes.
    """

    # Filter out the duplicates
    temp_dict = {}
    num_duplicates = 0
    for i in range(len(hom_positions)):
        cur_hom = hom_positions[i]
        cur_gsnv = gsnv_positions[i]
        cur_chr = chr_list[i]
        cur_bulk = bulk_genotypes[i]

        key = str(cur_chr) + "_" + str(cur_hom)
        if key not in temp_dict:
            temp_dict[key] = {'chr': cur_chr, 'hom': cur_hom, 'gsnv': cur_gsnv, 'bulk': cur_bulk}
        else:
            num_duplicates += 1
            if abs(cur_hom-cur_gsnv) < abs(cur_hom-temp_dict[key]['gsnv']):
                temp_dict[key]['gsnv'] = cur_gsnv
                temp_dict[key]['bulk'] = cur_bulk

    print("\tNumber of removed duplicate homozygous sites: ", num_duplicates)

    # Re-construct the lists
    valid_hom_positions = []
    valid_gsnv_positions = []
    valid_chr_list = []
    valid_bulk_genotype_list = []

    for key in temp_dict:
        valid_hom_positions.append(temp_dict[key]['hom'])
        valid_gsnv_positions.append(temp_dict[key]['gsnv'])
        valid_chr_list.append(temp_dict[key]['chr'])
        valid_bulk_genotype_list.append(temp_dict[key]['bulk'])

    return valid_hom_positions, valid_gsnv_positions, valid_chr_list, valid_bulk_genotype_list


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


def main():
    # code to process command line arguments
    # parser = argparse.ArgumentParser(description='Site detection.')
    # parser.add_argument('global_dir', help="Specify the directory.", type=str)
    # parser.add_argument('num_cells', help="Specify the number of cells.", type=int)
    # parser.add_argument('--bulk_depth_threshold', help="Specify the bulk depth threshold. Default: 20", type=int,
    #                     default=20)
    # parser.add_argument('--cell_depth_threshold', help="Specify the cell depth threshold. Default: 0", type=int,
    #                     default=0)
    # parser.add_argument('--chr_id', help="Specify the chromosome number. Default: 1", type=int, default=1)
    # parser.add_argument('--het_ratio_threshold', help="Specify the bulk heterozygous ratio threshold. Default: 0.2",
    #                     type=float, default=0.2)
    # parser.add_argument('--min_line', help="Specify the line number of min het position. Default: 0", type=int,
    #                     default=0)
    # parser.add_argument('--max_line', help="Specify the line number of max het position. Default: 0", type=int,
    #                     default=0)
    # parser.add_argument('--nuc_depth_threshold', help="Specify the minimum number of valid reads. Default: 2",
    #                     type=int, default=2)
    # parser.add_argument('--read_length', help="Specify the read length. Default: 100", type=int, default=100)
    # parser.add_argument('--seed_val', help="Specify the seed. Default: 123", type=int, default=123)
    # args = parser.parse_args()

    seed_list = [1]
    num_cells_list = [50]
    mut_type_list = [2, 4] # 1: 1gen / branch. 2: 10gen / branch. 3: 20gen / branch. 4: mut = gsnv, 5: mut = gsnv(1 + Poi(mut_poi_rate))
    p_ado_list = [0, 0.2, 0.4]
    p_ae_list = [0, 0.0001]
    phred_type_list = [0, 2] # 0: p_err = 0, 1: p_err = 0.0001, 2: p_error\ in [0.0001, 0.01]

    for seed_idx in range(len(seed_list)):
        for mut_type_idx in range(len(mut_type_list)):
            for num_cell_idx in range(len(num_cells_list)):
                for p_ado_idx in range(len(p_ado_list)):
                    for p_ae_idx in range(len(p_ae_list)):
                        for phred_idx in range(len(phred_type_list)):
                            print("\n\n*** NEW SITE DETECTION ***\n\n")

                            seed_val = seed_list[seed_idx]
                            mut_type = mut_type_list[mut_type_idx] # 1: 1gen / branch.2: 10gen / branch.3: 20gen / branch.4: mut = gsnv, 5: mut = gsnv(1 + Poi(mut_poi_rate))
                            num_cells = num_cells_list[num_cell_idx]
                            p_ado = p_ado_list[p_ado_idx]
                            p_ae = p_ae_list[p_ae_idx]
                            phred_type = phred_type_list[phred_idx] # 0: p_err = 0, 1: p_err = 0.0001, 2: p_error\ in [0.0001, 0.01]

                            chr_id = 1
                            read_length = 100

                            global_dir = "../../../data_simulator/data/seed_" + str(seed_val)  + "_cells_" + str(num_cells) + "_mut_" \
                                         + str(mut_type) + "_ado_" + str(p_ado) + "_ae_" + str(p_ae) + "_phred_" + str(phred_type) + "/";

                            min_line = 0
                            max_line = 0

                            start_time_global = time.time()

                            print("Global directory: ", global_dir)
                            bam_dir = global_dir + "bam/"
                            proc_dir = global_dir + "processed_data/"
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
                            line_range = [min_line, num_total_gsnv]
                            if max_line != 0:
                                line_range[1] = max_line

                            all_gsnv_list = all_gsnv_list[line_range[0]: line_range[1]]
                            all_chr_list = all_chr_list[line_range[0]: line_range[1]]
                            print("\tLimiting the number of analysed gSNVs to: ", len(all_gsnv_list))

                            # STEP 2. Validate gSNV sites
                            start_time = time.time()
                            print("\n***** STEP 2\n")
                            bulk_bam_filename = bam_dir + "bulk.bam"

                            valid_gsnv_list, valid_chr_list = get_valid_gsnv(chr_list=all_chr_list, gsnv_list=all_gsnv_list,
                                                                             bulk_bam_filename=bulk_bam_filename, num_cells=num_cells,
                                                                             bam_dir=bam_dir)

                            print("\tTotal number of valid gSNV sites: ", len(valid_gsnv_list))
                            # print("\tValid gSNV sites: ", valid_gsnv_list)
                            end_time = time.time()
                            print("\n\tTotal time: ", end_time - start_time, "\n*****")

                            # STEP 3. Check pairs of sites
                            start_time = time.time()
                            print("\n***** STEP 3\n")
                            valid_hom_positions, valid_gsnv_positions, valid_chr_list, bulk_genotype_list = get_valid_site_pair(
                                chr_list=valid_chr_list, gsnv_list=valid_gsnv_list, bulk_bam_filename=bulk_bam_filename,
                                num_cells=num_cells, bam_dir=bam_dir, read_length=read_length)

                            print("\tTotal number of valid site pairs: ", len(valid_hom_positions))
                            # for i in range(len(valid_hom_positions)):
                            #    print("\t\tChr: ", valid_chr_list[i], "\tgSNV: ", valid_gsnv_positions[i], "\tHom: ", valid_hom_positions[i])

                            # STEP 4. Save valid sites
                            generate_final_dicts(proc_dir=proc_dir, chr_list=valid_chr_list, gsnv_list=valid_gsnv_positions,
                                                 hom_list=valid_hom_positions, bulk_genotypes=bulk_genotype_list, num_cells=num_cells,
                                                 bam_dir=bam_dir)

                            end_time = time.time()
                            print("\n\tTotal time: ", end_time - start_time, "\n*****")

                            ###
                            print("\n***** DONE!")
                            end_time_global = time.time()
                            print("\tTotal global time: ", end_time_global - start_time_global, "\n*****")


if __name__ == "__main__":
    main()
