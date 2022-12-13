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


def retrieve_double_pos_coverage(bam_file, cur_chr, cur_gsnv, hom_pos):
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

    site_coverage = np.zeros(16)  # Stores the number of nucleotide pairs appears in reads

    for read in bam_file.fetch(str(cur_chr), min(cur_gsnv, hom_pos)-1, max(cur_gsnv, hom_pos)-1):
        cigar_tup = read.cigartuples
        if len(cigar_tup) == 1 and cigar_tup[0][0] == 0:
            seq_len = read.infer_read_length()
            seq = read.query_alignment_sequence

            start_pos = read.pos
            end_pos = start_pos + seq_len

            if start_pos < cur_gsnv < end_pos and start_pos < hom_pos <= end_pos:
                het_idx = cur_gsnv - start_pos - 1  # because of indexing differences between pysam and igv
                nucleotide_het = seq[het_idx]

                hom_idx = hom_pos - start_pos - 1  # because of indexing differences between pysam and igv
                nucleotide_hom = seq[hom_idx]

                try:
                    map_idx = nucleotide_map[nucleotide_het+nucleotide_hom]
                    site_coverage[map_idx] += 1
                except:  # The except part is needed. Sometimes there are nucleotides other than ACGT, i.e. N.
                    num_invalid_nucleotide += 1

    return site_coverage


def retrieve_double_pos_coverage_around_gsnvs(bam_filename, cur_chr, cur_gsnv, read_length):
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

    site_coverage = np.zeros((2 * (read_length - 1), 16))  # Stores the number of nucleotide pairs appears in reads
    hom_pos_list = np.zeros(2 * (read_length - 1))

    bam_file = pysam.AlignmentFile(bam_filename, "rb")

    site_idx = 0
    for hom_pos in range(cur_gsnv - read_length + 1, cur_gsnv + read_length):
        # TODO we can also limit the maximum range to be the genome length
        if hom_pos >= 1 and hom_pos != cur_gsnv:
            hom_pos_list[site_idx] = int(hom_pos)
            site_coverage[site_idx, :] = retrieve_double_pos_coverage(bam_file, cur_chr, cur_gsnv, hom_pos)
            site_idx += 1

    bam_file.close()
    return site_coverage, hom_pos_list


def retrieve_double_pos_coverage_around_gsnvs_orig(bam_filename, cur_chr, cur_gsnv, read_length):
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

    site_coverage = np.zeros((2 * (read_length - 1), 16))  # Stores the number of nucleotide pairs appears in reads
    hom_pos_list = np.zeros(2 * (read_length - 1))

    bam_file = pysam.AlignmentFile(bam_filename, "rb")

    site_idx = 0
    for hom_pos in range(cur_gsnv - read_length + 1, cur_gsnv + read_length):
        # TODO we can also limit the maximum range to be the genome length
        if hom_pos >= 1 and hom_pos != cur_gsnv:
            hom_pos_list[site_idx] = int(hom_pos)
            site_coverage[site_idx, :] = retrieve_double_pos_coverage(bam_file, cur_chr, cur_gsnv, hom_pos)
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

        for read in bam_file.fetch(str(cur_chr), min(cur_gsnv, cur_hom) - 1, max(cur_gsnv, cur_hom)-1):
            cigar_tup = read.cigartuples
            if len(cigar_tup) == 1 and cigar_tup[0][0] == 0:
                seq_len = read.infer_read_length()
                seq = read.query_alignment_sequence
                qual = read.query_qualities

                start_pos = read.pos
                end_pos = start_pos + seq_len

                if start_pos < cur_gsnv < end_pos and start_pos < cur_hom <= end_pos:
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


def get_valid_gsnv_pileup(chr_list, gsnv_list, bulk_bam_filename, num_cells, bam_dir, genome_length=None):
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
    data = parse_pileup(bam_dir, num_cells, genome_length)

    num_all_gsnv = len(gsnv_list)

    valid_gsnv_list = []
    valid_chr_list = []

    for gsnv_idx in range(num_all_gsnv):
        cur_gsnv = gsnv_list[gsnv_idx]
        cur_chr = chr_list[gsnv_idx]

        idx = np.where((data[:, 0] == int(cur_chr)) & (data[:, 1] == cur_gsnv))[0]
        if len(idx) > 0:
            contents = data[idx[0], :]

            # Get bulk's coverage matrix
            bulk_het_coverage = contents[3:3 + 4]

            # Get single-cells' coverage matrices
            sc_indices = list(range(1, num_cells + 1))
            cell_het_coverage = np.zeros((num_cells, 4), dtype=int)
            for cell_idx in range(num_cells):
                sc_idx = sc_indices[cell_idx]
                cell_het_coverage[cell_idx, :] = contents[3 + sc_idx * 5:3 + sc_idx * 5 + 4]

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


def get_valid_site_pair(chr_list, gsnv_list, bulk_bam_filename, num_cells, bam_dir, read_length, genome_length=None):
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
        bulk_pair_coverage, hom_pos_list = retrieve_double_pos_coverage_around_gsnvs(
            bam_filename=bulk_bam_filename, cur_chr=cur_chr, cur_gsnv=cur_gsnv, read_length=read_length)

        # Get single-cells' coverage matrices
        cell_pair_coverage = np.zeros((num_cells, 2 * (read_length - 1), 16))
        for cell_idx in range(num_cells):
            cell_bam_filename = bam_dir + "cell_idx_" + str(cell_idx) + ".bam"
            cell_pair_coverage[cell_idx, :, :], _ = retrieve_double_pos_coverage_around_gsnvs(
                bam_filename=cell_bam_filename, cur_chr=cur_chr, cur_gsnv=cur_gsnv, read_length=read_length)

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

    #print("\tBulk pairs dictionary is saved to: \t", bulk_out_filename)

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

        #print("\tCell pairs dictionary is saved to: \t", cell_filename)

    temp_filename = proc_dir + "all_cell_dict_pair.pickle"
    save_dictionary(temp_filename, all_cell)
    print("\tAll cell pairs dictionary is saved to: \t", temp_filename)


def get_valid_hom(chr_list, pos_list, bulk_bam_filename, num_cells, bam_dir):
    """
    Given a list of positions, this function checks whether the cells are differ than bulk
    and returns the list of valid positions and the corresponding chromosomes.
    :param chr_list: Python list.
    :param pos_list: Python list of integers. Same length as chr_list.
    :param bulk_bam_filename: String. Path of bulk BAM file.
    :param num_cells: Integer. Number of cells.
    :param bam_dir: String. Path to cell BAM files.
    :return: valid_gsnv_list: Python list of integers. Contains the valid gSNV positions.
    :return: valid_chr_list: Python list. Same length as valid_gsnv_list. Contains the corresponding chromosomes.
    """
    num_all_pos = len(pos_list)

    valid_pos_list = []
    valid_chr_list = []
    bulk_genotype_list = []

    for pos_idx in range(num_all_pos):
        cur_pos = pos_list[pos_idx]
        cur_chr = str(chr_list[pos_idx])
        # Get bulk's coverage matrix
        bulk_hom_coverage = retrieve_single_pos_coverage(bam_filename=bulk_bam_filename,
                                                         cur_chr=cur_chr, cur_gsnv=cur_pos)

        # Get single-cells' coverage matrices
        cell_hom_coverage = np.zeros((num_cells, 4), dtype=int)
        for cell_idx in range(num_cells):
            cell_bam_filename = bam_dir + "cell_idx_" + str(cell_idx) + ".bam"
            cell_hom_coverage[cell_idx, :] = retrieve_single_pos_coverage(bam_filename=cell_bam_filename,
                                                                          cur_chr=cur_chr, cur_gsnv=cur_pos)
        # Check validity
        is_valid, bulk_gen = check_hom_validity(bulk_hom_coverage, cell_hom_coverage)
        if is_valid:
            valid_pos_list.append(cur_pos)
            valid_chr_list.append(cur_chr)
            bulk_genotype_list.append(bulk_gen)

    return valid_pos_list, valid_chr_list, bulk_genotype_list


def parse_pileup(bam_dir, num_cells, genome_length=None):
    filename = bam_dir + "cells_pile.mpileup"
    out_filename = bam_dir + "cells_pile_parsed.txt"

    try:
        data = np.loadtxt(out_filename, delimiter='\t')
        print("\tLoaded parsed pileup file")

    except:
        nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        if genome_length is None:
            # Read file once and gather stats
            num_lines = sum(1 for line in open(filename))
            print("num_lines: ", num_lines)

        # Create file to store counts
        data = np.empty((genome_length, (num_cells+1)*5 + 3), dtype=int)

        # Parse pileup
        with open(filename, "r") as pileup_file:
            i = 0
            for line in pileup_file:
                contents = line.split("\t")
                ref_nuc = contents[2]

                data[i, 0] = int(contents[0])  # Chr
                data[i, 1] = int(contents[1])  # Pos
                try:
                    data[i, 2] = int(nuc_map[ref_nuc])  # Ref nuc idx
                except:
                    data[i, 2] = -1  # Invalid reference nuc

                for s in range(num_cells+1):
                    #sample_contents = contents[(s+1)*3:(s+2)*3]
                    sample_nucleotides = contents[(s+1)*3+1]
                    #print("\ts: ", s, "\t", sample_nucleotides)

                    counts = np.zeros(5, dtype=int)
                    for ch in sample_nucleotides:
                        #print("ch: ", ch)
                        if ch == 'A' or ch == 'a':
                            counts[0] += 1
                        elif ch == 'C' or ch == 'c':
                            counts[1] += 1
                        elif ch == 'G' or ch == 'g':
                            counts[2] += 1
                        elif ch == 'T' or ch == 't':
                            counts[3] += 1
                        elif (ch == '.' or ch == ',') and ref_nuc in nuc_map:
                            counts[nuc_map[ref_nuc]] += 1
                    counts[4] = np.sum(counts) # Depth

                    data[i, 3+s*5:3+(s+1)*5] = counts

                i += 1

        np.savetxt(out_filename, data, fmt='%d', delimiter='\t')
        print("\tParsed pileup file is saved to: ", out_filename)

    return data


def get_valid_hom_pileup(chr_list, pos_list, bulk_bam_filename, num_cells, bam_dir, genome_length=None):
    """
    Given a list of positions, this function checks whether the cells are differ than bulk
    and returns the list of valid positions and the corresponding chromosomes.
    :param chr_list: Python list.
    :param pos_list: Python list of integers. Same length as chr_list.
    :param bulk_bam_filename: String. Path of bulk BAM file.
    :param num_cells: Integer. Number of cells.
    :param bam_dir: String. Path to cell BAM files.
    :return: valid_gsnv_list: Python list of integers. Contains the valid gSNV positions.
    :return: valid_chr_list: Python list. Same length as valid_gsnv_list. Contains the corresponding chromosomes.
    """
    data = parse_pileup(bam_dir, num_cells, genome_length)

    num_all_pos = len(pos_list)

    valid_pos_list = []
    valid_chr_list = []
    bulk_genotype_list = []

    for pos_idx in range(num_all_pos):
        cur_pos = pos_list[pos_idx]
        cur_chr = str(chr_list[pos_idx])

        idx = np.where((data[:, 0] == int(cur_chr)) & (data[:, 1] == cur_pos))[0]
        if len(idx) > 0:
            contents = data[idx[0], :]

            # Get bulk's coverage matrix
            bulk_hom_coverage = contents[3:3+4]

            # Get single-cells' coverage matrices
            sc_indices = list(range(1, num_cells+1))
            cell_hom_coverage = np.zeros((num_cells, 4), dtype=int)
            for cell_idx in range(num_cells):
                sc_idx = sc_indices[cell_idx]
                cell_hom_coverage[cell_idx, :] = contents[3 + sc_idx * 5:3 + sc_idx * 5 + 4]

             # Check validity
            is_valid, bulk_gen = check_hom_validity(bulk_hom_coverage, cell_hom_coverage)
            if is_valid:
                valid_pos_list.append(cur_pos)
                valid_chr_list.append(cur_chr)
                bulk_genotype_list.append(bulk_gen)

    return valid_pos_list, valid_chr_list, bulk_genotype_list


def check_hom_validity(bulk_hom_coverage, cell_hom_coverage):
    """
    Given bulk and single-cell coverage matrices of a homozygous site,
    it checks whether the sites are worh for dataset inclusion or not.
    :param bulk_hom_coverage: Numpy (4, ) array of integers. Contains how many ACGT are observed in reads.
    :param cell_hom_coverage: Numpy (num_cells, 4) array of integers.
                              Contains how many ACGT are observed in each single-cell reads.
    :return: Boolean. If True, site is eligible for read-phasing.
    """
    # Get the total number of reads at the site.
    marg_bulk_hom_coverage = np.sum(bulk_hom_coverage)
    marg_cell_hom_coverage = np.sum(cell_hom_coverage, axis=1, keepdims=True)

    # Get the frequency of reads at the site.
    # Note: Shows warning due to division by zero, as expected. It doesn't effect the performance.
    bulk_hom_freq = np.divide(bulk_hom_coverage, marg_bulk_hom_coverage)
    cell_hom_freq = np.divide(cell_hom_coverage, marg_cell_hom_coverage)

    # Filter the matrices.
    bulk_hom_coverage[bulk_hom_coverage < 10] = 0
    bulk_hom_coverage[bulk_hom_coverage >= 10] = 1
    bulk_hom_freq[bulk_hom_freq < 0.8] = 0
    bulk_hom_freq[bulk_hom_freq >= 0.8] = 1
    temp_bulk = np.multiply(bulk_hom_coverage, bulk_hom_freq)

    # Here, we put 0 to bulk genotype and 1 to any other genotype.
    temp_bulk_inv = np.copy(temp_bulk)
    temp_bulk_inv[temp_bulk_inv == 0] = -1
    temp_bulk_inv[temp_bulk_inv == 1] = 0
    temp_bulk_inv[temp_bulk_inv == -1] = 1

    cell_hom_coverage[cell_hom_coverage < 2] = 0
    cell_hom_coverage[cell_hom_coverage >= 2] = 1
    cell_hom_freq[cell_hom_freq < 0.2] = 0
    cell_hom_freq[cell_hom_freq >= 0.2] = 1
    temp = np.multiply(cell_hom_coverage, cell_hom_freq)

    # Check visibility. At least two single-cells should be visible in order to allow read-phasing.
    temp_vis = np.multiply(temp_bulk_inv, temp)
    marg_temp_vis = np.sum(temp_vis, axis=0)
    if len(np.where(marg_temp_vis >= 2)[0]) >= 1:
        bulk_gen = np.where(temp_bulk == 1)[0][0]
        return True, bulk_gen
    else:
        return False, ""


def generate_final_singleton_dicts(proc_dir, chr_list, hom_list, bulk_genotypes, num_cells, bam_dir):
    bulk_pairs_dict = {}
    for pos_idx in range(len(chr_list)):
        pos_pair = str(hom_list[pos_idx])
        bulk_pairs_dict[pos_pair] = bulk_genotypes[pos_idx]

    bulk_out_filename = proc_dir + "bulk_singleton.pickle"
    save_dictionary(bulk_out_filename, bulk_pairs_dict)

    #print("\tBulk pairs dictionary is saved to: \t", bulk_out_filename)

    all_cell = {}
    for cell_idx in range(num_cells):
        all_cell[str(cell_idx)] = {}

    for cell_idx in range(num_cells):
        cell_bam_filename = bam_dir + "cell_idx_" + str(cell_idx) + ".bam"
        all_reads, all_quals = retrieve_singleton_reads_qualities(bam_filename=cell_bam_filename, chr_list=chr_list,
                                                             hom_list=hom_list)

        for pos_idx in range(len(chr_list)):
            pos_pair = str(hom_list[pos_idx])
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
        cell_filename = proc_dir + "cell_" + str(cell_idx) + "_singleton.pickle"
        save_dictionary(cell_filename, all_cell[str(cell_idx)])

        #print("\tCell pairs dictionary is saved to: \t", cell_filename)

    temp_filename = proc_dir + "all_cell_dict_singleton.pickle"
    save_dictionary(temp_filename, all_cell)
    print("\tAll cell pairs dictionary is saved to: \t", temp_filename)


def retrieve_singleton_reads_qualities(bam_filename, chr_list, hom_list):
    """
    Given a site, this function retrieves the reads covering the site and their Phred scores.
    :param bam_filename: String. Path of the BAM file.
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
        cur_hom = hom_list[idx]

        pair_reads = []
        pair_quals = []

        for read in bam_file.fetch(str(cur_chr), cur_hom - 1, cur_hom):
            cigar_tup = read.cigartuples
            if len(cigar_tup) == 1 and cigar_tup[0][0] == 0:
                seq_len = read.infer_read_length()
                seq = read.query_alignment_sequence
                qual = read.query_qualities

                start_pos = read.pos
                end_pos = start_pos + seq_len

                if start_pos < cur_hom <= end_pos:
                    hom_idx = cur_hom - start_pos - 1  # because of indexing differences between pysam and igv
                    nucleotide_hom = seq[hom_idx]
                    qual_hom = qual[hom_idx]

                    try:
                        hom_map_idx = nucleotide_map[nucleotide_hom]

                        pair_reads.append(nucleotide_hom)
                        pair_quals.append(qual_hom)
                    except:  # The except part is needed. Sometimes there are nucleotides other than ACGT, i.e. N.
                        num_invalid_nucleotide += 1

        all_reads.append(pair_reads)
        all_quals.append(pair_quals)

    bam_file.close()
    return all_reads, all_quals


def combine_dictionaries(proc_dir, scuphr_strategy):
    pair_dict = load_dictionary(proc_dir + "all_cell_dict_pair.pickle")
    if scuphr_strategy == "singleton" or scuphr_strategy == "hybrid":

        bulk_pair_dict = load_dictionary(proc_dir + "bulk_pairs.pickle")
        bulk_singleton_dict = load_dictionary(proc_dir + "bulk_singleton.pickle")

        bulk_pair_dict.update(bulk_singleton_dict)
        bulk_out_filename = proc_dir + "bulk.pickle"
        save_dictionary(bulk_out_filename, bulk_pair_dict)
        print("\tBulk dictionary is saved to: \t", bulk_out_filename)

        singleton_dict = load_dictionary(proc_dir + "all_cell_dict_singleton.pickle")
        for key in pair_dict.keys():
            pair_dict[key].update(singleton_dict[key])

            cell_filename = proc_dir + "cell_" + key + ".pickle"
            save_dictionary(cell_filename, pair_dict[key])

    out_filename = proc_dir + "all_cell_dict.pickle"
    save_dictionary(out_filename, pair_dict)
    print("\tAll cell dictionary is saved to: \t", out_filename)


def main():
    # code to process command line arguments
    parser = argparse.ArgumentParser(description='Site detection.')
    parser.add_argument('global_dir', help="Specify the directory.", type=str)
    parser.add_argument('num_cells', help="Specify the number of cells.", type=int)
    parser.add_argument('--genome_length', help="Specify the genome length.", type=int, default=0)
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
    parser.add_argument('--scuphr_strategy',
                        help="Specify the strategy for Scuphr (paired, singleton, hybrid). Default: paired",
                        type=str, default="paired")
    parser.add_argument('--seed_val', help="Specify the seed. Default: 123", type=int, default=123)
    args = parser.parse_args()

    start_time_global = time.time()

    print("Global directory: ", args.global_dir)
    bam_dir = args.global_dir + "bam/"
    proc_dir = args.global_dir + "processed_data/"
    if not os.path.exists(proc_dir):
        os.makedirs(proc_dir)
        print("Directory is created")
    pileup_filename = bam_dir + "cells_pile.mpileup"

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

    # STEP 2. Validate gSNV sites
    start_time = time.time()
    print("\n***** STEP 2\n")
    bulk_bam_filename = bam_dir + "bulk.bam"

    # Find the genome length.
    genome_length = args.genome_length
    # TODO There must be a more efficient way.
    if genome_length == 0:
        bam_file = pysam.AlignmentFile(bulk_bam_filename, "rb")
        for read in bam_file.fetch(str(args.chr_id)):
            genome_length = max(genome_length, read.pos + read.infer_read_length())
    print("\tGenome length is: ", genome_length)

    if os.path.exists(pileup_filename):
        valid_gsnv_list, valid_chr_list = get_valid_gsnv_pileup(chr_list=all_chr_list, gsnv_list=all_gsnv_list,
                                                                bulk_bam_filename=bulk_bam_filename,
                                                                num_cells=args.num_cells, bam_dir=bam_dir,
                                                                genome_length=genome_length)
    else:
        valid_gsnv_list, valid_chr_list = get_valid_gsnv(chr_list=all_chr_list, gsnv_list=all_gsnv_list,
                                                         bulk_bam_filename=bulk_bam_filename, num_cells=args.num_cells,
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
        num_cells=args.num_cells, bam_dir=bam_dir, read_length=args.read_length)

    print("\tTotal number of valid site pairs: ", len(valid_hom_positions))
    # for i in range(len(valid_hom_positions)):
    #    print("\t\tChr: ", valid_chr_list[i], "\tgSNV: ", valid_gsnv_positions[i], "\tHom: ", valid_hom_positions[i])

    # STEP 4. Save valid sites
    print("\n***** STEP 4\n")
    generate_final_dicts(proc_dir=proc_dir, chr_list=valid_chr_list, gsnv_list=valid_gsnv_positions,
                         hom_list=valid_hom_positions, bulk_genotypes=bulk_genotype_list, num_cells=args.num_cells,
                         bam_dir=bam_dir)

    # If the scuphr_strategy is singleton or hybrid, rin site detection away from gSNVs as well.
    if args.scuphr_strategy == "singleton" or args.scuphr_strategy == "hybrid":
        print("\n***** STEP 5\n")

        # Remove gSNVs and their closeby locations from positions-to-investigate
        no_closeby_positions = list(np.arange(args.genome_length)+1)
        for cur_gsnv in all_gsnv_list:
            if cur_gsnv in no_closeby_positions:
                no_closeby_positions.remove(cur_gsnv)
            for i in range(args.read_length):
                cur_pos = cur_gsnv - i
                if cur_pos in no_closeby_positions:
                    no_closeby_positions.remove(cur_pos)
                cur_pos = cur_gsnv + i
                if cur_pos in no_closeby_positions:
                    no_closeby_positions.remove(cur_pos)
        print("\tNumber of no-closeby positions: ", len(no_closeby_positions))
        #print("\tno-closeby positions: ", no_closeby_positions)

        all_hom_chr_list = np.ones(len(no_closeby_positions), dtype=int)

        if os.path.exists(pileup_filename):
            valid_hom_list, valid_hom_chr_list, bulk_hom_genotype_list = get_valid_hom_pileup(
                chr_list=all_hom_chr_list, pos_list=no_closeby_positions, bulk_bam_filename=bulk_bam_filename,
                num_cells=args.num_cells, bam_dir=bam_dir, genome_length=genome_length)
        else:
            valid_hom_list, valid_hom_chr_list, bulk_hom_genotype_list = get_valid_hom(
                chr_list=all_hom_chr_list, pos_list=no_closeby_positions, bulk_bam_filename=bulk_bam_filename,
                num_cells=args.num_cells, bam_dir=bam_dir)
        print("\tTotal number of valid hom sites: ", len(valid_hom_list))
        print("\tValid hom sites: ", valid_hom_list)
        #print("\tBulk hom genotypes: ", bulk_hom_genotype_list)

        generate_final_singleton_dicts(proc_dir=proc_dir, chr_list=valid_hom_chr_list, hom_list=valid_hom_list,
                                       bulk_genotypes=bulk_hom_genotype_list, num_cells=args.num_cells, bam_dir=bam_dir)

    #  Combine dictionaries.
    print("\n***** STEP 6\n")
    combine_dictionaries(proc_dir, args.scuphr_strategy)

    end_time = time.time()
    print("\n\tTotal time: ", end_time - start_time, "\n*****")

    ###
    print("\n***** DONE!")
    end_time_global = time.time()
    print("\tTotal global time: ", end_time_global - start_time_global, "\n*****")


if __name__ == "__main__":
    main()
