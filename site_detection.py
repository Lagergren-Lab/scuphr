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


def convert_base_to_idx(read):
    if len(read) == 2:
        read_idx = [0, 0]
        if read[0] == "C":
            read_idx[0] = 1
        elif read[0] == "G":
            read_idx[0] = 2
        elif read[0] == "T":
            read_idx[0] = 3
        if read[1] == "C":
            read_idx[1] = 1
        elif read[1] == "G":
            read_idx[1] = 2
        elif read[1] == "T":
            read_idx[1] = 3
    # If the read is singleton
    else:
        if read == "A":
            read_idx = 0
        elif read == "C":
            read_idx = 1
        elif read == "G":
            read_idx = 2
        elif read == "T":
            read_idx = 3
    return read_idx


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


def parse_pileup(bam_dir, filename, out_filename, num_cells, genome_length=0):
    #out_filename = bam_dir + "cells_pile_parsed.txt"

    #try:
    #    data = np.loadtxt(out_filename, delimiter='\t')
    #    print("\tLoaded parsed pileup file")

    #except:
    if True:
        nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        if genome_length == 0:
            # Read file once and gather stats
            num_lines = sum(1 for line in open(filename))
            print("num_lines: ", num_lines)
        else:
            num_lines = genome_length

        # Create file to store counts
        data = np.empty((num_lines, (num_cells+1)*5 + 3), dtype=int)

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
                    counts[4] = np.sum(counts)  # Depth

                    data[i, 3+s*5:3+(s+1)*5] = counts

                i += 1

        np.savetxt(out_filename, data, fmt='%d', delimiter='\t')
        print("\tParsed pileup file is saved to: ", out_filename)

    return data.astype(int)


def get_valid_gsnv_pileup(data, key_dict, chr_list, gsnv_list, num_cells):
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
    bulk_genotype_list = []

    min_pos = data[0, 1]
    max_pos = data[-1, 1]

    for gsnv_idx in range(num_all_gsnv):
        cur_gsnv = gsnv_list[gsnv_idx]
        cur_chr = chr_list[gsnv_idx]

        if cur_gsnv > max_pos:
            break
        if cur_gsnv < min_pos:
            continue

        key_str = str(cur_chr) + "_" + str(cur_gsnv)
        idx = key_dict[key_str]
        contents = data[idx, :]

        #idx = np.where((data[:, 0] == int(cur_chr)) & (data[:, 1] == cur_gsnv))[0]
        #if len(idx) > 0:
        #    contents = data[idx[0], :]

        # Get bulk's coverage matrix
        bulk_het_coverage = contents[3:3 + 4]

        # Get single-cells' coverage matrices
        sc_indices = list(range(1, num_cells + 1))
        cell_het_coverage = np.zeros((num_cells, 4), dtype=int)
        for cell_idx in range(num_cells):
            sc_idx = sc_indices[cell_idx]
            cell_het_coverage[cell_idx, :] = contents[3 + sc_idx * 5:3 + sc_idx * 5 + 4]

        # Check validity
        is_valid, bulk_gen = check_gsnv_validity(bulk_het_coverage, cell_het_coverage)
        if is_valid:
            valid_gsnv_list.append(cur_gsnv)
            valid_chr_list.append(cur_chr)
            bulk_genotype_list.append(bulk_gen)

    return valid_gsnv_list, valid_chr_list, bulk_genotype_list


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
        bulk_gen = np.where(temp_bulk == 1)[0]
        return True, bulk_gen
    else:
        return False, ""


def get_valid_hom_pileup(data, key_dict, chr_list, num_cells, genome_length, gsnv_list=None, pos_list=None):
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
    if pos_list is not None:
        num_all_pos = len(pos_list)
    else:
        #num_all_pos = genome_length
        pos_list = list(data[:, 1])
        num_all_pos = len(pos_list)

    valid_pos_list = []
    valid_chr_list = []
    bulk_genotype_list = []

    for pos_idx in range(num_all_pos):
        cur_pos = -9

        if pos_list is not None:
            cur_pos = pos_list[pos_idx]
        elif pos_idx+1 not in gsnv_list:
            cur_pos = pos_idx + 1

        if cur_pos != -9:
            try:
                cur_chr = str(int(chr_list[pos_idx]))
            except:
                cur_chr = "1"

            key_str = cur_chr + "_" + str(cur_pos)
            idx = key_dict[key_str]
            contents = data[idx, :]

            #idx = np.where((data[:, 0] == int(cur_chr)) & (data[:, 1] == cur_pos))[0]
            #if len(idx) > 0:
            #    contents = data[idx[0], :]

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


def get_valid_hom_pileup_avoid(data, key_dict, chr_list, num_cells, all_gsnv_list, genome_length, read_length):
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
    num_gsvs = len(all_gsnv_list)

    valid_pos_list = []
    valid_chr_list = []
    bulk_genotype_list = []

    for cur_pos in range(1, genome_length+1): # Since genome is 1-based system, starts with position 1.
        # TODO Also add chr search
        cur_chr = "1"
        # Find closest gSNV from all_gsnv_list
        closest_gsnv_idx = np.searchsorted(all_gsnv_list, cur_pos) - 1
        if closest_gsnv_idx < 0:
            closest_gsnv_idx += 1
            closest_gsnv = all_gsnv_list[closest_gsnv_idx]
        elif closest_gsnv_idx + 1 == num_gsvs:
            closest_gsnv = all_gsnv_list[closest_gsnv_idx]
        else:
            closest_gsnv = all_gsnv_list[closest_gsnv_idx]
            g2 = all_gsnv_list[closest_gsnv_idx + 1]
            if np.abs(g2 - cur_pos) < np.abs(closest_gsnv - cur_pos):
                closest_gsnv = g2
                closest_gsnv_idx += 1

        # If the cur_pos is away from gSNVs, continue
        if np.abs(closest_gsnv - cur_pos) >= read_length:
            key_str = cur_chr + "_" + str(cur_pos)
            idx = key_dict[key_str]
            contents = data[idx, :]

            #idx = np.where((data[:, 0] == int(cur_chr)) & (data[:, 1] == cur_pos))[0]
            #if len(idx) > 0:
            #    contents = data[idx[0], :]

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
    marg_temp_vis = np.nansum(temp_vis, axis=0)
    if len(np.where(marg_temp_vis >= 2)[0]) >= 1:
        try:
            bulk_gen = np.where(temp_bulk == 1)[0][0]
            return True, bulk_gen
        except:
            return False, ""
    else:
        return False, ""


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

            flag = True
            for op, count in cigar_tup:
                if op > 2:
                    flag = False
                    break

            if flag:
            #if len(cigar_tup) == 1 and cigar_tup[0][0] == 0:
                seq_len = read.infer_read_length()
                seq_orig = read.query_alignment_sequence
                qual_orig = read.query_qualities

                seq = ""
                qual = []
                i_ = 0
                for op, count in cigar_tup:
                    if op == 0:
                        for c in range(count):
                            seq += seq_orig[i_]
                            qual.append(qual_orig[i_])
                            i_ += 1
                    elif op == 1:
                        for c in range(count):
                            i_ += 1
                    elif op == 2:
                        for c in range(count):
                            seq += "-"
                            qual.append(0)

                start_pos = read.pos
                end_pos = start_pos + len(seq)

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


def combine_dictionaries(proc_dir, scuphr_strategy, base_range_min=0, base_range_max=0):
    if scuphr_strategy == "paired":
        final_dict = load_dictionary(proc_dir + "all_cell_dict_pair.pickle")
        bulk_final_dict = load_dictionary(proc_dir + "bulk_pairs.pickle")
    elif scuphr_strategy == "singleton":
        final_dict = load_dictionary(proc_dir + "all_cell_dict_singleton.pickle")
        bulk_final_dict = load_dictionary(proc_dir + "bulk_singleton.pickle")

    else:  # hybrid
        bulk_pair_dict = load_dictionary(proc_dir + "bulk_pairs.pickle")
        bulk_singleton_dict = load_dictionary(proc_dir + "bulk_singleton.pickle")

        pair_dict = load_dictionary(proc_dir + "all_cell_dict_pair.pickle")
        singleton_dict = load_dictionary(proc_dir + "all_cell_dict_singleton.pickle")

        # Remove redundant homozygous sites
        hom_pos_list = []
        for key in bulk_pair_dict.keys():
            hom_pos_list.append(key.split("_")[1])

        num_removed = 0
        for hom_pos in hom_pos_list:
            try:
                del bulk_singleton_dict[hom_pos]
                for cell in singleton_dict.keys():
                    del singleton_dict[cell][hom_pos]
                num_removed += 1
            except KeyError:
                pass
        print("\tNumber of removed duplicate sites: \t", num_removed)

        bulk_pair_dict.update(bulk_singleton_dict)
        bulk_out_filename = proc_dir + "bulk.pickle"
        save_dictionary(bulk_out_filename, bulk_pair_dict)
        print("\tBulk dictionary is saved to: \t", bulk_out_filename)

        for key in pair_dict.keys():
            pair_dict[key].update(singleton_dict[key])

        final_dict = pair_dict
        bulk_final_dict = bulk_pair_dict

    # Save for current range
    out_filename = proc_dir + "all_cell_dict" + "_" + str(base_range_min) + "_" + str(base_range_max) + ".pickle"
    save_dictionary(out_filename, final_dict)
    out_filename = proc_dir + "bulk" + "_" + str(base_range_min) + "_" + str(base_range_max) + ".pickle"
    save_dictionary(out_filename, bulk_final_dict)
    for key in final_dict.keys():
        cell_filename = proc_dir + "cell_" + key + "_" + str(base_range_min) + "_" + str(base_range_max) + ".pickle"
        save_dictionary(cell_filename, final_dict[key])

    # Save as final
    out_filename = proc_dir + "all_cell_dict.pickle"
    save_dictionary(out_filename, final_dict)
    out_filename = proc_dir + "bulk.pickle"
    save_dictionary(out_filename, bulk_final_dict)
    for key in final_dict.keys():
        cell_filename = proc_dir + "cell_" + key + ".pickle"
        save_dictionary(cell_filename, final_dict[key])

    print("\tAll cell dictionary is saved to: \t", out_filename)


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

            flag = True
            for op, count in cigar_tup:
                if op > 2:
                    flag = False
                    break

            if flag:
            #if len(cigar_tup) == 1 and cigar_tup[0][0] == 0:
                seq_len = read.infer_read_length()
                seq_orig = read.query_alignment_sequence
                qual_orig = read.query_qualities

                seq = ""
                qual = []
                i_ = 0
                for op, count in cigar_tup:
                    if op == 0:
                        for c in range(count):
                            seq += seq_orig[i_]
                            qual.append(qual_orig[i_])
                            i_ += 1
                    elif op == 1:
                        for c in range(count):
                            i_ += 1
                    elif op == 2:
                        for c in range(count):
                            seq += "-"
                            qual.append(0)

                start_pos = read.pos
                end_pos = start_pos + len(seq)

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


def get_site_pairs(chr_list, all_gsnv_list, hom_list, num_cells, bam_dir, bulk_gsnv_genotype_list, bulk_hom_genotype_list):
    nucleotide_map = {0: [0, 0], 1: [0, 1], 2: [0, 2], 3: [0, 3], 4: [1, 0], 5: [1, 1], 6: [1, 2], 7: [1, 3],
                      8: [2, 0], 9: [2, 1], 10: [2, 2], 11: [2, 3], 12: [3, 0], 13: [3, 1], 14: [3, 2], 15: [3, 3]}

    num_all_gsnv = len(all_gsnv_list)
    all_gsnv_list = np.array(all_gsnv_list)
    gsnv_list = []

    bulk_pairs_dict = {}
    for pos_idx in range(len(hom_list)):
        hom_pos = hom_list[pos_idx]
        # Find closest gSNV from all_gsnv_list and store
        closest_gsnv_idx = np.searchsorted(all_gsnv_list, hom_pos) - 1
        if closest_gsnv_idx < 0:
            closest_gsnv_idx += 1
            closest_gsnv = all_gsnv_list[closest_gsnv_idx]
        else:
            closest_gsnv = all_gsnv_list[closest_gsnv_idx]
            if closest_gsnv_idx+1 != num_all_gsnv:
                g2 = all_gsnv_list[closest_gsnv_idx+1]
                if np.abs(g2 - hom_pos) < np.abs(closest_gsnv - hom_pos):
                    closest_gsnv = g2
                    closest_gsnv_idx += 1

        gsnv_list.append(closest_gsnv)

        pos_pair = str(closest_gsnv) + "_" + str(hom_pos)

        bulk_gen = [[bulk_gsnv_genotype_list[closest_gsnv_idx][0], bulk_hom_genotype_list[pos_idx]],
                    [bulk_gsnv_genotype_list[closest_gsnv_idx][1], bulk_hom_genotype_list[pos_idx]]]
        bulk_pairs_dict[pos_pair] = np.array(bulk_gen)

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

    return bulk_pairs_dict, all_cell


def get_valid_site_pair(bulk_dict, all_cell_dict, num_cells):
    nucleotide_map = {str([0, 0]): 0, str([0, 1]): 1, str([0, 2]): 2, str([0, 3]): 3, str([1, 0]): 4, str([1, 1]): 5,
                      str([1, 2]): 6, str([1, 3]): 7, str([2, 0]): 8, str([2, 1]): 9, str([2, 2]): 10,
                      str([2, 3]): 11, str([3, 0]): 12, str([3, 1]): 13, str([3, 2]): 14, str([3, 3]): 15}

    key_list = list(bulk_dict.keys())
    for pos_pair in key_list:
        bulk_gen = bulk_dict[pos_pair]
        temp_bulk = np.zeros((1, 16))
        temp_bulk[0, nucleotide_map[str(list(bulk_gen[0]))]] = 1
        temp_bulk[0, nucleotide_map[str(list(bulk_gen[1]))]] = 1

        # Here, we put 0 to bulk genotype and 1 to any other genotype.
        temp_bulk_inv = np.ones((1, 16))
        temp_bulk_inv[0, nucleotide_map[str(list(bulk_gen[0]))]] = 0
        temp_bulk_inv[0, nucleotide_map[str(list(bulk_gen[1]))]] = 0

        # Get single-cells' coverage matrices
        cell_pair_coverage = np.zeros((num_cells, 16))
        for cell_idx in range(num_cells):
            cell_reads = all_cell_dict[str(cell_idx)][pos_pair]['reads']
            for read in cell_reads:
                read_idx = convert_base_to_idx(read)
                cell_pair_coverage[cell_idx, nucleotide_map[str(read_idx)]] += 1

        # Then, check homozygous sites and find the single-cells' disagreement with bulk.
        # Get the total number of reads at the site-pair.
        marg_cell_pair_coverage = np.sum(cell_pair_coverage, axis=1, keepdims=True)

        # Get the frequency of reads at the site-pair.
        # Note: Shows warning due to division by zero, as expected. It doesn't effect the performance.
        cell_pair_freq = np.divide(cell_pair_coverage, marg_cell_pair_coverage)

        # Filter the single-cell matrix.
        cell_pair_coverage[cell_pair_coverage < 2] = 0
        cell_pair_coverage[cell_pair_coverage >= 2] = 1
        cell_pair_freq[cell_pair_freq < 0.2] = 0
        cell_pair_freq[cell_pair_freq >= 0.2] = 1
        temp = np.multiply(cell_pair_coverage, cell_pair_freq)

        # Check visibility. At least two, at most num_cells-1 single-cells should be visible.
        temp_vis = np.multiply(temp_bulk_inv, temp)
        marg_temp_vis = np.sum(np.nan_to_num(temp_vis), axis=0, keepdims=True)
        temp_indices, votes = np.where(marg_temp_vis >= 2)
        temp_indices_2 = np.where(marg_temp_vis < num_cells)[0]
        intersect_indices = list(set(temp_indices) & set(temp_indices_2))

        for ind in intersect_indices:
            # Additional filter to avoid mutations in gSNV sites.
            # TODO This part might need more filters. For instance,
            #  if a subset of cells vote for mutation at gSNV site, others vote for mutation in homozygous site.
            if (np.where(temp_bulk[ind, :] == 1)[0][0] % 4) == (votes[list(temp_indices).index(ind)] % 4):
                del bulk_dict[pos_pair]
                for cell_idx in range(num_cells):
                    del all_cell_dict[str(cell_idx)][pos_pair]

        if len(intersect_indices) == 0:
            del bulk_dict[pos_pair]
            for cell_idx in range(num_cells):
                del all_cell_dict[str(cell_idx)][pos_pair]

    return bulk_dict, all_cell_dict


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
    parser.add_argument('--base_range_min', help="Specify the line number of min base position. Default: 0", type=int,
                        default=0)
    parser.add_argument('--base_range_max', help="Specify the line number of max base position. Default: 0", type=int,
                        default=0)
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

    print("Scuphr strategy: ", args.scuphr_strategy)

    pileup_filename = bam_dir + "cells_pile.mpileup"

    if args.base_range_min != 0 or args.base_range_max != 0:
        print("\nSTEP 0: Creating a new, smaller pileup file")
        start_time = time.time()
        orig_pileup_filename = bam_dir + "cells_pile_orig.mpileup"
        pileup_filename = bam_dir + "cells_pile.mpileup" + "_" + str(args.base_range_min) + "_" + str(args.base_range_max)
        #cmd = "awk \'NR>=" + str(args.base_range_min) + " && NR<=" + str(args.base_range_max) \
        #      + "\' " + orig_pileup_filename + " > " + pileup_filename
        cmd = "sed -n \'" + str(args.base_range_min) + "," + str(args.base_range_max) \
              + "p\' " + orig_pileup_filename + " > " + pileup_filename
        #print("cmd: ", cmd)
        os.system(cmd)
        print("\tTotal time: ", time.time() - start_time)

    # STEP 1. Parse pileup file
    print("\nSTEP 1: Parsing pileup file")
    start_time = time.time()

    out_parse_filename = bam_dir + "cells_pile_parsed.txt"
    if args.base_range_min != 0 or args.base_range_max != 0:
        out_parse_filename = bam_dir + "cells_pile_parsed.txt" + "_" + str(args.base_range_min) + "_" + str(args.base_range_max)

    data = parse_pileup(bam_dir, pileup_filename, out_parse_filename, args.num_cells, args.genome_length)

    # Create a small dictionary for fast access to data rows later on
    key_dict = {}
    for pos_idx in range(data.shape[0]):
        key_str = str(data[pos_idx, 0]) + "_" + str(data[pos_idx, 1])
        key_dict[key_str] = pos_idx
    print("\tTotal time: ", time.time() - start_time)

    # STEP 2. Load gSNV data
    print("\nSTEP 2: Loading gSNV data")
    start_time = time.time()
    regions_filename = bam_dir + "gsnv_vars_regions.bed"
    all_gsnv_list, all_chr_list = get_all_gsnv_list(regions_filename=regions_filename)
    num_total_gsnv = len(all_gsnv_list)
    print("\tTotal number of gSNV sites: ", num_total_gsnv)
    print("\tTotal time: ", time.time() - start_time)

    # STEP 2.1. Limit the gSNV sites based on the given arguments.
    print("\nSTEP 2.1: Limiting the gSNV range")
    start_time = time.time()
    line_range = [args.min_line, num_total_gsnv]
    if args.max_line != 0:
        line_range[1] = args.max_line
    all_gsnv_list = all_gsnv_list[line_range[0]: line_range[1]]
    all_chr_list = all_chr_list[line_range[0]: line_range[1]]
    print("\tLimiting the number of analysed gSNVs to: ", len(all_gsnv_list))
    print("\tTotal time: ", time.time() - start_time)

    # PAIRED SITE DETECTION
    if args.scuphr_strategy == "paired" or args.scuphr_strategy == "hybrid":
        print("\nPaired site detection starts...")
        start_time_paired = time.time()

        # STEP 3. Get valid gSNV positions (suitable for phasing)
        print("\nSTEP 3: Getting valid gSNV positions")
        start_time = time.time()
        valid_gsnv_list, valid_chr_list, bulk_gsnv_genotype_list = get_valid_gsnv_pileup(
            data=data, key_dict=key_dict, chr_list=all_chr_list, gsnv_list=all_gsnv_list, num_cells=args.num_cells)
        print("\tTotal number of valid gSNV sites: ", len(valid_gsnv_list))
        print("\tTotal time: ", time.time() - start_time)

        # STEP 4. Get valid hom positions around valid gSNVs
        print("\nSTEP 4: Getting candidate nearby hom positions")
        start_time = time.time()
        nearby_chr_list = []
        nearby_gsnv_list = []
        nearby_hom_list = []
        for i in range(len(valid_gsnv_list)):
            cur_gsnv = valid_gsnv_list[i]
            for hom_pos in range(cur_gsnv - args.read_length + 1, cur_gsnv + args.read_length):
                if 1 <= hom_pos <= data[-1, 1] and hom_pos != cur_gsnv:
                    nearby_chr_list.append(valid_chr_list[i])
                    nearby_gsnv_list.append(cur_gsnv)
                    nearby_hom_list.append(hom_pos)

        nearby_candidate_hom_list, nearby_candidate_chr_list, nearby_bulk_hom_genotype_list = get_valid_hom_pileup(
            data=data, key_dict=key_dict, chr_list=nearby_chr_list, num_cells=args.num_cells,
            genome_length=args.genome_length, pos_list=nearby_hom_list)
        print("\tTotal number of nearby candidate hom sites: ", len(nearby_candidate_hom_list))
        print("\tTotal time: ", time.time() - start_time)

        print("\nSTEP 5: Getting pairs data")
        start_time = time.time()
        bulk_pairs, sc_pairs = get_site_pairs(chr_list=nearby_candidate_chr_list, all_gsnv_list=valid_gsnv_list,
                                              hom_list=nearby_candidate_hom_list, num_cells=args.num_cells,
                                              bam_dir=bam_dir, bulk_gsnv_genotype_list=bulk_gsnv_genotype_list,
                                              bulk_hom_genotype_list=nearby_bulk_hom_genotype_list)
        print("\tTotal time: ", time.time() - start_time)

        # TODO Remove non-valid pairs
        bulk_pairs, sc_pairs = get_valid_site_pair(bulk_pairs, sc_pairs, num_cells=args.num_cells)
        print("\tTotal number of valid site pairs: ", len(list(bulk_pairs.keys())))

        # STEP 7. Save valid site pairs
        print("\nSTEP 7: Saving valid site-pairs")
        start_time = time.time()
        save_dictionary(proc_dir + "bulk_pairs.pickle", bulk_pairs)
        save_dictionary(proc_dir + "all_cell_dict_pair.pickle", sc_pairs)
        print("\tTotal time: ", time.time() - start_time)

        print("\nTotal time (paired): ", time.time() - start_time_paired)

    # SINGLETON SITE DETECTION
    if args.scuphr_strategy == "singleton" or args.scuphr_strategy == "hybrid":
        print("\nSingleton site detection starts...")
        start_time_singleton = time.time()

        # STEP 8. Getting list of hom positions
        # TODO Find a more efficient way
        print("\nSTEP 8: Getting list of hom positions")
        start_time = time.time()
        if False:
            # Remove gSNVs and their closeby locations from positions-to-investigate
            no_closeby_positions = list(np.arange(args.genome_length) + 1)
            for cur_gsnv in all_gsnv_list[::-1]:
                no_closeby_positions.remove(cur_gsnv)

                if args.scuphr_strategy == "hybrid":
                    for i in range(args.read_length):
                        cur_pos = cur_gsnv - i
                        if cur_pos in no_closeby_positions:
                            no_closeby_positions.remove(cur_pos)
                        cur_pos = cur_gsnv + i
                        if cur_pos in no_closeby_positions:
                            no_closeby_positions.remove(cur_pos)
            # TODO Fix here
            all_hom_chr_list = np.ones(len(no_closeby_positions), dtype=int)
            print("\tNumber of no-closeby positions: ", len(no_closeby_positions))
        print("\tTotal time: ", time.time() - start_time)

        # STEP 9. Find valid hom positions
        print("\nSTEP 9: Finding valid hom positions")
        start_time = time.time()
        #valid_hom_list, valid_hom_chr_list, bulk_hom_genotype_list = get_valid_hom_pileup_avoid(
        #    data=data, key_dict=key_dict, chr_list=[], num_cells=args.num_cells,
        #    all_gsnv_list=all_gsnv_list, genome_length=args.genome_length, read_length=args.read_length)

        valid_hom_list, valid_hom_chr_list, bulk_hom_genotype_list = get_valid_hom_pileup(
            data=data, key_dict=key_dict, chr_list=args.chr_id*np.ones(data.shape[0]), num_cells=args.num_cells,
            genome_length=args.genome_length, gsnv_list=all_gsnv_list)
        print("\tTotal number of valid hom sites: ", len(valid_hom_list))
        #print("\tValid hom sites: ", valid_hom_list)
        print("\tTotal time: ", time.time() - start_time)

        # STEP 10. Save valid singletons
        print("\nSTEP 10: Saving valid singletons")
        start_time = time.time()
        generate_final_singleton_dicts(proc_dir=proc_dir, chr_list=valid_hom_chr_list, hom_list=valid_hom_list,
                                       bulk_genotypes=bulk_hom_genotype_list, num_cells=args.num_cells, bam_dir=bam_dir)
        print("\tTotal time: ", time.time() - start_time)

        print("\nTotal time (singleton): ", time.time() - start_time_singleton)

    # STEP 11. Combine dictionaries.
    print("\nSTEP 11: Combining dictionaries")
    start_time = time.time()
    combine_dictionaries(proc_dir, args.scuphr_strategy, args.base_range_min, args.base_range_max)
    print("\tTotal time: ", time.time() - start_time)

    #
    print("\nSite detection finished!\n\tTotal global time: ", time.time() - start_time_global)


if __name__ == "__main__":
    main()
