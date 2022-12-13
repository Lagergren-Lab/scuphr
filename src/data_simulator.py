import os
import sys
import time
import pysam
import pickle
import argparse
import datetime
import matplotlib
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def recursive_binary_search(a_list, target, start, end):
    if end - start + 1 <= 0:
        return False
    else:
        midpoint = start + (end - start) // 2
        if a_list[midpoint] <= target < a_list[midpoint + 1]:
            return midpoint
        else:
            if target < a_list[midpoint]:
                return recursive_binary_search(a_list, target, start, midpoint - 1)
            else:
                return recursive_binary_search(a_list, target, midpoint + 1, end)


def get_gsnv_mut_counts(num_bp, gsnv_rate, mut_poisson_rate, seed_val):
    np.random.seed(seed_val)

    num_gsnv = int(num_bp * gsnv_rate)
    num_mut = num_gsnv * (1 + np.random.poisson(lam=mut_poisson_rate))

    if num_bp < num_gsnv:
        print("\nERROR: Couldn't generate num_gsnv and num_mut. num_gsnv > numBp.\t", num_gsnv, num_bp)
        print("Check numBp, gsnv_rate.")
        sys.exit()

    if num_bp < num_gsnv + num_mut:
        print("\nWARNING: num_gsnv + num_mut > numBp: \t", num_gsnv, num_mut, num_bp)
        max_trial = 10
        for i in range(max_trial):
            print("Regenerating the counts...", )
            num_mut = num_gsnv * (1 + np.random.poisson(lam=mut_poisson_rate))
            if num_bp >= num_gsnv + num_mut:
                break

        print("\nERROR: Couldn't generate num_gsnv and num_mut.")
        print("Check numBp, gsnv_rate, mut_poisson_rate")
        sys.exit()

    return num_gsnv, num_mut


def generate_bulk_genotype(num_bp, num_gsnv, num_mut, seed_val):
    np.random.seed(seed_val)

    special_locations = np.random.choice(num_bp, size=num_gsnv + num_mut, replace=False)
    gsnv_locations = sorted(special_locations[0:num_gsnv])
    mut_locations = sorted(special_locations[num_gsnv:])

    bulk_genome = np.zeros((num_bp, 2))
    for bp in range(num_bp):
        nucleotide = np.random.randint(4)
        bulk_genome[bp, :] = [nucleotide, nucleotide]

    for bp in gsnv_locations:
        nucleotide = bulk_genome[bp, 0]
        alternatives = list(range(4))
        alternatives.remove(nucleotide)
        alt_nucleotide = np.random.choice(alternatives)
        bulk_genome[bp, 1] = alt_nucleotide

    mut_genome = np.copy(bulk_genome)
    for mut_idx in range(num_mut):
        mut_loc = mut_locations[mut_idx]
        mut_allele = np.random.choice(2)
        nucleotide = bulk_genome[mut_loc, mut_allele]

        alternatives = list(range(4))
        alternatives.remove(nucleotide)
        alt_nucleotide = np.random.choice(alternatives)
        mut_genome[mut_loc, mut_allele] = alt_nucleotide

    bulk_genome = bulk_genome.astype(int)
    mut_genome = mut_genome.astype(int)
    return bulk_genome, mut_genome, gsnv_locations, mut_locations


def location_distance_statistics(gsnv_locations, mut_locations, read_length):
    num_mut = len(mut_locations)

    num_close_gsnv = 0
    find_closest_idices = np.searchsorted(np.array(gsnv_locations), mut_locations)

    for mut_idx in range(num_mut):
        mut_loc = mut_locations[mut_idx]

        left_gsnv_loc = find_closest_idices[mut_idx] - 1
        right_gsnv_loc = find_closest_idices[mut_idx]

        min_dist = np.Inf
        if left_gsnv_loc > 0:
            min_dist = min(min_dist, np.abs(mut_loc - gsnv_locations[left_gsnv_loc]))
        if right_gsnv_loc < len(gsnv_locations):
            min_dist = min(min_dist, np.abs(mut_loc - gsnv_locations[right_gsnv_loc]))

        if min_dist <= read_length:
            num_close_gsnv += 1

    return num_close_gsnv


def save_dictionary(filename, cell_dict):
    with open(filename, 'wb') as fp:
        pickle.dump(cell_dict, fp)


def load_dictionary(filename):
    with open(filename, 'rb') as fp:
        cell_dict = pickle.load(fp)
    return cell_dict


def get_fragment_string(fragment, nucleotides=None):
    if nucleotides is None:
        nucleotides = ["A", "C", "G", "T"]
    fragment_string = ""
    for idx in fragment:
        fragment_string = fragment_string + nucleotides[idx]
    return fragment_string


def generate_bulk_fasta(global_dir, bulk_genome, chr_id):
    num_bp = bulk_genome.shape[0]
    fragment_length = num_bp - 1

    header = {'HD': {'VN': '1.0'},
              'SQ': [{'LN': num_bp, 'SN': str(chr_id)}]}

    filename = global_dir + "bulk_ref.sam"
    with pysam.AlignmentFile(filename, "w", header=header) as outf:
        fragment_pos = 0
        fragment = bulk_genome[fragment_pos:fragment_pos + fragment_length, 0]
        fragment_string = get_fragment_string(fragment)
        fragment_quality = "=" * len(fragment_string)

        a = pysam.AlignedSegment()
        a.query_name = str(chr_id)
        a.reference_id = 0
        a.reference_start = fragment_pos
        a.query_sequence = fragment_string
        a.query_qualities = pysam.qualitystring_to_array(fragment_quality)
        a.cigar = ((0, fragment_length),)
        outf.write(a)

    # samtools view bulk_ref.sam | awk '{OFS="\t"; print ">"$1"\n"$10}' - > bulk_ref.fasta
    out_filename = global_dir + "bulk_ref.fasta"
    cmd = "samtools view " + filename + " | awk '{OFS=\"\\t\"; print \">\"$1\"\\n\"$10}' - > " + out_filename
    print("\tRunning: ", cmd)
    os.system(cmd)


def generate_bulk_sam(global_dir, bulk_genome, chr_id, read_length):
    num_bp, num_alleles = bulk_genome.shape

    header = {'HD': {'VN': '1.0'},
              'SQ': [{'LN': num_bp, 'SN': str(chr_id)}],
              'RG': [{'ID': -1, 'SM': 'bulk'}]}

    filename = global_dir + "bulk.sam"
    with pysam.AlignmentFile(filename, "w", header=header) as outf:

        for fragment_pos in range(num_bp - read_length):
            for allele in range(num_alleles):
                fragment = bulk_genome[fragment_pos:fragment_pos + read_length, allele]
                fragment_string = get_fragment_string(fragment)
                fragment_quality = "=" * len(fragment_string)

                a = pysam.AlignedSegment()
                a.query_name = "read_" + str(fragment_pos) + "_" + str(allele)
                a.reference_id = 0
                a.reference_start = fragment_pos
                a.query_sequence = fragment_string
                a.query_qualities = pysam.qualitystring_to_array(fragment_quality)
                a.cigar = ((0, read_length),)
                outf.write(a)

    # Convert sam to bam
    # samtools view -h -b -S filename > bam_filename
    bam_filename = global_dir + "bulk.bam"
    cmd = "samtools view -h -b -S " + filename + " > " + bam_filename
    print("\tRunning: ", cmd)
    os.system(cmd)
    cmd = "samtools index -b " + bam_filename
    print("\tRunning: ", cmd)
    os.system(cmd)


def generate_bed_file(filename, locations, chr_id):
    bed_file = open(filename, "w")
    for pos in locations:
        entry = str(chr_id) + "\t" + str(pos) + "\t" + str(pos + 1) + "\n"
        bed_file.write(entry)
    bed_file.close()


def generate_regions_bed_file(filename, locations, num_bp, read_length, chr_id):
    bed_file = open(filename, "w")
    for pos in locations:
        entry = str(chr_id) + "\t" + str(max(pos + 1 - read_length, 0)) + "\t" + str(
            min(pos + 1 + read_length, num_bp)) + "\n"
        bed_file.write(entry)
    bed_file.close()


def generate_genome_bias(num_bp, is_flat, num_rep, num_max_mid_points, seed_val):
    if is_flat == "True":
        return np.ones(num_bp), np.ones(num_bp)

    np.random.seed(seed_val)
    # num_max_mid_points = max(2, int(numBp / 100))#10000)
    num_max_mid_points = max(2, num_max_mid_points)

    div = np.random.randint(1, 5) * 100

    y_sin = np.zeros((num_rep, num_bp))
    for rep in range(num_rep):
        np.random.seed(seed_val + rep)

        num_mid_points = np.random.randint(1, num_max_mid_points)
        mid_points = np.sort(np.random.choice(num_bp, num_mid_points))

        print("\tMiddle points for repetition ", rep, ": ", mid_points)
        for part_i in range(num_mid_points + 1):
            if part_i == 0:
                min_val = 0
                max_val = mid_points[part_i]
            elif part_i < num_mid_points:
                min_val = mid_points[part_i - 1] + 1
                max_val = mid_points[part_i]
            else:
                min_val = mid_points[part_i - 1] + 1
                max_val = num_bp - 1

            part_i_freq = np.random.rand() / div
            for val in range(min_val, max_val + 1):
                y_sin[rep, val] = np.sin(2 * np.pi * part_i_freq * val) / 2 + 0.5

    y_mult = y_sin[0, :]
    for rep in range(1, num_rep):
        y_mult = np.multiply(y_mult, y_sin[rep, :])

    y_sin_ave = np.sum(y_sin, axis=0) / num_rep
    return y_sin_ave, y_mult


def save_amplification_plot(global_dir, genome_amp_wave):
    plt.figure(figsize=(20, 3))
    plt.plot(np.arange(len(genome_amp_wave)), genome_amp_wave)
    plt.ylim([0, 1])
    plt.xlabel("Positions")
    title_str = "genome_amplification_wave"
    plt.title(title_str)
    filename = global_dir + str(title_str) + ".png"
    plt.savefig(filename)
    plt.close()


def generate_real_bdp_tree(num_cells, seed_val):
    np.random.seed(seed_val)

    leaf_nodes = []
    parent_nodes = np.zeros((2 * num_cells - 1))

    root_node_id = 0
    leaf_nodes.append(root_node_id)

    cur_id = 1
    while len(leaf_nodes) != num_cells:
        pos = np.random.randint(0, len(leaf_nodes))
        parent = leaf_nodes[pos]
        parent_nodes[cur_id] = parent
        parent_nodes[cur_id + 1] = parent

        leaf_nodes.pop(pos)
        leaf_nodes.append(cur_id)
        leaf_nodes.append(cur_id + 1)

        cur_id += 2

    parent_nodes = parent_nodes.astype(int)
    return parent_nodes, leaf_nodes


def assign_mutations_to_nodes(parent_nodes, num_mut, seed_val):
    np.random.seed(seed_val)

    num_edges = len(parent_nodes) - 1
    mut_origin_nodes = np.zeros(num_mut)

    if num_mut < num_edges:
        print("\nWARNING: num_mut < num_edges.\t", num_mut, num_edges)

        fixed_muts = np.random.choice(np.arange(1, len(parent_nodes)), size=num_mut, replace=False)
        mut_origin_nodes[0:num_edges] = fixed_muts

    else:
        fixed_muts = np.random.choice(np.arange(1, len(parent_nodes)), size=num_edges, replace=False)
        mut_origin_nodes[0:num_edges] = fixed_muts

        num_extra_muts = num_mut - num_edges
        extra_muts = np.random.choice(np.arange(1, len(parent_nodes)), size=num_extra_muts)
        mut_origin_nodes[num_edges:] = extra_muts

    np.random.shuffle(mut_origin_nodes)
    mut_origin_nodes = mut_origin_nodes.astype(int)
    return mut_origin_nodes


def get_node_ancestors(cell_id, parent_nodes):
    ancestor_list = [cell_id]

    cur_node = cell_id
    while cur_node != 0:
        parent_id = parent_nodes[cur_node]
        ancestor_list.append(parent_id)
        cur_node = parent_id

    return ancestor_list


def generate_cell_genotype(leaf_idx, leaf_nodes, bulk_genome, mut_genome, mut_origin_nodes, mut_locations, parent_nodes,
                           seed_val):
    np.random.seed(seed_val)

    cell_id = leaf_nodes[leaf_idx]
    ancestor_list = get_node_ancestors(cell_id, parent_nodes)
    all_mutations = set()

    cell_genome = np.copy(bulk_genome)
    for ancestor_id in ancestor_list:
        cell_mutations_idx = np.where(mut_origin_nodes == ancestor_id)[0]
        for mut_idx in cell_mutations_idx:
            mut_loc = mut_locations[mut_idx]
            cell_genome[mut_loc, :] = mut_genome[mut_loc, :]
            all_mutations.add(mut_loc)

    # print("\tCell: ", leaf_idx, "\tMutations: ", sorted(list(all_mutations)))
    return cell_genome, sorted(list(all_mutations))


def mask_genome(genome, ado_type, p_ado, ado_poisson_rate, seed_val):
    np.random.seed(seed_val)

    if ado_type == 0:  # no ado
        p_ado = 0
    elif ado_type == 1:  # random
        p_ado = np.random.beta(a=1, b=30)
    elif ado_type == 2:  # random
        p_ado = np.random.beta(a=1, b=50)
        # else: #fixed ado

    # print("\tp_ado for cell is: ", p_ado)

    np.random.seed(seed_val)  # reset the seed

    mask_info = {}
    num_mask_events = 0
    num_covered_bp = 0

    num_bp, num_allele = genome.shape
    masked_genome = np.copy(genome)

    for allele in range(num_allele):
        max_idx = -1
        for bp in range(num_bp):
            if bp > max_idx:
                window_width = 1 + np.random.poisson(ado_poisson_rate)
                max_idx = bp + window_width - 1

                u = np.random.rand()
                if u < p_ado:
                    if bp + window_width >= num_bp:  # if the window exceeds the genome
                        window_width = num_bp - bp

                    masked_genome[bp:bp + window_width, allele] = -1
                    num_mask_events += 1
                    num_covered_bp = num_covered_bp + window_width

                    if window_width not in mask_info:
                        mask_info[window_width] = 1
                    else:
                        mask_info[window_width] += 1

    return masked_genome, num_mask_events, num_covered_bp, mask_info, p_ado


def mask_genome2(genome, ado_type, p_ado, ado_poisson_rate, seed_val):
    np.random.seed(seed_val)

    if ado_type == 0:  # no ado
        p_ado = 0
    elif ado_type == 1:  # random
        p_ado = np.random.beta(a=1, b=30)
    elif ado_type == 2:  # random
        p_ado = np.random.beta(a=1, b=50)
        # else: #fixed ado

    # print("\tp_ado for cell is: ", p_ado)

    np.random.seed(seed_val)  # reset the seed

    mask_info = {}
    num_mask_events = 0
    num_covered_bp = 0

    num_bp, num_allele = genome.shape
    masked_genome = np.copy(genome)

    for bp in range(num_bp):
        for allele in range(num_allele):
            u = np.random.rand()
            if u < p_ado:
                window_width = 1 + np.random.poisson(ado_poisson_rate)
                if bp + window_width >= num_bp:  # if the window exceeds the genome
                    window_width = num_bp - bp
                masked_genome[bp:bp + window_width, allele] = -1
                num_mask_events += 1
                num_covered_bp = num_covered_bp + window_width

                if window_width not in mask_info:
                    mask_info[window_width] = 1
                else:
                    mask_info[window_width] += 1

    return masked_genome, num_mask_events, num_covered_bp, mask_info, p_ado


def amplify_genome(genome, genome_amp_wave, p_ae, num_iter, read_length, amp_method, seed_val):
    total_time_search = 0
    total_time_accept = 0
    total_time_for = 0

    np.random.seed(seed_val)

    num_bp, num_allele = genome.shape

    min_frag_length = read_length

    if amp_method == "mda":
        max_frag_length = max(2 * min_frag_length, min(10000, num_bp))
    elif amp_method == "malbac":
        max_frag_length = max(2 * min_frag_length, min(2000, num_bp))
    else:
        max_frag_length = max(2 * min_frag_length, int(num_bp / 10))

    num_fragments = 0
    num_origin_from_frag = 0
    num_amp_errors = 0

    fragments_list = []
    fragments_allele_list = []
    fragments_start_locations_list = []
    fragments_amp_errored_list = []
    fragment_cumulative_length_list = [0]

    fragment_length_dict = {}

    num_reject_wave = 0

    num_amp = 0
    while num_amp <= num_iter:
        possible_positions = 2 * num_bp + fragment_cumulative_length_list[-1]  # num_fragments
        current_pos = np.random.randint(possible_positions)
        error_stat = 0
        # print("\tPossible positions: ", possible_positions, "current: ", current_pos)
        # print("\nSelected current pos: ", current_pos)

        start_time_temp_2 = time.time()

        if current_pos < num_bp:  # amplify from genome (allele 0)
            # print("\tallele 0")
            start_pos = current_pos
            allele_idx = 0

            read_length = np.random.choice(min(num_bp - start_pos, max_frag_length))
            fragment = genome[start_pos:start_pos + read_length, allele_idx]

        elif num_bp <= current_pos < 2 * num_bp:  # amplify from genome (allele 1)
            # print("\tallele 1")
            start_pos = current_pos - num_bp
            allele_idx = 1

            read_length = np.random.choice(min(num_bp - start_pos, max_frag_length))
            fragment = genome[start_pos:start_pos + read_length, allele_idx]

        else:  # amplify from fragment
            # print("\tno allele")
            # print("\tcumulative length list: ", fragment_cumulative_length_list)
            # num_origin_from_frag = num_origin_from_frag + 1

            temp_pos = current_pos - 2 * num_bp

            fragment_idx = recursive_binary_search(fragment_cumulative_length_list, temp_pos, 0,
                                                   len(fragment_cumulative_length_list))
            start_pos = temp_pos - fragment_cumulative_length_list[fragment_idx]
            # for i in range(len(fragment_cumulative_length_list)-1):
            #    if temp_pos >= fragment_cumulative_length_list[i] and temp_pos < fragment_cumulative_length_list[i+1]:
            #        fragment_idx = i
            #        start_pos = temp_pos-fragment_cumulative_length_list[fragment_idx]
            #        break

            if fragment_idx == -1 or start_pos == -1:
                print("\nERROR: Fragment index problem. fragment_idx, start_pos:\t", fragment_idx, start_pos)
                sys.exit()

            # fragment_idx = current_pos - 2*num_bp

            allele_idx = fragments_allele_list[fragment_idx]
            fragment_origin = fragments_list[fragment_idx]
            fragment_origin_start = fragments_start_locations_list[fragment_idx]
            # print("\tfragment (origin): ", fragment_origin)

            if fragments_amp_errored_list[fragment_idx] != 0:  # if originated fragment has errors in it
                error_stat = 2

            read_length = np.random.choice(min(len(fragment_origin) - start_pos, max_frag_length))
            fragment = fragment_origin[start_pos:start_pos + read_length]
            start_pos = start_pos + fragment_origin_start

        end_time_temp_2 = time.time()
        total_time_search += end_time_temp_2 - start_time_temp_2

        # print("\tstart pos: ", start_pos, "read length: ", read_length, "origin_length", len(fragment_origin))
        # print("\tstart_pos: ", start_pos, "end_pos", start_pos+read_length, "\tlength: ", read_length, "frag_idx:
        # ", fragment_idx) print("\tfragment: ", fragment)

        # For simplicity, I add the amplification sine-wave bias to here. The implementation can change. 

        start_time_temp_2 = time.time()
        start_pos_bias = genome_amp_wave[start_pos]
        u_wave = np.random.rand()
        if u_wave <= start_pos_bias:
            num_amp += 1
            if len(fragment) >= min_frag_length:
                read_length = len(fragment)
                if read_length not in fragment_length_dict:
                    fragment_length_dict[read_length] = 1
                else:
                    fragment_length_dict[read_length] += 1

                start_time_temp_2 = time.time()
                for idx in range(read_length):
                    if fragment[idx] != -1:
                        u = np.random.rand()
                        if u < p_ae:  # Add amplification errors
                            nucleotide = fragment[idx]

                            if nucleotide >= 10:  # TODO remove this if line in future
                                nucleotide -= 10

                            alternatives = list(range(4))
                            alternatives.remove(nucleotide)
                            alt_nucleotide = np.random.choice(alternatives)

                            alt_nucleotide += 10  # TODO remove this line in future

                            fragment[idx] = alt_nucleotide
                            num_amp_errors += 1
                            error_stat = 1

                end_time_temp_2 = time.time()
                total_time_for += end_time_temp_2 - start_time_temp_2

                fragments_list.append(fragment)
                fragments_allele_list.append(allele_idx)
                fragments_start_locations_list.append(start_pos)
                fragments_amp_errored_list.append(error_stat)
                fragment_cumulative_length_list.append(fragment_cumulative_length_list[-1] + read_length)
                # print("\tAdding fragment with the start loc: ", start_pos, " to the list: ")

                if current_pos >= 2 * num_bp:
                    num_origin_from_frag += 1

                num_fragments += 1
            # else:
            #    print("\nFragment size is too small. Not added!")
        else:
            num_reject_wave += 1

        end_time_temp_2 = time.time()
        total_time_accept += end_time_temp_2 - start_time_temp_2

    print("\n\tTotal time for cell's amplifyGenome: ", total_time_search + total_time_accept, total_time_search,
          total_time_accept, total_time_for)
    print("\tNumber of rejected fragments due to amplification wave: ", num_reject_wave)
    return fragments_list, fragments_start_locations_list, fragments_allele_list, fragments_amp_errored_list, \
           fragment_cumulative_length_list, fragment_length_dict, num_amp_errors, num_fragments, num_origin_from_frag


def cut_fragments(fragments_list, fragments_start_locations_list, read_length, seed_val):
    np.random.seed(seed_val)

    new_fragments_list = []
    new_fragments_start_locations_list = []
    num_masked = 0

    while len(fragments_list) > 0:
        fragment_idx = np.random.choice(len(fragments_list))
        fragment = fragments_list[fragment_idx]
        fragment_length = len(fragment)
        fragment_start_location = fragments_start_locations_list[fragment_idx]

        if fragment_length == read_length:
            start_pos = 0
        else:
            start_pos = np.random.choice(fragment_length - read_length)

        new_fragment = fragment[start_pos:start_pos + read_length]
        new_fragment_start_location = fragment_start_location + start_pos

        # add previous and after cuts to the fragment list
        prev_frag = fragment[0:start_pos]
        if len(prev_frag) >= read_length:
            fragments_list.append(prev_frag)
            fragments_start_locations_list.append(fragment_start_location)

        after_frag = fragment[start_pos + read_length:]
        if len(after_frag) >= read_length:
            fragments_list.append(after_frag)
            fragments_start_locations_list.append(fragment_start_location + start_pos + read_length)

            # remove fragment from the fragment list
        fragments_list.pop(fragment_idx)
        fragments_start_locations_list.pop(fragment_idx)

        if len(np.where(new_fragment == -1)[0]) == 0:  # if no masked region
            # TODO maybe add selective add.

            new_fragments_list.append(new_fragment)
            new_fragments_start_locations_list.append(new_fragment_start_location)
        else:
            num_masked += 1

    return new_fragments_list, new_fragments_start_locations_list, num_masked


def generate_reads(fragments_list, phred_type, mu_phred=40, sigma_phred=5):
    total_time_phred = 0
    total_time_for = 0
    total_time_if = 0

    total_time_q_err = 0
    total_time_p_err = 0

    phred_chars = ["!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4",
                   "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H",
                   "I", "J", "K"]

    num_seq_errors = 0

    reads_list = []
    phred_list = []
    phred_char_list = []
    error_probability_list = []

    if phred_type != 0 and phred_type != 1:
        q_error_rv = stats.truncnorm((0 - mu_phred) / sigma_phred, (42 - mu_phred) / sigma_phred, loc=mu_phred,
                                     scale=sigma_phred)

    for fragment in fragments_list:
        fragment_length = fragment.shape[0]

        start_time_temp = time.time()
        if phred_type == 0:
            q_error = (np.ones(fragment_length) * 42).astype(int)
            p_error = np.zeros(q_error.shape)
        elif phred_type == 1:
            q_error = (np.ones(fragment_length) * 42).astype(int)
            p_error = np.power((10 * np.ones(q_error.shape)), (-0.1 * q_error))
        else:
            # Moved this part out for making the code faster. q_error_rv = stats.truncnorm( (0-mu_phred)/sigma_phred,
            # (42-mu_phred)/sigma_phred, loc=mu_phred, scale=sigma_phred)

            start_time_temp_3 = time.time()
            q_error = q_error_rv.rvs(fragment_length).astype(int)
            end_time_temp_3 = time.time()
            total_time_q_err += end_time_temp_3 - start_time_temp_3

            start_time_temp_3 = time.time()
            p_error = np.power((10 * np.ones(q_error.shape)), (-0.1 * q_error))
            end_time_temp_3 = time.time()
            total_time_p_err += end_time_temp_3 - start_time_temp_3

        end_time_temp = time.time()
        total_time_phred += end_time_temp - start_time_temp

        start_time_temp = time.time()
        reads = np.copy(fragment)

        phred_frag_char = np.array(phred_chars)[q_error]
        p_vals_all = np.tile(p_error / 3, (4, 1))

        for i in range(fragment_length):
            sample = fragment[i]
            # p_vals = np.repeat(p_error[i]/3, 4) # moved out for better speed
            p_vals = p_vals_all[:, i]
            p_vals[sample] = 1 - p_error[i]

            start_time_temp_3 = time.time()
            reads[i] = np.argmax(np.random.multinomial(1, p_vals))
            end_time_temp_3 = time.time()
            total_time_if += end_time_temp_3 - start_time_temp_3

            if reads[i] != sample and sample != -1:
                num_seq_errors += 1

        end_time_temp = time.time()
        total_time_for += end_time_temp - start_time_temp

        phred_list.append(q_error)
        phred_char_list.append(phred_frag_char)
        error_probability_list.append(p_error)
        reads_list.append(reads)

    print("\n\tTotal time for cell's generateReads: ", total_time_phred + total_time_for, total_time_phred,
          total_time_for, total_time_if)
    print("\n\tTotal time generateReads subsets: ", total_time_q_err, total_time_p_err)

    return reads_list, phred_list, phred_char_list, error_probability_list, num_seq_errors


def generate_cell_sam(global_dir, num_bp, leaf_idx, fragments_list, phred_list, fragments_start_locations_list,
                      read_length, chr_id):
    header = {'HD': {'VN': '1.0'},
              'SQ': [{'LN': num_bp, 'SN': str(chr_id)}],
              'RG': [{'ID': leaf_idx, 'SM': str(leaf_idx)}]}

    filename = global_dir + "cell_idx_" + str(leaf_idx) + ".sam"
    with pysam.AlignmentFile(filename, "w", header=header) as outf:
        sort_idx = np.argsort(fragments_start_locations_list)
        fragments_start_locations_list = np.array(fragments_start_locations_list)[sort_idx]
        fragments_list = np.array(fragments_list)[sort_idx]
        phred_list = np.array(phred_list)[sort_idx]

        for fragment_idx in range(len(fragments_list)):
            fragment_pos = fragments_start_locations_list[fragment_idx]
            fragment = fragments_list[fragment_idx]
            phred = phred_list[fragment_idx]

            fragment_string = get_fragment_string(fragment)
            fragment_length = len(fragment_string)
            fragment_quality = phred

            if read_length != len(fragment_quality) or read_length != len(fragment_string):
                print("\nERROR: Fragment size mismatch. \tread_length, len(fragment_quality), len(fragment_string): \t",
                      read_length, len(fragment_quality), len(fragment_string))
                sys.exit()

            a = pysam.AlignedSegment()
            a.query_name = "read_" + str(fragment_idx)
            a.reference_id = 0
            a.reference_start = fragment_pos
            a.query_sequence = fragment_string
            # a.query_qualities = pysam.qualitystring_to_array(fragment_quality_str)
            a.query_qualities = fragment_quality
            a.cigar = ((0, fragment_length),)
            outf.write(a)

    cell_bam_filename = global_dir + "cell_idx_" + str(leaf_idx) + ".bam"
    # Convert sam to bam
    # samtools view -h -b -S filename > cell_bam_filename
    cmd = "samtools view -h -b -S " + filename + " > " + cell_bam_filename
    print("\tRunning: ", cmd)
    os.system(cmd)
    cmd = "samtools index -b " + cell_bam_filename
    print("\tRunning: ", cmd)
    os.system(cmd)


def save_arguments(args):
    filename = args.global_dir + "args_simulation.pickle"
    save_dictionary(filename, args)

    filename = args.global_dir + "args_simulation.txt"

    file = open(filename, "w")
    file.write("Arguments\n\n")

    for arg in vars(args):
        write_str = str(arg) + "\t" + str(getattr(args, arg)) + "\n"
        file.write(write_str)

    file.close()


def clean_fragment_list(fragments_list):
    temp_fragments = np.array(fragments_list)

    amp_a_idx = np.where(np.array(temp_fragments) == 10)
    temp_fragments[amp_a_idx] = 0

    amp_c_idx = np.where(np.array(temp_fragments) == 11)
    temp_fragments[amp_c_idx] = 1

    amp_g_idx = np.where(np.array(temp_fragments) == 12)
    temp_fragments[amp_g_idx] = 2

    amp_t_idx = np.where(np.array(temp_fragments) == 13)
    temp_fragments[amp_t_idx] = 3

    return temp_fragments


def get_amplification_error_count(fragments_list):
    temp_fragments = np.array(fragments_list)

    num_fragments, fragment_length = temp_fragments.shape
    total_bases = num_fragments * fragment_length

    if total_bases == 0:
        print("\tWARNING: There are no reads.")
        return 0, 0, -1

    total_amp_errors = len(np.where(np.array(temp_fragments) >= 10)[0])
    return total_amp_errors, total_bases, total_amp_errors / total_bases


def main():
    dt = datetime.datetime.now()
    default_dir = "../data/%s_%s_%s/" % (dt.year, dt.month, dt.day)

    # Code to process command line arguments
    parser = argparse.ArgumentParser(description='Generate synthetic data (BAM) files.')
    parser.add_argument('--ado_poisson_rate', help="Specify the ado poisson rate. Default: 0", type=float, default=0)
    parser.add_argument('--ado_type',
                        help="Specify the allelic dropout type "
                             "(0 for no ado, 1 for random Beta, 2 for even smaller Beta, 3 for fixed. Default: 3).",
                        type=int, default=3)
    parser.add_argument('--amp_method', help="Specify the amplification method. Default: mda", type=str, default="mda")
    parser.add_argument('--chr_id', help="Specify the chromosome number. Default: 1", type=int, default=1)
    parser.add_argument('--genome_length', help="Specify the length of genome. Default: 50000", type=int, default=50000)
    parser.add_argument('--global_dir', help="Specify the directory.", type=str, default=default_dir)
    parser.add_argument('--gsnv_rate', help="Specify the gSNV rate. Default: 0.005", type=float, default=0.005)
    parser.add_argument('--is_flat',
                        help="Specify the amplification bias type (True for flat, False for sine wave). Default: False",
                        default=False)
    parser.add_argument('--mut_poisson_rate', help="Specify the mutation poisson rate. Default: 3", type=float,
                        default=3)
    parser.add_argument('--num_cells', help="Specify the number of cells. Default: 10", type=int, default=10)
    parser.add_argument('--num_iter', help="Specify the number of iteration. Default: 5000", type=int, default=5000)
    parser.add_argument('--num_max_mid_points',
                        help="Specify the number of division points of amplification bias. Default: 10", type=int,
                        default=10)
    parser.add_argument('--num_rep_amp_bias',
                        help="Specify the number of repetitions of amplification bias. Default: 3", type=int, default=3)
    parser.add_argument('--p_ado', help="Specify the allelic dropout probability of a base. Default: 0.005", type=float,
                        default=0.005)
    parser.add_argument('--p_ae', help="Specify the amplification error probability of a base. Default: 0.00001",
                        type=float, default=0.00001)
    parser.add_argument('--phred_type',
                        help="Specify the phred score type "
                             "(0 for no errors, 1 for all 42, 2 for truncated normal). Default: 2",
                        type=int, default=2)
    parser.add_argument('--read_length', help="Specify the read length. Default: 100", type=int, default=100)
    parser.add_argument('--seed_val', help="Specify the seed. Default: 123", type=int, default=123)
    args = parser.parse_args()

    start_time_global = time.time()

    print("The data directory: ", args.global_dir)
    if not os.path.exists(args.global_dir):
        os.makedirs(args.global_dir)
        print("Directory is created: ", args.global_dir)

    # PART 0: Save arguments
    start_time = time.time()
    print("\nPart 0: Save arguments\n")

    save_arguments(args)

    end_time = time.time()
    print("\nTotal time: ", end_time - start_time)
    print("Part 0 ends...")
    ###

    print("\nData simulation starts...")

    # PART 1: Amplification bias
    start_time = time.time()
    print("\nPart 1: Amplification bias\n")

    _, genome_amp_wave = generate_genome_bias(args.genome_length, is_flat=args.is_flat, num_rep=args.num_rep_amp_bias,
                                              num_max_mid_points=args.num_max_mid_points, seed_val=args.seed_val)
    filename = args.global_dir + "genome_amplification_bias.pickle"
    save_dictionary(filename, genome_amp_wave)

    save_amplification_plot(args.global_dir, genome_amp_wave)

    end_time = time.time()
    print("\nTotal time: ", end_time - start_time)
    print("Part 1 ends...")
    ###

    # PART 2: Reference, bulk and mutation generation
    start_time = time.time()
    print("\nPart 2: Reference, bulk and mutation generation\n")

    num_gsnv, num_mut = get_gsnv_mut_counts(num_bp=args.genome_length, gsnv_rate=args.gsnv_rate,
                                            mut_poisson_rate=args.mut_poisson_rate, seed_val=args.seed_val)
    print("\tnumBp: ", args.genome_length)
    print("\tgSNV count: ", num_gsnv)
    print("\tmutation count: ", num_mut)

    num_edges = 2 * (args.num_cells - 1)
    if num_mut < num_edges:
        print("\nWARNING: num_mut < numEdges.\t", num_mut, num_edges)

    bulk_genome, mut_genome, gsnv_locations, mut_locations = generate_bulk_genotype(num_bp=args.genome_length,
                                                                                    num_gsnv=num_gsnv, num_mut=num_mut,
                                                                                    seed_val=args.seed_val)

    num_close_gsnv = location_distance_statistics(gsnv_locations, mut_locations, args.read_length)
    print("\tTotal gSNV count: ", num_gsnv, "\tTotal mutation count: ", num_mut, "\tMutations close to a gSNV: ",
          num_close_gsnv)

    filename = args.global_dir + "bulk_genome.pickle"
    save_dictionary(filename, bulk_genome)
    filename = args.global_dir + "mut_genome.pickle"
    save_dictionary(filename, mut_genome)
    filename = args.global_dir + "gsnv_locations.pickle"
    save_dictionary(filename, gsnv_locations)
    filename = args.global_dir + "mut_locations.pickle"
    save_dictionary(filename, mut_locations)

    generate_bulk_fasta(args.global_dir, bulk_genome, args.chr_id)
    generate_bulk_sam(args.global_dir, bulk_genome, args.chr_id, args.read_length)

    generate_bed_file(args.global_dir + "gsnv_positions.bed", gsnv_locations, chr_id=args.chr_id)
    generate_regions_bed_file(args.global_dir + "gsnv_vars_regions.bed", gsnv_locations, args.genome_length,
                              read_length=args.read_length, chr_id=args.chr_id)
    generate_bed_file(args.global_dir + "mutation_positions.bed", mut_locations, chr_id=args.chr_id)

    end_time = time.time()
    print("\nTotal time: ", end_time - start_time)
    print("Part 1 ends...")
    ###

    # PART 3: Tree generation and mutation partition
    start_time = time.time()
    print("\nPart 3: Tree generation and mutation partition\n")

    parent_nodes, leaf_nodes = generate_real_bdp_tree(num_cells=args.num_cells, seed_val=args.seed_val)
    print("\tparent nodes: ", parent_nodes)
    print("\tleaf nodes: ", leaf_nodes)

    filename = args.global_dir + "parent_nodes.pickle"
    save_dictionary(filename, parent_nodes)
    filename = args.global_dir + "leaf_nodes.pickle"
    save_dictionary(filename, leaf_nodes)

    mut_origin_nodes = assign_mutations_to_nodes(parent_nodes, num_mut=num_mut, seed_val=args.seed_val)
    filename = args.global_dir + "mut_origin_nodes.pickle"
    save_dictionary(filename, mut_origin_nodes)

    end_time = time.time()
    print("\nTotal time: ", end_time - start_time)
    print("Part 3 ends...")
    ###

    # PART 4: Cell simulation
    start_time = time.time()
    print("\nPart 4: Cell simulation\n")

    all_cells_amp_error = 0
    all_cells_total_bases = 0

    for leaf_idx in range(len(leaf_nodes)):
        start_time_cell = time.time()
        print("\n\tSimulating cell: ", leaf_idx)
        cell_seed_val = args.seed_val * (leaf_idx + 1)

        start_time_temp = time.time()
        cell_genome, all_mutations = generate_cell_genotype(leaf_idx, leaf_nodes, bulk_genome, mut_genome,
                                                            mut_origin_nodes, mut_locations, parent_nodes,
                                                            seed_val=cell_seed_val)
        # print("\n\tMutations: ", all_mutations)
        print("\n\tNumber of Mutations: ", len(all_mutations))

        filename = args.global_dir + "cell_" + str(leaf_idx) + "_genome.pickle"
        save_dictionary(filename, cell_genome)
        end_time_temp = time.time()
        print("\n\tTotal time for cell's generateCellGenotype: ", end_time_temp - start_time_temp)

        start_time_temp = time.time()
        masked_genome, num_mask, num_covered_bp, mask_info, p_ado = mask_genome(cell_genome, ado_type=args.ado_type,
                                                                                p_ado=args.p_ado,
                                                                                ado_poisson_rate=args.ado_poisson_rate,
                                                                                seed_val=cell_seed_val)
        filename = args.global_dir + "cell_" + str(leaf_idx) + "_masked_genome.pickle"
        save_dictionary(filename, masked_genome)
        print("\tp_ado: ", p_ado)
        print("\tNumber of mask events: ", num_mask)
        print("\t\tNumber of covered basepairs: ", num_covered_bp)
        print("\t\tMask information: ", mask_info)
        end_time_temp = time.time()
        print("\n\tTotal time for cell's maskGenome: ", end_time_temp - start_time_temp)

        start_time_temp = time.time()
        fragments_list, fragments_start_locations_list, fragments_allele_list, fragments_amp_errored_list, \
            fragment_cumulative_length_list, fragment_length_dict, num_amp_errors, num_fragments, num_origin_from_frag \
            = amplify_genome(masked_genome, genome_amp_wave, p_ae=args.p_ae, num_iter=args.num_iter,
                             read_length=args.read_length, amp_method=args.amp_method, seed_val=cell_seed_val)
        print("\n\tTotal number of fragments: ", num_fragments)
        print("\t\tNumber of fragments without amplification error: ",
              len(np.where(np.array(fragments_amp_errored_list) == 0)[0]))
        print("\t\tNumber of amplification errors: ", num_amp_errors)
        print("\t\tNumber of fragments with amplification error: ",
              len(np.where(np.array(fragments_amp_errored_list) == 1)[0]))
        print("\t\tNumber of fragments with amplification error (inherited from another fragment): ",
              len(np.where(np.array(fragments_amp_errored_list) == 2)[0]))
        print("\t\tNumber of fragments originated from other fragments: ", num_origin_from_frag)
        end_time_temp = time.time()
        print("\n\tTotal time for cell's amplifyGenome: ", end_time_temp - start_time_temp)

        start_time_temp = time.time()
        fragments_list, fragments_start_locations_list, num_masked = cut_fragments(fragments_list,
                                                                                   fragments_start_locations_list,
                                                                                   read_length=args.read_length,
                                                                                   seed_val=cell_seed_val)
        print("\n\tNew number of fragments (reads): ", len(fragments_list))
        print("\t\tNumber of masked fragments: ", num_masked)
        end_time_temp = time.time()
        print("\n\tTotal time for cell's cutFragments: ", end_time_temp - start_time_temp)

        start_time_temp = time.time()
        if len(fragments_list) == 0:
            print("\nWARNING: There are no reads. cell, len(fragments_list): ", leaf_idx, len(fragments_list))

        else:
            total_amp_errors, total_bases, p_ae_final = get_amplification_error_count(fragments_list)
            all_cells_amp_error = all_cells_amp_error + total_amp_errors
            all_cells_total_bases = all_cells_total_bases + total_bases
            print("\n\tTotal number of bases in reads: ", total_bases)
            print("\tTotal number of bases with amplification error: ", total_amp_errors)
            print("\tAmplification error per read base ratio: ", p_ae_final)

            coverage = total_bases / args.genome_length
            print("\n\tApproximate cell coverage: ", coverage)

            fragments_list = clean_fragment_list(fragments_list)

        end_time_temp = time.time()
        print("\n\tTotal time for cell's ampErrorCount: ", end_time_temp - start_time_temp)

        start_time_temp = time.time()
        reads_list, phred_list, phred_char_list, error_probability_list, num_seq_errors \
            = generate_reads(fragments_list, args.phred_type)
        print("\n\tNumber of sequencing errors: ", num_seq_errors)
        end_time_temp = time.time()
        print("\n\tTotal time for cell's generateReads: ", end_time_temp - start_time_temp)

        start_time_temp = time.time()
        generate_cell_sam(args.global_dir, args.genome_length, leaf_idx, reads_list, phred_list,
                          fragments_start_locations_list, args.read_length, chr_id=args.chr_id)
        end_time_temp = time.time()
        print("\n\tTotal time for cell's generateCellSam: ", end_time_temp - start_time_temp)

        end_time_cell = time.time()
        print("\n\tTotal time for cell : ", end_time_cell - start_time_cell)

    print("\nAll cell totals")
    print("\tTotal number of bases in reads: ", all_cells_total_bases)
    print("\tTotal number of bases with amplification error: ", all_cells_amp_error)
    if all_cells_total_bases > 0:
        print("\tAmplification error per read base ratio: ", all_cells_amp_error / all_cells_total_bases)

    coverage = (all_cells_total_bases / args.num_cells) / args.genome_length
    print("\n\tApproximate average cell coverages: ", coverage)

    end_time = time.time()
    print("\nTotal time: ", end_time - start_time)
    print("Part 4 ends...")
    ###

    # PART 5: Relocate files
    start_time = time.time()
    print("\nPart 5: Relocate files\n")

    bam_dir = args.global_dir + "bam/"
    truth_dir = args.global_dir + "truth/"
    if not os.path.exists(bam_dir):
        os.makedirs(bam_dir)
        print("\tDirectory is created: ", bam_dir)

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)
        print("\tDirectory is created: ", truth_dir)

    cmd = "mv " + args.global_dir + "*.bam " + bam_dir
    os.system(cmd)
    cmd = "mv " + args.global_dir + "*.bai " + bam_dir
    os.system(cmd)
    cmd = "cp " + args.global_dir + "gsnv_positions.bed " + bam_dir
    os.system(cmd)
    cmd = "cp " + args.global_dir + "gsnv_vars_regions.bed " + bam_dir
    os.system(cmd)

    cmd = "mv " + args.global_dir + "*.sam " + truth_dir
    os.system(cmd)
    cmd = "mv " + args.global_dir + "*.pickle " + truth_dir
    os.system(cmd)
    cmd = "mv " + args.global_dir + "*.bed " + truth_dir
    os.system(cmd)
    cmd = "mv " + args.global_dir + "*.fasta " + truth_dir
    os.system(cmd)
    cmd = "mv " + args.global_dir + "*.png " + truth_dir
    os.system(cmd)
    cmd = "mv " + args.global_dir + "*.txt " + truth_dir
    os.system(cmd)

    end_time = time.time()
    print("\nTotal time: ", end_time - start_time)
    print("Part 5 ends...")
    ###

    end_time_global = time.time()
    print("\nTotal (global) time: ", end_time_global - start_time_global)
    print("Data simulation ends...")


if __name__ == "__main__":
    main()
