import os
import time
import json
import pysam
import argparse
import datetime
import numpy as np

from data_simulator import assign_mutations_to_nodes, generate_bed_file, generate_bulk_fasta, generate_cell_genotype, \
    generate_cell_sam, generate_real_bdp_tree, generate_regions_bed_file, get_fragment_string, get_node_ancestors
from real_tree_newick import create_branch_files, create_newick_files


def save_json(filename, cell_dict):
    with open(filename, 'w') as fp:
        json.dump(cell_dict, fp)


def load_json(filename):
    with open(filename) as fp:
        cell_dict = json.load(fp)
    return cell_dict


def generate_bulk_sam(global_dir, bulk_genome, chr_id, read_length):
    num_bp, num_alleles = bulk_genome.shape

    header = {'HD': {'VN': '1.0'},
              'SQ': [{'LN': num_bp, 'SN': str(chr_id)}],
              'RG': [{'ID': -1, 'SM': 'bulk'}]}

    filename = global_dir + "bulk.sam"
    with pysam.AlignmentFile(filename, "w", header=header) as outf:

        for fragment_pos_idx in range(int(num_bp/2)):
            fragment_pos = 2 * fragment_pos_idx
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
                for i in range(15):
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
    

def generate_bulk_genotype(num_bp, num_gsnv, num_mut, seed_val):
    np.random.seed(seed_val)

    #gsnv_locations = np.arange(0, num_bp, 2)
    gsnv_locations = np.sort(np.random.choice(np.arange(0, num_bp, 2), num_gsnv, replace=False))
    mut_locations = np.sort(np.random.choice(np.arange(1, num_bp, 2), num_mut, replace=False))

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


def mask_genome(cell_genome, p_ado, read_length):
    masked_genome = np.copy(cell_genome)

    for i in range(int(len(cell_genome)/2)):
        cur_gsnv = read_length * i
        cur_hom = cur_gsnv + 1

        u1 = np.random.rand()
        u2 = np.random.rand()

        # First allele drops
        if u1 <= p_ado:
            masked_genome[cur_gsnv:cur_hom+1][:, 0] = [-1, -1]
        # Second allele drops
        if u2 <= p_ado:
            masked_genome[cur_gsnv:cur_hom+1][:, 1] = [-1, -1]

    return masked_genome


def simulate_cell(cell_genotype, p_ae, mean_coverage):
    reads_list = []
    phred_list = []
    phred_char_list = []

    allele_1_drop = np.array_equal(cell_genotype[:, 0], -1*np.ones_like(cell_genotype[:, 0]))
    allele_2_drop = np.array_equal(cell_genotype[:, 1], -1*np.ones_like(cell_genotype[:, 1]))
    
    # Both alleles drop, no reads
    if allele_1_drop and allele_2_drop:
        return reads_list, phred_list, phred_char_list
    
    else:
        num_reads = np.random.poisson(lam=mean_coverage)
        
        # First allele drops, reads are coming from second allele
        if allele_1_drop:
            reads_list, phred_list, phred_char_list = simulate_reads([cell_genotype[:, 1]], num_reads, p_ae)
        # Second allele drops, reads are coming from first allele
        elif allele_2_drop:
            reads_list, phred_list, phred_char_list = simulate_reads([cell_genotype[:, 0]], num_reads, p_ae)
        # Reads from both alleles                                             
        else:
            reads_list, phred_list, phred_char_list = simulate_reads([cell_genotype[:, 0], cell_genotype[:, 1]],
                                                                     num_reads, p_ae)
    
    return reads_list, phred_list, phred_char_list


def simulate_cell_old(cell_genotype, p_ae, mean_coverage):
    reads_list = []
    phred_list = []
    phred_char_list = []
    
    u1 = np.random.rand()
    u2 = np.random.rand()
    
    # Both alleles drop, no reads
    if u1 <= p_ado and u2 <= p_ado:
        return reads_list, phred_list, phred_char_list
    
    else:
        num_reads = np.random.poisson(lam=mean_coverage)
        
        # First allele drops, reads are coming from second allele
        if u1 <= p_ado:
            reads_list, phred_list, phred_char_list = simulate_reads([cell_genotype[:, 1]], num_reads, p_ae)
        # Second allele drops, reads are coming from first allele
        if u2 <= p_ado:
            reads_list, phred_list, phred_char_list = simulate_reads([cell_genotype[:, 0]], num_reads, p_ae)
        # Reads from both alleles                                             
        else:
            reads_list, phred_list, phred_char_list = simulate_reads([cell_genotype[:, 0], cell_genotype[:, 1]],
                                                                     num_reads, p_ae)
    
    return reads_list, phred_list, phred_char_list


def simulate_reads(fragment_list, num_reads, p_ae):
    for i in range(num_reads):
        new_fragment = fragment_list[np.random.randint(0, len(fragment_list))] 
        
        if new_fragment[0] not in list(range(4)) or new_fragment[1] not in list(range(4)):
            print("ERR: ", new_fragment, list(range(4)))
            
        u1 = np.random.rand()
        u2 = np.random.rand()
        
        if u1 <= p_ae: 
            alternatives = list(range(4))
            alternatives.remove(new_fragment[0])
            alt_nucleotide = np.random.choice(alternatives)
            new_fragment[0] = alt_nucleotide
        if u2 <= p_ae: 
            alternatives = list(range(4))
            alternatives.remove(new_fragment[1])
            alt_nucleotide = np.random.choice(alternatives)
            new_fragment[1] = alt_nucleotide
            
        fragment_list.append(new_fragment)
    
    reads_list = fragment_list[2:]
    
    phred_chars = np.array(["!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2",
                            "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@", "A", "B", "C", "D",
                            "E", "F", "G", "H", "I", "J", "K"])
    phred_list = []
    phred_char_list = []
    
    phred_idx = np.random.randint(30, 43, size=(num_reads, 2))
    for i in range(num_reads):
        q_error = phred_idx[i, :]
        p_error = np.power((10 * np.ones(q_error.shape)), (-0.1 * q_error))
        
        phred_list.append(q_error)
        phred_char_list.append(phred_chars[q_error])

        for j in range(2):
            sample = reads_list[i][j]
            p_vals = np.tile(p_error[j] / 3, 4)
            p_vals[sample] = 1 - p_error[j]
            reads_list[i][j] = np.argmax(np.random.multinomial(1, p_vals))
    
    return np.array(reads_list), np.array(phred_list), np.array(phred_char_list)


def create_cnvs(genome_length, parent_nodes, cnv_rate, seed_val, min_cnv_lenght=2):
    np.random.seed(seed_val)
    num_edges = len(parent_nodes) - 1

    num_cnvs = num_edges * np.random.poisson(5)
    cnv_origin_nodes = np.random.choice(np.arange(1, len(parent_nodes)), size=num_cnvs).astype(int)

    total_cnv_length = int(np.ceil(genome_length * cnv_rate))
    if total_cnv_length % 2 != 0:
        total_cnv_length += 1

    cnv_lengths = min_cnv_lenght * np.ones(num_cnvs, dtype=int)
    ratios = np.random.choice(np.arange(1, 11), size=num_cnvs)
    per_length = (total_cnv_length - min_cnv_lenght*num_cnvs) / np.sum(ratios)
    cnv_lengths += np.floor(per_length * ratios).astype(int)
    diff = total_cnv_length - np.sum(cnv_lengths)
    cnv_lengths[0] += diff

    total_non_cnv_length = genome_length - total_cnv_length
    ratios = np.random.choice(np.arange(1, 101), size=num_cnvs+1)
    per_length = total_non_cnv_length / np.sum(ratios)
    non_cnv_lengths = np.floor(per_length * ratios).astype(int)
    diff = total_non_cnv_length - np.sum(non_cnv_lengths)
    non_cnv_lengths[0] += diff

    cnv_start_positions = np.zeros(num_cnvs, dtype=int)
    cnv_2_alleles = np.zeros(num_cnvs, dtype=int)
    cur_pos = non_cnv_lengths[0]
    for i in range(num_cnvs):
        cur_cnv_length = cnv_lengths[i]

        u = np.random.rand()
        if u > 0.5:
            cnv_2_alleles[i] = 1

        cnv_start_positions[i] = cur_pos
        #print("\tCNV no: ", i, "\tStart: ", cur_pos, "\tEnd: ", cur_pos + cur_cnv_length)

        cur_pos += cur_cnv_length + non_cnv_lengths[i+1]
        #print("\t\tNo CNV no: ", i+1, "\tUntil: ", cur_pos)

    return num_cnvs, cnv_origin_nodes, cnv_start_positions, cnv_lengths, cnv_2_alleles, total_cnv_length


def add_cnvs_to_genome(leaf_idx, leaf_nodes, parent_nodes, cell_genome,
                       cnv_origin_nodes, cnv_start_positions, cnv_lengths, cnv_2_alleles):
    num_total_effected_pos = 0
    num_total_cnv = 0

    cell_id = leaf_nodes[leaf_idx]
    ancestor_list = get_node_ancestors(cell_id, parent_nodes)

    for ancestor_id in ancestor_list:
        cell_mutations_idx = np.where(cnv_origin_nodes == ancestor_id)[0]
        for mut_idx in cell_mutations_idx:
            mut_loc = cnv_start_positions[mut_idx]
            cur_cnv_length = cnv_lengths[mut_idx]

            allele_idx = cnv_2_alleles[mut_idx]
            if allele_idx == 0:
                cell_genome[mut_loc:mut_loc + cur_cnv_length, 1] = cell_genome[mut_loc:mut_loc + cur_cnv_length, 0]
            else:
                cell_genome[mut_loc:mut_loc + cur_cnv_length, 0] = cell_genome[mut_loc:mut_loc + cur_cnv_length, 1]

            num_total_cnv += 1
            num_total_effected_pos += cur_cnv_length

    return cell_genome, num_total_cnv, num_total_effected_pos


def main():
    dt = datetime.datetime.now()
    default_dir = "../data/%s_%s_%s/" % (dt.year, dt.month, dt.day)

    # Code to process command line arguments
    parser = argparse.ArgumentParser(description='Generate synthetic data (BAM) files.')
    #parser.add_argument('--ado_type',
    #                    help="Specify the allelic dropout type "
    #                         "(0 for no ado, 1 for random Beta, 2 for even smaller Beta, 3 for fixed. Default: 3).",
    #                    type=int, default=3)
    parser.add_argument('--avg_mut_per_branch', help="Specify the average number of mutations per branch. Default: 3",
                        type=int, default=3)
    parser.add_argument('--chr_id', help="Specify the chromosome number. Default: 1", type=int, default=1)
    parser.add_argument('--cnv_ratio', help="Specify the CNV ratio in the genome in range [0,1]. Default: 0",
                        type=float, default=0)
    parser.add_argument('--genome_length', help="Specify the genome length. Default: 1", type=int, default=0)
    parser.add_argument('--global_dir', help="Specify the directory.", type=str, default=default_dir)
    parser.add_argument('--mean_coverage', help="Specify the mean read coverage. Default: 10", type=int, default=10)
    parser.add_argument('--no_mut_to_mut_ratio',
                        help="Specify the ratio of non-mutated homozygous sites / mutated sites. Default: 1",
                        type=float, default=1)
    parser.add_argument('--num_cells', help="Specify the number of cells. Default: 10", type=int, default=10)
    parser.add_argument('--p_ado', help="Specify the allelic dropout probability of a base. Default: 0.005", type=float,
                        default=0.005)
    parser.add_argument('--p_ae', help="Specify the amplification error probability of a base. Default: 0.00001",
                        type=float, default=0.00001)
    parser.add_argument('--phased_freq', help="Specify the frequency of phased sites ([0,1]). Default: 1", type=float,
                        default=1)
    #parser.add_argument('--phred_type',
    #                    help="Specify the phred score type "
    #                         "(0 for no errors, 1 for all 42, 2 for truncated normal). Default: 2",
    #                    type=int, default=2)
    parser.add_argument('--read_length', help="Specify the read length. Default: 2", type=int, default=2)  # It must be 2. Don't change.
    parser.add_argument('--seed_val', help="Specify the seed. Default: 123", type=int, default=123)
    args = parser.parse_args()

    start_time_global = time.time()

    np.random.seed(args.seed_val)

    # Step 0: Arrange folders
    bam_dir = args.global_dir + "bam/"
    truth_dir = args.global_dir + "truth/"
    if not os.path.exists(bam_dir):
        os.makedirs(bam_dir)
        print("\tDirectory is created: ", bam_dir)

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)
        print("\tDirectory is created: ", truth_dir)

    # Step 1: Simulate binary tree
    parent_nodes, leaf_nodes = generate_real_bdp_tree(args.num_cells, args.seed_val)
    filename = truth_dir + "parent_nodes.txt"
    np.savetxt(filename, parent_nodes, fmt='%d')
    filename = truth_dir + "leaf_nodes.txt"
    np.savetxt(filename, leaf_nodes, fmt='%d')

    # Step 2: Decide the number of mutated and not mutated sites, determine genome length
    num_mut = 2 * (args.num_cells-1) * args.avg_mut_per_branch
    if args.genome_length == 0:
        num_no_mut = int(np.ceil(num_mut * args.no_mut_to_mut_ratio))
    else:
        num_no_mut = int(np.ceil(args.genome_length/2)) - num_mut
    num_gsnv = int(np.ceil((num_mut + num_no_mut) * args.phased_freq))
    num_bp = 2 * (num_mut + num_no_mut)

    print("Number of basepairs: ", num_bp)
    print("\tNumber of gSNVs: ", num_gsnv)
    print("\tNumber of mutations: ", num_mut)
    print("\tNumber of non-mutations: ", num_no_mut)
    print("\tNumber of no-purpose basepairs: ", num_bp-num_gsnv-num_mut-num_no_mut)

    # Step 3: Assign mutations to branches
    mut_origin_nodes = assign_mutations_to_nodes(parent_nodes, num_mut, args.seed_val)
    filename = truth_dir + "mut_origin_nodes.txt"
    np.savetxt(filename, mut_origin_nodes, fmt='%d')

    # Step 3.1: Create CNVs and assign to branches
    if args.cnv_ratio != 0:
        num_cnvs, cnv_origin_nodes, cnv_start_positions, cnv_lengths, cnv_2_alleles, total_cnv_length = create_cnvs(
            num_bp, parent_nodes, args.cnv_ratio, args.seed_val)
        print("\tNumber or CNVs: ", num_cnvs, " ( CNV ratio: ", args.cnv_ratio, ")")
        print("\tNumber or basepairs effected by CNVs: ", total_cnv_length)
        filename = truth_dir + "cnv_origin_nodes.txt"
        np.savetxt(filename, cnv_origin_nodes, fmt='%d')
        filename = truth_dir + "cnv_start_positions.txt"
        np.savetxt(filename, cnv_start_positions, fmt='%d')
        filename = truth_dir + "cnv_lengths.txt"
        np.savetxt(filename, cnv_lengths, fmt='%d')
        filename = truth_dir + "cnv_2_alleles.txt"
        np.savetxt(filename, cnv_2_alleles, fmt='%d')

    # Step 4: Simulate bulk and mut genome
    print("Simulating bulk")
    bulk_genome, mut_genome, gsnv_locations, mut_locations = generate_bulk_genotype(num_bp, num_gsnv, num_mut,
                                                                                    args.seed_val)
    filename = truth_dir + "bulk_genome.txt"
    np.savetxt(filename, bulk_genome, fmt='%d')
    filename = truth_dir + "mut_genome.txt"
    np.savetxt(filename, mut_genome, fmt='%d')
    filename = truth_dir + "gsnv_locations.txt"
    np.savetxt(filename, gsnv_locations, fmt='%d')
    filename = truth_dir + "mut_locations.txt"
    np.savetxt(filename, mut_locations, fmt='%d')

    generate_bulk_fasta(truth_dir, bulk_genome, args.chr_id)
    generate_bulk_sam(truth_dir, bulk_genome, args.chr_id, args.read_length)

    generate_bed_file(truth_dir + "gsnv_positions.bed", gsnv_locations, chr_id=args.chr_id)
    generate_regions_bed_file(truth_dir + "gsnv_vars_regions.bed", gsnv_locations, num_bp,
                              read_length=args.read_length, chr_id=args.chr_id)
    generate_bed_file(truth_dir + "mutation_positions.bed", mut_locations, chr_id=args.chr_id)

    # Step 5: Simulate cell genomes
    print("Simulating cell genomes")
    cell_dict = {}   
    for leaf_idx in range(args.num_cells):
        print("\tSimulating cell genomes: ", leaf_idx)
        cell_seed_val = args.seed_val * (leaf_idx + 1)
        cell_genome, all_mutations = generate_cell_genotype(leaf_idx, leaf_nodes, bulk_genome, mut_genome,
                                                            mut_origin_nodes, mut_locations, parent_nodes,
                                                            seed_val=cell_seed_val)

        filename = truth_dir + "cell_" + str(leaf_idx) + "_genome.txt"
        np.savetxt(filename, cell_genome, fmt='%d')

        if args.cnv_ratio != 0:
            cell_cnv_genome, cell_num_total_cnv, cell_num_total_effected_pos = add_cnvs_to_genome(
                leaf_idx, leaf_nodes, parent_nodes, cell_genome,
                cnv_origin_nodes, cnv_start_positions, cnv_lengths, cnv_2_alleles)
            filename = truth_dir + "cell_" + str(leaf_idx) + "_cnv_genome.txt"
            np.savetxt(filename, cell_cnv_genome, fmt='%d')
            cell_genome = cell_cnv_genome  # To mask the genome with CNVs
            print("\tCell: ", leaf_idx, " has ", cell_num_total_cnv,
                  " CNV regions effecting ", cell_num_total_effected_pos, " basepairs.")

        masked_genome = mask_genome(cell_genome, args.p_ado, args.read_length)
        filename = truth_dir + "cell_" + str(leaf_idx) + "_masked_genome.txt"
        np.savetxt(filename, masked_genome, fmt='%d')
        
        cell_dict[str(leaf_idx)] = masked_genome

    # Step 7: Create cell SAM files
    print("Creating cell SAM files")
    for leaf_idx in range(args.num_cells):
        print("\tCell: ", leaf_idx)
        cell_seed_val = args.seed_val * (leaf_idx + 1)
        np.random.seed(cell_seed_val)
        reads_list = []
        phred_list = []
        fragments_start_locations_list = []

        for i in range(int(num_bp/2)):
            cur_gsnv = args.read_length * i
            cur_hom = cur_gsnv + 1

            r_list, p_list, _ = simulate_cell(cell_dict[str(leaf_idx)][cur_gsnv:cur_hom + 1, :],
                                              args.p_ae, args.mean_coverage)

            cur_cell_dict = {"reads": r_list, "p_error": p_list, "lc": len(r_list)}

            for l in range(cur_cell_dict['lc']):
                fragments_start_locations_list.append(cur_gsnv)
                reads_list.append(cur_cell_dict['reads'][l])
                phred_list.append(cur_cell_dict['p_error'][l])

        generate_cell_sam(truth_dir, num_bp, leaf_idx, reads_list, phred_list, fragments_start_locations_list,
                          args.read_length, args.chr_id)

    # Step 8: Relocate and delete files
    print("Relocating files")
    cmd = "mv " + truth_dir + "*.bam " + bam_dir
    os.system(cmd)
    cmd = "mv " + truth_dir + "*.bai " + bam_dir
    os.system(cmd)
    cmd = "cp " + truth_dir + "gsnv_positions.bed " + bam_dir
    os.system(cmd)
    cmd = "cp " + truth_dir + "gsnv_vars_regions.bed " + bam_dir
    os.system(cmd)
    cmd = "rm " + truth_dir + "*.sam " 
    os.system(cmd)

    # Step 9: Create real tree newick
    print("Creating newick tree")
    mid_filename, mid_filename_normed = create_branch_files(truth_dir)
    create_newick_files(truth_dir + "real.tre", truth_dir + "real_normed.tre", mid_filename, mid_filename_normed)

    print("\n***** DONE!")
    end_time_global = time.time()
    print("\tTotal global time: ", end_time_global - start_time_global, "\n*****")


if __name__ == "__main__":
    main()
