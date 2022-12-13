import json
import argparse
import time
import numpy as np


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def save_json(filename, cell_dict):
    with open(filename, 'w') as fp:
        json.dump(cell_dict, fp)


def load_json(filename):
    with open(filename) as fp:
        cell_dict = json.load(fp)
    return cell_dict


def get_paired_singleton_sites(dataset, num_total_positions):
    hom_pos_list = []
    positions_orig = np.arange(0, num_total_positions)
    positions_paired = []
    positions_singleton = []
    for pos in positions_orig:
        cur_bulk = np.array(dataset[str(pos)]["bulk"])
        if cur_bulk.shape == (2, 2):
            positions_paired.append(pos)
            hom_pos_list.append(int(dataset[str(pos)]['pos_pair'][1]))
        elif cur_bulk.shape == (2,):
            positions_singleton.append(pos)
            hom_pos_list.append(int(dataset[str(pos)]['pos_pair']))
    return hom_pos_list, positions_paired, positions_singleton


def read_vcf(filename, chr_id):
    char_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    variants_vcf = []
    ref_vcf = []
    alt_vcf = []
    vcf_file = open(filename)
    vcf_lines = vcf_file.readlines()
    for line in vcf_lines:
        contents = line.split(sep='\t')
        if contents[0] == str(chr_id):
            variants_vcf.append(int(contents[1]))
            ref_vcf.append(char_map[contents[3]])
            alt_vcf.append(char_map[contents[4]])
    variants_vcf = sorted(variants_vcf)
    vcf_file.close()
    return variants_vcf, ref_vcf, alt_vcf


def write_inclusion_file(filename, dataset, num_total_positions, chr_id, variants_vcf, ref_vcf, alt_vcf):
    nuc_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    inc_file = open(filename, 'w')

    for pos_idx in range(num_total_positions):
        cur_bulk = np.array(dataset[str(pos_idx)]['bulk'])
        is_paired = False
        if cur_bulk.shape == (2, 2):
            cur_bulk = np.array(dataset[str(pos_idx)]['bulk'])[:, 1]
            cur_pos = int(dataset[str(pos_idx)]['pos_pair'][1])
            is_paired = True
        else:
            cur_pos = int(dataset[str(pos_idx)]['pos_pair'])

        if cur_pos in variants_vcf:
            i = variants_vcf.index(cur_pos)
            ref = ref_vcf[i]
            ref_str = nuc_map[ref]
            alt = alt_vcf[i]
            alt_str = nuc_map[alt]
        else:
            ref = np.unique(cur_bulk)[0]
            ref_str = nuc_map[ref]
            counts = [0, 0, 0, 0]
            for cell in dataset[str(pos_idx)]["cell_list"]:
                for read in cell['reads']:
                    if is_paired:
                        counts[read[1]] += 1
                    else:
                        counts[read] += 1
            counts[ref] = 0
            alt = np.argmax(counts)
            alt_str = nuc_map[alt]
        inc_file.write(str(chr_id) + "\t" + str(cur_pos) + "\t." + "\t" + ref_str + "\t" + alt_str + "\n")
    inc_file.close()


def write_exclusion_file(filename, chr_id, genome_length, hom_pos_list):
    exc_file = open(filename, 'w')
    for i in range(genome_length):
        cur_pos = i + 1
        if cur_pos not in hom_pos_list:
            exc_file.write(str(chr_id) + "\t" + str(cur_pos) + "\n")
    exc_file.close()


def main():
    # code to process command line arguments
    parser = argparse.ArgumentParser(description='Creates VCF files for Sciphi, using Scuphr sites.')
    parser.add_argument('data_dir', help="Specify the data directory", type=str)
    parser.add_argument('monovar_vcf', help="Specify the VCF file (output of Monovar).", type=str)
    parser.add_argument('--chr_id', help="Specify the chromosome number. Default: 1", type=int, default=1)
    parser.add_argument('--genome_length', help="Specify the length of genome. Default: 50000", type=int, default=50000)

    args = parser.parse_args()
    start_time_global = time.time()

    print("\nSTEP 1: Load Scuphr dataset")
    start_time = time.time()
    dataset = load_json(args.data_dir + "processed_data_dict/" + "data_orig.txt")
    num_total_positions = len(dataset)
    print("\tTotal number of positions in Scuphr dataset: \t", num_total_positions)
    end_time = time.time()
    print("\nTotal time: ", end_time - start_time)

    print("\nSTEP 2: Separate paired and singleton sites")
    start_time = time.time()
    hom_pos_list, positions_paired, positions_singleton = get_paired_singleton_sites(dataset, num_total_positions)
    print("\tNumber of paired sites: \t", len(positions_paired))
    print("\tNumber of singleton sites: ", len(positions_singleton))
    end_time = time.time()
    print("\nTotal time: ", end_time - start_time)

    print("\nSTEP 3: Read VCF file (Monovar)")
    start_time = time.time()
    variants_vcf, ref_vcf, alt_vcf = read_vcf(args.monovar_vcf, args.chr_id)
    print("\tNumber of variants in VCF file (Monovar): ", len(variants_vcf))
    end_time = time.time()
    print("\nTotal time: ", end_time - start_time)

    print("\nSTEP 4: Write inclusion file")
    start_time = time.time()
    inclusion_filename = args.data_dir + "/bam/inclusion_list.vcf"
    write_inclusion_file(inclusion_filename, dataset, num_total_positions, args.chr_id, variants_vcf, ref_vcf, alt_vcf)
    print("\tInclusion file is saved to: \t", inclusion_filename)
    end_time = time.time()
    print("\nTotal time: ", end_time - start_time)

    print("\nSTEP 5: Write exclusion file")
    start_time = time.time()
    exclusion_filename = args.data_dir + "/bam/exclusion_list.vcf"
    write_exclusion_file(exclusion_filename, args.chr_id, args.genome_length, hom_pos_list)
    print("\tExclusion file is saved to: \t", inclusion_filename)
    end_time = time.time()
    print("\nTotal time: ", end_time - start_time)

    end_time_global = time.time()
    print("\nTotal (global) time: ", end_time_global - start_time_global)
    print("Creating Sciphi inclusion and exclusion files end...")


if __name__ == "__main__":
    main()

