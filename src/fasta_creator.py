import os
import sys
import time
import json
import pysam
import pickle
import argparse
import datetime
import matplotlib
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def save_json(filename, cell_dict):
    with open(filename, 'w') as fp:
        json.dump(cell_dict, fp)


def load_json(filename):
    with open(filename) as fp:
        cell_dict = json.load(fp)
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

    filename = global_dir + "truth/bulk_ref.sam"
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
    out_filename = global_dir + "truth/bulk_ref.fasta"
    cmd = "samtools view " + filename + " | awk '{OFS=\"\\t\"; print \">\"$1\"\\n\"$10}' - > " + out_filename
    print("\tRunning: ", cmd)
    os.system(cmd)


def main():
    dt = datetime.datetime.now()
    default_dir = "../data/%s_%s_%s/" % (dt.year, dt.month, dt.day)

    # Code to process command line arguments
    parser = argparse.ArgumentParser(description='Generates FASTA reference file.')
    parser.add_argument('global_dir', help="Specify the directory.", type=str)
    parser.add_argument('--chr_id', help="Specify the chromosome number. Default: 1", type=int, default=1)
    args = parser.parse_args()

    start_time_global = time.time()

    print("The data directory: ", args.global_dir)
    if not os.path.exists(args.global_dir):
        os.makedirs(args.global_dir)
        print("Directory is created: ", args.global_dir)

    print("\nFasta creation starts...")

    # PART 1: Amplification bias
    start_time = time.time()
    print("\nPart 1\n")

    filename = args.global_dir + "truth/bulk_genome.txt"
    bulk_genome = np.array(load_json(filename))

    generate_bulk_fasta(args.global_dir, bulk_genome, args.chr_id)

    end_time = time.time()
    print("\nTotal time: ", end_time - start_time)
    print("Part 1 ends...")
    ###

    end_time_global = time.time()
    print("\nTotal (global) time: ", end_time_global - start_time_global)
    print("Fasta creation ends...")


if __name__ == "__main__":
    main()
