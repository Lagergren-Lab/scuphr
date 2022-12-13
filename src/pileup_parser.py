######################################################################################################################
# A simple pileup file parser. It omits insertions, deletions etc. Only counts the observed number of A,C,G and T.
#
# Author : Hazal Koptagel
#          KTH
#          E-mail : koptagel@kth.se
#
######################################################################################################################

import time
import argparse
import numpy as np

def get_stats(filename, num_lines=None):
    """
    Given the filename, this function retrieves the line count and the number of samples from pileup file. 
    If the user specified these values, the function simply return them.
    :param filename: String. Path of pileup file.
    :param num_lines: None or Scalar. The number of lines in pileup file.
    :return: num_lines: Scalar. The number of lines in pileup file.
    :return: num_samples: Scalar. The number of samples in pileup file.
    """
    if num_lines is None:
        num_lines = sum(1 for line in open(filename))

    with open(filename) as f:
        line = f.readline()
        num_samples = int((len(line.split("\t")) - 3) / 3)
        
    print("\tnum_lines: ", num_lines, "\tnum_samples: ", num_samples)
    return num_lines, num_samples

def parse_pileup(filename, num_lines, num_samples):
    """
    Given the filename, number of lines and samples of pileup file; this function parses the pileup file. 
    :param filename: String. Path of pileup file.
    :param num_lines: Scalar. The number of lines in pileup file.
    :param num_samples: Scalar. The number of samples in pileup file.
    :return: data: ndarray (dtype=int). Parsed data of shape (num_lines, num_samples*5 + 3). 
                                        First three columns correspond to CHR, POS, REF. Rest are sample data.
                                        CHR POS REF A1 C1 G1 T1 DEPTH1 A2 C2 G2 T2 DEPTH2 ...
                                        If reference nucleotide is not A,C,G,T, we assign -1. 
                                        Otherwise 0,1,2,3 respectively.
    """
    nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # Create array to store counts
    data = np.empty((num_lines, num_samples*5 + 3), dtype=int)

    # Parse pileup
    with open(filename, "r") as pileup_file:
        i = 0
        for line in pileup_file:
            contents = line.split("\t")
            ref_nuc = contents[2]   

            data[i, 0] = int(contents[0]) # Chr
            data[i, 1] = int(contents[1]) # Pos
            try:
                data[i, 2] = int(nuc_map[ref_nuc]) # Ref nuc idx
            except:
                data[i, 2] = -1 # Invalid reference nuc

            for s in range(num_samples):
                #sample_contents = contents[(s+1)*3:(s+2)*3]
                sample_nucleotides = contents[(s+1)*3+1]

                counts = np.zeros(5, dtype=int)
                for ch in sample_nucleotides:
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
    return data

def main():
    # code to process command line arguments
    parser = argparse.ArgumentParser(description='Simple pileup file parser.')
    parser.add_argument('filename', help="Specify the pileup file path.", type=str)
    parser.add_argument('out_filename', help="Specify the output file path.", type=str)
    parser.add_argument('--num_lines', help="Specify the number of lines in pileup file.", type=int)
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("Running pileup parser...")
    # Get line count and number of samples
    num_lines, num_samples = get_stats(args.filename, args.num_lines)
    # Parse pileup file
    data = parse_pileup(args.filename, num_lines, num_samples)
    # Save parsed contents
    np.savetxt(args.out_filename, data, fmt='%d', delimiter='\t')
    print("Parsed file is saved to: ", args.out_filename)
    
    end_time = time.time()
    print("\nTotal time: ", end_time - start_time)
    
    
if __name__ == "__main__":
    main()