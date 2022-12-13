
import os
import time
import json
import pysam
import argparse
import datetime
import numpy as np


def save_json(filename, cell_dict):
    with open(filename, 'w') as fp:
        json.dump(cell_dict, fp)


def load_json(filename):
    with open(filename) as fp:
        cell_dict = json.load(fp)
    return cell_dict


def parse_original_dataset(filename):
    data_orig = load_json(filename)

    new_data = {}
    for pos in range(len(list(data_orig.keys()))):
        new_data[str(pos)] = data_orig[str(pos)]

        temp_bulk = np.array(data_orig[str(pos)]['bulk'], dtype=int)[:, 1]
        new_data[str(pos)]['bulk'] = [int(temp_bulk[0]), int(temp_bulk[1])]

        temp_z = np.array(data_orig[str(pos)]['common_z'], dtype=int)[:, 1]
        new_data[str(pos)]['common_z'] = [int(temp_z[0]), int(temp_z[1])]

        z_list = []
        for i in range(3):
            z_list.append([new_data[str(pos)]['bulk'][0], (new_data[str(pos)]['bulk'][0] + i + 1) % 4])
        new_data[str(pos)]['z_list'] = z_list

        for cell_idx in range(len(new_data[str(pos)]['cell_list'])):
            cell_info = new_data[str(pos)]['cell_list'][cell_idx]
            reads = []
            p_error = []
            for i in range(cell_info['lc']):
                reads.append(cell_info['reads'][i][1])
                p_error.append(cell_info['p_error'][i][1])

            new_data[str(pos)]['cell_list'][cell_idx]['reads'] = reads
            new_data[str(pos)]['cell_list'][cell_idx]['p_error'] = p_error

    save_json(filename, new_data)
    print("New dataset is saved to: ", filename)

    new_filename = filename + "_backup_single_version.txt"
    save_json(new_filename, new_data)
    print("New dataset is copied to: ", new_filename)


def main():
    dt = datetime.datetime.now()

    # Code to process command line arguments
    parser = argparse.ArgumentParser(description='Generate synthetic data (BAM) files.')
    parser.add_argument('global_dir', help="Specify the directory.", type=str)
    args = parser.parse_args()

    start_time_global = time.time()

    # Step 0: Arrange folders
    proc_dict_dir = args.global_dir + "processed_data_dict/"
    print("Output directory: ", proc_dict_dir)

    # Step 1: Backup original pairs dataset and read & logzcy results
    print("\nBacking up pair dataset and computed dictionaries...")

    filename = proc_dict_dir + "data.txt"
    backup_filename = filename + "_backup_paired_version.txt"
    cmd = "cp " + filename + " " + backup_filename
    print("\tRunning: ", cmd)
    os.system(cmd)

    read_prob_dir = proc_dict_dir + "read_probabilities/"
    backup_read_prob_dir = proc_dict_dir + "read_probabilities_backup_paired_version/"
    cmd = "mv " + read_prob_dir + " " + backup_read_prob_dir
    print("\tRunning: ", cmd)
    os.system(cmd)

    # Step 2: Parse the original dataset and save the new one.
    parse_original_dataset(filename)

    # Step 3: Move reads to

    print("\n***** DONE!")
    end_time_global = time.time()
    print("\tTotal global time: ", end_time_global - start_time_global, "\n*****")


if __name__ == "__main__":
    main()