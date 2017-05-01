import numpy as np
import os
import scipy.io as sio

import cut_path as cp

ground_truth_dir = '../data/ground-truth/locations/'
files = os.listdir(ground_truth_dir)
nr_files = len(files)
multiple_cut_words = []
for idx, file in enumerate(files):
    file_number = file[0:3]
    single_cut_word = cp.cut_path(file_number, ground_truth_dir)
    # also save filename
    multiple_cut_words.append([file, single_cut_word])

cut_words = np.array(multiple_cut_words)
sio.savemat('cut_words.mat', {'cutWords': cut_words})
