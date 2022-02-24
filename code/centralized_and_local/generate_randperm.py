import numpy as np
import os

def show_all_files_in_directory(input_path,extension):
    'This function reads the path of all files in directory input_path'
    files_list=[]
    for path, subdirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(extension):
               files_list.append(os.path.join(path, file))
    return files_list




path = '/home/batool/FL/data'
all_npz_files = show_all_files_in_directory(path,'.npz')

##all files have the same length
all_gps_npz_files = [a for a in all_npz_files if 'gps.npz' in a.split('/')]

for npz_file in all_gps_npz_files:
    print(npz_file)
    gps_npz = np.load(npz_file)['gps']
    randperm = np.random.permutation(len(gps_npz))
    # 'data/a.npy'
    save_directory = '/'.join(npz_file.split('/')[:-1])
    print('save directory',save_directory)
    np.save(save_directory+'/ranperm.npy',randperm)
