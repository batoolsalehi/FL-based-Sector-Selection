import cv2
import numpy as np
import os
from tqdm import tqdm

def show_all_files_in_directory(input_path,extension):
    'This function reads the path of all files in directory input_path'
    files_list=[]
    for path, subdirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(extension):
               files_list.append(os.path.join(path, file))
    return files_list



path = '/home/batool/FL/data_half_half_size'
all_npz_files = show_all_files_in_directory(path,'.npz')


all_image_npz_files = [a for a in all_npz_files if 'image.npz' in a.split('/')]
resize_factor = 2


for npz_file in tqdm(all_image_npz_files):
    print('npz_file',npz_file)
    image_npz = np.load(npz_file)['img']
    all_image = np.empty([image_npz.shape[0],int(image_npz.shape[1]/resize_factor),int(image_npz.shape[2]/2),3], dtype=float)
    print('compare shapes',image_npz.shape,all_image.shape)


    for l in range(len(image_npz)):
        this_image = image_npz[l]
        resized_image = cv2.resize(this_image, (0,0), fx=0.5, fy=0.5)
        all_image[l] = resized_image

    np.savez_compressed(npz_file, img=all_image)
