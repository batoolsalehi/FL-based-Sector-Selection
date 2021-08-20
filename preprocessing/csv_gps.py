## wriiten in python 3.7.6
## user input: bag file directory line 32
import bagpy
import os,glob 
from bagpy import bagreader
import pandas as pd
import numpy as np
import csv
import math
# origin for coordinates = 42.338128661705255, -71.08705289706138
# BS coordinates = 42.33783671678225, -71.08665391849328

def show_all_files_in_directory(input_path,extension):
    'This function reads the path of all files in directory input_path'
    files_list=[]
    for path, subdirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(extension):
               files_list.append(os.path.join(path, file))
    return files_list


def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False


###user inputs
bag_file_directory = '/media/batool/MULTIMODAL_DATA/Multimodal_Data/Cat4/car_20_lr_blockage_20_rl/bag_files'   ##cat1
# bag_file_directory = '/media/batool/Seagate/MultiModal_Data/Cat3/static_in_front/bag_files'   ##cat2

bag_files = show_all_files_in_directory(bag_file_directory,'.bag')
print("bag_files",bag_files)
print('*************Generating the csv file of all bag files')      ###coment line 34-39 if already genertaed
# generate csv files
for f in bag_files:
	b = bagreader(f)
	data = b.message_by_topic('/vehicle/gps/fix')
	print("File saved: {}".format(data))


csv_files = show_all_files_in_directory(bag_file_directory,'vehicle-gps-fix.csv')
convert_to_x_y = True


print('*************Attaching experiment information****************')
for files in csv_files:
	appended_data = []
	with open(files, 'r', encoding='UTF8', newline='') as f:
	    reader = csv.reader(f)
	    data = list(reader)

	file_name = files.split('/')[-2]         ###sample of file_name:Cat1_20mph_lr_same_9
	# experiment info incldues: Category, speed, direction, lane episode  
	experiment_info = [file_name.split('_')[0],file_name.split('_')[1],file_name.split('_')[2],file_name.split('_')[3],file_name.split('.')[0].split('_')[-1]]
	for d in data[1:]:
		time_header = str(int(d[2])+1e-9*int(d[3]))
		appended_data.append(experiment_info+[time_header]+d[:10])

	header = ['Category', 'speed', 'direction', 'lane','episode']+['time header']+data[0][:10]


	###the appended _data file is generated 
	if convert_to_x_y:
		print('*********Converting to x y **********')
		converted = []
		## find minimum
		# min_lat = min([float(d[13]) for d in appended_data])
		# min_long = min([float(d[14]) for d in appended_data])
		min_lat = 42.33812866170525
		min_long = -71.08705289706138

		print('min lat/long', min_lat,min_long)
		for d in appended_data:
			dx = 1000*(float(d[14])-min_long)*40000*math.cos((float(d[13])+min_lat)*math.pi/360)/360
			dy = 1000*(float(d[13])-min_lat)*40000/360
			converted.append(d+[dx,dy]) 
		appended_data = converted
		header = ['Category', 'speed', 'direction', 'lane','episode']+['time header']+data[0][:10] + ['x','y']
		print(header)


	###################################### writing
	episode_index = file_name.split('_')[-1]
	print(file_name)
	save_directory = bag_file_directory[:-9]+'Extract/'+file_name[:-2]+'/episode_'+str(episode_index)+'/'
	check_and_create(save_directory)
	with open(save_directory+'GPS_data.csv', 'w', encoding='UTF8', newline='') as f:
	    writer = csv.writer(f)
	    writer.writerow(header)
	    writer.writerows(appended_data)