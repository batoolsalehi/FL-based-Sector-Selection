import csv 
import numpy 
import time
import datetime
from decimal import Decimal
from operator import itemgetter
import math
import os
###user input line 107

def interpolate_gps(GPS_selected,target_time_slot,mid_points_time_stamp):
	coorinates_target_time_slot = []
	print('target time',type(target_time_slot[0]))
	for t in target_time_slot:
		##find two nearest GPS points to target time slot start and end
		diffrence = [abs(float(GPS_selected[i][5])-float(t)) for i in range(len(GPS_selected))]
		## find top 2 min values, closest ones
		top2 = list(sorted(enumerate(diffrence), reverse=True, key = itemgetter(1)))[-2:]     # find top 2 closest items
		# print('top2', top2)
		m = top2[1][1] #closest
		n = top2[0][1]
		lat1, long1 = float(GPS_selected[top2[1][0]][13]), float(GPS_selected[top2[1][0]][14])  #closest
		lat2, long2 = float(GPS_selected[top2[0][0]][13]), float(GPS_selected[top2[0][0]][14])
		# print('prins',m,n,long1,lat1,long2,lat2,GPS_selected[top2[1][0]],GPS_selected[top2[0][0]])
		# Estimate the GPS coordinates of taregt time slot start and end
		if top2[1][1]<t<top2[0][1]:  # in between
			# print('first')
			longx = (m*long2+n*long1)/(m+n)
			latx = (m*lat2+n*lat1)/(m+n)
		else: 
			# print('second')     # on one side
			longx = (n*long1-m*long2)/(n-m)
			latx = (n*lat1-m*lat2)/(n-m)

		coorinates_target_time_slot.append((latx,longx))
		print('The estimated GPS coordinats of target time slot start/end', coorinates_target_time_slot)

	## split the long and latitiudat of target time slot start and end
	lat1, long1 = coorinates_target_time_slot[0][0], coorinates_target_time_slot[0][1]
	lat2, long2 = coorinates_target_time_slot[1][0], coorinates_target_time_slot[1][1]
	print('traget time slot long/log start/end',lat1,lat2,long1,long2)
	## the points that we want to estimate are in between so the same as line 26 to 28
	inter_lat = []
	inter_log = []
	for mid_points in mid_points_time_stamp:
		mid_m = mid_points - target_time_slot[0]
		mid_n = target_time_slot[1]- mid_points

		inter_lat.append((mid_m*lat2+mid_n*lat1)/(mid_m+mid_n))
		inter_log.append((mid_m*long2+mid_n*long1)/(mid_m+mid_n))

	# print(inter_lat,inter_log)
	return inter_log, inter_lat


def open_csv_files(csv_files_directory):
	GPS = csv_files_directory+'/GPS_data.csv'
	Image = csv_files_directory+'/Image_data.csv'
	pcd = csv_files_directory+'/pcd_data.csv'
	RF = csv_files_directory+'/RF_data.csv' 
	print('*************opening GPS csv file****************')
	with open(GPS, 'r', encoding='UTF8', newline='') as f:
		reader = csv.reader(f)
		GPS_data = list(reader)

	print('*************opening Image csv file****************')
	with open(Image, 'r', encoding='UTF8', newline='') as f:
		reader = csv.reader(f)
		Image_data = list(reader)

	print('*************opening pcd csv file****************')
	with open(pcd, 'r', encoding='UTF8', newline='') as f:
		reader = csv.reader(f)
		pcd_data = list(reader)

	print('*************opening RF csv file****************')
	with open(RF, 'r', encoding='UTF8', newline='') as f:
		reader = csv.reader(f)
		RF_data = list(reader)
	print('# of samples for GPS image lidar and RF:',len(GPS_data),len(Image_data),len(pcd_data),len(RF_data))


	return GPS_data, Image_data, pcd_data, RF_data



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





path_directory = '/media/batool/MULTIMODAL_DATA/Multimodal_Data/Cat4/car_20_lr_blockage_20_rl/bag_files'
bag_files = show_all_files_in_directory(path_directory,'.bag')
print(bag_files)
convert_to_x_y = True

for bf in bag_files:

	bag_file_name = bf.split('/')[-1].split('.')[0]
	episode = bag_file_name.split('_')[4]
	path_of_expeirmnet = path_directory[:-9]+'Extract/'+bag_file_name[:-2]+'/episode_'+str(episode)
	print('path_of_expeirmnet',path_of_expeirmnet)
	GPS_data, Image_data, pcd_data, RF_data = open_csv_files(path_of_expeirmnet)

	# extracting same exoeriment data
	experiment_info = [bag_file_name.split('_')[0],bag_file_name.split('_')[1],bag_file_name.split('_')[2],bag_file_name.split('_')[3],bag_file_name.split('_')[4]]
	print('************ experiment_info*************')
	print('experiment_info',experiment_info)
	# experiment_info = [Catergory,speed,direction,lan,episodes]
	GPS_selected = [GPS_data[i] for i in range(len(GPS_data)) if GPS_data[i][0:5]==experiment_info]
	pcd_selected = [pcd_data[i] for i in range(len(pcd_data)) if pcd_data[i][0:5]==experiment_info]
	img_selected = [Image_data[i] for i in range(len(Image_data)) if Image_data[i][0:5]==experiment_info]
	RF_selected = [RF_data[i] for i in range(len(RF_data)) if RF_data[i][0:5]==experiment_info]

	synchornized_data_all = []
	for i in range(len(RF_selected)-1):   #range(len(RF_selected)-1) or range(6,7)
		synchornized_data = []

		target_time_slot = [float(RF_selected[i][5]),float(RF_selected[i+1][5])]
		print('target time slot/diffrence',target_time_slot,float(target_time_slot[1])-float(target_time_slot[0]))
		##find image and lidar modalities in this slot
		pcd_in_range = [pcd_selected[l] for l in range(len(pcd_selected)) if target_time_slot[0]<float(pcd_selected[l][5])<target_time_slot[1]]
		image_in_range = [img_selected[l] for l in range(len(img_selected)) if target_time_slot[0]<float(img_selected[l][5])<target_time_slot[1]]
		print('# of pcd and image files in range',len(pcd_in_range),len(image_in_range))
							
		if len(image_in_range) !=0:    ##added
			print('************ pair synchornized image and lidar data*****************')
			for l in range(len(pcd_in_range)):
				diffrence=[]
				for j in range(len(image_in_range)):
					diffrence.append(abs(float(pcd_in_range[l][5])-float(image_in_range[j][5])))
				closest_image = diffrence.index(min(diffrence))    # find minuum diatnce 
				# attach:experiment info+time stamp of lidar+pcd file path+image file path
				synchornized_data.append(experiment_info+[pcd_in_range[l][5],pcd_in_range[l][-1],image_in_range[closest_image][-1]])
			print('************ Interpolating gps data*****************')
			mid_points_time_stamp_to_calculate = [float(s[5]) for s in synchornized_data]
			inter_log, inter_lat = interpolate_gps(GPS_selected,target_time_slot,mid_points_time_stamp_to_calculate)
			print(len(inter_log),len(inter_lat),len(synchornized_data))
			print('check point',GPS_selected,target_time_slot,mid_points_time_stamp_to_calculate,inter_log,inter_lat)
			print('************ attaching gps interpolated data*****************')
			bias_index = len(synchornized_data_all)
			for l in range(len(synchornized_data)):
				synchornized_data[l].append(inter_lat[l])
				synchornized_data[l].append(inter_log[l])

			print('************ attaching RF data*****************')
			for l in range(len(synchornized_data)):
				for c in range(len(RF_selected[i][6:])):
					synchornized_data[l].append(RF_selected[i][6+c])

			# print(synchornized_data)    ### HERE  check appedn
			### unpack the synchronized_data
			[synchornized_data_all.append(s) for s in synchornized_data]


			header = ['Category','speed','direction','lane','episode','time_stamp','lida_file','image_file','lat','long','max_rssi','selected_sector','tx_mcs','rx_mcs','sqi','all_rssi', 'all_sector']

	###the appended _data file is generated 
	if convert_to_x_y:
		print('*********Converting to x y **********')
		converted = []
		## find minimum
		# min_lat = min([float(d[8]) for d in synchornized_data_all])
		# min_long = min([float(d[9]) for d in synchornized_data_all])
		min_lat = 42.33812866170525
		min_long = -71.08705289706138
		print('min lat/long', min_lat,min_long)
		for d in synchornized_data_all:
			dx = 1000*(float(d[9])-min_long)*40000*math.cos((float(d[8])+min_lat)*math.pi/360)/360
			dy = 1000*(float(d[8])-min_lat)*40000/360
			converted.append(d+[dx,dy]) 
		synchornized_data_all = converted
		header = header + ['x','y']
		print(header)

		print('# of generated data',len(synchornized_data_all))

	# check_and_create(path_of_expeirmnet)
	# print('************ write to csv file*****************')
	# with open(path_of_expeirmnet+'/'+'Synchornized_data.csv', 'w', encoding='UTF8', newline='') as f:
	# 	writer = csv.writer(f)
	# 	writer.writerow(header)
	# 	writer.writerows(synchornized_data_all)






# for episodes in range(9):

# 	print('*************Synchronization****************')
# 	for Catergory in ["Cat1","Cat2"]:
# 		for speed in ['5mph','10mph',"20mph"]:
# 			for direction in ['lr']:
# 				for lan in ['opposite','same']:
# 					synchornized_data_all = []
# 					path_of_expeirmnet = path_directory+'Extract/'+Catergory+'_'+speed+'_'+direction+'_'+lan+'/episode_'+str(episodes)
# 					GPS_data, Image_data, pcd_data, RF_data = open_csv_files(path_of_expeirmnet)

# 					# extracting same exoeriment data
# 					experiment_info = [Catergory,speed,direction,lan,episodes]
# 					GPS_selected = [GPS_data[i] for i in range(len(GPS_data)) if GPS_data[i][0:5]==experiment_info]
# 					pcd_selected = [pcd_data[i] for i in range(len(pcd_data)) if pcd_data[i][0:5]==experiment_info]
# 					img_selected = [Image_data[i] for i in range(len(Image_data)) if Image_data[i][0:5]==experiment_info]
# 					RF_selected = [RF_data[i] for i in range(len(RF_data)) if RF_data[i][0:5]==experiment_info]

# 					for i in range(len(RF_selected)-1):   #range(len(RF_selected)-1) or range(6,7)
# 						synchornized_data = []

# 						target_time_slot = [float(RF_selected[i][5]),float(RF_selected[i+1][5])]
# 						print('target time slot/diffrence',target_time_slot,float(target_time_slot[1])-float(target_time_slot[0]))
# 						##find image and lidar modalities in this slot
# 						pcd_in_range = [pcd_selected[l] for l in range(len(pcd_selected)) if target_time_slot[0]<float(pcd_selected[l][5])<target_time_slot[1]]
# 						image_in_range = [img_selected[l] for l in range(len(img_selected)) if target_time_slot[0]<float(img_selected[l][5])<target_time_slot[1]]
# 						print('# of pcd and image files in range',len(pcd_in_range),len(image_in_range))

							
# 						if len(image_in_range) !=0:    ##added
# 							print('************ pair synchornized image and lidar data*****************')
# 							for l in range(len(pcd_in_range)):
# 								diffrence=[]
# 								for j in range(len(image_in_range)):
# 									diffrence.append(abs(float(pcd_in_range[l][5])-float(image_in_range[j][5])))
# 								closest_image = diffrence.index(min(diffrence))    # find minuum diatnce 
# 								# attach:experiment info+time stamp of lidar+pcd file path+image file path
# 								synchornized_data.append(experiment_info+[pcd_in_range[l][5],pcd_in_range[l][-1],image_in_range[closest_image][-1]])
# 							print('************ Interpolating gps data*****************')
# 							mid_points_time_stamp_to_calculate = [float(s[5]) for s in synchornized_data]
# 							inter_log, inter_lat = interpolate_gps(GPS_selected,target_time_slot,mid_points_time_stamp_to_calculate)
# 							print(len(inter_log),len(inter_lat),len(synchornized_data))
# 							print('************ attaching gps interpolated data*****************')
# 							bias_index = len(synchornized_data_all)
# 							for l in range(len(synchornized_data)):
# 								synchornized_data[l].append(inter_lat[l])
# 								synchornized_data[l].append(inter_log[l])

# 							print('************ attaching RF data*****************')
# 							for l in range(len(synchornized_data)):
# 								for c in range(len(RF_selected[i][6:])):
# 									synchornized_data[l].append(RF_selected[i][6+c])

# 							print(synchornized_data)    ### HERE  check appedn
# 							### unpack the synchronized_data
# 							[synchornized_data_all.append(s) for s in synchornized_data]


# 							header = ['Category','speed','direction','lane','episode','time_stamp','lida_file','image_file','lat','long','max_rssi','selected_sector','tx_mcs','rx_mcs','sqi','all_rssi', 'all_sector']


# 							###the appended _data file is generated 
# 							if convert_to_x_y:
# 								print('*********Converting to x y **********')
# 								converted = []
# 								## find minimum
# 								# min_lat = min([float(d[8]) for d in synchornized_data_all])
# 								# min_long = min([float(d[9]) for d in synchornized_data_all])
# 								min_lat = 42.33812866170525
# 								min_long = -71.08705289706138
# 								print('min lat/long', min_lat,min_long)
# 								for d in synchornized_data_all:
# 									dx = 1000*(float(d[9])-min_long)*40000*math.cos((float(d[8])+min_lat)*math.pi/360)/360
# 									dy = 1000*(float(d[8])-min_lat)*40000/360
# 									converted.append(d+[dx,dy]) 
# 								synchornized_data_all = converted
# 								header = header + ['x','y']
# 								print(header)

# 							print('# of generated data',len(synchornized_data_all))

# 							print('chekc',path_of_expeirmnet+'/'+'Synchornized_data.csv')
# 							# print('************ write to csv file*****************')
# 							# with open(path_of_expeirmnet+'/'+'Synchornized_data.csv', 'w', encoding='UTF8', newline='') as f:
# 							# 	writer = csv.writer(f)
# 							# 	writer.writerow(header)
# 							# 	writer.writerows(synchornized_data_all)