#######geneare frames from videos
###user input: line 24. line 23 is the same for all

import cv2
import os
from tqdm import tqdm

def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False
def show_all_files_in_directory(input_path,extension):
    'This function reads the path of all files in directory input_path'
    files_list=[]
    for path, subdirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(extension):
               files_list.append(os.path.join(path, file))
    return files_list

open_path = '/home/batool/Desktop/DCIM/100GOPRO/' # Change directory  /media/jgu1/My Passport/Talon/DCIM/100GOPRO/
camera_mapping = "/media/batool/MULTIMODAL_DATA/Multimodal_Data/Cat4/car_20_lr_blockage_20_rl/camera_mapping"     ##change this input



all_videos_path = show_all_files_in_directory(open_path,'.MP4')
with open(camera_mapping) as f:     # Location of .txt file with all timestamps
    lines = f.readlines()
    for i in tqdm(lines):
        if 'Cat' in i and '{' not in i:
            # print('i',i)
            cat = i.split(' ')[0]     #experiment info
            video_filename = i.split(' ')[1]   #vvideo file name
            ts = i.split(' ')[2]        #time stamp;one is added because there is a 1 sec delay here: 1 removed was incorrect
            # print(cat,video_filename,ts)
            ###find the corrosponfing video
            target_video = [i for i in all_videos_path if i.split('/')[-1]==video_filename][0]
            # print(target_video)
            #opeining the video
            cap = cv2.VideoCapture(target_video)
            while not cap.isOpened():
                cap = cv2.VideoCapture(target_video)
                cv2.waitKey(1000)
                print("Wait for the header")

            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            while True:
                flag, frame = cap.read()
                if flag:
                    # The frame is ready and already captured
                    save_directory = camera_mapping[:-14]+'bag_files/'+ cat + '/camera_images/'
                    # print(save_directory+ts+'.jpg')
                    check_and_create(save_directory)
                    cv2.imwrite(save_directory+ts+'.jpg', frame)
                    ts = float(ts)+0.0333333 # Changing types
                    ts = str(ts)
                    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    print(target_video+" Frame "+str(int(pos_frame)))
                else:
                    # The next frame is not ready, so we try to read it again
                    cap.get(cv2.CAP_PROP_POS_FRAMES)
                    print("Frame is not ready")
                    # It is better to wait for a while for the next frame to be ready
                    cv2.waitKey(1000)

                    if cv2.waitKey(10) == 27:
                        break
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                        # If the number of captured frames is equal to the total number of frames,
                        # we stop
                        break