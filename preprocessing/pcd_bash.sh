#!/bin/bash
# Passing arguments to a function
echo "Ros Test START"
source /opt/ros/melodic/setup.bash
source /home/batool/catkin_ws/devel/setup.bash
find_bag_files () {
  for i in $(find $1 -maxdepth 5 -type f -name "*.bag");
  do
      bag_file_name_ext=(${i//// })
      bag_file_name=(${bag_file_name_ext[-1]//./ })
      echo ${arrIN2[0]}
      for j in /ns1/velodyne_points
      do 
	  rosrun pcl_ros bag_to_pcd $i $j $1/${bag_file_name[0]}/lidar_pcd/$j
      done
  done
}
find_bag_files $1
