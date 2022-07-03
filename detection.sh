source /opt/ros/melodic/setup.bash
source devel/setup.bash
catkin_make
rosrun mmdet3d inference.py
