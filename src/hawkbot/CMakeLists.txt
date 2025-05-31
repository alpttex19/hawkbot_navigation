cmake_minimum_required(VERSION 2.8.3)
project(hawkbot)

find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  roscpp
  rospy
)

#roslint_python()



catkin_package(

)

#catkin_python_setup()

include_directories(
  ${catkin_INCLUDE_DIRS}
)



install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY nodes
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

