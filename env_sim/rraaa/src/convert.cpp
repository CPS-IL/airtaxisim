#include "ros/ros.h"
#include "std_msgs/Float64MultiArray.h"
#include "std_msgs/MultiArrayDimension.h"

#include <octomap/octomap.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>

#include <vector>



/**
 * @class OctomapMsgConverter
 * @brief Receive the octomap msg and convert it into multidimensional array for 3D path planning.
 *
 * Detailed description of the class and its purpose.
 * 
 * The following are the resources that I used.
 * 
 * Since the octomap msg is working well with RVIZ, the following RVIZ plugin C++ source should be the useful.
 * LINE 481 <--- look at here!   https://github.com/OctoMap/octomap_rviz_plugins/blob/kinetic-devel/src/occupancy_grid_display.cpp
 * 
 * The above uses the following "converstions.h"
 * https://github.com/OctoMap/octomap_msgs/blob/melodic-devel/include/octomap_msgs/conversions.h
 * 
 * =====================
 * Some basic c++ facts:
 * =====================
 * 
 * (pointer_name)->(variable_name)
 * 
 * Operation: The -> operator in C or C++ gives the value held by variable_name to structure or union variable pointer_name.
 * Difference between Dot(.) and Arrow(->) operator:  
 *
 * The Dot(.) operator is used to normally access members of a structure or union.
 * The Arrow(->) operator exists to access the members of the structure or the unions using pointers.
 * 
 * 
 * ============================================
 * Look at the below for multi-dim pub with ROS
 * ============================================
 * https://answers.ros.org/question/234028/how-to-publish-a-2-dimensional-array-of-known-values/
 * 
 * 
 * 
 * 
 */


class OctomapMsgConverter {

  private:
    double* array; // Pointer to the array
    int m;         // M Columns
    int n;         // N Rows
    std::size_t size=0;  // Size of the vector to return
    bool recvd = false;

  public:

    void Callback(const octomap_msgs::OctomapConstPtr& msg){

      recvd = true;

      // Map information
      octomap::AbstractOcTree* tree = octomap_msgs::msgToMap(*msg);

      double minX, minY, minZ, maxX, maxY, maxZ; 
      tree->getMetricMin(minX, minY, minZ); //map covering space limits
      tree->getMetricMax(maxX, maxY, maxZ); //min, max, X, Y, Z
      double resolution = tree->getResolution(); //get resolution of octree

      // Tree for iterating nodes
      octomap::OcTree* te = dynamic_cast<octomap::OcTree*>(tree);

      // Dimensions of the 2D array
      m = 5;                ; // m coloums or width
      n = te->calcNumNodes(); // n rows or height

      // Declare a memory block of size m*n
      array = new double[m * n];

      int i = 0;
      for (octomap::OcTree::leaf_iterator it = te->begin_leafs(); it != te->end_leafs(); ++it){
        array[i * m + 0] = it.getX();
        array[i * m + 1] = it.getY();
        array[i * m + 2] = it.getZ();
        array[i * m + 3] = it->getValue();
        array[i * m + 4] = te->isNodeOccupied(*it);
        
        i++;
      }
      n = i; // Number of row after filling-in values.
      size = m*n; 

      delete tree;

    }

    std::vector<double> getVector(){
      
      if (recvd){
        std::vector<double> vec(size,0); // vector to return
        for (int i=0; i<size; i++){
          vec[i] = array[i];
        };
        delete[] array;

        recvd = false;

        return vec;
      }
      else{
        std::vector<double> vec(1,0); // vector to return
        return vec;

      };
      
    }



    int getHeight(){
      return n;
    }

    int getWidth(){
      return m;
    }

};



int main(int argc, char **argv)
{

  ros::init(argc, argv, "node_converter");

  ros::NodeHandle n;

  OctomapMsgConverter converter;

  ros::Subscriber sub = n.subscribe("/octomap_full", 1000, &OctomapMsgConverter::Callback, &converter);
  ros::Publisher pub = n.advertise<std_msgs::Float64MultiArray>("/octomap_conv_array", 1);
  ros::Rate loop_rate(10);

  std_msgs::Float64MultiArray dat;


  // fill out message:
  dat.layout.dim.push_back(std_msgs::MultiArrayDimension());
  dat.layout.dim.push_back(std_msgs::MultiArrayDimension());
  dat.layout.dim[0].label = "row";
  dat.layout.dim[1].label = "x,y,z,value";


  int count = 0;
  while (ros::ok())
  {
    std::vector<double> vec = converter.getVector();
    if (vec.size()>1)
    {
      int H = converter.getHeight();
      int W = converter.getWidth();
      dat.layout.dim[0].size = H;
      dat.layout.dim[1].size = W;
      dat.layout.dim[0].stride = H*W;
      dat.layout.dim[1].stride = W;
      dat.layout.data_offset = 0;
      dat.data = vec;
    }
  

    pub.publish(dat);

    vec.clear();


    ros::spinOnce();

    loop_rate.sleep();
    ++count;
  };

  return 0;
}