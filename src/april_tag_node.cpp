//
// adapted from ros example and april tag examples - palash
//
#include <ros/ros.h>
// #include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

// #include <Scalar.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "AprilTags/TagDetector.h"
#include "AprilTags/Tag36h11.h"

#include "april_tag/AprilTag.h" // rosmsg
#include "april_tag/AprilTagList.h" // rosmsg


#define DOWNSAMPLE_FACTOR     2     // 1 or 2 only

static const std::string OPENCV_WINDOW = "Image window";
 
const double PI = 3.14159265358979323846;
const double TWOPI = 2.0*PI;


/**
*   CLASS DEFINITION
*/
class AprilTagNode
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;

  ros::Publisher transform_pub;

  AprilTags::TagDetector* tag_detector;
  cv::Mat intrinsic_, distCoeff_;
  cv::Mat map1, map2;

  AprilTags::TagCodes tag_codes;
  int tag_id_;
  double camera_focal_length_x; // in pixels. late 2013 macbookpro retina = 700
  double camera_focal_length_y; // in pixels
  double camera_center_x;       // in pixels
  double camera_center_y;       // in pixels
  double tag_size; // tag side length of frame in meters 
  bool  show_debug_image;

  int rightCam;      // For stereo camera: false is left, true is right. false for monocular

public:
  AprilTagNode() : 
    it_(nh_), 
    tag_codes(AprilTags::tagCodes36h11), 
    tag_detector(NULL),
    camera_focal_length_y(700),
    camera_focal_length_x(700),
    camera_center_x(360),
    camera_center_y(360),
    tag_size(0.29), // 1 1/8in marker = 0.29m
    show_debug_image(false)
  {
    image_sub_ = it_.subscribe("/camera/image_raw", 1, &AprilTagNode::imageCb, this);

    image_pub_ = it_.advertise("/april_tag_debug/output_video", 1);
    
    // Use a private node handle so that multiple instances of the node can
    // be run simultaneously while using different parameters.
    ros::NodeHandle private_node_handle("~");

    private_node_handle.param<int>("camera", rightCam, 0);

    intrinsic_ = cv::Mat::eye(3,3,cv::DataType<double>::type);
    distCoeff_ = cv::Mat::zeros(5,1,cv::DataType<double>::type);

    // Read calibration parameters
    bool loadSuccess = true;

    if ( !rightCam )    
    {
      XmlRpc::XmlRpcValue intrinsics_cam0, distCoeff_cam0;
      loadSuccess &= private_node_handle.getParam("cam0/intrinsics", intrinsics_cam0);  // camera_intrinsics
      intrinsic_.at<double>(0,0) = (double)intrinsics_cam0[0] / DOWNSAMPLE_FACTOR;
      intrinsic_.at<double>(1,1) = (double)intrinsics_cam0[1] / DOWNSAMPLE_FACTOR;
      intrinsic_.at<double>(0,2) = (double)intrinsics_cam0[2] / DOWNSAMPLE_FACTOR;
      intrinsic_.at<double>(1,2) = (double)intrinsics_cam0[3] / DOWNSAMPLE_FACTOR;

      loadSuccess &= private_node_handle.getParam("cam0/distortion_coeffs", distCoeff_cam0);

      distCoeff_.at<double>(0,0) = (double)distCoeff_cam0[0];
      distCoeff_.at<double>(1,0) = (double)distCoeff_cam0[1];
      distCoeff_.at<double>(2,0) = (double)distCoeff_cam0[2];
      distCoeff_.at<double>(3,0) = (double)distCoeff_cam0[3];
      distCoeff_.at<double>(4,0) = (double)distCoeff_cam0[4];

      transform_pub = nh_.advertise<geometry_msgs::TransformStamped>("/tf_cam0", 10);
    } else {
      XmlRpc::XmlRpcValue intrinsics_cam1, distCoeff_cam1;

      loadSuccess &= private_node_handle.getParam("cam1/intrinsics", intrinsics_cam1);  // camera_intrinsics
      intrinsic_.at<double>(0,0) = (double)intrinsics_cam1[0] / DOWNSAMPLE_FACTOR;
      intrinsic_.at<double>(1,1) = (double)intrinsics_cam1[1] / DOWNSAMPLE_FACTOR;
      intrinsic_.at<double>(0,2) = (double)intrinsics_cam1[2] / DOWNSAMPLE_FACTOR;
      intrinsic_.at<double>(1,2) = (double)intrinsics_cam1[3] / DOWNSAMPLE_FACTOR;
      
      loadSuccess &= private_node_handle.getParam("cam1/distortion_coeffs", distCoeff_cam1);

      distCoeff_.at<double>(0,0) = (double)distCoeff_cam1[0];
      distCoeff_.at<double>(1,0) = (double)distCoeff_cam1[1];
      distCoeff_.at<double>(2,0) = (double)distCoeff_cam1[2];
      distCoeff_.at<double>(3,0) = (double)distCoeff_cam1[3];
      distCoeff_.at<double>(4,0) = (double)distCoeff_cam1[4];

      transform_pub = nh_.advertise<geometry_msgs::TransformStamped>("/tf_cam1", 10);
    }

    private_node_handle.param<int>("tag_id", tag_id_, 8);
    private_node_handle.param<double>("focal_length_px", camera_focal_length_x, intrinsic_.at<double>(0,0) );
    private_node_handle.param<double>("focal_length_py", camera_focal_length_y, intrinsic_.at<double>(1,1) );
    private_node_handle.param<double>("center_px", camera_center_x, intrinsic_.at<double>(0,2) );
    private_node_handle.param<double>("center_py", camera_center_y, intrinsic_.at<double>(1,2) );

    private_node_handle.param<double>("tag_size_cm", tag_size, 2.9);
    private_node_handle.param<bool>("show_debug_image", show_debug_image, false);

    tag_size = tag_size / 100.0; // library takes input in meters

    cv::initUndistortRectifyMap(intrinsic_, distCoeff_, cv::Mat::eye(3,3,cv::DataType<double>::type),
                                  intrinsic_, cv::Size(752 / DOWNSAMPLE_FACTOR, 480 / DOWNSAMPLE_FACTOR), CV_16SC2, map1, map2);

    cout << "got focal length " << camera_focal_length_x << endl;
    cout << "got tag size " << tag_size << endl;
    tag_detector = new AprilTags::TagDetector(tag_codes);
    if (show_debug_image) {
      cv::namedWindow(OPENCV_WINDOW);
      std::cout << "Intrinsics : " << intrinsic_.at<double>(0,0) << " " 
                              << intrinsic_.at<double>(1,1) << " "
                              << intrinsic_.at<double>(0,2) << " "
                              << intrinsic_.at<double>(1,2) << std::endl;
    }

  }

  ~AprilTagNode()  {
    if (show_debug_image) {
     cv::destroyWindow(OPENCV_WINDOW);
    }
  }

  void send_transform_msg(const tf::Transform& Tf, const cv_bridge::CvImagePtr& cv_ptr) const {
    static tf::TransformBroadcaster br;

    geometry_msgs::Quaternion pose_quat;
    tf::StampedTransform pose;

    if (!rightCam)
      pose = tf::StampedTransform(Tf, cv_ptr->header.stamp, "origin", "cam0");
    else
      pose = tf::StampedTransform(Tf, cv_ptr->header.stamp, "origin", "cam1");

    geometry_msgs::TransformStamped pose_trans;

    tf::transformStampedTFToMsg(pose, pose_trans);
    transform_pub.publish(pose_trans);
    br.sendTransform(pose_trans);
  }

  void convert_to_msg(AprilTags::TagDetection& detection, const cv_bridge::CvImagePtr& cv_ptr) {
    Eigen::Vector3d translation;
    Eigen::Matrix3d rotation;
    cv::Mat rvec, tvec;

    detection.getRelativeTranslationRotation(tag_size, 
                                             camera_focal_length_x, 
                                             camera_focal_length_y, 
                                             camera_center_x, 
                                             camera_center_y,
                                             translation, 
                                             rotation,
                                             rvec, tvec);
    april_tag::AprilTag tag_msg;

    tf::Matrix3x3 rot; tf::Vector3 trans;
    tf::matrixEigenToTF(rotation, rot);
    tf::vectorEigenToTF(translation, trans);

    tf::Transform t_tag_cam(rot, trans);

    send_transform_msg(t_tag_cam.inverse(), cv_ptr);
  }

  void processCvImage(cv_bridge::CvImagePtr cv_ptr)  {
    cv::Mat image_gray, image_ud;
    cv::cvtColor(cv_ptr->image, image_gray, CV_BGR2GRAY);

    vector<AprilTags::TagDetection> detections = tag_detector->extractTags(image_gray);

    for (int i=0; i<detections.size(); i++) {
      if (detections[i].id != tag_id_)  continue;
      std::cout << "Camera " << rightCam << " detects tag" << std:: endl;
      convert_to_msg(detections[i], cv_ptr);
    }
  }


  void imageCb(const sensor_msgs::ImageConstPtr& msg)  {
    cv_bridge::CvImagePtr cv_ptr;
    cv::Mat image_ud, image_down;

    // ros::Time start = ros::Time::now();

    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    if (DOWNSAMPLE_FACTOR != 1)
    {
      cv::pyrDown(cv_ptr->image, image_down, cv::Size(cv_ptr->image.cols/DOWNSAMPLE_FACTOR, cv_ptr->image.rows/DOWNSAMPLE_FACTOR)); 
      cv::remap(image_down, image_ud, map1, map2, CV_INTER_LINEAR);   
    } else {
      cv::remap(cv_ptr->image, image_ud, map1, map2, CV_INTER_LINEAR);
    }

    // cv::undistort(cv_ptr->image, image_ud, intrinsic_, distCoeff_);
    cv_ptr->image = image_ud;

    processCvImage(cv_ptr);

    if (show_debug_image) {
      // Update GUI Window
      cv::imshow(OPENCV_WINDOW, cv_ptr->image);
      cv::waitKey(3);
    }
    // ros::Time end = ros::Time::now();

    // std::cout << (end-start).toSec() << std::endl;

  }
};

/**
*   MAIN FUNCTION
*/
int main(int argc, char** argv)  {
  ros::init(argc, argv, "april_tag_node");
  AprilTagNode atn;

  std::cout << "Starting Detection \n";

  ros::spin();
  return 0;
}
