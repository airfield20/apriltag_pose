//
// adapted from ros example and april tag examples - palash
//
#include <ros/ros.h>
// #include <tf/tf.h>
#include <tf/transform_broadcaster.h>
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


static const std::string OPENCV_WINDOW = "Image window";

const double PI = 3.14159265358979323846;
const double TWOPI = 2.0*PI;

/**
 * Normalize angle to be within the interval [-pi,pi].
 */
inline double standardRad(double t) {
  if (t >= 0.) {
    t = fmod(t+PI, TWOPI) - PI;
  } else {
    t = fmod(t-PI, -TWOPI) + PI;
  }
  return t;
}

inline double radToDeg(double rads) {
  return ((rads * 180)/PI );
}

/**
 * Convert rotation matrix to Euler angles
 */
void wRo_to_euler(const Eigen::Matrix3d& wRo, double& yaw, double& pitch, double& roll) {
    yaw = standardRad(atan2(wRo(1,0), wRo(0,0)));
    double c = cos(yaw);
    double s = sin(yaw);
    pitch = standardRad(atan2(-wRo(2,0), wRo(0,0)*c + wRo(1,0)*s));
    roll  = standardRad(atan2(wRo(0,2)*s - wRo(1,2)*c, -wRo(0,1)*s + wRo(1,1)*c));
}


class AprilTagNode
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  // ros::Publisher tag_list_pub;

  ros::Publisher transform_pub;

  AprilTags::TagDetector* tag_detector;
  cv::Mat intrinsic_, distCoeff_;
  cv::Mat map1, map2;
  // allow configurations for these:  
  AprilTags::TagCodes tag_codes;
  double camera_focal_length_x; // in pixels. late 2013 macbookpro retina = 700
  double camera_focal_length_y; // in pixels
  double tag_size; // tag side length of frame in meters 
  bool  show_debug_image;

  bool rightCam;      // For stereo camera: false is left, true is right. false for monocular

public:
  AprilTagNode() : 
    it_(nh_), 
    tag_codes(AprilTags::tagCodes36h11), 
    tag_detector(NULL),
    camera_focal_length_y(700),
    camera_focal_length_x(700),
    tag_size(0.29), // 1 1/8in marker = 0.29m
    show_debug_image(false)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/camera/image_raw", 10, &AprilTagNode::imageCb, this);
    image_pub_ = it_.advertise("/april_tag_debug/output_video", 1);
    // tag_list_pub = nh_.advertise<april_tag::AprilTagList>("/april_tags", 100);
    transform_pub = nh_.advertise<geometry_msgs::TransformStamped>("/tf_cam0", 10);
    
    // Use a private node handle so that multiple instances of the node can
    // be run simultaneously while using different parameters.
    ros::NodeHandle private_node_handle("~");

    private_node_handle.param<bool>("camera", rightCam, false);

    intrinsic_ = cv::Mat::eye(3,3,cv::DataType<double>::type);
    distCoeff_ = cv::Mat::zeros(5,1,cv::DataType<double>::type);

    // Read calibration parameters
    bool loadSuccess = true;

    if ( !rightCam )    
    {
      XmlRpc::XmlRpcValue intrinsics_cam0;
      loadSuccess &= private_node_handle.getParam("cam0/intrinsics", intrinsics_cam0);  // camera_intrinsics
      intrinsic_.at<double>(0,0) = (double)intrinsics_cam0[0];
      intrinsic_.at<double>(1,1) = (double)intrinsics_cam0[1];
      intrinsic_.at<double>(0,2) = (double)intrinsics_cam0[2];
      intrinsic_.at<double>(1,2) = (double)intrinsics_cam0[3];

      // Radial Distortion coeffs (from Marco Karrer) [k1, k2, p1, p2, k3]
      double cam0_distCoeffs[5] = { -0.293099583176669, 0.101589055552550, -0.00004886151975809833, 0.00009521834372139675, -0.018417213000905 }; 
      distCoeff_.at<double>(0,0) = (double)cam0_distCoeffs[0];
      distCoeff_.at<double>(1,0) = (double)cam0_distCoeffs[1];
      distCoeff_.at<double>(2,0) = (double)cam0_distCoeffs[2];
      distCoeff_.at<double>(3,0) = (double)cam0_distCoeffs[3];
      distCoeff_.at<double>(4,0) = (double)cam0_distCoeffs[4];

    } else {
      XmlRpc::XmlRpcValue intrinsics_cam1;
      loadSuccess &= private_node_handle.getParam("cam1/intrinsics", intrinsics_cam1);  // camera_intrinsics
      intrinsic_.at<double>(0,0) = (double)intrinsics_cam1[0];
      intrinsic_.at<double>(1,1) = (double)intrinsics_cam1[1];
      intrinsic_.at<double>(0,2) = (double)intrinsics_cam1[2];
      intrinsic_.at<double>(1,2) = (double)intrinsics_cam1[3];

      // Radial Distortion coeffs (from Marco Karrer) [k1, k2, p1, p2, k3]
      double cam1_distCoeffs[5] = { -0.346781208941426, 0.171638088490686, 2.122745013661727e-04, -2.605449368212942e-04, -0.041879348897746 }; 
      distCoeff_.at<double>(0,0) = (double)cam1_distCoeffs[0];
      distCoeff_.at<double>(1,0) = (double)cam1_distCoeffs[1];
      distCoeff_.at<double>(2,0) = (double)cam1_distCoeffs[2];
      distCoeff_.at<double>(3,0) = (double)cam1_distCoeffs[3];
      distCoeff_.at<double>(4,0) = (double)cam1_distCoeffs[4];

    }

    private_node_handle.param<double>("focal_length_px", camera_focal_length_x, intrinsic_.at<double>(0,0) );
    private_node_handle.param<double>("focal_length_py", camera_focal_length_y, intrinsic_.at<double>(1,1) );

    private_node_handle.param<double>("tag_size_cm", tag_size, 2.9);
    private_node_handle.param<bool>("show_debug_image", show_debug_image, false);

    // camera_focal_length_y = camera_focal_length_x; // meh
    tag_size = tag_size / 100.0; // library takes input in meters

    cv::initUndistortRectifyMap(intrinsic_, distCoeff_, cv::Mat::eye(3,3,cv::DataType<double>::type),
                                  intrinsic_, cv::Size(752 / 2, 480 / 2), CV_16SC2, map1, map2);

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

  void send_transform_msg(const tf::Vector3& tvec, const tf::Quaternion& Q, const cv_bridge::CvImagePtr& cv_ptr) const {
    static tf::TransformBroadcaster br;
#if 0
    tf::Transform transform;
    transform.setOrigin( tvec );
    transform.setRotation(Q);
    if (!rightCam)
    {
      // br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "cam0"));
      transform_pub.publish(tf::StampedTransform(transform, ros::Time::now(), "world", "cam0"));
    } else {
      // br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "cam1"));
      transform_pub.publish(tf::StampedTransform(transform, ros::Time::now(), "world", "cam1"));
    }
#else
    geometry_msgs::Quaternion pose_quat;

    tf::quaternionTFToMsg(Q, pose_quat);

    geometry_msgs::TransformStamped pose_trans;

    pose_trans.header.stamp = cv_ptr->header.stamp;
    pose_trans.header.frame_id = "world_AT";
    if (!rightCam)
      pose_trans.child_frame_id = "cam0";
    else
      pose_trans.child_frame_id = "cam1";
    pose_trans.transform.translation.x = tvec[0];
    pose_trans.transform.translation.y = tvec[1];
    pose_trans.transform.translation.z = tvec[2];
    pose_trans.transform.rotation = pose_quat;
    transform_pub.publish(pose_trans);
    br.sendTransform(pose_trans);
#endif
  }

  void convert_to_msg(AprilTags::TagDetection& detection, const cv_bridge::CvImagePtr& cv_ptr) {
    // recovering the relative pose of camera w.r.t. the tag:

    // NOTE: for this to be accurate, it is necessary to use the
    // actual camera parameters here as well as the actual tag size
    // (m_fx, m_fy, m_px, m_py, m_tagSize)

    Eigen::Vector3d translation;
    Eigen::Matrix3d rotation;
    cv::Mat rvec, tvec;

    int width = cv_ptr->image.cols;
    int height = cv_ptr->image.rows;

    detection.getRelativeTranslationRotation(tag_size, 
                                             camera_focal_length_x, 
                                             camera_focal_length_y, 
                                             width / 2, 
                                             height / 2,
                                             translation, 
                                             rotation,
                                             rvec, tvec);
    april_tag::AprilTag tag_msg;

    tfScalar angle = sqrt( rvec.at<double>(0)*rvec.at<double>(0) + rvec.at<double>(1)*rvec.at<double>(1) + rvec.at<double>(2)*rvec.at<double>(2) );
    tf::Vector3 axis( rvec.at<double>(0) / angle, 
                      rvec.at<double>(1) / angle, 
                      rvec.at<double>(2) / angle);

    tf::Quaternion qCW(axis, angle);    // Rotation of world w.r.t camera
    tf::Vector3 CrCW = tf::Vector3(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));   // Vector of world w.r.t. camera in camera frame

    double yaw, pitch, roll;

    tf::Transform tCW, tWC, tW;
    tCW.setOrigin(CrCW);
    tCW.setRotation(qCW);
    // tf::Matrix3x3(qCW).getRPY(roll,pitch,yaw);
    // tCW.setRotation(tf::Quaternion(yaw, 0, 0));
    // tCW.setRotation(tf::createQuaternionFromRPY(roll-PI/2.2,pitch,yaw));
    
    tWC = tCW.inverse();
    tf::Vector3 WrWC = tWC.getOrigin();             // Vector of Camera w.r.t world (april tag) in world frame
    tf::Quaternion qWC = tWC.getRotation();         // Rotation of Camera w.r.t world (april tag)
    send_transform_msg( WrWC, qWC, cv_ptr);
    // send_transform_msg( WrWC, qWC * tf::createQuaternionFromRPY(PI/2.2,0,0), cv_ptr);
  }

  void processCvImage(cv_bridge::CvImagePtr cv_ptr)  {
    cv::Mat image_gray, image_ud;
    cv::cvtColor(cv_ptr->image, image_gray, CV_BGR2GRAY);

    vector<AprilTags::TagDetection> detections = tag_detector->extractTags(image_gray);

    for (int i=0; i<detections.size(); i++) {
      if (detections[i].id != 8)  continue;

      convert_to_msg(detections[i], cv_ptr);
    }
  }


  void imageCb(const sensor_msgs::ImageConstPtr& msg)  {
    cv_bridge::CvImagePtr cv_ptr;
    cv::Mat image_ud, image_down;

    ros::Time start = ros::Time::now();

    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    cv::pyrDown(cv_ptr->image, image_down, cv::Size(cv_ptr->image.cols/2, cv_ptr->image.rows/2)); 
    cv::remap(image_down, image_ud, map1, map2, CV_INTER_LINEAR);

    // cv::undistort(cv_ptr->image, image_ud, intrinsic_, distCoeff_);
    cv_ptr->image = image_ud;

    processCvImage(cv_ptr);

    if (show_debug_image) {
      // Update GUI Window
      cv::imshow(OPENCV_WINDOW, cv_ptr->image);
      cv::waitKey(3);
    }
    ros::Time end = ros::Time::now();

    std::cout << (end-start).toSec() << std::endl;

  }
};

int main(int argc, char** argv)  {
  ros::init(argc, argv, "april_tag_node");
  AprilTagNode atn;

  std::cout << "Starting Detection \n";

  ros::spin();
  return 0;
}
