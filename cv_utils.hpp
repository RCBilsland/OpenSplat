#ifndef CV_UTILS
#define CV_UTILS

#include <torch/torch.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

// Image I/O functions
cv::Mat imreadRGB(const std::string &filename);
void imwriteRGB(const std::string &filename, const cv::Mat &image);

// Tensor conversion functions
cv::Mat floatNxNtensorToMat(const torch::Tensor &t);
torch::Tensor floatNxNMatToTensor(const cv::Mat &m);
cv::Mat tensorToImage(const torch::Tensor &t);
torch::Tensor imageToTensor(const cv::Mat &image);

// Fisheye camera functions
cv::Point2f projectFisheye(const cv::Point3f &p3d, const cv::Mat &K, const cv::Mat &D);
cv::Point2f distortPointFisheye(const cv::Point2f& normalizedPt, float k1, float k2, float k3, float k4);
cv::Point2f undistortPointFisheye(const cv::Point2f& distortedPt, float k1, float k2, float k3, float k4);

#endif