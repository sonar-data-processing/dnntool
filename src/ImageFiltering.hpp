#ifndef dnntool_ImageFiltering
#define dnntool_ImageFiltering

#include <opencv2/opencv.hpp>

namespace dnntool
{

namespace image_filtering
{

void color_enhancement(const cv::Mat& src, cv::Mat& dst);

void border_filter(const cv::Mat& src, cv::Mat& dst);

void morphology(const cv::Mat& src, cv::Mat& dst, int type, cv::Size kernel_size, int iterations);

void erode(const cv::Mat& src, cv::Mat& dst, cv::Size kernel_size, int iterations);

void dilate(const cv::Mat& src, cv::Mat& dst, cv::Size kernel_size, int iterations);

} // namespace image_filtering

} // namespace dnntool

#endif
