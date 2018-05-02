#include <limits>
#include <numeric>
#include "ImageProc.hpp"
#include "ImageFiltering.hpp"
#include "Utils.hpp"
#include "TargetOrientationEstimator.hpp"

namespace dnntool {

TargetOrientationEstimator::TargetOrientationEstimator(
    size_t window_size)
    : window_size_(window_size)
{
}

TargetOrientationEstimator::~TargetOrientationEstimator()
{
}

float TargetOrientationEstimator::GetOrientationPCA(const cv::Mat& src) {
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(src, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

    cv::Point2d vec;
    double max_val = DBL_MIN;

    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area >= 1e3 && area <= 1e7) {
            std::vector<cv::Point2d> eigen_vecs;
            std::vector<double> eigen_val;

            image_proc::perform_pca_analisys(contours[i], eigen_vecs, eigen_val);

            if (max_val < eigen_val[0]) {
                vec = eigen_vecs[0];
                max_val = eigen_val[0];
            }
        }
    }

    float theta = atan2(vec.y, vec.x);

    float theta_diff = 0;
    if (!samples_.empty()) {
        float last = samples_.back();
        theta_diff = fabs(theta-last);
        if (theta_diff > M_PI) theta_diff = fabs(theta_diff - 2 * M_PI);
    }

    if (max_val > 0 && theta_diff < M_PI/2) {
        samples_.push_back(theta);
        if (samples_.size() >= window_size_) {
            float sum = std::accumulate(samples_.begin(), samples_.end(), 0.0);
            float mean = sum / samples_.size();
            samples_.pop_front();
            return mean;
        }
    }
    else {
        samples_.clear();
    }

    return std::numeric_limits<float>::quiet_NaN();

}

void TargetOrientationEstimator::PerformDenoise(const cv::Mat& src, cv::Mat& denoised)
{
    cv::Mat enhanced;
    image_filtering::color_enhancement(src, enhanced);
    cv::boxFilter(enhanced, denoised, CV_8U, cv::Size(5, 5));
}

void TargetOrientationEstimator::PerformBorderFilter(const cv::Mat& src, cv::Mat& border)
{
    image_filtering::border_filter(src, border);

    cv::Mat border_sm;
    cv::boxFilter(border, border_sm, CV_8U, cv::Size(5, 5));

    cv::Mat src_sm;
    cv::boxFilter(src, src_sm, CV_8U, cv::Size(50, 50));

    border = border_sm-src_sm;

    cv::normalize(border, border, 0, 255, cv::NORM_MINMAX);
}

float TargetOrientationEstimator::Compute(const cv::Mat& src)
{

    cv::Mat denoised;
    PerformDenoise(src, denoised);

    cv::Mat border;
    PerformBorderFilter(denoised, border);

    return GetOrientationPCA(border);
}

} // namespace dnntool
