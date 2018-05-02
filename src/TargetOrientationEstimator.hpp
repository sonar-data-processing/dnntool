#ifndef dnntool_TargetOrientationEstimator_hpp
#define dnntool_TargetOrientationEstimator_hpp

#include <opencv2/opencv.hpp>

namespace dnntool {

class TargetOrientationEstimator
{
public:
    TargetOrientationEstimator(size_t window_size = 5);
    virtual ~TargetOrientationEstimator();

    float Compute(const cv::Mat& src);

private:
    float GetOrientationPCA(const cv::Mat& src);

    void PerformDenoise(const cv::Mat& src, cv::Mat& denoised);

    void PerformBorderFilter(const cv::Mat& src, cv::Mat& border);

    size_t window_size_;
    float last_mean_;
    std::deque<float> samples_;
};

} // namespace dnntool

#endif
