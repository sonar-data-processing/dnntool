#include "ImageFiltering.hpp"

namespace dnntool
{

namespace image_filtering
{

void color_enhancement(const cv::Mat& src, cv::Mat& dst)
{
    dst = src.clone();

    // calculate the proportional mean of each image row
    std::vector<double> row_mean(src.rows, 0);
    for (size_t i = 0; i < src.rows; i++) {
        double value = cv::sum(src.row(i))[0] / src.cols;
        row_mean[i] = std::isnan(value) ? 0 : value;
    }

    // get the maximum mean between lines
    double max_mean = *std::max_element(row_mean.begin(), row_mean.end());

    // apply the insonification correction
    for (size_t i = 0; i < src.rows; i++) {
        if(row_mean[i]) {
            double factor = max_mean / row_mean[i];
            dst.row(i) *= factor;
        }
    }
    dst.setTo(255, dst > 255);
}

void border_filter(const cv::Mat& src, cv::Mat& dst) {
    cv::Mat G;
    cv::Mat Gx, Gy;
    cv::Mat Gx2, Gy2;

    cv::Sobel( src, Gx, CV_16S, 1, 0, CV_SCHARR, 0.5, 0, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( Gx, Gx2 );

    cv::Sobel( src, Gy, CV_16S, 0, 1, CV_SCHARR, 0.5, 0, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( Gy, Gy2 );

    cv::addWeighted(Gx2, 0.5, Gy2, 0.5, 0, G);

    G.copyTo(dst);
}

void morphology(const cv::Mat& src, cv::Mat& dst, int type, cv::Size kernel_size, int iterations) {
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, kernel_size);
    cv::morphologyEx(src, dst, type, kernel, cv::Point(-1, -1), iterations);
}

void erode(const cv::Mat& src, cv::Mat& dst, cv::Size kernel_size, int iterations) {
    morphology(src, dst, cv::MORPH_ERODE, kernel_size, iterations);
}


void dilate(const cv::Mat& src, cv::Mat& dst, cv::Size kernel_size, int iterations) {
    morphology(src, dst, cv::MORPH_DILATE, kernel_size, iterations);
}

} // namespace image_filtering

} // namespace dnntool
