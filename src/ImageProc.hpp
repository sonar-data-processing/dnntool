#ifndef image_proc_hpp
#define image_proc_hpp

#include <opencv2/opencv.hpp>

namespace dnntool
{

namespace image_proc
{

std::vector<cv::Point> find_countour(const cv::Mat& src);

void rotate(const cv::Mat& src, cv::Mat& dst, float angle, cv::Point2f center, cv::Size size = cv::Size(-1, -1));

void draw_text(cv::Mat& dst, const std::string& text, const cv::Point& pos, const cv::Scalar& color);

float get_color(int c, int x, int max);

cv::Scalar get_scalar(int x, int max);

double otsu_thresh(const cv::Mat& src);

void perform_pca_analisys(
    const std::vector<cv::Point> &pts,
    std::vector<cv::Point2d>& eigen_vecs,
    std::vector<double>& eigen_val,
    cv::Point2d& center);

void perform_pca_analisys(
    const std::vector<cv::Point> &pts,
    std::vector<cv::Point2d>& eigen_vecs,
    std::vector<double>& eigen_val);

void draw_axis(
    cv::Mat& img,
    cv::Point p,
    cv::Point q,
    cv::Scalar color,
    const float scale = 0.2);

void create_cartesian_mask(
    const cv::Size& size,
    double t0,
    double t1,
    cv::Mat& out
);

} // namespace image_proc

} // namespace dnntool

#endif
