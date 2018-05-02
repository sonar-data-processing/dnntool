#include "ImageProc.hpp"

namespace dnntool
{

namespace image_proc
{

std::vector<cv::Point> find_countour(const cv::Mat& src)
{
    cv::Mat mat;
    src.copyTo(mat);

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(mat, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    int last_size_area = 0;
    int biggest_index = -1;
    for( int i = 0; i < contours.size(); i++ ) {
        cv::RotatedRect box = cv::minAreaRect(cv::Mat(contours[i]));
        float size_area = box.size.area();
        if (size_area > last_size_area) {
            last_size_area = size_area;
            biggest_index = i;
        }
    }

    std::vector<cv::Point> contour;
    cv::convexHull(cv::Mat(contours[biggest_index]), contour, false );
    return contour;
}

void rotate(const cv::Mat& src, cv::Mat& dst, float angle, cv::Point2f center, cv::Size size)
{
    cv::Mat rot;
    cv::Rect bbox;

    rot = cv::getRotationMatrix2D(center, angle, 1.0);
    bbox = cv::RotatedRect(center, src.size(), angle).boundingRect();

    if (size == cv::Size(-1, -1)) {
        rot.at<double>(0,2) += bbox.width/2.0-center.x;
        rot.at<double>(1,2) += bbox.height/2.0-center.y;
        cv::warpAffine(src, dst, rot, bbox.size());
        return;
    }

    rot.at<double>(0,2) += size.width/2.0-center.x;
    rot.at<double>(1,2) += size.height/2.0-center.y;
    cv::warpAffine(src, dst, rot, size);
}

void draw_text(cv::Mat& dst, const std::string& text, const cv::Point& pos, const cv::Scalar& color)
{
    int baseline = 0;
    double font_scale = 1.0;
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    int thickness = 2;
    cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
    cv::Point textOrg = cv::Point(pos.x, pos.y-text_size.height);
    cv::putText(dst, text, textOrg, font_face, font_scale, color, thickness);
}

float get_color(int c, int x, int max)
{
    const float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    return r;

}

cv::Scalar get_scalar(int x, int max)
{
    float red = get_color(2,x,max) * 255;
    float green = get_color(1,x,max) * 255;
    float blue = get_color(0,x,max) * 255;
    return cv::Scalar(blue, green, red);
}

double otsu_thresh(const cv::Mat& _src)
{
    cv::Size size = _src.size();
    if( _src.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    const int N = 256;
    int i, j, s, h[N] = {0};

    s = size.width*size.height;
    for( i = 0; i < size.height; i++ ) {
        const uchar* src = _src.data + _src.step * i;
        for(j = 0; j < size.width; j++ ) {
            h[src[j]]++;
        }
    }

    double mu = 0, scale = 1./s;
    for( i = 0; i < N; i++ ) mu += i*(double)h[i];

    mu *= scale;
    double mu1 = 0, q1 = 0;
    double max_sigma = 0, max_val = 0;

    for( i = 0; i < N; i++ )
    {
        double p_i, q2, mu2, sigma;

        p_i = h[i]*scale;
        mu1 *= q1;
        q1 += p_i;
        q2 = 1. - q1;

        if( std::min(q1,q2) < FLT_EPSILON || std::max(q1,q2) > 1. - FLT_EPSILON ) continue;

        mu1 = (mu1 + i*p_i)/q1;
        mu2 = (mu - q1*mu1)/q2;
        sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
        if( sigma > max_sigma )
        {
            max_sigma = sigma;
            max_val = i;
        }
    }

    return max_val;
}

void perform_pca_analisys(
    const std::vector<cv::Point> &pts,
    std::vector<cv::Point2d>& eigen_vecs,
    std::vector<double>& eigen_val,
    cv::Point2d& center)
{
    cv::Mat data_pts = cv::Mat(pts.size(), 2, CV_64FC1);
    for (int i = 0; i < data_pts.rows; ++i) {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }

    cv::PCA pca(data_pts, cv::Mat(), CV_PCA_DATA_AS_ROW);
    center = cv::Point2d(pca.mean.at<double>(0, 0), pca.mean.at<double>(0, 1));

    eigen_vecs.resize(2, cv::Point2d(0, 0));
    eigen_val.resize(2, 0);

    for (int i = 0; i < 2; ++i) {
        eigen_vecs[i] = cv::Point2d(pca.eigenvectors.at<double>(i, 0), pca.eigenvectors.at<double>(i, 1));
        eigen_val[i] = pca.eigenvalues.at<double>(0, i);
    }
}

void perform_pca_analisys(
    const std::vector<cv::Point> &pts,
    std::vector<cv::Point2d>& eigen_vecs,
    std::vector<double>& eigen_val)
{
    cv::Point2d center;
    perform_pca_analisys(pts, eigen_vecs, eigen_val, center);
}


void draw_axis(cv::Mat& img, cv::Point p, cv::Point q, cv::Scalar color, const float scale)
{
    double angle;
    double hypotenuse;

    angle = atan2( (double) p.y - q.y, (double) p.x - q.x ); // angle in radians
    hypotenuse = sqrt( (double) (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));

    q.x = (int) (p.x - scale * hypotenuse * cos(angle));
    q.y = (int) (p.y - scale * hypotenuse * sin(angle));
    cv::line(img, p, q, color, 3, CV_AA);

    // create the arrow hooks
    p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
    cv::line(img, p, q, color, 3, CV_AA);
    p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
    cv::line(img, p, q, color, 3, CV_AA);
}

void create_cartesian_mask(const cv::Size& size, double t0, double t1, cv::Mat& out)
{

    int cy = size.height;
    int cx = size.width / 2;

    out = cv::Mat::zeros(size, CV_8UC1);

    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {

            if (!out.at<uchar>(y, x)) {
                float dx = cx - x;
                float dy = cy - y;
                float r = sqrt(dx * dx + dy * dy);
                float t = atan2(dy, dx) - M_PI_2;

                if (r <= size.height && r >= 0 && t >= t0 && t <= t1) {
                    out.at<uchar>(y, x) = 255;
                }
            }
        }
    }

}

} // namespace image_proc

} // namespace dnntool
