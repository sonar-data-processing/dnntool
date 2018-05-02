#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include "DetectorSettings.hpp"
#include "ClassifierSettings.hpp"
#include "CommonSettings.hpp"
#include "Detector.hpp"
#include "Classifier.hpp"
#include "TargetOrientationEstimator.hpp"
#include "ImageProc.hpp"
#include "Darknet.hpp"
#include "Utils.hpp"

using namespace dnntool;

namespace fs = boost::filesystem;

#define WINDOW_NAME "yolo_output"

void print_image_info(const cv::Mat& img)
{
    float  type = img.type();
    int channel = img.channels();
    int depth = img.depth();

    std::cout << "channels: " << channel << std::endl;
    std::cout << "depth: " << depth << std::endl;
    std::cout << "type: " << type << std::endl;

}

void print_detection_result(
    const cv::Rect boxs,
    const std::vector<float>& probs,
    const std::vector<int>& classes,
    const std::vector<std::string>& names)
{
    std::cout << "box: " << boxs << std::endl;
    for (int j = 0; j < probs.size(); j++) {
        std::cout << "prob: " << probs[j] << " ";
    }
    std::cout << "\n";
    for (int j = 0; j < classes.size(); j++) {
        if (j > 0) std::cout << ", ";
        std::cout << "class: " << classes[j] << ", name: " << names[j];
    }
    std::cout << "\n";
}

void print_classification_result(
    const std::vector<float>& predictions,
    const std::vector<int>& indexes,
    const std::vector<std::string>& labels,
    float step = 30.0f)
{
    for (size_t i = 0; i < indexes.size(); i++) {
        int angle = indexes[i] * step;
        printf("%d - %s - %f\n", angle, labels[i].c_str(), predictions[i]);
    }
}

void clip(cv::Point& pt, const cv::Rect& box)
{
    if (pt.x < box.x) pt.x=box.x;
    if (pt.x > box.x+box.width) pt.x=box.x + box.width;
    if (pt.y < box.y) pt.y=box.y;
    if (pt.y > box.y+box.height) pt.y=box.y+box.height;
}

int line_length(
    cv::Point c,
    float r,
    float t,
    const cv::Rect& box)
{
    cv::Point pt = cv::Point(r*cos(t)+c.x, r*sin(t)+c.y);
    clip(pt, box);
    int dx = pt.x-c.x;
    int dy = pt.y-c.y;
    return sqrt(dx*dx+dy*dy);
}

int full_line_length(cv::Point c, float r, float t, const cv::Rect& box)
{
    return line_length(c, -r, t, box) + line_length(c, r, t, box);
}

void draw_oriented_line(
    cv::Mat& im,
    cv::Point c,
    float r,
    float t,
    const cv::Rect& box,
    const cv::Scalar& color)
{
    cv::Point pt = cv::Point(r*cos(t)+c.x, r*sin(t)+c.y);
    clip(pt, box);
    cv::line(im, c, pt, color, 3);
}

float get_radius(cv::Rect box, cv::Point origin, float theta)
{

    float l = box.x;
    float t = box.y;
    float r = box.x + box.width;
    float b = box.y + box.height;

    float dx = 0, dy = 0;
    dy = origin.y - t;
    dx = origin.x - r;
    return sqrt(dx * dx + dy * dy);
}

void rotate(const cv::Mat& src, cv::Mat& dst, double angle, const cv::Rect roi = cv::Rect(0, 0, -1, -1))
{
    cv::Mat rot;
    image_proc::rotate(src, rot, angle, cv::Point2f(src.cols/2, src.rows/2));

    if (roi.width > 0 && roi.height > 0) {
        rot(roi).copyTo(dst);
        return;
    }

    rot.copyTo(dst);
}

cv::RotatedRect get_rotated_rect(
    const cv::Rect& box,
    const cv::Size& tw,
    float theta)
{
    cv::Point center = cv::Point(box.x+box.width/2, box.y+box.height/2);
    float radius = get_radius(box, center, theta);

    // rotated rect width
    int w = full_line_length(center, radius, -theta, box);
    // rotated rect height
    int h = (double)w  * tw.height / (double)tw.width;

    return cv::RotatedRect(center, cv::Size(w, h), utils::rad2deg(-theta));
}

void draw_rotated_rect(cv::Mat& mat, cv::Scalar color, const cv::RotatedRect& box)
{
    cv::Point2f rect_points[4];
    box.points( rect_points );
    for( int i = 0; i < 4; i++ ) {
        cv::line(mat, rect_points[i], rect_points[(i+1)%4], color, 3, 8 );
    }
}

void init_boundingbox_detector(const std::string& configfile, Detector& detector)
{
    DetectorSettings detector_settings;
    detector_settings.Load(configfile, "boundingbox-detector");
    detector.Load(detector_settings);
}

void init_orientation_classifier(const std::string& configfile, Classifier& classifier)
{
    ClassifierSettings classifier_settings;
    classifier_settings.Load(configfile, "orientation-classifier");
    classifier.Load(classifier_settings);
}

void init_cartesian_mask(const std::string& configfile, cv::Size frame_size, cv::Mat& mask)
{
   CommonSettings common_settings;
   common_settings.Load(configfile);

   double t0 = -utils::deg2rad(common_settings.sonar_beam_width()) / 2;
   double t1 = utils::deg2rad(common_settings.sonar_beam_width()) / 2;

   image_proc::create_cartesian_mask(frame_size, t0, t1, mask);
}

float predict_orientation(const cv::Mat& img, const Classifier& classifier, float& max_pred) {
    std::vector<float> pred_r, pred_f;
    std::vector<int> ind_r, ind_f;
    std::vector<std::string> labels_r, labels_f;

    cv::Mat rot;
    rotate(img, rot, 360.0);
    classifier.Predict(rot, pred_r, ind_r, labels_r);

    pred_f.insert(pred_f.end(), pred_r.begin(), pred_r.end());
    ind_f.insert(ind_f.end(), ind_r.begin(), ind_r.end());
    labels_f.insert(labels_f.end(), labels_r.begin(), labels_r.end());

    size_t max_i;
    utils::find_max<float>(pred_f, max_pred, max_i);


    const float step = 15.0f;
    return ind_f[max_i] * step;
}

void run_detector(const Detector& detector, const Classifier& classifier, const cv::Mat& mask, cv::Mat& src)
{
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float> > probs;
    std::vector<std::vector<int> > classes;
    std::vector<std::vector<std::string> > names;

    cv::Mat img;
    src.copyTo(img, mask);
    detector.Detect(img, boxes, probs, classes, names);

    cv::Size target_window = cv::Size(136, 32);

    for (int i = 0; i < boxes.size(); i++) {
        print_detection_result(boxes[i], probs[i],classes[i], names[i]);
        cv::Mat bbox_image;
        img(boxes[i]).copyTo(bbox_image);

        cv::Mat bbox_gray;
        cv::cvtColor(bbox_image, bbox_gray, CV_BGR2GRAY);

        float pred_val;
        float theta_deg = predict_orientation(bbox_image, classifier, pred_val);
        std::cout << "angle: " << theta_deg << " pred: " << pred_val << std::endl;

        int x = classes[i].front() * 123457 % detector.classes();
        std::string class_names = "";
        for (int j = 0; j < names[i].size(); j++) class_names += names[i][j] + " ";

        cv::Scalar color = image_proc::get_scalar(x, detector.classes());

        if (theta_deg == 0 || theta_deg == 180 || theta_deg == 90 || pred_val < 0.49) {
            cv::rectangle(src, boxes[i], color, 3);
            image_proc::draw_text(src, class_names, cv::Point(boxes[i].x, boxes[i].y), color);
        }
        else {
            cv::RotatedRect rbox = get_rotated_rect(boxes[i], target_window, utils::deg2rad(theta_deg));
            draw_rotated_rect(src, color, rbox);

            cv::Point2f rect_points[4];
            rbox.points(rect_points);

            int x = INT_MAX;
            int y = INT_MAX;

            if (theta_deg >= 90) {
                for( int i = 0; i < 4; i++ ) {
                    x = std::min(x, (int)rect_points[i].x);
                    y = std::min(y, (int)rect_points[i].y);
                }
            }
            else {
                y = INT_MIN;
                for( int i = 0; i < 4; i++ ) {
                    x = std::min(x, (int)rect_points[i].x);
                    y = std::max(y, (int)rect_points[i].y) + 15;
                }
            }



            image_proc::draw_text(src, class_names, cv::Point(x, y), color);


        }


    }

}

void run_detector(const Detector& detector, const Classifier& classifier, cv::Mat& src)
{
    cv::Mat mask = cv::Mat::ones(src.size(), CV_8UC1) * 255;
    run_detector(detector, classifier, mask, src);
}

void run_detector_from_video(const std::string& configfile, const std::string& videofile)
{
    // load video
    cv::VideoCapture cap(videofile);
    assert(cap.isOpened());

    // load bounding box detector
    Detector detector;
    init_boundingbox_detector(configfile, detector);

    // load orientation classifier
    Classifier classifier;
    init_orientation_classifier(configfile, classifier);

    // obtain frame size
    cv::Size frame_size = cv::Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH), (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

    // create cartesian mask
    cv::Mat mask;
    init_cartesian_mask(configfile, frame_size, mask);

    cv::Mat frame;
    int frame_count = 0;

    for (int i = 0; i < 1000; i++) cap.grab();
    for (;;) {
        // read frame
        cap >> frame;
        if (frame.empty()) break;

        // run bounding box detector and classify the orientation
        run_detector(detector, classifier, mask, frame);

        cv::imshow("frame", frame);
        if (frame_count == 0){
            int key;
            do {
                key = cv::waitKey() & 0xff;
            } while (key != 10);
        }
        else
            cv::waitKey(15);
        frame_count++;
    }
}

void run_detector_from_image(const std::string& configfile, const std::string& imagefile)
{
    // load image
    cv::Mat img = cv::imread(imagefile);

    // load bounding box detector
    Detector detector;
    init_boundingbox_detector(configfile, detector);

    // load orientation classifier
    Classifier classifier;
    init_orientation_classifier(configfile, classifier);

    // run bounding box detector and classify the orientation
    run_detector(detector, classifier, img);

    cv::imshow("image", img);
    cv::waitKey();
}

void run_detector_from_path(const std::string& configfile, const std::string& path)
{
    // load bounding box detector
    Detector detector;
    init_boundingbox_detector(configfile, detector);

    // load orientation classifier
    Classifier classifier;
    init_orientation_classifier(configfile, classifier);

    fs::path p(path);
    boost::regex expression(".*\\.png");

    fs::directory_iterator end_iter;
    for (fs::directory_iterator dir_itr(p); dir_itr != end_iter; ++dir_itr) {
        boost::smatch what;
        if(!boost::regex_match(dir_itr->path().filename().string(), what, expression)) continue;
        std::cout << dir_itr->path().string() << "\n";

        // load image
        cv::Mat img = cv::imread(dir_itr->path().string());

        // run bounding box detector and classify the orientation
        run_detector(detector, classifier, img);

        cv::imshow("image", img);
        cv::waitKey();
    }
}

void run_classifier_from_path(const std::string& configfile, const std::string& path)
{
    // load orientation classifier
    Classifier classifier;
    init_orientation_classifier(configfile, classifier);

    fs::path p(path);
    boost::regex expression(".*\\.png");

    fs::directory_iterator end_iter;
    for (fs::directory_iterator dir_itr(p); dir_itr != end_iter; ++dir_itr) {
        boost::smatch what;
        if(!boost::regex_match(dir_itr->path().filename().string(), what, expression)) continue;
        std::cout << dir_itr->path().string() << "\n";

        // load image
        cv::Mat img = cv::imread(dir_itr->path().string());

        // predict orientation
        float pred;
        float theta_deg = predict_orientation(img, classifier, pred);
        std::cout << "angle: " << theta_deg << " pred: " << pred << std::endl;

        cv::imshow("image", img);
        cv::waitKey();
    }
}
