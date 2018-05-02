#include "Detector.hpp"

namespace dnntool
{

Detector::Detector()
    : net_(NULL)
    , options_(NULL)
    , thresh_(0.24)
    , boxes_(NULL)
    , probs_(NULL)
    , num_boxes_(0)
{
}

Detector::Detector(const DetectorSettings& settings)
    : net_(NULL)
    , options_(NULL)
    , thresh_(0.24)
    , boxes_(NULL)
    , probs_(NULL)
    , num_boxes_(0)
{
    Load(settings);
}

Detector::~Detector()
{
    if (boxes_) {
        free(boxes_);
        boxes_ = NULL;
    }

    if (probs_) {
        for(int j = 0; j < num_boxes_; ++j) free(probs_[j]);
        free(probs_);
        probs_ = NULL;
    }

    if (options_) {
        free_list(options_);
        options_ = NULL;
    }

}

void Detector::Load(const DetectorSettings& settings)
{
    thresh_ = settings.confidence();
    options_ = darknet::read_data_cfg(settings.data_filepath());

    net_ = darknet::load_network(settings.cfg_filepath(), settings.weights_filepath(), 0);
    darknet::set_batch_network(net_, 1);

    srand(2222222);

    // get last network layer
    layer l = net_->layers[net_->n-1];
    num_boxes_ = l.w*l.h*l.n;

    boxes_ = (box*)calloc(num_boxes_, sizeof(box));
    probs_ = (float**)calloc(num_boxes_, sizeof(float*));
    for(int j = 0; j < num_boxes_; ++j) probs_[j] = (float*)calloc(l.classes+1, sizeof(float));

}

void Detector::Detect(
    const cv::Mat& src,
    std::vector<cv::Rect>& out_boxes,
    std::vector<std::vector<float> >& out_probs,
    std::vector<std::vector<int> >& out_classes,
    std::vector<std::vector<std::string> >& out_names) const
{
    out_probs.clear();
    out_classes.clear();
    out_names.clear();

    int classes = darknet::option_find_int(options_, "classes", 3);
    char *name_list = darknet::option_find_str(options_, "names", "data/names.list");
    char **names = darknet::get_labels(name_list);

    layer l = net_->layers[net_->n-1];

    image im = darknet::cv_to_image(src);
    image sized = darknet::letterbox_image(im, net_->w, net_->h);

    float *X = sized.data;

    double time = darknet::what_time_is_it_now();
    darknet::network_predict(net_, X);

    printf("Predicted in %f seconds.\n", darknet::what_time_is_it_now()-time);

    darknet::get_region_boxes(l, im.w, im.h, net_->w, net_->h, thresh_, probs_, boxes_, NULL, 0, 0, 0.5f, 1);
    darknet::do_nms_sort(boxes_, probs_, l.w*l.h*l.n, l.classes, 0.3);

    darknet::read_detections(
        im,
        num_boxes_,
        thresh_,
        boxes_,
        probs_,
        l.classes,
        out_boxes,
        out_probs,
        out_classes);

    for (int i = 0; i < out_boxes.size(); i++) {
        std::vector<std::string> name_list;
        for (int j = 0; j < out_classes[i].size(); j++) {
            name_list.push_back(names[out_classes[i][j]]);
        }
        out_names.push_back(name_list);
    }

    free_image(im);
    free_image(sized);
}

} // namespace dnntool
