#include "Classifier.hpp"

namespace dnntool
{

Classifier::Classifier()
    : net_(NULL)
{
}

Classifier::Classifier(const ClassifierSettings& settings)
    : net_(NULL)
{
    Load(settings);
}

Classifier::~Classifier()
{
}


void Classifier::Load(const ClassifierSettings& settings)
{
    /* load network */
    net_ = darknet::load_network(settings.cfg_filepath(), settings.weights_filepath(), 0);
    darknet::set_batch_network(net_, 1);
    srand(time(0));

    options_ = darknet::read_data_cfg(settings.data_filepath());
}

void Classifier::Predict(
    const cv::Mat& src,
    std::vector<float>& out_predictions,
    std::vector<int>& out_indexes,
    std::vector<std::string>& out_labels) const
{
    out_predictions.clear();
    out_indexes.clear();
    out_labels.clear();

    int topk = darknet::option_find_int(options_, "top", 1);
    int classes = darknet::option_find_int(options_, "classes", 2);
    char *label_list = darknet::option_find_str(options_, "labels", "data/labels.list");
    char **labels = darknet::get_labels(label_list);

    image im = darknet::cv_to_image(src);
    image resized = darknet::resize_min(im, net_->w);
    image crop = darknet::crop_image(resized, (resized.w - net_->w)/2, (resized.h - net_->h)/2, net_->w, net_->h);

    float *pred = darknet::network_predict(net_, crop.data);
    if(net_->hierarchy) hierarchy_predictions(pred, net_->outputs, net_->hierarchy, 1, 1);

    if(resized.data != im.data) free_image(resized);

    free_image(im);
    free_image(crop);

    int *indexes = (int*)calloc(topk, sizeof(int));

    top_k(pred, classes, topk, indexes);

    for(int j = 0; j < topk; j++) {
        out_predictions.push_back(pred[indexes[j]]);
        out_indexes.push_back(indexes[j]);
        out_labels.push_back(labels[indexes[j]]);
    }

    free(indexes);
}

} /* namespace dnntool */
