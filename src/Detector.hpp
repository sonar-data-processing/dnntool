#ifndef dnntool_Detector_hpp
#define dnntool_Detector_hpp

#include "Darknet.hpp"
#include "DetectorSettings.hpp"

namespace dnntool
{

class Detector
{
public:
    Detector();
    Detector(const DetectorSettings& settings);
    virtual ~Detector();

    void Load(const DetectorSettings& settings);

    void Detect(
        const cv::Mat& src,
        std::vector<cv::Rect>& out_boxes,
        std::vector<std::vector<float> >& out_probs,
        std::vector<std::vector<int> >& out_classes,
        std::vector<std::vector<std::string> >& out_names) const;

    int classes() const {
        return net_->layers[net_->n-1].classes;
    }

    const network *net() const {
        return net_;
    }

private:

    // deep neural network
    network *net_;

    // data file options
    list *options_;

    // confidence value
    float thresh_;

    // list of boxes
    box *boxes_;

    // number of grids
    int num_boxes_;

    // list of probabilities
    float **probs_;
};

} // namespace dnntool

#endif // dnntool_Detector_hpp
