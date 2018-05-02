#ifndef dnntool_Classifier_hpp
#define dnntool_Classifier_hpp

#include "Darknet.hpp"
#include "ClassifierSettings.hpp"

namespace dnntool
{

class Classifier
{

public:
    Classifier();
    Classifier(const ClassifierSettings& settings);

    virtual ~Classifier();

    void Load(const ClassifierSettings& settings);

    void Predict(
        const cv::Mat& src,
        std::vector<float>& out_predictions,
        std::vector<int>& out_indexes,
        std::vector<std::string>& out_labels) const;

private:

    // deep neural network
    network *net_;

    // data file options
    list *options_;

};

} /* namespace dnntool */

#endif /* dnntool_Classifier_hpp_hpp */
