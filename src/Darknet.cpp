#include <iostream>
#include <string>
#include "Darknet.hpp"

namespace dnntool
{

image darknet::cv_to_image(const cv::Mat& mat)
{
    IplImage src = mat;

    int h = src.height;
    int w = src.width;
    int c = src.nChannels;

    image out = ::make_image(w, h, c);

    ::ipl_into_image(&src, out);
    ::rgbgr_image(out);
    return out;
}

void darknet::read_detections(
    image im,
    int num,
    float thresh,
    box *boxes,
    float **probs,
    int classes,
    std::vector<cv::Rect>& out_boxes,
    std::vector<std::vector<float> >& out_probs,
    std::vector<std::vector<int> >& out_classes)
{
    for(int i = 0; i < num; ++i) {
        std::vector<float> prob_list;
        std::vector<int> class_list;
        for(int j = 0; j < classes; ++j) {
            if (probs[i][j] > thresh) {
                class_list.push_back(j);
                prob_list.push_back(probs[i][j]);
            }
        }

        if(!prob_list.empty()) {
            box b = boxes[i];

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

            out_boxes.push_back(cv::Rect(left, top, right-left, bot-top));
            out_probs.push_back(prob_list);
            out_classes.push_back(class_list);
        }
    }
}

} /* namespace dnntool */
