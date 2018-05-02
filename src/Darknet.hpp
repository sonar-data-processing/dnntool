#ifndef Darknet_hpp
#define Darknet_hpp

#include <opencv2/opencv.hpp>

#ifdef GPU
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "darknet.h"
void ipl_into_image(IplImage* src, image im);

#ifdef __cplusplus
}
#endif

namespace dnntool {

namespace darknet {

inline
network* load_network (
    const std::string& cfgfile,
    const std::string& weightfile,
    int clear)
{
    return ::load_network(
        const_cast<char*>(cfgfile.c_str()),
        const_cast<char*>(weightfile.c_str()),
        clear);
}

inline
list* read_data_cfg(const std::string& datafile)
{
    return ::read_data_cfg(const_cast<char*>(datafile.c_str()));
}

inline
int option_find_int(list *l, const char *key, int def)
{
    return ::option_find_int(l, const_cast<char*>(key), def);
}

inline
char* option_find_str(list *l, const char *key, const char *def)
{
    return ::option_find_str(l, const_cast<char*>(key), const_cast<char*>(def));
}

inline
char **get_labels(char *filename)
{
    return ::get_labels(filename);
}

inline
void save_image(image im, const std::string& name)
{
    ::save_image(im, const_cast<char*>(name.c_str()));
}

inline
void show_image(image im, const std::string& name)
{
    ::show_image(im, const_cast<char*>(name.c_str()));
}

inline
image** load_alphabet()
{
    return ::load_alphabet();
}

inline
image letterbox_image(image im, int w, int h)
{
    return ::letterbox_image(im, w, h);
}

inline
float* network_predict(network *net, float *input)
{
    return ::network_predict(net, input);
}

inline
void set_batch_network(network *net, int b)
{
    return ::set_batch_network(net, b);
}

inline
void get_region_boxes(
    layer l,
    int w,
    int h,
    int netw,
    int neth,
    float thresh,
    float **probs,
    box *boxes,
    float **masks,
    int only_objectness,
    int *map,
    float tree_thresh,
    int relative)
{
    return ::get_region_boxes(
        l, w, h, netw, neth,
        thresh, probs, boxes,
        masks, only_objectness,
        map, tree_thresh, relative
    );
}

inline
double what_time_is_it_now()
{
    return ::what_time_is_it_now();
}

inline
void draw_detections(
    image im,
    int num,
    float thresh,
    box *boxes,
    float **probs,
    float **masks,
    char **names,
    image **alphabet,
    int classes)
{
    return ::draw_detections(
        im, num ,thresh, boxes, probs, masks, names, alphabet, classes);
}

inline
void do_nms_sort(
    box *boxes,
    float **probs,
    int total,
    int classes,
    float thresh)
{
    ::do_nms_sort(boxes, probs, total, classes, thresh);
}

inline
image resize_min(
    image im,
    int min)
{
    return ::resize_min(im, min);
}

inline
image crop_image(
    image im,
    int dx,
    int dy,
    int w,
    int h)
{
    return ::crop_image(im, dx, dy, w, h);
}

image cv_to_image(
    const cv::Mat& mat);

void read_detections(
    image im,
    int num,
    float thresh,
    box *boxes,
    float **probs,
    int classes,
    std::vector<cv::Rect>& out_boxes,
    std::vector<std::vector<float> >& out_probs,
    std::vector<std::vector<int> >& out_classes);

} /* namespace darknet */

} /* namespace dnntool */

#endif
