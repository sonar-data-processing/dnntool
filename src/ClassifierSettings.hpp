#ifndef dnntool_ClassifierSettings_hpp
#define dnntool_ClassifierSettings_hpp

#include <fstream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "BaseSettings.hpp"
namespace  dnntool {

class ClassifierSettings : public BaseSettings {

public:
    virtual ~ClassifierSettings() {}
};

} /* namespace dnntool */

#endif /* dnntool_ClassifierSettings_hpp */
