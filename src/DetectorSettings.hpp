#ifndef dnntool_DetectorSettings_hpp
#define dnntool_DetectorSettings_hpp

#include <fstream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "BaseSettings.hpp"

namespace  dnntool {

class DetectorSettings : public BaseSettings {

public:

    virtual ~DetectorSettings(){}

    float confidence() const {
        return confidence_;
    }

    virtual std::string to_string() {
        std::stringstream ss;
        ss << BaseSettings::to_string();
        ss << "confidence: " << confidence_ << "\n";
        return ss.str();
    }

private:

    void LoadSettings(const YAML::Node& node);

    // confidence
    float confidence_;
};

} /* namespace dnntool */

#endif /* dnntool_DetectorSettings_hpp */
