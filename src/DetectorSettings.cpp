#include <iostream>
#include "DetectorSettings.hpp"

namespace  dnntool {

void DetectorSettings::LoadSettings(const YAML::Node& node)
{
    BaseSettings::LoadSettings(node);
    assert(node["confidence"]);
    confidence_ = node["confidence"].as<float>();
}

} /* namespace dnntool */
