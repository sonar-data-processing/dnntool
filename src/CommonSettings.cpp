#include <yaml-cpp/yaml.h>
#include "CommonSettings.hpp"

namespace dnntool
{

void CommonSettings::Load(const std::string& filepath)
{
    YAML::Node node = YAML::LoadFile(filepath);
    assert(node.Type() == YAML::NodeType::Map);
    assert(node["sonar-beam-width"]);
    sonar_beam_width_ = node["sonar-beam-width"].as<double>();
}


} /* namespace dnntool */
