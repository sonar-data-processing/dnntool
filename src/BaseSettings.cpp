#include <iostream>
#include "BaseSettings.hpp"

namespace  dnntool {

BaseSettings::BaseSettings()
{
}

BaseSettings::~BaseSettings()
{
}

void BaseSettings::Load(const std::string& filepath, std::string root_settings)
{
    YAML::Node node = YAML::LoadFile(filepath);
    assert(node.Type() == YAML::NodeType::Map);
    LoadSettings(root_node(node, root_settings));
}

void BaseSettings::LoadSettings(const YAML::Node& node)
{
    assert(node.Type() == YAML::NodeType::Map);

    assert(node["cfg"]);
    assert(node["data"]);
    assert(node["weights"]);

    cfg_filepath_ = node["cfg"].as<std::string>();
    data_filepath_ = node["data"].as<std::string>();
    weights_filepath_ = node["weights"].as<std::string>();
}


} /* namespace dnntool */
