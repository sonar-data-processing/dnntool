#ifndef dnntool_BaseSettings_hpp
#define dnntool_BaseSettings_hpp

#include <fstream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace  dnntool {

class BaseSettings {

public:
    BaseSettings();
    BaseSettings(const std::string& filepath);

    virtual ~BaseSettings();

    virtual void Load(const std::string& filepath, const std::string root_settings="");

    std::string cfg_filepath() const {
        return cfg_filepath_;
    }

    std::string data_filepath() const {
        return data_filepath_;
    }

    std::string weights_filepath() const {
        return weights_filepath_;
    }

    virtual std::string to_string() {
        std::stringstream ss;
        ss << "cfg: " << cfg_filepath_ << "\n";
        ss << "data: " << data_filepath_ << "\n";
        ss << "weights: " << weights_filepath_ << "\n";
        return ss.str();
    }

protected:

    virtual YAML::Node root_node(const YAML::Node& node, const std::string& root_settings) {
        assert(node[root_settings]);
        assert(node[root_settings].Type() == YAML::NodeType::Map);
        return node[root_settings];
    }

    virtual void LoadSettings(const YAML::Node& node);

    // cfg file path
    std::string cfg_filepath_;

    // data file path
    std::string data_filepath_;

    // weights filepath
    std::string weights_filepath_;
};

} /* namespace dnntool */

#endif /* dnntool_BaseSettings_hpp */
