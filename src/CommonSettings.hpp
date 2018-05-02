#ifndef dnntool_CommonSettings_hpp
#define dnntool_CommonSettings_hpp

#include <string>

namespace dnntool
{

class CommonSettings
{
public:
    CommonSettings()
        :  sonar_beam_width_(0.0) {}

    virtual ~CommonSettings() {}

    void Load(const std::string& filepath);

    double sonar_beam_width() const {
        return sonar_beam_width_;
    }

private:
    double sonar_beam_width_;


}; /* CommonSettings */

} /* namespace dnntool */

#endif
