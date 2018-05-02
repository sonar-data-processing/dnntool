#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#define APP_NAME "dnntool"

using namespace boost;

namespace po = boost::program_options;

extern void run_detector_from_video(const std::string& configfile, const std::string& videofile);
extern void run_detector_from_image(const std::string& configfile, const std::string& imagefile);
extern void run_detector_from_path(const std::string& configfile, const std::string& path);
extern void run_classifier_from_path(const std::string& configfile, const std::string& path);

std::string read_filepath(const program_options::variables_map& vm, const std::string& arg)
{
    std::string filepath;
    if (vm.count(arg)) {
        filepath = vm[arg].as<std::string>();
        if (!filesystem::exists(filepath) &&
            !filesystem::exists(filesystem::path(filesystem::current_path()).string() + "/" + filepath)){
            std::cerr << filepath << " file does not exist." << std::endl;
            exit(-1);
        }
    }
    return filepath;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::cout << "wrong number of arguments" << std::endl;
        return -1;
    }

    std::vector<std::string> opts;
    for (size_t i=2; i < argc; i++) {
        opts.push_back(std::string(argv[i]));
    }

    po::options_description detector_desc("detector options");
    detector_desc.add_options()
        ("help", "show detector options")
        ("conf,c", po::value<std::string>()->required(), "path to configuration file")
        ("video,v", po::value<std::string>(), "video input file")
        ("image,i", po::value<std::string>(), "image input file")
        ("path,p", po::value<std::string>(), "path to image files");

    po::variables_map vm;
    po::store(po::command_line_parser(opts).options(detector_desc).run(), vm);

    if (vm.count("help")) {
        std::cout << detector_desc << std::endl;
        return 0;
    }

    po::notify(vm);

    if (!strcmp(argv[1], "detector")) {
        if (vm.count("video")) {
            run_detector_from_video(read_filepath(vm, "conf"), read_filepath(vm, "video"));
        }
        else if (vm.count("image")) {
            run_detector_from_image(read_filepath(vm, "conf"), read_filepath(vm, "image"));
        }
        else if (vm.count("path")) {
            run_detector_from_path(read_filepath(vm, "conf"), read_filepath(vm, "path"));
        }
        else {
            std::cout << "You should inform an option!" << std::endl;
            std::cout << detector_desc << std::endl;
        }
    }
    else if (!strcmp(argv[1], "classifier")) {
        if (vm.count("path")) {
            run_classifier_from_path(read_filepath(vm, "conf"), read_filepath(vm, "path"));
        }
        else {
            std::cout << "You should inform an option!" << std::endl;
            std::cout << detector_desc << std::endl;
        }
    }


    return -1;

}
