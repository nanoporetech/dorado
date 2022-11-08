#include <map>
#include <string>
// Given a path to a space-delimited csv in `tempate_id complement_id` format,
// returns a map of template_id to  complement_id
std::map<std::string, std::string> load_pairs_file(std::string pairs_file);
