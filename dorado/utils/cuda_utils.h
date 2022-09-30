#pragma once

#include <string>
#include <vector>

// Given a string representing cuda devices (e.g "cuda:0,1,3") returns a vector of strings, one for
// each device (e.g ["cuda:0", "cuda:2", ..., "cuda:7"]
std::vector<std::string> parse_cuda_device_string(std::string device_string);

// Given a vector of devices in the format as returned by parse_cuda_device_string this function
// will report the largest amount of available memory (in bytes) common to all devices.
size_t available_memory(std::vector<std::string> devices);
int auto_gpu_batch_size(std::string model_path, std::vector<std::string> devices);
