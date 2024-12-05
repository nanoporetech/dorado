#!/bin/bash -e
#
# Helper script to run clang-tidy on the codebase.
#

# Parse any input args.
build_dir=build-tidy
num_jobs=$(( $( command -v nproc > /dev/null && nproc || sysctl -n hw.physicalcpu ) / 2 ))
apply_fixits=
while getopts j:B:f opt; do
  case $opt in
    j) # Number of jobs to use during build/analysis
       num_jobs="$OPTARG"
       ;;
    B) # Location to put the build dir
       build_dir="$OPTARG"
       ;;
    f) # Attempt to apply fixits
       apply_fixits="-fix"
       ;;
    ?) echo "Usage $0 [-j jobs] [-B build_dir] [-f]"
       grep " .) #" $0 | grep -v grep
       exit 1
       ;;
  esac
done

# Check that we can actually find clang-tidy.
if ! command -v clang-tidy || ! command -v run-clang-tidy || ! command -v clang || ! command -v clang++ ; then
  echo "Cannot find clang, clang-tidy, or run-clang-tidy"
  exit 1
fi

# Use clang as the compiler so that vbz doesn't trip up clang-tidy.
export CC=clang
export CXX=clang++

# Assuming the current script is in /scripts.
source_dir=$(dirname $(dirname $0))

# Make a new build folder to analyse.
cmake \
  -S ${source_dir} \
  -B ${build_dir} \
  -D CMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build dorado_utils so that its dependencies (htslib, etc...) get installed, otherwise
# clang-tidy can't find their headers.
cmake \
  --build ${build_dir} \
  --target dorado_utils \
  --target vbz_hdf_plugin \
  -j ${num_jobs}

# Remove any 3rdparty .clang-tidy's otherwise we check them for errors.
find ${source_dir}/dorado/3rdparty/* -name .clang-tidy -delete

# Print the current config to make sure it parses correctly.
clang-tidy --dump-config

# Run clang-tidy with our warnings.
# Note that we use run-clang-tidy to avoid the overhead of having to a full build too.
run-clang-tidy \
  -p ${build_dir} \
  -j ${num_jobs} \
  ${apply_fixits} \
  -quiet
