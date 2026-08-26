# Homebrew formula for Dorado.
#
# This formula is intended for a third-party tap, NOT for homebrew/core.
# See README.md in this directory for the list of homebrew/core acceptance
# rules that Dorado currently cannot satisfy.
#
# Usage:
#   brew tap nanoporetech/dorado https://github.com/nanoporetech/dorado
#   brew install --build-from-source ./packaging/homebrew/dorado.rb
class Dorado < Formula
  desc "Basecaller for Oxford Nanopore reads"
  homepage "https://github.com/nanoporetech/dorado"
  url "https://github.com/nanoporetech/dorado.git",
      tag:      "v2.1.2",
      revision: "8b8fc5d36a9c0baab262a743cb175b5878e38ca6"
  # Oxford Nanopore Technologies PLC. Public License Version 1.0.
  # MPL-2.0 derived, but section 2.1 restricts the grant to "Research Purposes",
  # so it is not an OSI-approved / DFSG-compatible licence and Homebrew has no
  # SPDX identifier for it.
  license :cannot_represent
  head "https://github.com/nanoporetech/dorado.git", branch: "master"

  # Dorado only builds without proprietary components on Apple Silicon:
  # the Linux and Windows builds link against libkoi, a closed-source binary
  # distributed from the ONT CDN, and require the CUDA toolkit.
  depends_on "autoconf" => :build # htslib configure
  depends_on "automake" => :build
  depends_on "cmake" => :build
  depends_on xcode: ["15.0", :build] # xcrun metal / metallib
  depends_on arch: :arm64
  depends_on macos: :sonoma # deployment target is macOS 14

  depends_on "openssl@3"

  # Versions and checksums below are mirrored from the cmake/ modules of the
  # pinned tag. They MUST be re-checked against cmake/Torch.cmake,
  # cmake/Pod5.cmake and cmake/Metal.cmake on every version bump.
  resource "libtorch" do
    url "https://cdn.oxfordnanoportal.com/software/analysis/torch-2.9.0.2-ont-macos-m1.zip"
    sha256 "84113189187bc3d3f4d3948a15b5c4ee13f05ffc4df1449e733e3dcf286e1c80"
  end

  resource "metal-cpp" do
    url "https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip"
    sha256 "4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4"
  end

  resource "pod5" do
    url "https://github.com/nanoporetech/pod5-file-format/releases/download/0.3.39/lib_pod5-0.3.39-osx-14.0-arm64.tar.gz"
    sha256 "710db33a00ca041094588b7c7b5355ad45611e6160e2e3ec8763e8be34f4ac91"
  end

  def install
    # cmake/DownloadAndExtract.cmake skips a download when the target directory
    # already exists under ${CMAKE_BINARY_DIR}/download, so pre-stage every
    # prebuilt dependency there and the build never touches the network.
    download = buildpath/"build/download"
    download.mkpath

    stage_prebuilt = lambda do |name, dest, single_root|
      resource(name).stage do
        root = Pathname.pwd
        root /= single_root if single_root && (root/single_root).directory?
        (download/dest).install root.children
      end
    end

    stage_prebuilt.call "libtorch", "torch-2.9.0.2-ont-macos-m1/libtorch", "libtorch"
    stage_prebuilt.call "metal-cpp", "metal-cpp_26/metal-cpp", "metal-cpp"
    stage_prebuilt.call "pod5", "pod5-0.3.39-Darwin", nil

    args = %W[
      -DDORADO_LIBTORCH_DIR=#{download}/torch-2.9.0.2-ont-macos-m1/libtorch
      -DOPENSSL_ROOT_DIR=#{formula_opt_prefix("openssl@3")}
      -DDORADO_DISABLE_TESTS=ON
      -DDORADO_DISABLE_PACKAGING=ON
    ]

    system "cmake", "-S", ".", "-B", "build", *args, *std_cmake_args
    system "cmake", "--build", "build", "--config", "Release"
    system "cmake", "--install", "build"

    # The build also installs a few static archives of vendored libraries
    # (spdlog, haplotag) that are of no use to a Homebrew consumer.
    rm_r(lib/"cmake") if (lib/"cmake").exist?
    Dir[lib/"*.a"].each { |f| rm f }
  end

  def caveats
    <<~EOS
      Dorado is distributed under the Oxford Nanopore Technologies Public
      License v1.0, which grants rights for Research Purposes only.

      Basecalling models are not bundled. Download them with:
        dorado download --model all --models-directory <dir>
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/dorado --version 2>&1")
    system bin/"dorado", "basecaller", "--help"
  end
end
