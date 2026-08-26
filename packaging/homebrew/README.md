# Homebrew packaging

`dorado.rb` is a Homebrew formula that builds Dorado from source on Apple
Silicon.

```bash
brew install autoconf automake cmake openssl@3
brew install --build-from-source ./packaging/homebrew/dorado.rb
```

The formula pre-stages every prebuilt dependency (libtorch, metal-cpp, pod5)
as Homebrew `resource` blocks into `build/download/`, which is where
`cmake/DownloadAndExtract.cmake` looks before fetching. The CMake configure
step therefore runs with no network access, and every artefact is pinned by
SHA-256.

Requirements:

* Apple Silicon, macOS >= 14.
* A full Xcode install: `xcrun metal` / `xcrun metallib` are needed to compile
  `dorado/nn/metal/nn.metal`, and the Command Line Tools alone do not ship the
  Metal toolchain.

## Why this is not a `homebrew/core` formula

Homebrew's [Acceptable Formulae](https://docs.brew.sh/Acceptable-Formulae)
rules rule Dorado out of `homebrew/core` on four independent counts. The first
is not fixable by any packaging work.

### 1. Licence (blocking, not fixable downstream)

> A formula in `homebrew/core` must be open source under a licence compatible
> with the Debian Free Software Guidelines.

`LICENCE.txt` is the Oxford Nanopore Technologies PLC. Public License v1.0. It
is derived from MPL-2.0, but section 2.1 grants the right to "use, reproduce,
make available, modify, display, perform, distribute, and otherwise exploit
**solely for Research Purposes**". A field-of-endeavour restriction fails DFSG
§6, so there is no SPDX identifier Homebrew can accept and the formula has to
declare `license :cannot_represent`.

### 2. Proprietary dependency

> A core formula must not depend on a cask, proprietary software or a runtime
> step that automatically installs either one.

On Linux and Windows, `cmake/Koi.cmake` downloads `libkoi`, a closed-source
static library published only as a binary on the ONT CDN (the source repo is
`git.oxfordnanolabs.local`, which is internal). macOS uses the Metal backend
instead and avoids Koi, which is why the formula is macOS-only, but it means
`homebrew/core`'s Linux bottles could never be produced.

### 3. Binary dependencies that cannot be built from source

> It must either build from source or install portable, platform-independent
> output.

The build consumes prebuilt binaries with no source-build path in this repo:

| Dependency | Origin |
| --- | --- |
| libtorch 2.9.0 (ONT-patched) | `cdn.oxfordnanoportal.com`, no published patch set |
| pod5 0.3.39 | `nanoporetech/pod5-file-format` release tarballs |
| metal-cpp | Apple developer download |
| OpenSSL 3.5.5 | `cdn.oxfordnanoportal.com` prebuilt archives |
| flash-attention (Linux/Windows) | `cdn.oxfordnanoportal.com` |

The formula replaces OpenSSL with Homebrew's `openssl@3`, but the ONT-patched
libtorch cannot be swapped for an upstream build, and pod5 and metal-cpp remain
binary-only.

### 4. Vendored dependencies

> Avoid bundling upstream libraries. Instead, use Homebrew-provided
> equivalents.

`dorado/3rdparty` carries 14 git submodules, several of which duplicate
existing core formulae (`htslib`, `spdlog`, `openssl`, `zlib`, `minimap2` via
`ont-minimap2`). `homebrew/core` would require these to be replaced with
`depends_on`, which is not possible for the ONT forks (`ont-minimap2`,
the patched htslib build flags).

## What would have to change

1. Relicense under an OSI-approved licence, or dual-licence the parts shipped
   in the formula. Without this, nothing else matters.
2. Publish Koi sources, or make the CUDA backend an optional component so a
   Koi-free Linux build exists.
3. Publish the libtorch patch set so Dorado can build against an upstream
   libtorch, and get pod5 buildable from source.
4. Un-vendor htslib/spdlog/zlib in favour of system packages.

Until then, a third-party tap is the correct distribution channel, and the
existing `dorado-<version>-osx-arm64.zip` release could alternatively be
shipped as a Homebrew **cask**, which has no source or licence requirement.
