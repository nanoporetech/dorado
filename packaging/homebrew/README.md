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
| pod5 0.3.39 | `nanoporetech/pod5-file-format` release tarballs (upstream source is MPL-2.0) |
| metal-cpp | Apple developer download |
| OpenSSL 3.5.5 | `cdn.oxfordnanoportal.com` prebuilt archives |
| flash-attention (Linux/Windows) | `cdn.oxfordnanoportal.com` |

The formula replaces OpenSSL with Homebrew's `openssl@3`. metal-cpp has no
source distribution at all.

pod5 is only binary-*here*: upstream,
[`nanoporetech/pod5-file-format`](https://github.com/nanoporetech/pod5-file-format)
is MPL-2.0 and buildable from source, so a core formula could plausibly depend
on a `pod5` formula rather than the release tarball. Nobody has packaged one
yet, but this is a gap rather than a hard blocker.

libtorch is the least certain of the three. `homebrew/core` does carry
`pytorch`, and `DORADO_LIBTORCH_DIR` accepts any libtorch tree, so a core
formula could in principle depend on a source-built libtorch. Whether Dorado
actually needs the ONT patches (`TORCH_PATCH_SUFFIX` in `cmake/Torch.cmake`)
or would link against a stock libtorch 2.9.0 has **not** been tested here, and
the patch set is not published, so this is an open question rather than a
demonstrated blocker.

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

## What actually blocks core, and what does not

`brew audit` does not hard-fail on `license :cannot_represent`: it is in
`SPDX::ALLOWED_LICENSE_SYMBOLS`, and 36 formulae in `homebrew/core` use it
(`abnfgen`, `cspice`, `flite`, `ispell`, `nmap`, `ngspice`, `pymol`, `texlive`
and others). Those are custom licences that SPDX cannot express but that are
still free; being unrepresentable is not itself disqualifying.

The disqualifier is the *content* of the licence, not its representability.
Homebrew's `INCOMPATIBLE_LICENSE_PREFIXES` lists `CC-BY-NC`, annotated with
Debian's ["no commercial use" DFSG FAQ entry](https://people.debian.org/~bap/dfsg-faq.html#no_commercial).
The ONT licence has the same defect: §1.10 limits use to "internal research and
not intended for or directed towards commercial advantages or monetary
compensation". A `:cannot_represent` declaration sidesteps the automated check,
so the rejection is a maintainer decision against the written rule rather than
an audit failure, but the written rule is not ambiguous.

### "But Homebrew is free, so it is not commercial"

This is the obvious objection, and it misidentifies the party the restriction
binds. DFSG §6 constrains what the *recipient* may do, not what the
distributor charges:

> **No Discrimination Against Fields of Endeavor.** The license must not
> restrict anyone from making use of the program in a specific field of
> endeavor. For example, it may not restrict the program from being used in a
> business, or from being used for genetic research.

Debian is itself a free, non-commercial project and still refuses
non-commercial licences from `main` for exactly this reason. DFSG §1 goes
further and requires that the licence permit *selling* the software as part of
an aggregate distribution.

There is also a concrete problem rather than a doctrinal one.
`homebrew/core` builds bottles automatically and redistributes the compiled
binaries from its own infrastructure (`https://ghcr.io/v2/homebrew/core`).
Homebrew would therefore be shipping Dorado binaries to every user, including
commercial ones, which a grant limited to "internal research ... not intended
for or directed towards commercial advantages" does not cover. A commercial
nanopore user running `brew install dorado` would be outside the licence.

Notability is not a problem: at 865 stars / 128 forks, Dorado clears the
[thresholds](https://docs.brew.sh/Package-Acceptance-Policy#notability) even
for a self-submission.

macOS-only is not a problem either: `homebrew/core` carries plenty of
macOS-only formulae, so restricting the formula to Apple Silicon to avoid Koi
would be acceptable on its own.

## About a cask

Shipping the existing `dorado-<version>-osx-arm64.zip` release as a Homebrew
**cask** is the closest thing to an official `brew install`, since casks have
no source-build or licence requirement. It is plausible but not automatic:
[Acceptable Casks](https://docs.brew.sh/Acceptable-Casks) says that
"open-source command-line-only software normally belongs in `homebrew/core` as
a formula built from source" and that "a rejection from `homebrew/core` does
not by itself make the software eligible for `homebrew/cask`". The argument
for a cask here is that Dorado is not DFSG-open-source and has no practical
source-build path, which is exactly the gap casks exist to fill.

Until one of those routes is taken, a third-party tap is the correct
distribution channel.
