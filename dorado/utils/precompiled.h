#ifdef _WIN32
// Unreachable code warnings are emitted from Torch's Optional class, even though they should be disabled by the
// MSVC /external:W0 setting.  This is a limitation of /external: for some C47XX backend warnings.  See:
// https://learn.microsoft.com/en-us/cpp/build/reference/external-external-headers-diagnostics?view=msvc-170#limitations
#pragma warning(push)
#pragma warning(disable : 4702)
#endif  // _WIN32
#include <ATen/ATen.h>
#ifdef _WIN32
#pragma warning(pop)
#endif  // _WIN32