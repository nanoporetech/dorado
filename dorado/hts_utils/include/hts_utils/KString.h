#pragma once

#include <htslib/sam.h>

#include <cstddef>
#include <memory>

struct kstring_t;

namespace dorado {

/// Wrapper for htslib kstring_t struct.
class KString {
public:
    /** Contains an uninitialized kstring_t.
         *  Useful for later assigning to a kstring_t returned by an htslib
         *  API function. If you need to pass object to an API function which
         *  will put data in it, then use the pre-allocate constructor instead,
         *  to avoid library conflicts on windows.
         */
    KString();

    /** Pre-allocate the string with n bytes of storage. If you pass the kstring
         *  into an htslib API function that would resize the kstring_t object as
         *  needed, when the API function does the resize this can result in
         *  strange errors like stack corruption due to differences between the
         *  implementation in the compiled library, and the implementation compiled
         *  into your C++ code using htslib macros. So make sure you pre-allocate
         *  with enough memory to insure that no resizing will be needed.
         */
    KString(size_t n);

    /** This object owns the storage in the internal kstring_t object.
         *  To avoid reference counting, we don't allow this object to be copied.
         */
    KString(const KString &) = delete;

    /** Take ownership of the data in the kstring_t object.
         *  Note that it is an error to create more than one KString object
         *  that owns the same kstring_t data.
         */
    KString(kstring_t &&data) noexcept;

    /// Move Constructor
    KString(KString &&other) noexcept;

    /// No copying allowed.
    KString &operator=(const KString &) = delete;

    /// Move assignment.
    KString &operator=(KString &&rhs) noexcept;

    /// Destroys the kstring_t data.
    ~KString();

    /// Returns the kstring_t object that points to the internal data.
    kstring_t &get() const;

private:
    std::unique_ptr<kstring_t> m_data;
};

}  // namespace dorado