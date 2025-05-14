/* The MIT License

   Copyright (c) 2008, 2011 Attractive Chaos <attractor@live.co.uk>

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

/*
  2011-04-10 (0.1.6):

  	* Added sample

  2011-03 (0.1.5):

	* Added shuffle/permutation

  2008-11-16 (0.1.4):

    * Fixed a bug in introsort() that happens in rare cases.

  2008-11-05 (0.1.3):

    * Fixed a bug in introsort() for complex comparisons.

	* Fixed a bug in mergesort(). The previous version is not stable.

  2008-09-15 (0.1.2):

	* Accelerated introsort. On my Mac (not on another Linux machine),
	  my implementation is as fast as std::sort on random input.

	* Added combsort and in introsort, switch to combsort if the
	  recursion is too deep.

  2008-09-13 (0.1.1):

	* Added k-small algorithm

  2008-09-05 (0.1.0):

	* Initial version

*/

#ifndef AC_KSORT_H
#define AC_KSORT_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

namespace kadayashi {

// The ksort uses POSIX rand48() which is not available on Windows.
#ifndef _MSC_VER
inline double drand48_wrap() { return drand48(); }
#else
inline double drand48_wrap() { return (static_cast<double>(rand()) / RAND_MAX); }
#endif

typedef struct {
    void *left, *right;
    int depth;
} ks_isort_stack_t;

#define KSORT_SWAP(type_t, a, b) \
    {                            \
        type_t t = (a);          \
        (a) = (b);               \
        (b) = t;                 \
    }

#define KSORT_INIT(name, type_t, __sort_lt)                                          \
    void ks_mergesort_##name(size_t n, type_t array[], type_t temp[]) {              \
        type_t *a2[2], *a, *b;                                                       \
        int64_t curr, shift;                                                         \
                                                                                     \
        a2[0] = array;                                                               \
        a2[1] = temp ? temp : (type_t *)malloc(sizeof(type_t) * n);                  \
        for (curr = 0, shift = 0; (((size_t)(1)) << shift) < n; ++shift) {           \
            a = a2[curr];                                                            \
            b = a2[1 - curr];                                                        \
            if (shift == 0) {                                                        \
                type_t *p = b, *i, *eb = a + n;                                      \
                for (i = a; i < eb; i += 2) {                                        \
                    if (i == eb - 1)                                                 \
                        *p++ = *i;                                                   \
                    else {                                                           \
                        if (__sort_lt(*(i + 1), *i)) {                               \
                            *p++ = *(i + 1);                                         \
                            *p++ = *i;                                               \
                        } else {                                                     \
                            *p++ = *i;                                               \
                            *p++ = *(i + 1);                                         \
                        }                                                            \
                    }                                                                \
                }                                                                    \
            } else {                                                                 \
                size_t i, step = ((size_t)(1)) << shift;                             \
                for (i = 0; i < n; i += step << 1) {                                 \
                    type_t *p, *j, *k, *ea, *eb;                                     \
                    if (n < i + step) {                                              \
                        ea = a + n;                                                  \
                        eb = a;                                                      \
                    } else {                                                         \
                        ea = a + i + step;                                           \
                        eb = a + (n < i + (step << 1) ? n : i + (step << 1));        \
                    }                                                                \
                    j = a + i;                                                       \
                    k = a + i + step;                                                \
                    p = b + i;                                                       \
                    while (j < ea && k < eb) {                                       \
                        if (__sort_lt(*k, *j))                                       \
                            *p++ = *k++;                                             \
                        else                                                         \
                            *p++ = *j++;                                             \
                    }                                                                \
                    while (j < ea)                                                   \
                        *p++ = *j++;                                                 \
                    while (k < eb)                                                   \
                        *p++ = *k++;                                                 \
                }                                                                    \
            }                                                                        \
            curr = 1 - curr;                                                         \
        }                                                                            \
        if (curr == 1) {                                                             \
            type_t *p = a2[0], *i = a2[1], *eb = array + n;                          \
            for (; p < eb; ++i)                                                      \
                *p++ = *i;                                                           \
        }                                                                            \
        if (temp == 0)                                                               \
            free(a2[1]);                                                             \
    }                                                                                \
    /* This function is adapted from: http://ndevilla.free.fr/median/ */             \
    /* 0 <= kk < n */                                                                \
    type_t ks_ksmall_##name(size_t n, type_t arr[], size_t kk) {                     \
        type_t *low, *high, *k, *ll, *hh, *mid;                                      \
        low = arr;                                                                   \
        high = arr + n - 1;                                                          \
        k = arr + kk;                                                                \
        for (;;) {                                                                   \
            if (high <= low)                                                         \
                return *k;                                                           \
            if (high == low + 1) {                                                   \
                if (__sort_lt(*high, *low))                                          \
                    KSORT_SWAP(type_t, *low, *high);                                 \
                return *k;                                                           \
            }                                                                        \
            mid = low + (high - low) / 2;                                            \
            if (__sort_lt(*high, *mid))                                              \
                KSORT_SWAP(type_t, *mid, *high);                                     \
            if (__sort_lt(*high, *low))                                              \
                KSORT_SWAP(type_t, *low, *high);                                     \
            if (__sort_lt(*low, *mid))                                               \
                KSORT_SWAP(type_t, *mid, *low);                                      \
            KSORT_SWAP(type_t, *mid, *(low + 1));                                    \
            ll = low + 1;                                                            \
            hh = high;                                                               \
            for (;;) {                                                               \
                do                                                                   \
                    ++ll;                                                            \
                while (__sort_lt(*ll, *low));                                        \
                do                                                                   \
                    --hh;                                                            \
                while (__sort_lt(*low, *hh));                                        \
                if (hh < ll)                                                         \
                    break;                                                           \
                KSORT_SWAP(type_t, *ll, *hh);                                        \
            }                                                                        \
            KSORT_SWAP(type_t, *low, *hh);                                           \
            if (hh <= k)                                                             \
                low = ll;                                                            \
            if (hh >= k)                                                             \
                high = hh - 1;                                                       \
        }                                                                            \
    }                                                                                \
    void ks_shuffle_##name(size_t n, type_t a[]) {                                   \
        int i, j;                                                                    \
        for (i = ((int)n); i > 1; --i) {                                             \
            type_t tmp;                                                              \
            j = (int)(drand48_wrap() * i);                                           \
            tmp = a[j];                                                              \
            a[j] = a[i - 1];                                                         \
            a[i - 1] = tmp;                                                          \
        }                                                                            \
    }                                                                                \
    void ks_sample_##name(size_t n, size_t r, type_t a[]) /* FIXME: NOT TESTED!!! */ \
    { /* reference: http://code.activestate.com/recipes/272884/ */                   \
        int i;                                                                       \
        size_t k, pop = n;                                                           \
        for (i = (int)r, k = 0; i >= 0; --i) {                                       \
            double z = 1., x = drand48_wrap();                                       \
            type_t tmp;                                                              \
            while (x < z)                                                            \
                z -= z * i / (pop--);                                                \
            if (k != n - pop - 1)                                                    \
                tmp = a[k], a[k] = a[n - pop - 1], a[n - pop - 1] = tmp;             \
            ++k;                                                                     \
        }                                                                            \
    }

#define ks_mergesort(name, n, a, t) ks_mergesort_##name(n, a, t)
#define ks_introsort(name, n, a) ks_introsort_##name(n, a)
#define ks_combsort(name, n, a) ks_combsort_##name(n, a)
#define ks_heapsort(name, n, a) ks_heapsort_##name(n, a)
#define ks_heapmake(name, n, a) ks_heapmake_##name(n, a)
#define ks_heapadjust(name, i, n, a) ks_heapadjust_##name(i, n, a)
#define ks_ksmall(name, n, a, k) ks_ksmall_##name(n, a, k)
#define ks_shuffle(name, n, a) ks_shuffle_##name(n, a)

#define ks_lt_generic(a, b) ((a) < (b))
#define ks_lt_str(a, b) (strcmp((a), (b)) < 0)

typedef const char *ksstr_t;

#define KSORT_INIT_GENERIC(type_t) KSORT_INIT(type_t, type_t, ks_lt_generic)
#define KSORT_INIT_STR KSORT_INIT(str, ksstr_t, ks_lt_str)

#define RS_MIN_SIZE 64
#define RS_MAX_BITS 8

#define KRADIX_SORT_INIT(name, rstype_t, rskey, sizeof_key)                        \
    typedef struct {                                                               \
        rstype_t *b, *e;                                                           \
    } rsbucket_##name##_t;                                                         \
    void rs_insertsort_##name(rstype_t *beg, rstype_t *end) {                      \
        rstype_t *i;                                                               \
        for (i = beg + 1; i < end; ++i)                                            \
            if (rskey(*i) < rskey(*(i - 1))) {                                     \
                rstype_t *j, tmp = *i;                                             \
                for (j = i; j > beg && rskey(tmp) < rskey(*(j - 1)); --j)          \
                    *j = *(j - 1);                                                 \
                *j = tmp;                                                          \
            }                                                                      \
    }                                                                              \
    void rs_sort_##name(rstype_t *beg, rstype_t *end, int n_bits, int s) {         \
        rstype_t *i;                                                               \
        int size = 1 << n_bits, m = size - 1;                                      \
        rsbucket_##name##_t *k, b[1 << RS_MAX_BITS], *be = b + size;               \
        assert(n_bits <= RS_MAX_BITS);                                             \
        for (k = b; k != be; ++k)                                                  \
            k->b = k->e = beg;                                                     \
        for (i = beg; i != end; ++i)                                               \
            ++b[rskey(*i) >> s & m].e;                                             \
        for (k = b + 1; k != be; ++k)                                              \
            k->e += (k - 1)->e - beg, k->b = (k - 1)->e;                           \
        for (k = b; k != be;) {                                                    \
            if (k->b != k->e) {                                                    \
                rsbucket_##name##_t *l;                                            \
                if ((l = b + (rskey(*k->b) >> s & m)) != k) {                      \
                    rstype_t tmp = *k->b, swap;                                    \
                    do {                                                           \
                        swap = tmp;                                                \
                        tmp = *l->b;                                               \
                        *l->b++ = swap;                                            \
                        l = b + (rskey(tmp) >> s & m);                             \
                    } while (l != k);                                              \
                    *k->b++ = tmp;                                                 \
                } else                                                             \
                    ++k->b;                                                        \
            } else                                                                 \
                ++k;                                                               \
        }                                                                          \
        for (b->b = beg, k = b + 1; k != be; ++k)                                  \
            k->b = (k - 1)->e;                                                     \
        if (s) {                                                                   \
            s = s > n_bits ? s - n_bits : 0;                                       \
            for (k = b; k != be; ++k)                                              \
                if (k->e - k->b > RS_MIN_SIZE)                                     \
                    rs_sort_##name(k->b, k->e, n_bits, s);                         \
                else if (k->e - k->b > 1)                                          \
                    rs_insertsort_##name(k->b, k->e);                              \
        }                                                                          \
    }                                                                              \
    void radix_sort_##name(rstype_t *beg, rstype_t *end) {                         \
        if (end - beg <= RS_MIN_SIZE)                                              \
            rs_insertsort_##name(beg, end);                                        \
        else                                                                       \
            rs_sort_##name(beg, end, RS_MAX_BITS, (sizeof_key - 1) * RS_MAX_BITS); \
    }

}  // namespace kadayashi

#endif
