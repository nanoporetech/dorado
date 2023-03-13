#include <cstddef>
#include <cstdlib>
#include <iostream>

// These functions are linked into dorado and replace the default allocator with jemalloc.

extern "C" {

void *je_malloc(size_t size);
void *je_calloc(size_t num, size_t size);
int je_posix_memalign(void **memptr, size_t alignment, size_t size);
void *je_aligned_alloc(size_t alignment, size_t size);
void *je_realloc(void *ptr, size_t size);
void je_free(void *size);

void *je_mallocx(size_t size, int flags);
void *je_rallocx(void *ptr, size_t size, int flags);
size_t je_xallocx(void *ptr, size_t size, size_t extra, int flags);
size_t je_sallocx(const void *ptr, int flags);
void je_dallocx(void *ptr, int flags);
void je_sdallocx(void *ptr, size_t size, int flags);
size_t je_nallocx(size_t size, int flags);

size_t je_malloc_usable_size(void *ptr);
//size_t je_malloc_size(const void *ptr);

void *je_memalign(size_t alignment, size_t size);
void *je_valloc(size_t size);
//void *je_pvalloc(size_t size);

void *malloc(size_t size) { return je_malloc(size); }
void *calloc(size_t num, size_t size) { return je_calloc(num, size); }
int posix_memalign(void **memptr, size_t alignment, size_t size) {
    return je_posix_memalign(memptr, alignment, size);
}
void *aligned_alloc(size_t alignment, size_t size) { return je_aligned_alloc(alignment, size); }
void *realloc(void *ptr, size_t size) { return je_realloc(ptr, size); }
void free(void *ptr) { je_free(ptr); }

void *jmallocx(size_t size, int flags) { return je_mallocx(size, flags); }
void *rallocx(void *ptr, size_t size, int flags) { return je_rallocx(ptr, size, flags); }
size_t xallocx(void *ptr, size_t size, size_t extra, int flags) {
    return je_xallocx(ptr, size, extra, flags);
}
size_t sallocx(const void *ptr, int flags) { return je_sallocx(ptr, flags); }
void dallocx(void *ptr, int flags) { return je_dallocx(ptr, flags); }
void sdallocx(void *ptr, size_t size, int flags) { return je_sdallocx(ptr, size, flags); }
size_t nallocx(size_t size, int flags) { return je_nallocx(size, flags); }

size_t malloc_usable_size(void *ptr) { return je_malloc_usable_size(ptr); }
/*size_t malloc_size(const void *ptr) {
    return je_malloc_size(ptr);
}*/

void *memalign(size_t alignment, size_t size) { return je_memalign(alignment, size); }
void *valloc(size_t size) { return je_valloc(size); }
/*void *pvalloc(size_t size) {
    return je_pvalloc(size);
}*/
}