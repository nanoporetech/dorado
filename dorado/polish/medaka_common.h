#ifndef _MEDAKA_COMMON_H
#define _MEDAKA_COMMON_H

#include <stdint.h>

/** Simple integer min/max
 * @param a
 * @param b
 *
 * @returns the min/max of a and b
 *
 */
static inline int max(int a, int b) { return a > b ? a : b; }
static inline int min(int a, int b) { return a < b ? a : b; }

/** Allocates zero-initialised memory with a message on failure.
 *
 *  @param num number of elements to allocate.
 *  @param size size of each element.
 *  @param msg message to describe allocation on failure.
 *  @returns pointer to allocated memory
 *
 */
void *xalloc(size_t num, size_t size, const char *msg);

/** Reallocates memory with a message on failure.
 *
 *  @param ptr pointer to realloc.
 *  @param size size of each element.
 *  @param msg message to describe allocation on failure.
 *  @returns pointer to allocated memory
 *
 */
void *xrealloc(void *ptr, size_t size, const char *msg);

/** Format a uint32_t to a string
 *
 * @param value to format.
 * @param dst destination char.
 * @returns length of string.
 *
 */
size_t uint8_to_str(uint8_t value, char *dst);

#endif
