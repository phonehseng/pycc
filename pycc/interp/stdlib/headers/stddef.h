/* stddef.h - common definitions (pycc's simulated C library).
 *
 * size_t is the type of sizes and counts (what sizeof gives you); ptrdiff_t
 * is the type of the difference of two pointers.
 */
#ifndef CV_STDDEF_H
#define CV_STDDEF_H

#ifndef CV_SIZE_T_DEFINED
#define CV_SIZE_T_DEFINED
typedef unsigned long size_t;
#endif

typedef long ptrdiff_t;

#ifndef NULL
#define NULL ((void *)0)
#endif

/* The byte offset of a member inside a struct. */
#define offsetof(type, member) ((size_t)&((type *)0)->member)

#endif /* CV_STDDEF_H */
