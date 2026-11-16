/* errno.h - error numbers (pycc's simulated C library).
 *
 * When a library function fails it stores a code in errno. The value only
 * means something right after a call that reported failure; strerror and
 * perror turn it into words.
 */
#ifndef CV_ERRNO_H
#define CV_ERRNO_H

extern int errno;

#define ENOENT 2
#define EIO 5
#define EBADF 9
#define ENOMEM 12
#define EACCES 13
#define EEXIST 17
#define ENOTDIR 20
#define EISDIR 21
#define EINVAL 22
#define EDOM 33
#define ERANGE 34

#endif /* CV_ERRNO_H */
