/* limits.h - the range of each integer type (pycc's simulated C library).
 *
 * Going past these values wraps around. The numbers are those of 64-bit
 * Linux: char is 8 bits and signed, int is 32 bits, long is 64 bits.
 */
#ifndef CV_LIMITS_H
#define CV_LIMITS_H

#define CHAR_BIT 8
#define SCHAR_MIN (-128)
#define SCHAR_MAX 127
#define UCHAR_MAX 255
#define CHAR_MIN SCHAR_MIN
#define CHAR_MAX SCHAR_MAX
#define SHRT_MIN (-32767 - 1)
#define SHRT_MAX 32767
#define USHRT_MAX 65535
/* Written as a subtraction because 2147483648 alone does not fit in an int. */
#define INT_MIN (-2147483647 - 1)
#define INT_MAX 2147483647
#define UINT_MAX 4294967295U
#define LONG_MIN (-9223372036854775807L - 1L)
#define LONG_MAX 9223372036854775807L
#define ULONG_MAX 18446744073709551615UL
#define LLONG_MIN (-9223372036854775807LL - 1LL)
#define LLONG_MAX 9223372036854775807LL
#define ULLONG_MAX 18446744073709551615ULL

#endif /* CV_LIMITS_H */
