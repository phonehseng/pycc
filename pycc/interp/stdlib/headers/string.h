/* string.h - C strings and raw memory (pycc's simulated C library).
 *
 * A C string is an array of char that ends with a '\0' byte. None of these
 * functions know how big your arrays are: making room is your job.
 */
#ifndef CV_STRING_H
#define CV_STRING_H

#ifndef CV_SIZE_T_DEFINED
#define CV_SIZE_T_DEFINED
typedef unsigned long size_t;
#endif

#ifndef NULL
#define NULL ((void *)0)
#endif

/* Length, copying and joining. */
size_t strlen(const char *s);
char *strcpy(char *dest, const char *src);
char *strncpy(char *dest, const char *src, size_t n);
char *strcat(char *dest, const char *src);
char *strncat(char *dest, const char *src, size_t n);
char *strdup(const char *s);

/* Comparing and searching. */
int strcmp(const char *s1, const char *s2);
int strncmp(const char *s1, const char *s2, size_t n);
char *strchr(const char *s, int c);
char *strrchr(const char *s, int c);
char *strstr(const char *haystack, const char *needle);
size_t strspn(const char *s, const char *accept);
size_t strcspn(const char *s, const char *reject);
char *strtok(char *str, const char *delim);

/* Raw bytes. */
void *memset(void *s, int c, size_t n);
void *memcpy(void *dest, const void *src, size_t n);
void *memmove(void *dest, const void *src, size_t n);
int memcmp(const void *s1, const void *s2, size_t n);

/* The text for an errno value. */
char *strerror(int errnum);

#endif /* CV_STRING_H */
