/* stdlib.h - general utilities (pycc's simulated C library).
 *
 * Heap memory, number parsing, random numbers, sorting and searching, and
 * ways to end the program.
 */
#ifndef CV_STDLIB_H
#define CV_STDLIB_H

#ifndef CV_SIZE_T_DEFINED
#define CV_SIZE_T_DEFINED
typedef unsigned long size_t;
#endif

#ifndef NULL
#define NULL ((void *)0)
#endif

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1
#define RAND_MAX 32767

/* Heap memory: every successful malloc, calloc or realloc needs one free. */
void *malloc(size_t size);
void *calloc(size_t nmemb, size_t size);
void *realloc(void *ptr, size_t size);
void free(void *ptr);

/* Text to numbers. */
int atoi(const char *nptr);
long atol(const char *nptr);
double atof(const char *nptr);
long strtol(const char *nptr, char **endptr, int base);
double strtod(const char *nptr, char **endptr);

/* Integer helpers. */
int abs(int j);
long labs(long j);

/* Pseudo-random numbers: the same seed always gives the same sequence. */
int rand(void);
void srand(unsigned int seed);

/* Ending the program. */
void exit(int status);
void abort(void);

/* Sorting and searching arrays of any element type. */
void qsort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *));
void *bsearch(const void *key, const void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *));

/* The environment. */
char *getenv(const char *name);
int system(const char *command);

#endif /* CV_STDLIB_H */
