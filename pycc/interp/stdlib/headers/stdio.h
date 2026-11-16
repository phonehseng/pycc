/* stdio.h - standard input and output (pycc's simulated C library).
 *
 * Everything that reads or writes text lives here: the terminal (stdin,
 * stdout, stderr) and files opened with fopen. A FILE is an opaque handle:
 * your program only ever holds a pointer to one.
 */
#ifndef CV_STDIO_H
#define CV_STDIO_H

#ifndef CV_SIZE_T_DEFINED
#define CV_SIZE_T_DEFINED
typedef unsigned long size_t;
#endif

#ifndef NULL
#define NULL ((void *)0)
#endif

/* Returned by the reading functions at end of file (and on errors). */
#define EOF (-1)
#define BUFSIZ 8192

typedef struct _CVFILE { int _id; } FILE;

extern FILE *stdin;
extern FILE *stdout;
extern FILE *stderr;

/* Formatted output. */
int printf(const char *format, ...);
int fprintf(FILE *stream, const char *format, ...);
int sprintf(char *str, const char *format, ...);
int snprintf(char *str, size_t size, const char *format, ...);

/* Character and line output. */
int puts(const char *s);
int putchar(int c);
int fputs(const char *s, FILE *stream);
int fputc(int c, FILE *stream);
int putc(int c, FILE *stream);

/* Formatted input. */
int scanf(const char *format, ...);
int fscanf(FILE *stream, const char *format, ...);
int sscanf(const char *str, const char *format, ...);

/* Character and line input. */
char *fgets(char *s, int size, FILE *stream);
int fgetc(FILE *stream);
int getc(FILE *stream);
int getchar(void);
int ungetc(int c, FILE *stream);

/* Files. */
FILE *fopen(const char *pathname, const char *mode);
int fclose(FILE *stream);
size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream);
int feof(FILE *stream);
int ferror(FILE *stream);
void clearerr(FILE *stream);
int fflush(FILE *stream);
int remove(const char *pathname);
int rename(const char *oldpath, const char *newpath);
void perror(const char *s);

#endif /* CV_STDIO_H */
