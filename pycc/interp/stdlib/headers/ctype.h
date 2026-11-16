/* ctype.h - classifying and converting characters (pycc's simulated C library).
 *
 * Each "is" function (isalpha, isdigit...) answers non-zero for true and 0 for
 * false. They take an int so that EOF can be passed in too.
 */
#ifndef CV_CTYPE_H
#define CV_CTYPE_H

int isalpha(int c);
int isdigit(int c);
int isalnum(int c);
int isspace(int c);
int isupper(int c);
int islower(int c);
int ispunct(int c);
int isxdigit(int c);
int isprint(int c);
int toupper(int c);
int tolower(int c);

#endif /* CV_CTYPE_H */
