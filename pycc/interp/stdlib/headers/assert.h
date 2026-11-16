/* assert.h - checking what must be true (pycc's simulated C library).
 *
 * assert(expression) stops the program with a message when the expression
 * is false. Defining NDEBUG before the #include turns every assert off.
 *
 * How it works here: the preprocessor sees __CV_ASSERT__ and rewrites each
 * assert(e) into a call of __cv_assert with the text of e, the file, the line
 * and the function name, so the message can quote exactly what you wrote.
 */
#ifndef CV_ASSERT_H
#define CV_ASSERT_H

#ifdef NDEBUG
#define assert(expression) ((void)0)
#else
#define __CV_ASSERT__ 1
#endif

void __cv_assert(int ok, const char *expr, const char *file, int line, const char *func);

#endif /* CV_ASSERT_H */
