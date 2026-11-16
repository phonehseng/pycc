/* time.h - time and clocks (pycc's simulated C library).
 *
 * time() counts the attached device's ticks (0 when no device is attached) and
 * clock() counts the statements your program has executed so far.
 */
#ifndef CV_TIME_H
#define CV_TIME_H

#ifndef NULL
#define NULL ((void *)0)
#endif

typedef long time_t;
typedef long clock_t;

#define CLOCKS_PER_SEC 1000000L

time_t time(time_t *tloc);
clock_t clock(void);

#endif /* CV_TIME_H */
