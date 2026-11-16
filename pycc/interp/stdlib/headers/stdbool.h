/* stdbool.h - the bool type (pycc's simulated C library).
 *
 * C's own boolean type is spelled _Bool; this header gives it the friendlier
 * name bool, with the values true (1) and false (0).
 */
#ifndef CV_STDBOOL_H
#define CV_STDBOOL_H

#define bool _Bool
#define true 1
#define false 0
#define __bool_true_false_are_defined 1

#endif /* CV_STDBOOL_H */
