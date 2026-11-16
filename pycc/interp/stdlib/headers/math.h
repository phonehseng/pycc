/* math.h - floating-point mathematics (pycc's simulated C library).
 *
 * Every function takes and returns double. Impossible requests such as
 * sqrt(-1.0) return NaN ("not a number") and set errno to EDOM; results too
 * big to represent return infinity and set errno to ERANGE.
 */
#ifndef CV_MATH_H
#define CV_MATH_H

#define M_PI 3.14159265358979323846
#define M_E 2.7182818284590452354

double sqrt(double x);
double pow(double x, double y);
double fabs(double x);
double floor(double x);
double ceil(double x);
double round(double x);
double sin(double x);
double cos(double x);
double tan(double x);
double atan2(double y, double x);
double fmod(double x, double y);
double exp(double x);
double log(double x);

#endif /* CV_MATH_H */
