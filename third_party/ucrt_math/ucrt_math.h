/* Force-included into libsoxr when building with MinGW on Windows.
 *
 * The python-soxr wheels librosa uses on Windows are built with MSVC, so libsoxr's filter design
 * and FFT twiddles call the Universal CRT's math functions. MinGW links its own libm instead,
 * whose results occasionally round differently, which perturbs the resampled audio by an ulp.
 * Redirecting these calls to ucrtbase.dll keeps the resampler bit-identical to the wheel. */
#ifndef BPP_UCRT_MATH_H
#define BPP_UCRT_MATH_H
#if defined(_WIN32) && !defined(_MSC_VER)
#include <math.h>
double bpp_ucrt_sin(double), bpp_ucrt_cos(double), bpp_ucrt_exp(double), bpp_ucrt_log(double);
double bpp_ucrt_pow(double, double), bpp_ucrt_asin(double), bpp_ucrt_atan(double);
float bpp_ucrt_sinf(float), bpp_ucrt_cosf(float);
#define sin bpp_ucrt_sin
#define cos bpp_ucrt_cos
#define exp bpp_ucrt_exp
#define log bpp_ucrt_log
#define pow bpp_ucrt_pow
#define asin bpp_ucrt_asin
#define atan bpp_ucrt_atan
#define sinf bpp_ucrt_sinf /* pffft's float build maps sin/cos to sinf/cosf */
#define cosf bpp_ucrt_cosf
#endif
#endif
