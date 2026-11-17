/* See ucrt_math.h. Resolves the math functions from ucrtbase.dll on first use. */
#if defined(_WIN32) && !defined(_MSC_VER)
#include <windows.h>

typedef double (*unary_fn)(double);
typedef float (*unary_fn_f)(float);
typedef double (*binary_fn)(double, double);

static FARPROC resolve(const char* name) {
    static HMODULE ucrt;
    if (!ucrt) ucrt = LoadLibraryA("ucrtbase.dll");
    return ucrt ? GetProcAddress(ucrt, name) : 0;
}

#define UNARY(TYPE, FN_T, NAME)                                  \
    TYPE bpp_ucrt_##NAME(TYPE x) {                               \
        static FN_T fn;                                          \
        if (!fn) fn = (FN_T)(void (*)(void))resolve(#NAME);      \
        return fn(x);                                            \
    }

UNARY(double, unary_fn, sin)
UNARY(double, unary_fn, cos)
UNARY(double, unary_fn, exp)
UNARY(double, unary_fn, log)
UNARY(double, unary_fn, asin)
UNARY(double, unary_fn, atan)
UNARY(float, unary_fn_f, sinf)
UNARY(float, unary_fn_f, cosf)

double bpp_ucrt_pow(double x, double y) {
    static binary_fn fn;
    if (!fn) fn = (binary_fn)(void (*)(void))resolve("pow");
    return fn(x, y);
}
#endif
