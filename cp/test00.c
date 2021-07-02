#include <stdio.h>

void cfun(const double *indatav, size_t size, double *outdatav) 
{
    size_t i;
    printf("cfun\n");
    for (i = 0; i < size; ++i)
        outdatav[i] = indatav[i] * 3.0;
}
// gcc -fPIC -shared -o test00.so test00.c