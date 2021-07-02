#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <complex.h>

#include "fft-complex.h"
#include "test_f.h"


void wrapper_lv(const double *conf, size_t size, const double *indatav, double *out_abs_v, double *out_arg_v)
{
    
}

void cfun(const double *indatav, size_t size, double *outdatav)
{
    size_t i;
    printf("cfun\n");
    for (i = 0; i < size; ++i)
        outdatav[i] = indatav[i] * 3.0;
}

void spline_c(size_t x_n, const double *x, const double *y, double *sp_b, double *sp_c, double *sp_d)
{

    //--------------------------------------------------------------------------
    // Spline from
    // https://fac.ksu.edu.sa/sites/default/files/numerical_analysis_9th.pdf//page=167
    //--------------------------------------------------------------------------

    double *h = malloc((x_n - 1) * sizeof(double));
    double *a = malloc((x_n - 1) * sizeof(double));
    double *l = malloc((x_n) * sizeof(double));
    double *u = malloc((x_n) * sizeof(double));
    double *z = malloc((x_n) * sizeof(double));

    if ((h == NULL) || (a == NULL) || (l == NULL) || (u == NULL) || (z == NULL))
    {
        printf("malloc failed\n");
    }

    //--------------------------------------------------------------------------
    // Calculate h = dx between samples
    // Can be omitted if h = 1
    //--------------------------------------------------------------------------
    //for x_i in np.arange(0,x_n-1):
    for (int x_i = 0; x_i < x_n - 1; x_i++)
    {
        h[x_i] = x[x_i + 1] - x[x_i];
    }

    //--------------------------------------------------------------------------
    // A, sparse matrix
    //--------------------------------------------------------------------------
    //for x_i in np.arange(1,x_n-1):
    for (int x_i = 1; x_i < x_n - 1; x_i++)
    {
        a[x_i] = 3.0 * (y[x_i + 1] - y[x_i]) / h[x_i] - 3.0 * (y[x_i] - y[x_i - 1]) / h[x_i - 1];
    }

    //--------------------------------------------------------------------------
    // l, u, z
    //--------------------------------------------------------------------------
    l[0] = 1;
    u[0] = 0;
    z[0] = 0;

    //for x_i in np.arange(1, x_n-1):
    for (int x_i = 1; x_i < x_n - 1; x_i++)
    {
        l[x_i] = 2.0 * (x[x_i + 1] - x[x_i - 1]) - h[x_i - 1] * u[x_i - 1];
        u[x_i] = h[x_i] / l[x_i];
        z[x_i] = (a[x_i] - h[x_i - 1] * z[x_i - 1]) / l[x_i];
    }

    l[x_n - 1] = 1;
    z[x_n - 1] = 0;
    sp_c[x_n - 1] = 0;

    //--------------------------------------------------------------------------
    // b, c, d
    // y(x) = yi + bi*(x-xi) + ci*(x-xi)^2 + di*(x-xi)^3
    //--------------------------------------------------------------------------

    // for x_i in np.arange(x_n-2, -1, -1):  //x_n-1:-1:1
    for (int x_i = x_n - 2; x_i > -1; x_i--)
    {
        sp_c[x_i] = z[x_i] - u[x_i] * sp_c[x_i + 1];
        sp_b[x_i] = (y[x_i + 1] - y[x_i]) / h[x_i] - h[x_i] *
                                                         (sp_c[x_i + 1] + 2.0 * sp_c[x_i]) / 3.0;
        sp_d[x_i] = (sp_c[x_i + 1] - sp_c[x_i]) / 3.0 / h[x_i];
    }

    free(h);
    free(a);
    free(l);
    free(u);
    free(z);
}

void interpolate_c(const double *sp_b, const double *sp_c, const double *sp_d,
                   size_t x_n, const double *x_v, const double *y_v, size_t xx_n, const double *xx_v, double *yy_v)
{
    //interpolate(sp_b, sp_c, sp_d, x_n, x_v, y_v, xx_n, xx_v):
    int x0_i = 0;
    //for xx_i in np.arange(0, xx_n):
    for (int xx_i = 0; xx_i < xx_n; xx_i++)
    {
        double x = xx_v[xx_i];
        //  % Search, but can be calculated if h = 1
        // for x_i in np.arange(x0_i, x_n):
        int x_i;
        for (x_i = x0_i; x_i < x_n; x_i++)
        {
            // if x_v[x_i] > x:
            if (x_v[x_i] > x)
            {
                x_i = x_i - 1;
                x0_i = x_i;
                break;
            }
        }
        double xd = x - x_v[x_i];
        double xd2 = xd * xd;
        double xd3 = xd2 * xd;

        //#%    disp(sprintf('%f, %f, %f, %d', x_i, x, x_v(x_i), xd))
        yy_v[xx_i] = y_v[x_i] + sp_b[x_i] * xd + sp_c[x_i] * xd2 + sp_d[x_i] * xd3;
    }
}

void movemean_c(size_t x_n, const double *x_v, size_t b_n, double *x2_v)
{
    //def movemean(x_n, x_v, b_n):

    double *b_v = calloc(b_n, sizeof(double));

    int b2_n = (b_n - 1) / 2;
    double avg = 0.0;

    //for b_i in np.arange(b_n-1):
    for (int b_i = 0; b_i < b_n - 1; b_i++)
    {
        int i_i = b_i - b2_n;
        //if i_i < 0:
        if (i_i < 0)
        {
            b_v[b_i] = x_v[0] / b_n;
        }
        else
        {
            b_v[b_i] = x_v[i_i] / b_n;
        }
        avg += b_v[b_i];
    }

    b_v[b_n - 1] = 0.0;
    int b_i = b_n - 1;
    //for x_i in np.arange(x_n):
    for (int x_i = 0; x_i < x_n; x_i++)
    {
        int i_i = x_i + b2_n;
        if (i_i >= x_n)
        {
            i_i = x_n - 1;
        }

        double last = b_v[b_i];
        b_v[b_i] = x_v[i_i] / b_n;
        //# print('avg', avg, -last, b_v[b_i])
        avg += -last + b_v[b_i];
        x2_v[x_i] = avg;
        ++b_i;
        if (b_i >= b_n)
            b_i = 0;
    }

    free(b_v);
}

int signd(double x)
{
    if (x > 0.0)
        return 1;
    if (x < 0.0)
        return -1;
    return 0;
}

double absd(double x)
{
    if (x >= 0.0)
        return x;
    else
        return -x;
}

void find_frequency_c(size_t x_n, const double *x_v, const double *y_v,
                      const double *sp_b, const double *sp_c, int conf_sample_buffer, int conf_signal_period_n, double conf_ds,
                      double *fn, double *fp, double *favg, double *xzcp_v, double *xzcn_v)
{
    //def find_frequency(x_n, x_v, y_v, sp_b, sp_c, conf_sample_buffer,
    //               conf_signal_period_n, conf_ds):
    //--------------------------------------------------------------------------
    // Search for sign changes of signal, scp, scn
    //--------------------------------------------------------------------------
    //scp_iv = np.zeros(x_n, dtype=int)
    //scn_iv = np.zeros(x_n, dtype=int)
    int *scp_iv = calloc(x_n, sizeof(int));
    int *scn_iv = calloc(x_n, sizeof(int));

    int scp_i = 0;
    int scn_i = 0;

    //   for x_i in np.arange(conf_sample_buffer, x_n-1):
    for (int x_i = conf_sample_buffer; x_i < x_n - 1; x_i++)
    {

        if (signd(y_v[x_i]) != signd(y_v[x_i + 1]) && signd(y_v[x_i]) != 0)
        {
            if (signd(y_v[x_i]) < 0)
            {
                if (absd(y_v[x_i]) < absd(y_v[x_i + 1]))
                    scp_iv[scp_i] = x_i;
                else
                    scp_iv[scp_i] = x_i + 1;
                scp_i += 1;
            }
            else
            {
                if (absd(y_v[x_i]) < absd(y_v[x_i + 1]))
                    scn_iv[scn_i] = x_i;
                else
                    scn_iv[scn_i] = x_i + 1;
                scn_i += 1;
            }
        }
    }

    //for i_i in np.arange(0, conf_signal_period_n+1):
    for (int i_i = 0; i_i < conf_signal_period_n + 1; i_i++)
    {
        int x_i = scn_iv[i_i];
        xzcn_v[i_i] = x_v[x_i] + (-sp_b[x_i] -
                                  sqrt(sp_b[x_i] * sp_b[x_i] - 4.0 * y_v[x_i] * sp_c[x_i])) /
                                     2.0 / sp_c[x_i];

        x_i = scp_iv[i_i];
        xzcp_v[i_i] = x_v[x_i] + (-sp_b[x_i] +
                                  sqrt(sp_b[x_i] * sp_b[x_i] - 4.0 * y_v[x_i] * sp_c[x_i])) /
                                     2.0 / sp_c[x_i];
    }

    *fn = 1.0 / ((xzcn_v[conf_signal_period_n] - xzcn_v[0]) / (conf_signal_period_n)*conf_ds);
    *fp = 1.0 / ((xzcp_v[conf_signal_period_n] - xzcp_v[0]) / (conf_signal_period_n)*conf_ds);
    *favg = (*fn + *fp) / 2;

    free(scp_iv);
    free(scn_iv);
}

//def fft_aa(fft_y_v, conf_fft_n):
// void fft_aa_c(const double *x_v, size_t conf_fft_n, double *freq_sabs_v, double *freq_sangle_v)
// {
//     double *freq_dc_v = malloc((conf_fft_n) * sizeof(double));
//     // double sided complex
//     freq_dc_v = scipy.fft.fft(fft_y_v);
    
//     freq_dabs_v = np.abs(freq_dc_v/conf_fft_n);

//     // single sided angle/amplitude
//     freq_sangle_v = np.angle(freq_dc_v[0:int(conf_fft_n/2)])
//     freq_sabs_v = freq_dabs_v[0:int(conf_fft_n/2)]
//     // correct amplitude
//     freq_sabs_v[1:-2] = 2*freq_sabs_v[1:-2]
    
//     free(freq_dc_v);
// }

// double abs_test(double complex a) 
// {
//     return cabs(a);
// }

void fft_test_c(const double *in_v, size_t n, double *out_abs_v, double *out_arg_v)
{
    double complex *actual = malloc(n * sizeof(double complex));
    for (size_t i = 0; i < n; i++)
    {
        actual[i] = in_v[i];
    }
    
    Fft_transform(actual, n, false);
    for (size_t i = 0; i < n; i++)
    {
        out_abs_v[i] = abs_test(actual[i]);
        out_arg_v[i] = carg(actual[i]);
    }
    free(actual);
}
// gcc -fPIC -shared -o test10.so test10.c