# %%
import ctypes
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
import pandas as pd

# import os
# os.chdir('c:/Users/niklaser2/Documents/repo/c/cp/')
# import test01

# lib = ctypes.cdll.LoadLibrary('./test10.so')
# spline_c = lib.spline_c
# interpolate_c = lib.interpolate_c
# movemean_c = lib.movemean_c
# find_frequency_c = lib.find_frequency_c
# fft_test_c = lib.fft_test_c

def fft_test_cf(x_n, x):
    
    ab = np.zeros(x_n);
    ar = np.zeros(x_n);

    t0 = time.perf_counter()
    fft_test_c( ctypes.c_void_p(x.ctypes.data), x_n, ctypes.c_void_p(ab.ctypes.data), ctypes.c_void_p(ar.ctypes.data))
    t1 = time.perf_counter()
    print(f'fft_test_cf: {(t1-t0)*1e3:2f} ms')
    return ab, ar

def spline_cf(x_n, x, y):
    
    sp_b = np.zeros(x_n);
    sp_c = np.zeros(x_n);
    sp_d = np.zeros(x_n);

    t0 = time.perf_counter()
    spline_c(x_n, ctypes.c_void_p(x.ctypes.data), ctypes.c_void_p(y.ctypes.data),
    ctypes.c_void_p(sp_b.ctypes.data), ctypes.c_void_p(sp_c.ctypes.data), ctypes.c_void_p(sp_d.ctypes.data))
    t1 = time.perf_counter()
    print(f'spline_cf: {(t1-t0)*1e3:2f} ms')
    return sp_b, sp_c, sp_d

def interpolate_cf(sp_b, sp_c, sp_d, x_n, x_v, y_v, xx_n, xx_v):
    
    yy_v = np.zeros(xx_n)

    t0 = time.perf_counter()
    interpolate_c(ctypes.c_void_p(sp_b.ctypes.data), ctypes.c_void_p(sp_c.ctypes.data), ctypes.c_void_p(sp_d.ctypes.data),
    x_n, ctypes.c_void_p(x_v.ctypes.data), ctypes.c_void_p(y_v.ctypes.data), xx_n, ctypes.c_void_p(xx_v.ctypes.data), ctypes.c_void_p(yy_v.ctypes.data))
    t1 = time.perf_counter()
    print(f'interpolate_cf: {(t1-t0)*1e3:2f} ms')
    return yy_v

def movemean_cf(x_n, x_v, b_n):
    
    x2_v = np.zeros(x_n)

    t0 = time.perf_counter()
    movemean_c(x_n, ctypes.c_void_p(x_v.ctypes.data), b_n, ctypes.c_void_p(x2_v.ctypes.data))
    t1 = time.perf_counter()
    print(f'movemean_cf: {(t1-t0)*1e3:2f} ms')
    
    return x2_v

def find_frequency_cf(x_n, x_v, y_v, sp_b, sp_c, conf_sample_buffer,
                   conf_signal_period_n, conf_ds):
    
    # void find_frequency_c(size_t x_n, const double *x_v, const double *y_v,
    #                   const double *sp_b, const double *sp_c, int conf_sample_buffer, int conf_signal_period_n, double conf_ds,
    #                   double *fn, double *fp, double *favg, double *xzcp_v, double *xzcn_v)
    xzcn_v = np.zeros(conf_signal_period_n+1);
    xzcp_v = np.zeros(conf_signal_period_n+1);
    fn = ctypes.c_double()
    fp = ctypes.c_double()
    favg = ctypes.c_double()

    t0 = time.perf_counter()
    find_frequency_c(x_n, ctypes.c_void_p(x_v.ctypes.data), ctypes.c_void_p(y_v.ctypes.data),
                     ctypes.c_void_p(sp_b.ctypes.data), ctypes.c_void_p(sp_c.ctypes.data), conf_sample_buffer, conf_signal_period_n, ctypes.c_double(conf_ds), 
                     ctypes.byref(fn), ctypes.byref(fp), ctypes.byref(favg), ctypes.c_void_p(xzcp_v.ctypes.data), ctypes.c_void_p(xzcn_v.ctypes.data))
    t1 = time.perf_counter()
    print(f'find_frequency_cf: {(t1-t0)*1e3:2f} ms')
    
    return fn.value, fp.value, favg.value, xzcp_v, xzcn_v

def spline_f(x_n, x, y):

    #--------------------------------------------------------------------------
    # Spline from
    # https://fac.ksu.edu.sa/sites/default/files/numerical_analysis_9th.pdf#page=167
    #--------------------------------------------------------------------------

    h = np.zeros((x_n-1));
    a = np.zeros(x_n-1);
    l = np.zeros(x_n);

    u = np.zeros(x_n);
    z = np.zeros(x_n);
    sp_b = np.zeros(x_n);
    sp_c = np.zeros(x_n);
    sp_d = np.zeros(x_n);


    #--------------------------------------------------------------------------
    # Calculate h = dx between samples
    # Can be omitted if h = 1
    #--------------------------------------------------------------------------

    for x_i in np.arange(0,x_n-1):
        h[x_i] = x[x_i + 1] - x[x_i]

    #--------------------------------------------------------------------------
    # A, sparse matrix
    #--------------------------------------------------------------------------

    for x_i in np.arange(1,x_n-1):
        a[x_i] = 3.0 * (y[x_i + 1] - y[x_i]) / h[x_i] - 3.0 * (y[x_i] - y[x_i - 1]) / h[x_i - 1]


    #--------------------------------------------------------------------------
    # l, u, z
    #--------------------------------------------------------------------------
    l[0] = 1
    u[0] = 0
    z[0] = 0

    for x_i in np.arange(1, x_n-1):
        l[x_i] = 2.0 * (x[x_i + 1] - x[x_i - 1]) - h[x_i - 1] * u[x_i - 1]
        u[x_i] = h[x_i] / l[x_i]
        z[x_i] = (a[x_i] - h[x_i - 1] * z[x_i - 1]) / l[x_i]



    l[x_n-1] = 1
    z[x_n-1] = 0
    sp_c[x_n-1] = 0

    #--------------------------------------------------------------------------
    # b, c, d
    # y(x) = yi + bi*(x-xi) + ci*(x-xi)^2 + di*(x-xi)^3
    #--------------------------------------------------------------------------

    for x_i in np.arange(x_n-2, -1, -1):  #x_n-1:-1:1
        sp_c[x_i] = z[x_i] - u[x_i] * sp_c[x_i + 1]
        sp_b[x_i] = (y[x_i + 1] - y[x_i]) / h[x_i] - h[x_i] * \
            (sp_c[x_i + 1] + 2.0 * sp_c[x_i]) / 3.0
        sp_d[x_i] = (sp_c[x_i + 1] - sp_c[x_i]) / 3.0 / h[x_i]

    return sp_b, sp_c, sp_d

def interpolate(sp_b, sp_c, sp_d, x_n, x_v, y_v, xx_n, xx_v):
    yy_v = np.zeros(xx_v.shape)
    x0_i = 0
    for xx_i in np.arange(0, xx_n):
        x = xx_v[xx_i]
       #  % Search, but can be calculated if h = 1
        for x_i in np.arange(x0_i, x_n):
            if x_v[x_i] > x:
                x_i = x_i-1
                x0_i = x_i
                break



        xd = x-x_v[x_i];
        #%    disp(sprintf('%f, %f, %f, %d', x_i, x, x_v(x_i), xd))
        yy_v[xx_i] = y_v[x_i] + sp_b[x_i]*xd + sp_c[x_i]*xd**2 + sp_d[x_i]*xd**3

    return yy_v

def movemean(x_n, x_v, b_n):
    x2_v = np.zeros(x_n)
    b_v = np.zeros(b_n)
    b2_n = int((b_n-1)/2)
    # print(b2_n)
    avg = 0

    for b_i in np.arange(b_n-1):
        i_i = b_i-b2_n
        if i_i < 0:
            b_v[b_i] = x_v[0]/b_n
        else:
            b_v[b_i] = x_v[i_i]/b_n
        avg += b_v[b_i]

    b_v[b_n-1] = 0
    b_i = b_n-1
    # print('a', avg, b_v, b_v)
    for x_i in np.arange(x_n):
        i_i = x_i+b2_n
        if i_i >= x_n:
            i_i = x_n-1

        last = b_v[b_i]
        b_v[b_i] = x_v[i_i]/b_n
        # print('avg', avg, -last, b_v[b_i])
        avg = avg - last + b_v[b_i]
        x2_v[x_i] = avg
        b_i += 1
        if b_i >= b_n:
            b_i = 0
    return x2_v

def find_frequency(x_n, x_v, y_v, sp_b, sp_c, conf_sample_buffer,
                   conf_signal_period_n, conf_ds):
    #--------------------------------------------------------------------------
    # Search for sign changes of signal, scp, scn
    #--------------------------------------------------------------------------
    scp_iv = np.zeros(x_n, dtype=int)
    scn_iv = np.zeros(x_n, dtype=int)

    scp_i = 0
    scn_i = 0

    for x_i in np.arange(conf_sample_buffer, x_n-1):
        if np.sign(y_v[x_i]) != np.sign(y_v[x_i+1]) and np.sign(y_v[x_i]) != 0:
            if np.sign(y_v[x_i]) < 0:
                if np.abs(y_v[x_i]) < np.abs(y_v[x_i+1]):
                    scp_iv[scp_i] = x_i
                else:
                    scp_iv[scp_i] = x_i+1
                scp_i += 1

            else:
                if np.abs(y_v[x_i]) < np.abs(y_v[x_i+1]):
                    scn_iv[scn_i] = x_i
                else:
                    scn_iv[scn_i] = x_i+1
                scn_i +=1

    # print(scp_iv[:scp_i])
    # print(scn_iv[:scn_i])

    xzcn_v = np.zeros(conf_signal_period_n+1)
    xzcp_v = np.zeros(conf_signal_period_n+1)

    for i_i in np.arange(0, conf_signal_period_n+1):
        x_i = scn_iv[i_i];
        xzcn_v[i_i] = x_v[x_i] + (-sp_b[x_i] -
            np.sqrt(sp_b[x_i]**2-4.0*y_v[x_i]*sp_c[x_i]))/2.0/sp_c[x_i]

        x_i = scp_iv[i_i];
        xzcp_v[i_i] = x_v[x_i] + (-sp_b[x_i] +
            np.sqrt(sp_b[x_i]**2-4.0*y_v[x_i]*sp_c[x_i]))/2.0/sp_c[x_i]


    # print(xzcn_v)
    # print(xzcp_v)

    fn = 1/((xzcn_v[conf_signal_period_n]-xzcn_v[0])/(conf_signal_period_n)*conf_ds);
    fp = 1/((xzcp_v[conf_signal_period_n]-xzcp_v[0])/(conf_signal_period_n)*conf_ds);
    favg = (fn+fp)/2;

    # print(fn, fp, favg)
    return fn, fp, favg, xzcp_v, xzcn_v



def fft_aa(fft_y_v, conf_fft_n):

    # double sided complex
    freq_dc_v = scipy.fft.fft(fft_y_v);
    freq_dabs_v = np.abs(freq_dc_v/conf_fft_n);

    # single sided angle/amplitude
    freq_sangle_v = np.angle(freq_dc_v[0:int(conf_fft_n/2)])
    freq_sabs_v = freq_dabs_v[0:int(conf_fft_n/2)]
    # correct amplitude
    freq_sabs_v[1:-2] = 2*freq_sabs_v[1:-2]

    return freq_sabs_v, freq_sangle_v

def angle_wrap(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi

# %%
conf_filter_n = 15
conf_fft_n = 1024
conf_f0 = 50
conf_f_min = 45
conf_f_sample = 4000
conf_signal_period_n = 10
conf_sample_buffer = 10

conf_sample_max = np.ceil(2*conf_sample_buffer + 1/conf_f_min*(conf_signal_period_n+1)*conf_f_sample)

conf_ds = 1/conf_f_sample

fft_tw = conf_signal_period_n/conf_f0
fft_f = 1/(fft_tw/conf_fft_n)

conf_freq_f_v = np.arange(0, conf_fft_n/2)/conf_fft_n*fft_f/conf_f0

raw_x_v = np.arange(-conf_sample_buffer, conf_sample_max-conf_sample_buffer-1)
raw_n = raw_x_v.size

df = pd.DataFrame(raw_x_v)
df.to_csv('raw_x_v.csv',index=False)

conf_v = [conf_filter_n, conf_fft_n, conf_f0, conf_f_min, conf_f_sample, conf_signal_period_n, conf_sample_buffer, conf_sample_max, conf_ds]
df = pd.DataFrame(conf_v)
df.to_csv('conf_v.csv',index=False)

df = pd.DataFrame(conf_freq_f_v)
df.to_csv('conf_freq_f_v.csv',index=False)

# %%
run_n = 128
n_n = 20

sf0m_v = 50+10*(-0.5+np.random.rand(run_n,1))

amp_over_v = np.zeros((1, n_n));

amp_over_v[0, 0] = 100
amp_over_v[0, 2] = 2
amp_over_v[0, 8] = 1
amp_over_v[0, 14] = 0.3

amp_over_v[0, [4, 6]] = 2
amp_over_v[0, [10, 12]] = 1.5
amp_over_v[0, [16, 18]] = 1.0

amp_over_v[0, [1, 3]] = 1
amp_over_v[0, 5] = 0.5
amp_over_v[0, 7:20:2] = 0.2

amp_over_v = 0.01*amp_over_v

sam_vv = np.zeros((run_n, n_n))
sphim_vv = np.zeros((run_n, n_n))

sam_vv[:,0] = 1 #np.ones((run_n, 1))
sam_vv[:,1:n_n] = amp_over_v[0, 1:n_n]*(4*np.random.rand(run_n, n_n-1))

sphim_vv = 2*np.pi*np.random.rand(run_n, n_n)



y_vv = np.zeros((run_n, raw_n))
for i_i in np.arange(0, n_n):
    y_vv = y_vv + sam_vv[:,i_i, np.newaxis] * np.cos(2*np.pi*(i_i+1)*sf0m_v*raw_x_v*conf_ds + sphim_vv[:,i_i,np.newaxis])

df = pd.DataFrame(y_vv)
df.to_csv('y_vv.csv',index=False)

print('ä')

# %%
name = 'raw'
run_v = [np.random.randint(0, run_n)]



fd_v = np.zeros((run_n))
ad_v = np.zeros((run_n, n_n))
pd_v = np.zeros((run_n, n_n))
ii_v = np.linspace(10, n_n*10, n_n, dtype=int)

# run_v = np.arange(0, run_n)
for run_i in run_v:

    sf0m = sf0m_v[run_i]
    raw_y_v = y_vv[run_i, :]

    t0 = time.perf_counter()
    raw_sp_b, raw_sp_c, raw_sp_d = spline_f(raw_n, raw_x_v, raw_y_v)
    t1 = time.perf_counter()
    print(f'spline_f: {(t1-t0)*1e3:2f} ms')
    
    craw_sp_b, craw_sp_c, craw_sp_d = spline_cf(raw_n, raw_x_v, raw_y_v)
    
    

    t0 = time.perf_counter()
    yy_v = interpolate(raw_sp_b, raw_sp_c, raw_sp_d,
                       raw_n, raw_x_v, raw_y_v, raw_n, raw_x_v)
    t1 = time.perf_counter()
    print(f'interpolate: {(t1-t0)*1e3:2f} ms')
    
    cyy_v = interpolate_cf(craw_sp_b, craw_sp_c, craw_sp_d,
                       raw_n, raw_x_v, raw_y_v, raw_n, raw_x_v)

    t0 = time.perf_counter()
    lp_y_v = movemean(raw_n, raw_y_v, conf_filter_n)
    t1 = time.perf_counter()
    print(f'movemean: {(t1-t0)*1e3:2f} ms')

    clp_y_v = movemean_cf(raw_n, raw_y_v, conf_filter_n)

    t0 = time.perf_counter()
    lp_sp_b, lp_sp_c, lp_sp_d = spline_f(raw_n, raw_x_v, lp_y_v)
    t1 = time.perf_counter()
    print(f'spline_f: {(t1-t0)*1e3:2f} ms')

    clp_sp_b, clp_sp_c, clp_sp_d = spline_cf(raw_n, raw_x_v, clp_y_v)
    


    t0 = time.perf_counter()
    fn, fp, favg, xzcp_v, xzcn_v = find_frequency(raw_n, raw_x_v, lp_y_v,
                                                  lp_sp_b, lp_sp_c,
                   conf_sample_buffer, conf_signal_period_n, conf_ds)
    t1 = time.perf_counter()
    print(f'find_frequency: {(t1-t0)*1e3:2f} ms')

    cfn, cfp, cfavg, cxzcp_v, cxzcn_v = find_frequency_cf(raw_n, raw_x_v, clp_y_v,
                                                  clp_sp_b, clp_sp_c,
                   conf_sample_buffer, conf_signal_period_n, conf_ds)
    


    fft_x_v = np.linspace(0, conf_fft_n-1, conf_fft_n)*10/favg/conf_ds/(conf_fft_n)
    t0 = time.perf_counter()
    fft_y_v = interpolate(raw_sp_b, raw_sp_c, raw_sp_d,
                          raw_n, raw_x_v, raw_y_v, conf_fft_n, fft_x_v)
    t1 = time.perf_counter()
    print(f'interpolate: {(t1-t0)*1e3:2f} ms')

    cfft_y_v = interpolate_cf(craw_sp_b, craw_sp_c, craw_sp_d,
                          raw_n, raw_x_v, raw_y_v, conf_fft_n, fft_x_v)
    
    # print(favg, fft_y_v[0:3])
    # print(cfavg, cfft_y_v[0:3])

    ab, ar = fft_test_cf(raw_n, cfft_y_v)
    
    print(cfft_y_v[0:3])
    print(ab[0:3])
    print(ar[0:3])

    freq_sabs_v, freq_sangle_v = fft_aa(fft_y_v, conf_fft_n)

    fd_v[run_i] = (favg-sf0m)/sf0m


    ad_v[run_i, :] = (freq_sabs_v[ii_v]-sam_vv[run_i, :]) / sam_vv[run_i, 0]
    phi = sphim_vv[run_i, :]
    pd_v[run_i, :] = angle_wrap((freq_sangle_v[ii_v]-phi))

    if run_i == 0:
        plt.ion()
        f_h = plt.figure(num=f'{name}')
        f_h.clf()
        a_h = f_h.add_subplot(1, 1, 1)

        a_h.plot(raw_x_v, raw_y_v,
                 linestyle='-',  label=f'raw')
        a_h.plot(raw_x_v, lp_y_v,
                 linestyle='-',  label=f'lp')
        a_h.plot(raw_x_v, yy_v,
                 linestyle=':', marker=None,  label=f'interp')
        a_h.plot(xzcp_v, np.zeros(xzcp_v.shape),
                 linestyle='none', marker='s',  label=f'pos')
        a_h.plot(xzcn_v, np.zeros(xzcn_v.shape),
                 linestyle='none', marker='s',  label=f'neg')
        a_h.plot(fft_x_v, fft_y_v,
                 linestyle='none', marker='.',  label=f'pos')

        a_h.legend()
        a_h.set_xlabel('Tid [-]')
        a_h.set_ylabel('Signal [-]')
        a_h.set_title(f'{name}')

        f_h = plt.figure(num=f'{name} fft')
        f_h.clf()
        a_h = f_h.add_subplot(1, 1, 1)

        a_h.loglog(freq_sabs_v,
                 linestyle='-', marker='.',  label=f'raw')


        a_h.legend()
        a_h.set_xlabel('Frekvens [-]')
        a_h.set_ylabel('Signal [-]')
        a_h.set_title(f'{name} fft')

# %%
amax_v = np.max(ad_v, axis=0)
pmax_v = np.max(pd_v, axis=0)

f_h = plt.figure(num=f'{name} amplitud')
f_h.clf()
a_h = f_h.add_subplot(1, 1, 1)
a_h.plot(amax_v)
a_h.set_xlabel('Harmonic [-]')
a_h.set_ylabel('Amplitudfel [-]')
a_h.set_title(f'{name} amplitud')

f_h = plt.figure(num=f'{name} fas')
f_h.clf()
a_h = f_h.add_subplot(1, 1, 1)
a_h.plot(pd_v.T)
a_h.set_xlabel('Harmonic [-]')
a_h.set_ylabel('Fasfel [rad]')
a_h.set_title(f'{name} fas')

f_h = plt.figure(num=f'{name} frekvens')
f_h.clf()
a_h = f_h.add_subplot(1, 1, 1)

a_h.hist(fd_v)


a_h.set_xlabel('Frekvens [-]')
a_h.set_ylabel('Antal [-]')
a_h.set_title(f'{name} frekvens')