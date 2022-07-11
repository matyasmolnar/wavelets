"""Plotting functions

Inspired by https://github.com/alsauve/scaleogram
"""


import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import pywt


def _wavelet_instance(wavelet):
    """Function responsible for returning the correct pywt.ContinuousWavelet
    """
    if isinstance(wavelet, pywt.ContinuousWavelet):
        return wavelet
    if isinstance(wavelet, str):
        return pywt.ContinuousWavelet(wavelet)
    else:
        raise ValueError('Expecting a string name for the wavelet, '+
                         'or pywt.ContinuousWavelet')


def plot_wav_time(wav='cmor1-1.5', ax=None, plt_complex=True, clearx=False,
                  xlabel='Time [s]', alpha=1, ylabel=None, title=None, xlim=None,
                  ylim=None, figsize=None):
    """Plot wavelet wavefunction in time domain
    """
    wav  = _wavelet_instance(wav)
    wavefun, time = wav.wavefun(length=int(1e5))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if plt_complex and np.iscomplexobj(wavefun):
        ax.plot(time, wavefun.real, label=r'$\mathfrak{Re}$', alpha=alpha)
        ax.plot(time, wavefun.imag, color='red', label=r'$\mathfrak{Im}$', alpha=alpha)
        ax.legend(loc='best')
    else:
        ax.plot(time, wavefun.real, label=r'$\mathfrak{Re}$', alpha=alpha)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if title is None:
        ax.set_title(wav.name)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if clearx:
        ax.set_xticks([])
    else:
        ax.set_xlabel(xlabel)

    return ax


def plot_wav_freq(wav='cmor1-1.5', ax=None, plt_complex=True, yscale='linear',
                  annotate=True, clearx=False, alpha=1, title=None, xlim=None,
                  ylim=None, figsize=None):
    """Plot wavelet frequency support
    """
    wav  = _wavelet_instance(wav)
    wavefun, time = wav.wavefun(length=int(1e5))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    wt = scipy.fft.fftshift(scipy.fft.fft(wavefun, n=int(1e7)))  # increase sampling on FFT end
    nrm = wt.max()
    wt /= nrm  # normalize to unity amplitude
    df = np.median(np.ediff1d(time))  # get sampling
    wt_frqs = scipy.fft.fftshift(scipy.fft.fftfreq(wavefun.size, df))
    wt_frqs = np.interp(np.arange(wt.size)/wt.size, np.arange(wt_frqs.size)/wt_frqs.size, wt_frqs)

    if plt_complex and np.iscomplexobj(wt):
        ax.plot(wt_frqs, wt.real, label=r'$\mathfrak{Re}$', alpha=alpha)
        ax.plot(wt_frqs, wt.imag, color='red', label=r'$\mathfrak{Im}$', alpha=alpha)
        ax.legend(loc='best')
    else:
        ax.plot(wt_frqs, np.abs(wt))

    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if title is None:
        ax.set_title('Frequency support')

    if clearx:
        ax.set_xticks([])
    else:
        ax.set_xlabel('Frequency [Hz]')

    ax.set_yscale(yscale)

    central_frequency = wav.center_frequency
    if not central_frequency:
        central_frequency = pywt.central_frequency(wav)
    bandwidth_frequency = wav.bandwidth_frequency if wav.bandwidth_frequency else 0
    ax.axvline(wav.center_frequency, color='orange')

    if annotate:
        ax.text(0.05, 0.85, f'Centr. freq. = {central_frequency:.1f} Hz\n'+
                f'Bwidth param. = {bandwidth_frequency:.1f}', ha='left',
                transform=ax.transAxes)

    return ax


def plot_wav(wav='cmor1-1.5', axes=None, t_complex=True, f_complex=True, yscale='linear',
             t_ylabel='Amplitude', t_alpha=1, f_alpha=1, annotate=True, clearx=False,
             figsize=None, dpi=100):
    """Plot wavelet function and frequency support side by side
    """
    wav  = _wavelet_instance(wav)
    wavefun, time = wav.wavefun(length=int(1e5))

    if axes is None:
        fig, axes = plt.subplots(ncols=2, figsize=figsize, dpi=dpi)

    plot_wav_time(wav, ax=axes[0], plt_complex=t_complex, ylabel=t_ylabel, alpha=t_alpha,
                  clearx=clearx)
    plot_wav_freq(wav, ax=axes[1], plt_complex=f_complex, yscale=yscale, alpha=f_alpha,
                  annotate=annotate, clearx=clearx)

    return axes


CBAR_DEFAULTS = {'vertical': {'aspect': 30, 'pad': 0.03, 'fraction':0.05},
                 'horizontal': {'aspect': 40, 'pad': 0.12, 'fraction':0.05}}

COI_DEFAULTS = {'alpha': 0.5, 'hatch': '/'}


def cws(time, signal=None, scales=None, wavelet=None, periods=None, spectrum='amp',
        coi=True, coikw=None, yaxis='period', cscale='linear', cmap='jet', clim=None,
        cbar='vertical', cbarlabel=None, cbarkw=None, xlim=None, ylim=None, yscale=None,
        xlabel=None, ylabel=None, title=None, ax=None, vlims=None, figsize=None):

    # allow to build the spectrum for signal only
    if signal is None:
        signal = time
        time   = np.arange(len(time))

    # build a default scales array
    if scales is None:
        scales = np.arange(1, min(len(time)/10, 100))
    if scales[0] <= 0:
        raise ValueError(f'scales[0] must be > 0, found: {str(scales[0])}')

    dt = np.median(np.ediff1d(time))  # calculate sampling
    coefs, scales_freq = pywt.cwt(signal, scales, wavelet, sampling_period=dt)

    # create plot area or use the one provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # adjust y axis ticks
    scales_period = 1./scales_freq  # needed also for COI mask
    xmesh = time #np.concatenate([time, [time[-1]+dt]])
    if yaxis == 'period':
        # ymesh = np.concatenate([scales_period, [scales_period[-1]+dt]])
        ymesh = scales_period
        ylim  = ymesh[[-1, 0]] if ylim is None else ylim
        ax.set_ylabel('Period' if ylabel is None else ylabel)
    elif yaxis == 'frequency':
        # df    = scales_freq[-1]/scales_freq[-2]
        # ymesh = np.concatenate([scales_freq, [scales_freq[-1]*df]])
        ymesh = scales_freq
        # set a useful yscale default: the scale freqs appears evenly in logscale
        yscale = 'log' if yscale is None else yscale
        ylim   = ymesh[[-1, 0]] if ylim is None else ylim
        ax.set_ylabel('Frequency' if ylabel is None else ylabel)
        #ax.invert_yaxis()
    elif yaxis == 'scale':
        ymesh = scales
        # ds = scales[-1]-scales[-2]
        # ymesh = np.concatenate([scales, [scales[-1] + ds]])
        ylim  = ymesh[[-1, 0]] if ylim is None else ylim
        ax.set_ylabel('Scale' if ylabel is None else ylabel)
    else:
        raise ValueError("yaxis must be one of 'scale', 'frequency' or 'period', found "
                          + str(yaxis)+" instead")

    # limit of visual range
    xr = [time.min(), time.max()]
    if xlim is None:
        xlim = xr
    else:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # adjust logarithmic scales on request (set automatically in Frequency mode)
    if yscale is not None:
        ax.set_yscale(yscale)

    # choose the correct spectrum display function and name
    if spectrum == 'amp':
        values = np.abs(coefs)
        sp_title = 'Amplitude'
        cbarlabel= 'abs(CWT)' if cbarlabel is None else cbarlabel
    elif spectrum == 'real':
        values = np.real(coefs)
        sp_title = 'Real'
        cbarlabel= 'real(CWT)' if cbarlabel is None else cbarlabel
    elif spectrum == 'imag':
        values = np.imag(coefs)
        sp_title = 'Imaginary'
        cbarlabel= 'imaginary(CWT)' if cbarlabel is None else cbarlabel
    elif spectrum == 'power':
        sp_title = 'Power'
        cbarlabel= 'abs(CWT)$^2$' if cbarlabel is None else cbarlabel
        values = np.power(np.abs(coefs),2)
    elif hasattr(spectrum, '__call__'):
        sp_title = 'Custom'
        values = spectrum(coefs)
    else:
        raise ValueError("The spectrum parameter must be one of 'amp', 'real', 'imag',"+
                         "'power' or a lambda() expression")

    # labels and titles
    ax.set_title('Continuous Wavelet Transform '+sp_title+' Spectrum'
                 if title is None else title)
    ax.set_xlabel('Time/spatial domain' if xlabel is None else xlabel)


    if cscale == 'log':
        isvalid = (values > 0)
        if vlims is None:
            cnorm = LogNorm(values[isvalid].min(), values[isvalid].max())
        else:
            cnorm = LogNorm(vlims[0], vlims[1])
    elif cscale == 'linear':
        cnorm = None
    else:
        raise ValueError("Color bar cscale should be 'linear' or 'log', got: "+
                         str(cscale))

    # plot the 2D spectrum using a pcolormesh to specify the correct Y axis
    # location at each scale
    qmesh = ax.pcolormesh(xmesh, ymesh, values, cmap=cmap, norm=cnorm)

    if clim:
        qmesh.set_clim(*clim)

    # fill visually the Cone Of Influence
    # (locations subject to invalid coefficients near the borders of data)
    if coi:
        # convert the wavelet scales frequency into time domain periodicity

        # here extend the scale range for which the CoI is computed, for better plotting
        nscales = np.append(scales, np.arange(scales[-1], scales[-1]+5)[1:])
        nscales_freq = pywt.scale2frequency(wavelet, nscales)/dt
        nscales_period = 1./nscales_freq

        scales_coi = nscales_period
        max_coi  = scales_coi[-1]

        # produce the line and the curve delimiting the COI masked area
        mid = int(len(xmesh)/2)
        time0 = np.abs(xmesh[:mid+1]-xmesh[0])
        ymask = np.zeros(len(xmesh))
        ymhalf = ymask[:mid+1]  # compute the left part of the mask
        ws = np.argsort(scales_coi) # ensure np.interp() works
        minscale, maxscale = sorted(ax.get_ylim())

        # * sqrt(2) as in Torrence and Compo with CoI = sqrt(2) * scale
        if yaxis == 'period':
            ymhalf[:] = np.interp(time0, scales_coi[ws]*2**0.5, scales_coi[ws])
            yborder = np.zeros(len(xmesh)) + maxscale
            ymhalf[time0 > max_coi*2**0.5] = np.nan
        elif yaxis == 'frequency':

            f1 = scipy.interpolate.interp1d(scales_coi[ws]*2**0.5 , 1./scales_coi[ws], \
                                            kind='cubic', fill_value='extrapolate')
            # ymhalf[:] = np.interp(time0, scales_coi[ws], 1./scales_coi[ws])
            ymhalf[:] = f1(time0)
            yborder = np.zeros(len(xmesh)) + minscale
            ymhalf[time0 > max_coi*2**0.5] = np.nan
        elif yaxis == 'scale':
            ymhalf[:] = np.interp(time0, scales_coi*2**0.5, nscales)
            yborder = np.zeros(len(xmesh)) + maxscale
            ymhalf[time0 > max_coi*2**0.5] = np.nan
        else:
            raise ValueError(f'yaxis = {yaxis}')

        # complete the right part of the mask by symmetry
        ymask[-mid:] = ymhalf[:mid][::-1]

        # plot the mask and forward user parameters
        ax.plot(xmesh, ymask)
        coikw = COI_DEFAULTS if coikw is None else coikw
        ax.fill_between(xmesh, yborder, ymask, **coikw )

    # color bar stuff
    if cbar:
        cbarkw   = CBAR_DEFAULTS[cbar] if cbarkw is None else cbarkw
        colorbar = plt.colorbar(qmesh, orientation=cbar, ax=ax, **cbarkw)
        if cbarlabel:
            colorbar.set_label(cbarlabel)

    return ax, qmesh, values


def scaleogram_tf(time, signal, wavelet='cmor1.5-1.0', scales=None, freq_plot='ps',
                 coi=True, scg_title='', scg_xlabel='Time [s]', scg_ylabel='Frequency [Hz]',
                 t_label='', f_label='', savefig=None, figsize=(8, 8), dpi=125):
    """Scaleogram with time and frequency domains also plotted on sides
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True, dpi=dpi)

    gs = fig.add_gridspec(3, 2, height_ratios=[1, 2.5, 0.1], width_ratios=[2.5, 1])

    # 1) Scaleogram
    ax1 = plt.subplot(gs[1, 0])
    ax1, qmesh, values = cws(time, signal, wavelet=wavelet, scales=scales, \
        cscale='log', coi=coi, title=scg_title, ax=ax1, spectrum='power', yaxis='frequency', \
        cbar=False, xlabel=scg_xlabel, ylabel=scg_ylabel, yscale='log', \
        cbarkw={'aspect':40, 'pad':0.12, 'fraction':0.05}, coikw={'alpha':0.5, 'hatch':'/'})

    cax1 = plt.subplot(gs[2, 0])
    plt.colorbar(qmesh, cax=cax1, orientation='horizontal', label='$|\mathrm{CWT}|^2$')


    # 2) Visibility Signal
    ax0 = plt.subplot(gs[0, 0])
    if np.iscomplexobj(signal):
        ax0.plot(time, signal.real, label=r'$\mathfrak{Re}$')
        ax0.plot(time, signal.imag, label=r'$\mathfrak{Im}$')
        ax0.legend(loc='best')
    else:
        ax0.plot(time, signal)
    ax0.set_ylabel(t_label)
    ax0.set_xlim(*ax1.get_xlim())
    ax0.tick_params(labelbottom=False)


    # 3) PS or FT
    sampling = np.median(np.ediff1d(time))
    if freq_plot.lower() == 'ps':
        # 3a) PS
        delay, pspec = scipy.signal.periodogram(signal, fs=1/sampling, window='blackmanharris',
            scaling='spectrum', nfft=signal.size, detrend=False, return_onesided=False)
        delay_sort = np.argsort(delay)
        delay = delay[delay_sort]
        pspec = pspec[delay_sort]
        pspec[np.abs(delay) < ax1.get_ylim()[0] - np.ediff1d(delay).mean()] *= np.nan

        ax2 = plt.subplot(gs[1, 1])
        z_idx = np.where(delay == 0)[0][0]
        ax2.plot(pspec[z_idx:], delay[z_idx:], label=r'$+$', c='deeppink', alpha=0.8)
        ax2.plot(pspec[:z_idx+1], -delay[:z_idx+1], label=r'$-$', c='purple', alpha=0.8)
        ax2.set_ylim(*ax1.get_ylim())
        ax2.set_xscale('log')
        ax2.set_yscale('log')

        ax2.legend(loc='best')
        # ax2.set_title('Power Spectrum')
        ax2.set_xlabel(f_label)
        ax2.tick_params(labelleft=False)

    elif freq_plot.lower() == 'ft':
        # 3b) FT
        vft = scipy.fft.fft(signal*scipy.signal.blackmanharris(signal.size))
        dly = scipy.fft.fftfreq(signal.size, sampling)

        dly_sort = np.argsort(dly)
        dly = dly[dly_sort]
        vft = vft[dly_sort]
        vft[np.abs(dly) < ax1.get_ylim()[0] - np.ediff1d(dly).mean()] *= np.nan

        ax2 = plt.subplot(gs[1, 1])
        z_idx = np.where(dly == 0)[0][0]
        ax2.plot(np.abs(vft[z_idx:]), dly[z_idx:], label=r'$+$', c='deeppink', alpha=0.8)
        ax2.plot(np.abs(vft[:z_idx+1]), -dly[:z_idx+1], label=r'$-$', c='purple', alpha=0.8)
        ax2.set_ylim(*ax1.get_ylim())
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        # ax2.set_xlim((0.15, 150))
        ax2.set_xlim((0.5, 50))

        ax2.legend(loc='upper right')
        # ax2.set_title('Power Spectrum')
        ax2.set_xlabel(f_label)
        ax2.tick_params(labelleft=False)

    else:
        raise ValueError('Specify either "ps" or "ft" for frequency plot.')

    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight')

    return fig, (ax0, ax1, ax2)
