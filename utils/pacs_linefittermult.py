#!/usr/bin/env python
# IDENT         pacs_linefittermult.py
# LANGUAGE      Python 3.X
# AUTHOR        E.PUGA
# PURPOSE       Class to fit a group of lines on PACS spectra using astropy models
#               and fitter explicitly.
#               Ported from the analogue Jython Class LineFitterMult
#               created by Pierre Royer (KUL) 08/03/2010.
#               PACS was optimized for lines, not continuum, therefore the continuum
#               fitting is local and we do not try to fit the entire spectral range.
#               Continuum is modeled as Polynomial1D
#               Requires getSpecResolution and access to PACS calTree
#               Features: 1. The constructor contains model initialization parameters
#                         2. Methods to retrieve parameters
#                         3. Continuum is Polynomial1D and blending line cluster
#                         wavelength is input as a list of reference wavelengths
#               Differences: 1. wavelength sorted is assumed
#                            2. input is specutils Spectrum1D which works with masked NaNs
#
# VERSION
# 1.0.0 11/08/2020 EP Creation
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter
from astropy import units as u
from specutils.fitting import fit_lines
from specutils import SpectralRegion, Spectrum1D
from specutils.manipulation import extract_region, noise_region_uncertainty

from specutils.fitting import fit_generic_continuum, fit_continuum

class LineFitterMult(object):
    '''

    Parameters
    ----------
    spectrumIn: specutils.Spectrum1D
        specutils Spectrum1D instance in units of astropy.units.Jy and astropy.units.microns
    wline: float/list
        Estimated line(s) centre(s). Can be a single line or a list of lines in microns
    band: str
        PACS band for the observation ['B2A'|'B3A'|'B2B'|'R1'] used to estimate spectral resolution
    polyOrder: int
        Order of the polynomial by which the continuum will be locally approximated. Default=1
    widthFactor: int
        Number of fwhm kept on each side of wline for the line fit. Default=4
    verbose: bool
        print output-debug aid. Default=0

    Attributes
    ----------
    spectrumIn: :class:`specutils.Spectrum1D`
        input instance of :class:`specutils.Spectrum1D` with populated or defaulted mask extension
    spectrum:  :class:`specutils.Spectrum1D`
        instance of :class:`specutils.Spectrum1D` with masked NaN values
    wmax: float
        maximum wavelength in microns
    wmin: float
        minimum wavelength in microns
    centre: numpy.array
        fitted line(s) centre in microns
    peak: numpy.array
        line(s) amplitude of a Gaussian1D model in units of Jy
    sigma: numpy.array
        fitted line(s) sigma in Hz
    width: numpy.array
        line(s) width or fwhm in microns
    fwhm: numpy.array
        line(s) full-width at half maximum in microns
    continuum: numpy.array
        array of continuum values at the line central wavelength (estimated or fitted,
        depending on findline)
    flux: numpy.array
        line(s) flux determined analytically from Gaussian fit parameters in the frequency
        domain : Integral(F(nu)dNu) in units of Watt * m**-2
    fluxError: numpy.array
        error in the line(s) flux propagated from amplitude and sigma parameter uncertainties
    fluxErrorRelative: numpy.array
        relative error in the line(s) flux
    polyOrder: int
        order of polynomial
    model: :class:`astropy.modeling.models.CompoundModel`
        model for PACS line spectrum consisting of a local continuum and `n` Gaussian profiles
        as `~astropy.units.Quantity`
    chiSquared: float
        chi-squared of goodness of fit to the input spectrum
    stddev: numpy.array
        standard deviations pertaining to all model compound parameters
    fitResult: :class:`astropy.units.Quantity`
        global model fit 'specutils.Spectrum1D.flux' evaluated at 'specutils.Spectrum1D.spectral_axis'
    fitter: :class:`astropy.fitting.LevMarLSQFitter`
        using the fitter explicitly, gives access to the covariance matrix and thus to
        the parameters uncertainties
    line_spec: :class:`specutils.Spectrum1D`
        line model fit component (continuum subtracted) as `~astropy.units.Quantity` evaluated
        at 'specutils.Spectrum1D.spectral_axis'
    findLine: bool
        indicates if a line has been detected
    continuumModel: :class:`astropy.modeling.models.Polynomial1D`
        fitted continuum model using a Polynomial1D of a certain order, and always model [0]

    '''

    def __init__(self, spectrumIn, wline, band, polyOrder=1, widthFactor=4., verbose=0):
        # =======================================================================
        # Initial check in spectral_axis
        # =======================================================================
        if not (isinstance(spectrumIn.spectral_axis, u.Quantity) and isinstance(spectrumIn.flux, u.Quantity)):
            raise ValueError("Spectral axis must be a `Quantity` object.")
            if not spectrumIn.spectral_axis.unit == u.um:
                raise ValueError("Spectral axis is not in units of microns")

        self.polyOrder = polyOrder
        resv = self.get_spec_resolution(int(band[1]), np.array(wline, dtype=float))
        # =======================================================================
        # Convert delta velocity to delta lambda in microns with astropy units equivalencies
        # =======================================================================
        # c_kms = const.c.to('km/s')
        # fwhmum = (resv * u.kilometer/ u.s / c_kms) * wline
        rest_wline = wline * u.um
        fwhmum_q = (resv * u.km / u.s).to(u.um, equivalencies=u.doppler_optical(rest_wline)) - rest_wline
        fwhmum = fwhmum_q.value
        fwhm_to_sigma = 1./(8 * np.log(2))**0.5
        lineWidthEstimate = fwhmum * fwhm_to_sigma
        wmin, wmax = wline[0] - (widthFactor * fwhmum), wline[-1] + (widthFactor * fwhmum)
        if verbose: print("wline, wmin, wmax, resv, fwhmum, lineWidthEstimate: ", wline, wmin, wmax, resv, fwhmum, lineWidthEstimate)

        self.wmin = wmin
        self.wmax = wmax
        self.spectrumIn = spectrumIn
        # =======================================================================
        # mask non finite elements
        # =======================================================================
        spectrum = self.__finite(spectrumIn, verbose=verbose)
        self.spectrum = spectrum

        wunit = self.spectrum.spectral_axis.unit
        region = SpectralRegion(self.wmin * wunit, self.wmax * wunit)
        spectrum_region = extract_region(self.spectrum, region)

        # =======================================================================
        # Compute peak flux estimates for model parameters starting values in the region
        # =======================================================================
        peakFluxEstimate = []
        for wsline in wline:
            wave = spectrum_region.spectral_axis.value #just the ndarray and not Quantity
            flux = spectrum_region.flux.value
            wdist = np.abs(wsline - wave)
            if verbose: print(wsline, min(wdist), max(wdist))
            indexLine = np.where(wdist == min(wdist))[0][0]
            if verbose: print("indexLine= {}".format(indexLine))
            peakEstimate = np.mean(flux[indexLine - 1:indexLine + 1])
            if verbose: print('Estimates for peak init {}'.format(peakEstimate))
            cont_sample = np.concatenate((flux[:5], flux[-5:]), axis=None)
            continuumEstimate = np.median(np.concatenate((flux[:5], flux[-5:]), axis=None))
            peakFluxEstimate = np.append(peakFluxEstimate, peakEstimate - continuumEstimate)
            if verbose: print('Estimates for peak & continuum {}, {}'.format(peakFluxEstimate, continuumEstimate))

        # =======================================================================
        # Construct model compound (branching off lines+continuum or continuum)
        # =======================================================================

        try:
            lineModel_init = models.Polynomial1D(self.polyOrder, c0=continuumEstimate, name='cont')
            for xi in range(len(wline)):
                lineModel_init += models.Gaussian1D(amplitude=peakFluxEstimate[xi], mean=wline[xi],
                                                    stddev=lineWidthEstimate[xi], name='g{}'.format(xi+1))
            fitter = LevMarLSQFitter()
            lineModel = fit_lines(self.spectrum, lineModel_init, fitter=fitter, window=region)
            fitResult = lineModel(self.spectrum.spectral_axis)
            findLine = 1

            self.flux = []
            self.sigma = []

            for idx in range(len(wline)):
                #momentarily taking advantage of astropy units for conversion
                line_amp = (lineModel.unitless_model[idx + 1].amplitude.value * u.Jy).to(u.Watt / u.m ** 2 / u.Hz)
                line_sig = (lineModel.unitless_model[idx + 1].stddev.value * u.um).to(u.Hz, equivalencies=u.spectral())
                self.flux = np.append(self.flux, (line_amp * line_sig * np.sqrt(2. * np.pi)).value)
                self.sigma = np.append(self.sigma, line_sig.value)
        except:
            if verbose: print('Exception')
            lineModel_init = models.Polynomial1D(self.polyOrder, c0=continuumEstimate, name='cont')
            fitter = LevMarLSQFitter()
            lineModel = fit_lines(self.spectrum, lineModel_init, fitter=fitter, window=region) #the problem is narrow window where the contribution of the continuum sample is small
            fitResult = lineModel(self.spectrum.spectral_axis)
            findLine = 0
        self.model = lineModel
        self.fitResult = fitResult
        self.findLine = findLine
        self.fitter = fitter
        # =======================================================================
        # Preserve continuum Polynomial model
        # =======================================================================
        # there are two types of models, those that are based on
        # `~astropy.modeling.models.PolynomialModel` and therefore
        # require the ``degree`` parameter when instantiating the
        # class , and "everything else" that does not require an
        # "extra" parameter for class instantiation.
        compound_model = lineModel.n_submodels > 1
        if compound_model:
            self.continuumModel = lineModel.unitless_model[0]
        else:
            self.continuumModel = lineModel.unitless_model
        if findLine:
            self.continuum = []
            self.peak = []
            self.centre = []
            self.sigma = []
            self.fwhm = []
            self.chiSquared = (self.fitter.fit_info['fvec']**2).sum()/(len(self.fitter.fit_info['fvec'])-len(self.fitter.fit_info['param_cov'].data))
            self.stddev = np.sqrt(np.diag(fitter.fit_info['cov_x'])) #standard deviations pertaining to all parameters.
            params_idx = [int(param.split('_', -1)[-1]) for param in self.model.param_names]
            self.fluxError = []
            self.fluxErrorRelative = []
            for idx in range(len(wline)):
                self.continuum = np.append(self.continuum, self.continuumModel(lineModel.unitless_model[idx + 1].mean.value))
                self.peak = np.append(self.peak, lineModel.unitless_model[idx + 1].amplitude.value)
                self.centre = np.append(self.centre, lineModel.unitless_model[idx + 1].mean.value)
                self.sigma = np.append(self.sigma, lineModel.unitless_model[idx + 1].stddev.value)
                self.fwhm = np.append(self.fwhm, self.sigma/fwhm_to_sigma)
                line_amp = (lineModel.unitless_model[idx + 1].amplitude.value * u.Jy).to(u.Watt / u.m ** 2 / u.Hz)
                line_sig = (lineModel.unitless_model[idx + 1].stddev.value * u.um).to(u.Hz, equivalencies=u.spectral())
                param_idx = [i for i, value in enumerate(params_idx) if value == (idx + 1)]
                self.fluxErrorRelative = np.append(self.fluxErrorRelative,
                                                   np.sqrt(np.sum((self.stddev/self.model.parameters)[np.array([param_idx])][np.array([0,-1])] ** 2.)))
            self.fluxError = self.fluxErrorRelative * self.flux
            self.width = self.fwhm
        else:
            self.continuum = np.array([np.median(flux) for i in range(len(wline))])
            self.flux = self.peak = self.sigma = self.fwhm = self.width = np.array([0. for i in range(len(wline))])
            self.centre = wline
            if verbose: print('Line Not Detected. Continuum: {}'.format(self.continuum))
        self.line_spec = self.get_line_spec()
        self.chiSquared = (self.fitter.fit_info['fvec']**2).sum()/(len(self.fitter.fit_info['fvec'])-len(self.fitter.fit_info['param_cov'].data))

        return

    def __finite(self, spectrum, verbose=0):
        '''
        Mask non-finite values for `:class:specutils.Spectrum1D` instance in
        either `Spectrum1D.flux` or `Spectrum1D.spectral_axis` and include it
        into input mask.
        Note: specutils will recognize flux values input as NaN and set the mask to True for those values
        unless explicitly overridden.
        Parameters
        ----------
        spectrum : :class:`specutils.Spectrum1D`
        verbose

        Returns `:class:specutils.Spectrum1D` instance
        -------

        '''
        if verbose: print('Dimensions: input {},{}'.format(spectrum.spectral_axis.shape, spectrum.flux.shape))
        mask = np.isnan(spectrum.flux) | np.isnan(spectrum.spectral_axis)
        spectrum_masked = spectrum._copy()
        if spectrum.mask is None:
            spectrum_masked.mask = mask #mask populated anyways
        else:
            spectrum_masked.mask = spectrum_masked.mask & mask
        spectrum_masked.flux[spectrum_masked.mask] = np.nan
        if verbose: print('Dimensions: finite {},{}'.format(spectrum_masked.spectral_axis[~spectrum_masked.mask].shape,
                                                            spectrum_masked.flux[~spectrum_masked.mask].shape))
        return spectrum_masked

    def get_wmin(self):
        return self.wmin
    def get_wmax(self):
        return self.wmax
    def get_fit_result(self):
        return self.fitResult
    def get_continuum(self):
        return self.continuum
    def get_continuum_model(self):
        return self.continuumModel
    def get_line_peak(self):
        return self.peak
    def get_line_centre(self):
        return self.centre
    # def get_line_width(self):
    #     return self.width
    def get_line_fwhm(self):
        return self.fwhm
    def get_line_model(self):
        return self.model
    def get_line_flux(self):
        return self.flux
    def get_find_line(self):
        return self.findLine
    def get_fitter(self):
        return self.fitter
    def get_plot(self):
        #TODO: determine if it is necessary to get a plot of the fit
        return self.plot
    def get_wave(self):
        '''
        Returns wavelength array as `specutils.Spectrum1D.spectral_axis` with units of :class:`astropy.units.Quantity`
        -------
        '''
        return self.spectrum.spectral_axis
    def get_flux(self):
        '''
        Returns flux array as `specutils.Spectrum1D.flux` with units of :class:`astropy.units.Quantity`
        -------
        '''
        return self.spectrum.flux
    def get_line_spec(self):
        '''
        Returns Line spectrum component (continuum subtracted) as :class:`specutils.Spectrum1D` with units of :class:`astropy.units.Quantity`
        -------
        '''
        fitted_continuum = self.continuumModel(self.spectrum.spectral_axis.value)
        line_spectrum = self.spectrum - Spectrum1D(flux=fitted_continuum * u.Jy, spectral_axis=self.spectrum.spectral_axis.value * u.um)
        return line_spectrum
    def get_chi_squared(self):
        return self.chiSquared
    def get_standard_deviation(self):
        return self.stddev
    def get_spec_resolution(self, order, wavelength):
        '''
        function ported from the java task GetSpecResolution() that reads the cal file specProperties
        and returns the spectral resolution in km/s at a given order and wavelength in micron
        using the equation described in PICC-ME-SD-004
        Parameters
        ----------
        order: int
            order of the PACS band [1,2]
        wavelength: float/list
            scalar or list to be converted to array

        Returns
        -------
            np.array with effective resolution in km/s
        '''
        _calfile = 'http://archives.esac.esa.int/hsa/legacy/cal/PACS/user/PCalSpectrometer_SpecProperties_FM_v1.fits'
        hdul = fits.open(_calfile)

        c = hdul['lightSpeed'].data
        g = hdul['gratingConstant'].data
        coll = hdul['beamDiameter'].data
        scale = hdul['scale'].data

        wave = np.asarray([wavelength]) if np.isscalar(wavelength) else np.asarray(wavelength)
        fac = order * g * wave / 1000.
        velRes = c * np.sqrt(1. - (fac / 2.) ** 2.0) / (order * g * coll)
        velResPix = c * 2.0 * np.sqrt(1. - (fac / 2.) ** 2.0) / (fac * scale)
        effRes = np.sqrt((velRes ** 2.0) + (velResPix ** 2.0))

        return effRes

if __name__ == "__main__":
    wline = [185.999]
    band = 'R1'
    order = int(band[1])
    np.random.seed(0)
    x = np.linspace(180., 190., 100)
    y = 3 * np.exp(-0.5 * (x - 185.999) ** 2 / 0.1 ** 2)
    y += np.random.normal(0., 0.2, x.shape)

    y_continuum = 3.2 * np.exp(-0.5 * (x - 5.6) ** 2 / 4.8 ** 2)
    y += y_continuum

    #create spectrum to fit
    spectrum = Spectrum1D(flux=y * u.Jy, spectral_axis=x * u.um)
    noise_region = SpectralRegion(180. * u.um, 184. * u.um)
    spectrum = noise_region_uncertainty(spectrum, noise_region)

    #line_region = [(185.52059807*u.um, 186.47740193*u.um)]
    g1_fit = fit_generic_continuum(spectrum, model=models.Polynomial1D(1))
    y_continuum_fitted = g1_fit(x * u.um)

    plt.plot(x, y, label='spectrum')
    plt.errorbar(x, y, yerr=spectrum.uncertainty.array, color='b')
    plt.plot(x, y_continuum_fitted, label='cont_0')
    plt.title('Continuum+line Fitting')
    plt.grid(True)

    line = LineFitterMult(spectrum, wline, band, polyOrder=1, widthFactor=10, verbose=1)
    print(line.get_line_centre(), line.get_line_peak(), line.get_line_fwhm())
    plt.plot(spectrum.spectral_axis, line.get_line_spec().flux, label='line model', color='r') #with units
    plt.plot(spectrum.spectral_axis, line.continuumModel(spectrum.spectral_axis.value), label='continuum model', color='g')
    plt.legend()
