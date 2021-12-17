# Import required packages
import csv
import os
from brian2 import ms, Hz, defaultclock, second
import numpy as np
import shutil
import time
import sparse

from scipy import ndimage
import matplotlib.cm as cm


class DVSmonitor:
    """Summary
    Attributes:
        pol (TYPE): Description
        t (TYPE): Description
        xi (TYPE): Description
        yi (TYPE): Description
    """

    def __init__(self, xi, yi, t, pol, unit=None):
        """Summary
        Args:
            xi (TYPE): Description
            yi (TYPE): Description
            t (TYPE): Description
            pol (TYPE): Description
        """
        if unit is not None:
            self.t = np.squeeze(np.asarray(t)) * unit
        else:
            try:
                if t.dim == second.dim:
                    self.t = t

                else:
                    # this means it has a brian2 dim that is not second
                    # or it is of some other type that has a .dim!
                    raise Exception('t does not have time as dimension/unit')
            except AttributeError:
                self.t = np.squeeze(np.asarray(t)) * ms

        self.xi = np.squeeze(xi)
        self.yi = np.squeeze(yi)
        self.pol = np.squeeze(pol)


class Plotter2d(object):
    """
    Attributes:
        cols (TYPE): Description
        dims (TYPE): Description
        mask (TYPE): Description
        plotrange (TYPE): Description
        pol (TYPE): Description
        rows (TYPE): Description
        shape (TYPE): Description
    """

    def __init__(self, monitor, dims, plotrange=None, frames=None, frame_timestamps=None):
        """Summary
        Args:
            monitor (TYPE): Description
            dims (TYPE): Description
            plotrange (None, optional): Description
        """
        self.rows = dims[0]
        self.cols = dims[1]
        self.dims = dims

        self._t = monitor.t  # times of spikes
        self.shape = (dims[0], dims[1], len(monitor.t))

        self.monitor = monitor  # mainly for debugging!

        # self.name = monitor.name
        try:  # that should work if the monitor is a Brian2 Spikemonitor
            self._i = monitor.i  # neuron index number of spike
            # print(self._i)
            self._xi, self._yi = np.unravel_index(self._i, (dims[0], dims[1]))
            # assert(len(self._i) == len(self._t))
        except ValueError as e:
            print('You probably did not set the correct dimensions for your input!')
            raise e
        except AttributeError:  # that should work, if it is a DVSmonitor (it has xi and yi instead of y)
            self._xi = np.asarray(monitor.xi, dtype='int')
            self._yi = np.asarray(monitor.yi, dtype='int')
            self._i = np.ravel_multi_index((self._xi, self._yi), dims)  # neuron index number of spike
            try:  # check, if _t has a unit (dvs raw data is given in ms)
                self._t[0].dim
            except:
                self._t = self._t * ms

        try:
            self._pol = monitor.pol
        except:
            self._pol = np.zeros_like(self._i)

        self.mask = range(len(monitor.t))  # [True] * (len(monitor.t))

        self._plotrange = (0 * ms, 0 * ms)
        self.set_range(plotrange)

        self._frames = frames
        self._frame_timestamps = frame_timestamps

    @property
    def plotrange(self):
        return self._plotrange

    @plotrange.setter
    def plotrange(self, plotrange):
        self.set_range(plotrange)

    @property
    def pol(self):
        """Summary
        Returns:
            TYPE: Description
        """
        return self._pol[self.mask]

    @property
    def t(self):
        """Summary
        Returns:
            TYPE: Description
        """
        return self._t[self.mask]

    @property
    def t_(self):
        """
        unitless t in ms
        Returns:
            TYPE: Description
        """
        return self._t[self.mask] / ms

    @property
    def i(self):
        """Summary
        Returns:
            TYPE: Description
        """
        return self._i[self.mask]

    @property
    def xi(self):
        """Summary
        Returns:
            TYPE: Description
        """
        return self._xi[self.mask]

    @property
    def yi(self):
        """Summary
        Returns:
            TYPE: Description
        """
        return self._yi[self.mask]

    @property
    def frames(self):
        return self._frames[np.where((self._frame_timestamps < self.plotrange[1]) &
                                     (self._frame_timestamps > self.plotrange[0]))]

    @property
    def frame_timestamps(self):
        return self._frame_timestamps[np.where((self._frame_timestamps < self.plotrange[1]) &
                                               (self._frame_timestamps > self.plotrange[0]))]

    @property
    def plotlength(self):
        """Summary
        Returns:
            TYPE: Description
        """
        # if self.plotrange is not None:
        plotlength = self.plotrange[1] - self.plotrange[0]
        # else:
        #    plotlength = np.max(self.t)
        return plotlength

    def plotshape(self, dt):
        """Summary
        Args:
            dt (TYPE): Description
        Returns:
            TYPE: Description
        """
        plottimesteps = int(np.ceil(0.0001 + self.plotlength / dt))
        return plottimesteps, self.dims[0], self.dims[1]
    #
    def set_range(self, plotrange=None):
        '''
        set a range with unit that is applied for all computations with this monitor
        Args:
            plotrange (TYPE): Description
        '''
        if plotrange is None:
            self.mask = range(len(self._t))  # slice(len(self._t))  # [True] * (len(self._t))
            if len(self.t) > 0:
                self._plotrange = (np.min(self.t), np.max(self.t))
            else:
                self._plotrange = (0 * ms, 0 * ms)
        else:
            self._plotrange = plotrange
            self.mask = np.where((self._t <= plotrange[1]) & (self._t >= plotrange[0]))[0]

    def get_sparse3d(self, dt, align_to_min_t=True):
        if len(self.t) > 0:
            if align_to_min_t:
                min_t = np.min(self.t)
            else:
                min_t = 0 * ms

            try:
                sparse_spikemat = sparse.COO((np.ones(len(self.t), dtype=int), ((self.t - min_t) / dt, np.asarray(self.xi,dtype=int), np.asarray(self.yi,dtype=int))), shape=self.plotshape(dt))
            except:
                coords = ((self.t - min_t) / dt, np.asarray(self.xi,dtype=int), np.asarray(self.yi,dtype=int))
                coords = np.asarray(coords, dtype=int)
                data = np.ones(len(self.t), dtype=int)
                plotshape = self.plotshape(dt)

                sparse_spikemat = sparse.COO(
                    coords=coords,
                    data=data,
                    shape=plotshape)
        else:
            print('Your monitor is empty!')
            # just create a matrix of zeros, hope, this does not lead to other problems
            sparse_spikemat = sparse.COO(([0], ([0], [0], [0])), shape=self.plotshape(dt))
        return sparse_spikemat


    def get_dense3d(self, dt):

        sparse3d = self.get_sparse3d(dt)
        return sparse3d.todense()

    def get_filtered(self, dt, filtersize):

        dense3d = self.get_dense3d(dt)
        filtered = ndimage.uniform_filter1d(dense3d, size=int(filtersize / dt),
                                            axis=0, mode='constant') * second / dt
        return filtered

    def plot3d(self, plot_dt=defaultclock.dt, filtersize=10 * ms):

        try:
            video_filtered = self.get_filtered(plot_dt, filtersize)
        except MemoryError:
            raise MemoryError("the dt you have set would generate a too large matrix for your memory")

        return video_filtered
