from read_nmnist import *
from brian2 import us, ms, second
from dvs_utils import Plotter2d, DVSmonitor
import cv2
import os
import matplotlib.pyplot as plt


# Load Data
a = read_dataset('data/00004.bin')

# Get events from data
ev_x = a.data.x
ev_y = a.data.y
ev_t = a.data.ts - a.data.ts[0]
ev_p = a.data.p.astype(int)

# Frame Size of input data
frame_height = a.height
frame_width = a.width

# Save events as images - similar to the DVS exercise #

dvs_monitor = DVSmonitor(ev_x, ev_y, ev_t, ev_p, unit=us)

# Choose plotting parameters.
# You have to select these in such a way such that you can recognise
# the digits once you save them as frames

plot_dt = 100000
filtersize = 1
xy_dimensions_dvs = [frame_height, frame_width]
start_end_times = [0, 10]

dvs_plotter = Plotter2d(dvs_monitor, dims=(xy_dimensions_dvs[0], xy_dimensions_dvs[1]),
                        plotrange=(start_end_times[0] * second, start_end_times[1] * second))

# Save event stream as numpy arrays
# video_dvs is numpy array version of events.
video_dvs = dvs_plotter.plot3d(plot_dt=plot_dt * us, filtersize=plot_dt * us * filtersize)

_, x_dim, y_dim = video_dvs.shape

# Save numpy arrays as frames in order to see if you can clearly recognise the digits from the data
save_path = 'frames'
if not os.path.exists(save_path):
    os.mkdir(save_path)

print('Saving Frames...')
for iFrame in range(len(video_dvs)):
    filename = save_path + '/frame' + str(iFrame) + '.png'
    cv2.imwrite(filename, video_dvs[iFrame])

for i in range(10):
    plt.imshow(video_dvs[i])