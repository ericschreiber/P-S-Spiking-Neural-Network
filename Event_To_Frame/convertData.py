from read_nmnist import *
from brian2 import us, ms, second
from dvs_utils import Plotter2d, DVSmonitor
import cv2
import os
import matplotlib

numbers = [str(0), str(1), str(2), str(3), str(4), str(5), str(6), str(7), str(8), str(9)]

for num in numbers:
    #path = r'C:\Users\erics\Documents\Programme\PundS_Spiking_Neural_Networks\Data\Train\Train\5'
    path = os.path.join(r'C:\Users\erics\Documents\Programme\PundS_Spiking_Neural_Networks\Data\Train\Train\\' , num)

    list_of_files = []
    filenames = []
    iterator = -1

    for root, dirs, files in os.walk(path):
        for file in files:
            list_of_files.append(os.path.join(root,file))
            filenames.append(file[:-4])

    for name in list_of_files:
        # print(filenames[iterator])
        # iterator += 1
        #print(name)
        

        # Load Data
        iterator += 1
        try:
            #a = read_dataset('C:/Users/erics/Documents/Programme/PundS_Spiking_Neural_Networks/Data/Test/Test/2/00002.bin')
            a = read_dataset(name)
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

            plot_dt = 13000000
            filtersize = 2
            xy_dimensions_dvs = [frame_height, frame_width]
            start_end_times = [0, 10]

            dvs_plotter = Plotter2d(dvs_monitor, dims=(xy_dimensions_dvs[0], xy_dimensions_dvs[1]),
                                    plotrange=(start_end_times[0] * second, start_end_times[1] * second))

            # Save event stream as numpy arrays
            # video_dvs is numpy array version of events.
            video_dvs = dvs_plotter.plot3d(plot_dt=plot_dt * us, filtersize=plot_dt * us * filtersize)

            _, x_dim, y_dim = video_dvs.shape


            # Save numpy arrays as frames in order to see if you can clearly recognise the digits from the data
            if x_dim == 34:
                filename = 'C:/Users/erics/Documents/Programme/PundS_Spiking_Neural_Networks/Event_To_Frame/Converted/Train/' + num + '/frame' + filenames[iterator] + '.png'
                matplotlib.image.imsave(filename, video_dvs[0])
            else:
                print("Error in frame x_dim = 33")

        except:
            print("Error in frame " + filenames[iterator])