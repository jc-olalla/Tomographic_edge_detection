from scipy.ndimage.filters import laplace
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import griddata


def boxcar(array, filt_len=10):
    cumsum, moving_aves = [0], []

    for i, x in enumerate(array, 1):
        cumsum.append(cumsum[i - 1] + x)

        if i >= filt_len:
            moving_ave = (cumsum[i] - cumsum[i - filt_len]) / filt_len
            # can do stuff with moving_ave here
            moving_aves.append(moving_ave)
        else:
            moving_ave = x
            moving_aves.append(moving_ave)

    smooth_array = np.array(moving_aves)
    return smooth_array


def rotate(original_list, n_rotation):
    return original_list[n_rotation:] + original_list[:n_rotation]


def split_interface(xint, zint):
    index_min = np.argmin(xint)
    try:
        x_int_new = rotate(xint.tolist(), index_min)
        z_int_new = rotate(zint.tolist(), index_min)
    except:
        x_int_new = rotate(xint, index_min)
        z_int_new = rotate(zint, index_min)

    index_max = np.argmax(x_int_new)
    list_interfaces = []
    if index_max <= len(x_int_new) - 1:
        array1 = np.zeros((index_max+1, 2))
        array1[:, 0] = x_int_new[0:index_max+1]
        array1[:, 1] = z_int_new[0:index_max+1]
        list_interfaces.append(array1)

        len2 = len(x_int_new) - index_max -1
        array2 = np.zeros((len2, 2))
        array2[:, 0] = x_int_new[index_max+1:]
        array2[:, 1] = z_int_new[index_max+1:]
        list_interfaces.append(array2)

    return list_interfaces

class tomo_laplace():
    def __init__(self, X, Z, RHO):
        self.X = X
        self.Z = Z
        self.RHO = RHO
        self.dl = np.abs(X[0, 0] - X[0, 1])
        self.lap_edge()

    def lap_edge(self):
        self.lap_grid = laplace(self.RHO)
        # Find gradient magnitude
        sx = ndimage.sobel(self.RHO, axis=0, mode='reflect')  # this is actually not the gradient, it should be divided by 2dl?
        sy = ndimage.sobel(self.RHO, axis=1, mode='reflect')  # this is actually not the gradient
        # I should try to use only the vertical gradient.
        # Then, I could do something like np.gradient(self.RHO, self.Z, axis=0)
        # Should I take into account the sign of the gradient?
        self.sobel = np.hypot(sx, sy)

    def find_all_interfaces_with_threshold(self, T=None, nmin=4):
        points = np.vstack((np.ndarray.flatten(self.X), np.ndarray.flatten(self.Z))).transpose()
        values = np.ndarray.flatten((self.sobel))
        # Find laplace contours
        contours = measure.find_contours(self.lap_grid, 0.0)
        fig = plt.figure()
        fig.suptitle('Laplacian zero crossings')
        ax0 = fig.add_subplot(211)
        ax0.set_title('Raw contours')
        ax = fig.add_subplot(212)
        ax.set_title('Processed contours')
        count = 0
        contours2 = []
        # Interpolate gradient
        x_int_all = []
        z_int_all = []
        for n, contour in enumerate(contours):
            xint = (self.X[0, 0] + contour[:, 1] * self.dl) * (self.X[0, 1] > self.X[0, 0]) + (self.X[0, 0] - contour[:, 1] * self.dl) * (self.X[0, 1] < self.X[0, 0])
            zint = (self.Z[0, 0] + contour[:, 0] * self.dl) * (self.Z[1, 0] > self.Z[0, 0]) + (self.Z[0, 0] - contour[:, 0] * self.dl) * (self.Z[1, 0] < self.Z[0, 0])
            x_int_all.append(xint)
            z_int_all.append(zint)

        flat_xint = []
        flat_zint = []
        for sublist_x, sublist_z in zip(x_int_all, z_int_all):
            for item_x, item_z in zip(sublist_x, sublist_z):
                flat_xint.append(item_x)
                flat_zint.append(item_z)

        x_int_all_array = np.array(flat_xint)
        z_int_all_array = np.array(flat_zint)
        xz_int = np.array([x_int_all_array, z_int_all_array]).transpose()
        print('Interpolating gradient...')
        grad_int_all = griddata(points, values, xz_int)  # this takes some time to compute
        print('Done with gradient interpolation')

        # Thresholding
        low_bound = 0
        for n, contour in enumerate(contours):
            print('Analizing contour ', n + 1, ' out of ', len(contours))
            xint = (self.X[0, 0] + contour[:, 1] * self.dl) * (self.X[0, 1] > self.X[0, 0]) + (self.X[0, 0] - contour[:, 1] * self.dl) * (self.X[0, 1] < self.X[0, 0])
            zint = (self.Z[0, 0] + contour[:, 0] * self.dl) * (self.Z[1, 0] > self.Z[0, 0]) + (self.Z[0, 0] - contour[:, 0] * self.dl) * (self.Z[1, 0] < self.Z[0, 0])
            nint = len(xint)
            up_bound = low_bound + nint
            grad_int = grad_int_all[low_bound:up_bound]
            ax0.plot(xint, zint, '-x', label='n='+str(n))
            low_bound = low_bound + nint
            if T is not None:
                n_larger_than_T = len(np.where(grad_int >= T)[0])
                xint_thres = np.zeros((n_larger_than_T, ))
                zint_thres = np.zeros((n_larger_than_T, ))
                count_nT = 0
                if n_larger_than_T > 0:
                    for i_gradi, gradi in enumerate(grad_int):
                        if gradi >= T:
                            xint_thres[count_nT] = xint[i_gradi]
                            zint_thres[count_nT] = zint[i_gradi]
                            count_nT = count_nT + 1

            else:
                xint_thres = xint
                zint_thres = zint

            # Break contour lines
            try:  # I should replace this try by somethig like : if xint_thers and zint_thres are not empty
                list_interfaces_n = split_interface(xint_thres, zint_thres)
                for arr_k in list_interfaces_n:
                    if np.shape(arr_k)[0] > nmin:
                        ax.plot(arr_k[:, 0], arr_k[:, 1], '-x', label='n='+str(count))
                        contours2.append(arr_k)
                        count = count + 1
            except:
                pass

        print('Total number of interfaces:', count)
        ax.legend(ncol=5, prop={'size': 6})
        ax.set_ylim(np.min(self.Z), np.max(self.Z))
        ax0.set_ylim(np.min(self.Z), np.max(self.Z))
        ax0.grid()
        ax.grid()
        self.contours = contours2  # contains all the contours (after some filtering)

    def pick_interfaces(self, n_list, form_list=None, xmin=None, xmax=None, dxresample=0.5, nresample=200):
        if form_list == None:
            form_list = []
            for ni in n_list:
                form_list.append('n'+str(ni))

        if xmin == None:
            xmin = []
            for ni in n_list:
                xmin.append(np.min(self.contours[ni][:, 0]))
        if xmax == None:
            xmax = []
            for ni in n_list:
                xmax.append(np.max(self.contours[ni][:, 0]))

        self.form_list = form_list
        self.n_list = n_list

        x_int_list = []
        z_int_list = []
        for i, n in enumerate(self.n_list):
            x_int_list_i = []
            z_int_list_i = []
            xint = self.contours[n][:, 0]
            zint = self.contours[n][:, 1]

            # Resample is it really needed? uncomment if needed
            # points = np.column_stack((xint, zint))
            # distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
            # distance = np.insert(distance, 0, 0) / distance[-1]
            # alpha = np.linspace(0, 1, nresample)
            # fint = interp1d(distance, points, kind='quadratic', axis=0)
            # interpolated_points = fint(alpha)
            # x_int_list.append(interpolated_points[:, 0].tolist())
            # z_int_list.append(interpolated_points[:, 1].tolist())
            x_int_list.append(xint)
            z_int_list.append(zint)

        self.x_int_list = x_int_list  # contains only the selected contours
        self.z_int_list = z_int_list  # contains only the selected contours

    def plot_selected_interfaces(self, n_split=20):
        for i_list, (x_int, z_int) in enumerate(zip(self.x_int_list, self.z_int_list)):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x_int, z_int, '-x', label=str(self.n_list[i_list]))
            n_plot = int(len(x_int) / n_split)
            count = 0
            for ip in range(n_split):
                print(count)
                ax.plot(x_int[count], z_int[count], 'ro')
                ax.annotate(str(count), (x_int[count], z_int[count]))
                count = ip * n_plot

            ax.legend()
            ax.grid()



