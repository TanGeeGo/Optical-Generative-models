import numpy as np
import scipy.io as scio
import datetime

from configobj import ConfigObj

def init_params():
    '''
    Initialize network training parameters.
    tc: training_configurations
    always define x as the prior dimension (matrix rows, along the height),
    y as the following inner (last) dimension
    '''

    tc = ConfigObj()

    # basic optical parameters
    tc.c = 299792458 # 3e8 in meter per second
    tc.freq = 400e9 
    tc.ridx_air = 1.0
    tc.wlength_vc = tc.c / tc.freq              # wavelength
    tc.amp_modulation = False                   # Consider or not the absorption of the masks
    tc.ridx_mask, tc.attenu_factor = extract_material_parameter(tc.freq, tc.amp_modulation)

    # mask parameters (unit in meter)
    tc.num_masks = 5 # number of masks
    tc.object_mask_dist, tc.mask_mask_dist, tc.mask_sensor_dist = 4e-2, 20e-3, 7e-3 # unit in meter
    tc.mask_base_thick = 0.2e-3
    tc.total_x_num, tc.total_y_num = 400, 400   # number of pixels on the mask
    tc.mask_x_num, tc.mask_y_num = 240, 240     # number of trainable pixels
    tc.dx, tc.dy = 0.3e-3, 0.3e-3               # pixel size of the mask, about 0.3mm
    tc.theta0 = 75                              # limitation on the spatial frequency angle

    # training parameter
    tc.normalize_output = True
    tc.mask_init_method = 'zero'
    tc.batch_size, tc.test_batch_size = 50, 400
    tc.lr = 3e-3
    tc.max_epoch = 200
    tc.validation_ratio = 0.2
    tc.seed = None
    tc.checkpoint_save = 5
    tc.checkpoint_print = 1
    datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    tc.image_save_dir = './logs/' + datetime_str + '/image'
    tc.model_save_dir = './logs/' + datetime_str + '/model'
    tc.log_save_dir = './logs/' + datetime_str + '/tsboard'
    tc.ckpt_to_load = None

    # training data parameters
    # tc.image_data_path = './mnist_data'
    # tc.image_data_path = './0304-ddpm-butterflies-32/generation_datasets/timestep_09to10'
    # tc.image_data_path = './0307-ddpm-butterflies-160/generation_datasets/timestep_09to10'
    tc.image_data_path = 'random_noise'
    
    tc.image_transform = None                   # list ['rotate90', 'fliplr', 'flipud']
    # tc.datax_num, tc.datay_num = 28, 28         # pixel number of the input image
    # tc.datax_num, tc.datay_num = 32, 32         # pixel number of the input image
    tc.datax_num, tc.datay_num = 160, 160         # pixel number of the input image
    # tc.obj_wx, tc.obj_wy = 4.8e-2, 4.8e-2       # the total size of the image in object plane, about 48mm
    # tc.obj_wx, tc.obj_wy = 9.6e-3, 9.6e-3       # the total size of the image in object plane, about 48mm
    tc.obj_wx, tc.obj_wy = 1.92e-2, 1.92e-2       # the total size of the image in object plane, about 48mm
    tc.obj_x_num = int(np.round(tc.obj_wx / tc.dx)) # pixel number of the object plane
    tc.obj_y_num = int(np.round(tc.obj_wy / tc.dy)) # pixel number in y
    tc.image_size = 64
    return tc

def extract_material_parameter(frequency, mask_amp_modulation):
    measurement_file = scio.loadmat('./refractiveindex/RefIndexMeasurements_1.mat')
    x_freq = measurement_file['f'][0, :]
    y_refidx = measurement_file['n_plastic'][0, :]      # measured refractive index with regard of frequence x
    ridx_mask = np.interp(frequency, x_freq, y_refidx)  # determine the refractive index at the given frequency
    
    if mask_amp_modulation:
        y_attenu = measurement_file['k_plastic'][0, :]  # measured attenuation factor with regard of frequence x
        attenu_factor = np.interp(frequency, x_freq, y_attenu) # the attenuation of the given frequency
    else:
        attenu_factor = 0.

    return ridx_mask, attenu_factor

if __name__ == '__main__':
    i, k = extract_material_parameter(400e9, True)
    print(i, k)