# batch gen
import random
import h5py
import numpy as np
from scipy.ndimage.interpolation import rotate, shift, affine_transform, zoom
from numpy.random import random_sample, rand, random_integers, uniform
# import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
# from astropy.nddata.utils import block_reduce
# from staintools.LuminosityStandardizer import standardize
import staintools
import matplotlib.pyplot as plt
import scipy
from copy import deepcopy


# quite slow -> Don't use this! Not optimized and doesn't fit our problem!
def add_affine_transform2(input_im, output, max_deform):
    random_20 = uniform(-max_deform, max_deform, 2)
    random_80 = uniform(1 - max_deform, 1 + max_deform, 2)

    mat = np.array([[1, 0, 0],
                    [0, random_80[0], random_20[0]],
                    [0, random_20[1], random_80[1]]]
                   )
    input_im[:, :, :, 0] = affine_transform(input_im[:, :, :, 0], mat, output_shape=np.shape(input_im[:, :, :, 0]))
    output[:, :, :, 0] = affine_transform(output[:, :, :, 0], mat, output_shape=np.shape(input_im[:, :, :, 0]))
    output[:, :, :, 1] = affine_transform(output[:, :, :, 1], mat, output_shape=np.shape(input_im[:, :, :, 0]))

    output[output < 0.5] = 0
    output[output >= 0.5] = 1

    return input_im, output


"""
###
input_im:		input image, 5d ex: (1,64,256,256,1) , (dimi0, z, x, y, channel)
output:			ground truth, 5d ex: (1,64,256,256,2), (dimi0, z, x, y, channel)
max_shift:		the maximum amount th shift in a direction, only shifts in x and y dir
###
"""


# faster and sexier? - mvh André :)
def add_shift2(input_im, output, max_shift):
    # randomly choose which shift to set for each axis (within specified limit)
    sequence = [round(uniform(-max_shift, max_shift)), round(uniform(-max_shift, max_shift)), 0]

    # apply shift to RGB-image
    input_im = shift(input_im.copy(), sequence, order=0, mode='constant')
    output[..., 1] = shift(output[..., 1], sequence[:2], order=0, mode='constant')
    output[..., 0] = shift(output[..., 0], sequence[:2], order=0, mode='constant', cval=1)

    return input_im, output


"""
####
input_im:		input image, 5d ex: (1,64,256,256,1) , (dimi0, z, x, y, channel)
output:			ground truth, 5d ex: (1,64,256,256,2), (dimi0, z, x, y, channel)
min/max_angle: 	minimum and maximum angle to rotate in deg, positive integers/floats.
####
"""


# -> Only apply rotation in image plane -> faster and unnecessairy to rotate xz or yz

def add_rotation2(input_im, output, max_angle):
    # randomly choose how much to rotate for specified max_angle
    angle_xy = round(uniform(-max_angle, max_angle))

    # rotate chunks
    input_im = rotate(input_im, angle_xy, axes=(0, 1), reshape=False, mode='constant', order=1)
    output[..., 1] = rotate(output[..., 1], angle_xy, axes=(0, 1), reshape=False, mode='constant', order=1)

    # threshold gt again, as we used linear interpolation after rotation
    output[..., 1][output[..., 1] <= 0.5] = 0
    output[..., 1][output[..., 1] > 0.5] = 1
    output[..., 0] = 1 - output[..., 1]
    output = output.astype(int).astype(np.float32)

    return input_im, output


"""
flips the array along random axis, no interpolation -> super-speedy :)
"""


def add_flip2(input_im, output):
    # randomly choose whether or not to flip
    if (random_integers(0, 1) == 1):
        # randomly choose which axis to flip against
        flip_ax = random_integers(0, high=1)

        # flip CT-chunk and corresponding GT
        input_im = np.flip(input_im, flip_ax)
        output = np.flip(output, flip_ax)

    return input_im, output


"""
performs intensity transform on the chunk, using gamma transform with random gamma-value
"""


def add_gamma2(input_im, output, r_limits):
    # limits
    r_min, r_max = r_limits

    # randomly choose gamma factor
    r = uniform(r_min, r_max)

    # RGB: float [0,1] -> uint8 [0,1]
    input_im = (np.round(255. * input_im.copy())).astype(np.uint8)

    # RGB -> HSV
    input_im = cv2.cvtColor(input_im, cv2.COLOR_RGB2HSV).astype(np.float32)

    input_im[..., 2] = np.clip(np.round(input_im[..., 2] ** r), a_min=0, a_max=255)

    # HSV -> RGB
    input_im = cv2.cvtColor(input_im.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # need to normalize again after augmentation
    input_im = (input_im.astype(np.float32) / 255)

    return input_im, output


def add_scaling2(input_im, output, r_limits):
    min_scaling, max_scaling = r_limits
    scaling_factor = np.random.uniform(min_scaling, max_scaling)

    def crop_or_fill(image, shape):
        image = np.copy(image)
        for dimension in range(2):
            if image.shape[dimension] > shape[dimension]:
                # Crop
                if dimension == 0:
                    image = image[:shape[0], :]
                elif dimension == 1:
                    image = image[:, :shape[1], :]
            else:
                # Fill
                if dimension == 0:
                    new_image = np.zeros((shape[0], image.shape[1], shape[2]))
                    new_image[:image.shape[0], :, :] = image
                elif dimension == 1:
                    new_image = np.zeros((shape[0], shape[1], shape[2]))
                    new_image[:, :image.shape[1], :] = image
                image = new_image
        return image

    input_im[..., :3] = crop_or_fill(scipy.ndimage.zoom(input_im[..., :3], [scaling_factor, scaling_factor, 1], order=1), input_im[..., :3].shape)  # RGB only
    input_im[..., 3] = np.squeeze(crop_or_fill(scipy.ndimage.zoom(np.expand_dims(input_im[..., 3], axis=-1), [scaling_factor, scaling_factor, 1], order=0), input_im.shape[:2] + (1,)), axis=-1)  # heatmap only
    output = crop_or_fill(scipy.ndimage.zoom(output, [scaling_factor, scaling_factor, 1], order=0), output.shape)

    return input_im, output


def add_brightness_mult2(input_im, output, r_limits):
    # limits
    r_min, r_max = r_limits

    # randomly choose multiplication factor
    r = uniform(r_min, r_max)

    # RGB: float [0,1] -> uint8 [0,1]
    input_im = (np.round(255. * input_im.copy())).astype(np.uint8)

    # RGB -> HSV
    input_im = cv2.cvtColor(input_im, cv2.COLOR_RGB2HSV).astype(np.float32)

    input_im[..., 2] = np.clip(np.round(input_im[..., 2] * r), a_min=0, a_max=255)

    # HSV -> RGB
    input_im = cv2.cvtColor(input_im.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # need to normalize again after augmentation
    input_im = (input_im.astype(np.float32) / 255)

    return input_im, output


def add_HEstain2(input_im, output):
    # RGB: float [0,1] -> uint8 [0,1] to use staintools
    input_im = (np.round(255. * input_im.astype(np.float32))).astype(np.uint8)
    # input_im = input_im.astype(np.uint8)

    # standardize brightness (optional -> not really suitable for augmentation?
    # input_im = staintools.LuminosityStandardizer.standardize(input_im)

    # define augmentation algorithm -> should only do this the first time!
    if not 'augmentor' in globals():
        global augmentor
        # input_im = input_im[...,::-1]
        # augmentor = staintools.StainAugmentor(method='vahadane', sigma1=0.2, sigma2=0.2) # <- best, but slow
        augmentor = staintools.StainAugmentor(method='macenko', sigma1=0.1, sigma2=0.1)  # <- faster but worse

    # fit augmentor on current image
    augmentor.fit(input_im)

    # extract augmented image
    input_im = augmentor.pop()

    input_im = input_im.astype(np.float32) / 255.

    return input_im, output


'''
def add_HEstain2_all(input_im, output):
	input_shape = input_im.shape

	# for each image in stack -> transform to RGB uint8
	for i in range(input_im.shape[0]):
		input_im[i] *= 255
	input_im = input_im.astype(np.uint8)

	# define augmentation algorithm -> should only do this the first time!
	if not 'augmentor' in globals():
		global augmentor
		augmentor = staintools.StainAugmentor(method='vahadane', sigma1=0.2, sigma2=0.2)

	# apply augmentation on all slices
	augmentor.fit(input_im)

	# for each image extract augmented images
	input_out = np.zeros(input_shape)
	#for i in range()
'''


def add_rotation2_ll(input_im, output):
    # randomly choose rotation angle: 0, +-90, +,180, +-270
    k = random_integers(0, high=3)  # -> 0 means no rotation

    # rotate
    input_im = np.rot90(input_im, k)
    output = np.rot90(output, k)

    return input_im, output


def add_hsv2(input_im, output, max_shift):
    # RGB: float [0,1] -> uint8 [0,1]
    input_im = (np.round(255. * input_im.copy())).astype(np.uint8)

    # RGB -> HSV
    input_im = cv2.cvtColor(input_im, cv2.COLOR_RGB2HSV)

    input_im = input_im.astype(np.float32)

    ## augmentation, only on Hue and Saturation channel
    # hue

    input_im[..., 0] = np.mod(input_im[..., 0] + round(uniform(-max_shift, max_shift)), 180)

    # saturation
    input_im[..., 1] = np.clip(input_im[..., 1] + round(uniform(-max_shift, max_shift)), a_min=0, a_max=255)

    # input_im = (np.round(255*maxminscale(input_im.astype(np.float32)))).astype(np.uint8)

    # input_im = np.round(input_im).astype(np.uint8)
    input_im = input_im.astype(np.uint8)

    # HSV -> RGB
    input_im = cv2.cvtColor(input_im, cv2.COLOR_HSV2RGB)

    # need to normalize again after augmentation
    input_im = (input_im.astype(np.float32) / 255)

    return input_im, output


import numba as nb


@nb.jit(nopython=True)
def sc_any(array):
    for x in array.flat:
        if x:
            return True
    return False


def maxminscale(tmp):
    if sc_any(tmp):
        tmp = tmp - np.amin(tmp)
        tmp = tmp / np.amax(tmp)
    return tmp


"""
aug: 		dict with what augmentation as key and what degree of augmentation as value
		->  'rotate': 20 , in deg. slow
		->	'shift': 20, in pixels. slow
		->	'affine': 0.2 . should be between 0.05 and 0.3. slow
		->	'flip': 1, fast
"""


def batch_gen2(file_list, batch_size, aug={}, shuffle_list=True, epochs=1):
    while (True):
        batch = 0
        for filename in file_list:
            file = h5py.File(filename, 'r')
            input_shape = file['data'].shape
            output_shape = file['label'].shape
            file.close()

        # for each epoch, clear batch
        im = np.zeros((batch_size, input_shape[1], input_shape[2], input_shape[3]))
        gt = np.zeros((batch_size, output_shape[1], output_shape[2], output_shape[3]))

        for i in range(epochs):

            if shuffle_list:
                random.shuffle(file_list)

            for filename in file_list:
                file = h5py.File(filename, 'r')
                input_im = np.array(file['data'], dtype=np.float32)
                output = np.array(file['label'], dtype=np.float32)
                heatmap = np.array(file['heatmap'], dtype=np.float32)
                file.close()

                input_im = np.squeeze(input_im, axis=0)
                output = np.squeeze(output, axis=0)
                heatmap = np.squeeze(heatmap, axis=0)

                input_im = np.concatenate([input_im, heatmap], axis=-1)

                # preprocessing
                # input_im = input_im[0] # <- slow!
                # input_im = np.squeeze(input_im, axis=0)
                # RGB uint8 (to use staintools)
                # input_im = (np.round(255 * maxminscale(input_im.copy()))).astype(np.uint8)

                # standardize brightness
                # input_im = staintools.LuminosityStandardizer.standardize(input_im.astype(np.uint8)).astype(np.float32)

                # maxminscale # <- something wrong with this for RGB images???
                # input_im[pat] = maxminscale(input_im[pat].copy())

                # cv2.imshow('image', cv2.cvtColor(input_im[0], cv2.COLOR_RGB2BGR))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # input_im = np.expand_dims(input_im, axis=0)

                # del input_im, output

                # apply specified agumentation on both image stack and ground truth
                if 'rotate_ll' in aug:
                    input_im, output = add_rotation2_ll(input_im, output)

                if 'flip' in aug:
                    input_im, output = add_flip2(input_im, output)

                if 'stain' in aug:  # <- do this first
                    input_im, output = add_HEstain2(input_im, output)

                if 'hsv' in aug:  # <- do this first
                    input_im, output = add_hsv2(input_im, output, aug['hsv'])

                if 'gamma' in aug:
                    input_im, output = add_gamma2(input_im, output, aug['gamma'])

                if 'mult' in aug:
                    input_im, output = add_brightness_mult2(input_im, output, aug['mult'])

                if 'scale' in aug:
                    input_im, output = add_scaling2(input_im, output, aug['scale'])

                if 'rotate' in aug:  # -> do this last maybe?
                    input_im, output = add_rotation2(input_im, output, aug['rotate'])

                if 'affine' in aug:
                    input_im, output = add_affine_transform2(input_im, output, aug['affine'])

                if 'shift' in aug:
                    input_im, output = add_shift2(input_im, output, aug['shift'])

                # normalize at the end
                # im[batch] = im[batch] / 255.

                # print(input_im.shape)
                # print(output.shape)

                # fig, ax = plt.subplots(1, 3)
                ##ax[0].imshow(input_im)
                # ax[1].imshow(output[..., 0], cmap="gray")
                # ax[2].imshow(output[..., 1], cmap="gray")
                # plt.show()

                im[batch] = np.expand_dims(input_im, axis=0)
                gt[batch] = np.expand_dims(output, axis=0)

                del input_im, output

                batch += 1
                if batch == batch_size:
                    batch = 0
                    yield im, gt


def batch_length(file_list):
    length = len(file_list)
    print('images in generator:', length)
    return length


# file_list, batch_size, aug={}, shuffle_list=True, epochs=1


# @threadsafe_generator
from tensorflow.python.keras.utils.data_utils import Sequence
import multiprocessing as mp
from timeit import time
class mpBatchGeneratorCustom(Sequence):
    def __init__(self, file_list, batch_size=1, verbose=False, aug=None, input_shape=(), nb_classes=2, N=1, classes=None, max_q_size=20, max_proc=8, heatmap_guiding=False,
                 deep_supervision=False, hprob=False, arch=None):
        self.file_list = file_list
        self.batch_size = batch_size
        self.sample_list = []
        self.verbose = verbose
        self.transforms = []
        self.batch_transforms = {}
        self.aug = aug
        self.N = N
        self.classes = classes
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.output_shape = (self.batch_size, self.nb_classes)
        self.patch_list = []
        self.max_q_size = max_q_size
        self.max_proc = max_proc
        self.heatmap_guiding = heatmap_guiding
        self.deep_supervision = deep_supervision
        self.hprob = hprob
        self.arch = arch

        file = h5py.File(file_list[0], 'r')
        self.input_shape = file['data'].shape
        self.output_shape = file['label'].shape
        file.close()

    #'''
    def __len__(self):
        return int(np.ceil(self.N / self.batch_size))

    def __getitem__(self, batch_index):
        sample_indices_for_batch = self.file_list[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        return self._generate_batch(sample_indices_for_batch)  # generator batch based on selected patch indices

    '''
    def __iter__(self):
        """Creates an infinite generator that iterate over the Sequence.
        Yields:
          Sequence items.
        """

        while True:
            for item in (self[i] for i in range(len(self))):
                yield item
    '''

    #'''

    def mpGenerator(self):
        """ Use multiprocessing to generate batches in parallel. """
        try:
            #print("starting queue")
            queue = mp.Queue(maxsize=self.max_q_size)

            # define producer (putting items into queue)
            def producer():

                try:
                    #print("trying")

                    seed_N = None
                    #self.generate_random_patch(seed_N)

                    inputs = []
                    for i in range(self.batch_size):
                        inputs.append(self.generate_random_patch(seed_N))

                    X, y = self._generate_batch(inputs)
                    queue.put((X, y))

                except:
                    print("Nothing here")

            processes = []

            def start_process():
                print("starting process")
                for i in range(len(processes), self.max_proc):
                    thread = mp.Process(target=producer)
                    time.sleep(0.01)
                    thread.start()
                    processes.append(thread)

            # run as consumer (read items from queue, in current thread)
            while True:
                processes = [p for p in processes if p.is_alive()]

                if len(processes) < self.max_proc:
                    start_process()

                yield queue.get()

        except:
            # print("Finishing")
            for th in processes:
                th.terminate()
            queue.close()
            raise

    def on_epoch_end(self):
        self.on_epoch_begin()  # generate patch_list before next epoch starts

    def _transform_batch(self, input_im, output):
        aug = self.aug

        # apply specified agumentation on both image stack and ground truth
        if 'rotate_ll' in aug:
            if np.random.choice([0, 1]) == 1:
                input_im, output = add_rotation2_ll(input_im, output)

        if 'flip' in aug:
            if np.random.choice([0, 1]) == 1:
                input_im, output = add_flip2(input_im, output)

        if 'stain' in aug:  # <- do this first
            if np.random.choice([0, 1]) == 1:
                input_im[..., :3], output = add_HEstain2(input_im[..., :3], output)  # assumed first three channels correspond to the RGB image

        if 'hsv' in aug:  # <- do this first
            if np.random.choice([0, 1]) == 1:
                input_im[..., :3], output = add_hsv2(input_im[..., :3], output, aug['hsv'])

        if 'gamma' in aug:
            if np.random.choice([0, 1]) == 1:
                input_im, output = add_gamma2(input_im, output, aug['gamma'])

        if 'mult' in aug:
            if np.random.choice([0, 1]) == 1:
                input_im[..., :3], output = add_brightness_mult2(input_im[..., :3], output, aug['mult'])

        if 'scale' in aug:
            if np.random.choice([0, 1]) == 1:
                input_im, output = add_scaling2(input_im, output, aug['scale'])

        if 'rotate' in aug:  # -> do this last maybe?
            if np.random.choice([0, 1]) == 1:
                input_im, output = add_rotation2(input_im, output, aug['rotate'])

        if 'affine' in aug:
            if np.random.choice([0, 1]) == 1:
                input_im, output = add_affine_transform2(input_im, output, aug['affine'])

        if 'shift' in aug:
            if np.random.choice([0, 1]) == 1:
                input_im, output = add_shift2(input_im, output, aug['shift'])

        return input_im, output

    # def __iter__(self):
    #    return self.__next__()

    def on_epoch_begin(self):
        np.random.shuffle(self.file_list)

    def _generate_batch(self, samples):
        # for each epoch, clear batch
        #im = np.zeros((self.batch_size, self.input_shape[1], self.input_shape[2], self.input_shape[3] + int(self.heatmap_guiding)))
        #gt = np.zeros((self.batch_size, self.output_shape[1], self.output_shape[2], self.output_shape[3]))
        im = []
        gt = []

        for batch, filename in enumerate(samples):

            file = h5py.File(filename, 'r')
            input_im = np.array(file['data'], dtype=np.float32)
            output = np.array(file['label'], dtype=np.float32)
            heatmap = np.array(file['heatmap'], dtype=np.float32)
            file.close()

            if not self.hprob:
                heatmap = (heatmap >= 0.5).astype(np.float32)  # threshold to produce binary heatmap

            input_im = np.squeeze(input_im, axis=0)
            output = np.squeeze(output, axis=0)
            heatmap = np.squeeze(heatmap, axis=0)

            if self.heatmap_guiding:
                input_im = np.concatenate([input_im, heatmap], axis=-1)

            # augment (image and GT only, no heatmap)
            #input_im[..., :3], output = self._transform_batch(input_im[..., :3], output)  # @TODO: Need to fix this to handle heatmap stuff also given as input...
            input_im, output = self._transform_batch(input_im, output)

            # if deep supervision is enabled, need to generate downsampled versions of the original GT
            if self.deep_supervision:
                hierarchical_gt = []
                hierarchical_gt.append(output)
                for i in range(1, 7):  # @TODO: Should make this generic, iterable here should be an input to the generator
                    tmp = deepcopy(output)
                    limit = int(pow(2, i))
                    new_gt = tmp[0::limit, 0::limit]
                    #new_gt = np.expand_dims(new_gt, axis=0)
                    hierarchical_gt.append(new_gt)
                output = hierarchical_gt
            #else:
                #output = np.expand_dims(output, axis=0)

            # @TODO: Create downsampled versions of the GT for deep supervision
            #   ... add some code her ...

            #im.append(np.expand_dims(input_im, axis=0))
            im.append(input_im)
            gt.append(output)

            #im[batch] = np.expand_dims(input_im, axis=0)
            #gt[batch] = np.expand_dims(output, axis=0)

        if self.deep_supervision:
            new_batches_y = []
            for c in range(len(gt[0])):
                flipped = []
                for b in range(len(gt)):
                    flipped.append(gt[b][c])
                new_batches_y.append(np.asarray(flipped))
            gt = new_batches_y.copy()
            del new_batches_y, flipped
        else:
            gt = np.array(gt)

        im = np.array(im)
        #gt = np.array(gt)

        # for DoubleU-Net, we need to provide two identical outputs
        if self.arch == "doubleunet":
            gt = [gt, gt]

        #print(im.shape)
        #for g in gt:
        #    print(g.shape)

        #print(im.shape)
        #print(gt.shape)

        #print(im.shape)

        return im, gt  # [gt[i] for i in range(gt.shape[0])])
