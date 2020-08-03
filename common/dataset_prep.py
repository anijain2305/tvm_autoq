try:
  # %tensorflow_version only exists in Colab.
  import tensorflow.compat.v2 as tf
except Exception:
  pass
tf.enable_v2_behavior()

# from tensorflow import keras
import numpy as np
import glob
import random
 
import mxnet as mx
# from tflite_inference import TFLiteExecutor
# from accuracy_measurement import AccuracyAggregator

class ImagenetDatasetPreparator(object):
    def __init__(self, val_path, num_calib_samples, num_val_samples):
        all_class_path = sorted(glob.glob(val_path + '/*'))
        
        filenames = list()
        for cur_class in all_class_path:
            all_image = glob.glob(cur_class+'/*')
            filenames.extend(all_image)
        
        random.seed(0)
        random.shuffle(filenames)

        self.calib_filenames = filenames[0:num_calib_samples]
        self.val_filenames = filenames[num_calib_samples:num_calib_samples + num_val_samples]

        self.categories = dict()
        # FIXME - Get relative path
        with open("common/categories.txt", "r") as fh:
            lines = fh.readlines()
            for line in lines:
                line = line.rstrip()
                jpeg_name, index = line.split()
                self.categories[jpeg_name] = int(index)

    def _preprocess(self, filenames, data_shape, dtype='float32'):
        ret = dict()
        gt = lambda filename: self.categories[filename.split('/')[-1]]
        for filename in filenames:
            label = gt(filename)
            ret[filename] = (self.preprocess(filename, data_shape), label)
        return ret

    def preprocess_calib(self, data_shape, dtype='float32'):
        return self._preprocess(self.calib_filenames, data_shape, dtype)

    def preprocess_val(self, data_shape, dtype='float32'):
        return self._preprocess(self.val_filenames, data_shape, dtype)


class MXNetImagenetDatasetPreparator(ImagenetDatasetPreparator):
    def __init__(self, val_path, num_calib_samples, num_val_samples):
        super().__init__(val_path, num_calib_samples, num_val_samples)

    def preprocess(self, raw_image, data_shape):
        image = mx.image.imread(raw_image)
        resized = mx.image.resize_short(image, 224) #minimum 224x224 images
        cropped, crop_info = mx.image.center_crop(resized, (224, 224))
        normalized = mx.image.color_normalize(cropped.astype(np.float32)/255,
                                              mean=mx.nd.array([0.485, 0.456, 0.406]),
                                              std=mx.nd.array([0.229, 0.224, 0.225]))
        # the network expect batches of the form (N,3,224,224)
        transposed = normalized.transpose((2,0,1))  # Transposing from (224, 224, 3) to (3, 224, 224)
        batchified = transposed.expand_dims(axis=0) # change the shape from (3, 224, 224) to (1, 3, 224, 224)
        np_image = batchified.asnumpy()
        return np_image

class TFImagenetDatasetPreparator(ImagenetDatasetPreparator):
    def __init__(self, val_path, num_calib_samples, num_val_samples):
        super().__init__(val_path, num_calib_samples, num_val_samples)

    def preprocess(self, raw_image, data_shape):
        height = width = data_shape
        image = tf.io.read_file(raw_image)
        image = tf.image.decode_jpeg(image, channels=3)
        central_crop = True
        central_fraction = 0.875
        with tf.name_scope('eval_image'):
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            # if use_grayscale:
            #     image = tf.image.rgb_to_grayscale(image)
            # Crop the central region of the image with an area containing 87.5% of
            # the original image.
            if central_crop and central_fraction:
                image = tf.image.central_crop(image, central_fraction=central_fraction)
                
            if height and width:
                # Resize the image to the specified height and width.
                image = tf.expand_dims(image, 0)
                image = tf.compat.v1.image.resize(image, [height, width],
                        align_corners=False)
                image = tf.image.resize(image, [height, width])
                image = tf.squeeze(image, [0])
                # image = tf.subtract(image, 0.5)
                # image = tf.multiply(image, 2.0)
                image = tf.expand_dims(image, axis=0)
            return image


