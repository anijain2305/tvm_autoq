import tvm
from tvm import relay
from mxnet import gluon

from common.dataset_prep import MXNetImagenetDatasetPreparator as DatasetPreparator
from common.model_compiler import compile_and_run
from common.quantize_helper import quantize
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--only_inference", action="store_true", default=False, help="Only compilation")
parser.add_argument("--only_compile", action="store_true", default=False, help="Only compilation")
args = parser.parse_args()

batch_size = 1
model_name = "resnet50_v1"
target = 'llvm'
ctx = tvm.context(target)

###############################################################################
# The calibration dataset should be an iterable object. We define the
# calibration dataset as a generator object in Python. In this tutorial, we
# only use a few samples for calibration.

def calib_dataset_iter(dataset, input_name):
    for _, record in dataset.items():
        # record[0] is tensor, record[1] is label
        yield {input_name: record[0]}

###############################################################################
# Import the model
# ----------------
# We use the Relay MxNet frontend to import a model from the Gluon model zoo.
def get_model():
    gluon_model = gluon.model_zoo.vision.get_model(model_name, pretrained=True)
    img_size = 299 if model_name == 'inceptionv3' else 224
    data_shape = (batch_size, 3, img_size, img_size)
    mod, params = relay.frontend.from_mxnet(gluon_model, {"data": data_shape})
    return mod, params


def main():
    val_path = '/home/ubuntu/tensorflow_datasets/downloads/manual/imagenet2012/val'
    num_calib_samples = 100
    num_test_samples = 1000
    dataset_preparator = DatasetPreparator(val_path, num_calib_samples, num_test_samples)
    
    val_dataset = dataset_preparator.preprocess_val(224, 'float32')

    # Original 
    fp32_mod, params = get_model()
    compile_and_run(fp32_mod, params, target, "mxnet_" + model_name + "_fp32", val_dataset, 'data', args)
    

    # # Non data aware 
    # fp32_mod, params = get_model()
    # mod = quantize(fp32_mod, params, False, None)
    # compile_and_run(mod, params, target, "mxnet_" + model_name + "_no_data", val_dataset, 'data', args)


    # Non data aware 
    calib_dataset = dataset_preparator.preprocess_calib(224, 'float32')
    c = calib_dataset_iter(calib_dataset, 'data')
    fp32_mod, params = get_model()
    mod = quantize(fp32_mod, params, True, c)
    compile_and_run(mod, params, target, "mxnet_" + model_name + "_data", val_dataset, 'data', args)


if __name__ == '__main__':
    main()
