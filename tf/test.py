try:
  # %tensorflow_version only exists in Colab.
  import tensorflow.compat.v2 as tf
except Exception:
  pass
# tf.enable_v2_behavior()
import tensorflow_hub as hub

import tvm
from tvm import relay

from common.dataset_prep import TFImagenetDatasetPreparator as DatasetPreparator
from common.model_compiler import compile_and_run
from common.quantize_helper import quantize
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--only_inference", action="store_true", default=False, help="Only compilation")
parser.add_argument("--only_compile", action="store_true", default=False, help="Only compilation")
args = parser.parse_args()

batch_size = 1
model_name = "resnet_50"
target = 'llvm -mcpu=cascadelake'
ctx = tvm.context(target)


##############################
# Original FP32 TF/Keras model
##############################
tf_hub_links = {
    "resnet_50"             : "https://tfhub.dev/tensorflow/resnet_50/classification/1",
    "resnet_v2_50"          : "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/4",
    "mobilenet_v1"          : "https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/4",
    "mobilenet_v2"          : "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4",
    "inception_v1"          : "https://tfhub.dev/google/imagenet/inception_v1/classification/4",
    "inception_v2"          : "https://tfhub.dev/google/imagenet/inception_v2/classification/4",
    "inception_v3"          : "https://tfhub.dev/google/imagenet/inception_v3/classification/4",
    "inception_v3_preview"  : "https://tfhub.dev/google/tf2-preview/inception_v3/classification/4",
    "mobilenet_v2_preview"  : "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4",
    # "efficientnet_b0"       : "https://tfhub.dev/tensorflow/efficientnet/b0/classification/1",
}


###############################################################################
# The calibration dataset should be an iterable object. We define the
# calibration dataset as a generator object in Python. In this tutorial, we
# only use a few samples for calibration.

def calib_dataset_iter(dataset, input_name):
    for _, record in dataset.items():
        # record[0] is tensor, record[1] is label
        # print(record[0])
        yield {input_name: record[0]}

###############################################################################
# Import the model
# ----------------
# We use the Relay MxNet frontend to import a model from the Gluon model zoo.
def get_model():

    relay_file = "relay.json"
    relay_params = "relay.params"
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    model = tf.keras.Sequential([
        hub.KerasLayer(tf_hub_links[model_name], output_shape=[1001])
    ])
    img_size = 299 if model_name == 'inceptionv3' else 224
    np_image = np.random.rand(1, img_size, img_size, 3).astype('float32')
    model._set_inputs(np_image)


    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="data"))

    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./.tf_saved_model/" + model_name,
                      name="frozen_graph.pb",
                      as_text=False)

    parser = tvm.relay.frontend.TFParser("./.tf_saved_model/"
                                         + model_name +  "/frozen_graph.pb")
    graph_def = parser.parse()
    mod, params = relay.frontend.from_tensorflow(graph_def,
                                                 shape={"data": (1, img_size, img_size, 3)})

    # with open(relay_file, "w") as fo:
    #     fo.write(tvm.ir.save_json(mod))
    # with open(relay_params, "wb") as fo:
    #     fo.write(relay.save_param_dict(params))

    # with open(relay_file, "r") as fi:
    #     mod = tvm.ir.load_json(fi.read())
    # with open(relay_params, "rb") as fi:
    #     params = relay.load_param_dict(fi.read())
    return mod, params


def main():
    val_path = '/home/ubuntu/tensorflow_datasets/downloads/manual/imagenet2012/val'
    num_calib_samples = 100
    num_test_samples = 1000
    dataset_preparator = DatasetPreparator(val_path, num_calib_samples, num_test_samples)
    val_dataset = dataset_preparator.preprocess_val(224, 'float32')

    # Original 
    fp32_mod, params = get_model()
    compile_and_run(fp32_mod, params, target, "tf_" + model_name + "_fp32", val_dataset, 'data', args)
    

    # # Non data aware 
    # fp32_mod, params = get_model()
    # mod = quantize(fp32_mod, params, False, None)
    # compile_and_run(mod, params, target, "tf_" + model_name + "_no_data", val_dataset, 'data', args)


    # Non data aware 
    calib_dataset = dataset_preparator.preprocess_calib(224, 'float32')
    c = calib_dataset_iter(calib_dataset, 'data')
    fp32_mod, params = get_model()
    mod = quantize(fp32_mod, params, True, c)
    compile_and_run(mod, params, target, "tf_" + model_name + "_data", val_dataset, 'data', args)


if __name__ == '__main__':
    main()
