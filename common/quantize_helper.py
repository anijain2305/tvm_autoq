import tvm
from tvm import relay

def quantize(mod, params, data_aware, iterator):
    if data_aware:
        with relay.quantize.qconfig(calibrate_mode='avg_min_max', weight_scale='max',
                skip_conv_layers=[0]):
            mod = relay.quantize.quantize(mod, params, dataset=iterator)
    else:
        with relay.quantize.qconfig(calibrate_mode='global_scale', global_scale=7.9):
            mod = relay.quantize.quantize(mod, params)
    return mod


