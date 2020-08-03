import tvm
from tvm import relay
import tvm.contrib.graph_runtime as runtime

import os
import pathlib
from pathlib import Path
import numpy as np
from .accuracy_measurement import AccuracyAggregator


class ModelCompiler(object):
    def compile(self, mod, params, target, artifact_name):
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)


            base_dir = os.getcwd() + "/compiled_models"
            pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)

            base = base_dir + "/" + artifact_name

            path_lib = base + '_deploy_lib.tar'
            path_graph =  base + '_deploy_graph.json'
            path_params = base + '_deploy_params.params'

            try:
                os.remove(path_lib)
                os.remove(path_graph)
                os.remove(path_params)
            except OSError:
                pass
            lib.export_library(path_lib)
            with open(path_graph, 'w') as fo:
                fo.write(graph)
            with open(path_params, 'wb') as fo:
                fo.write(relay.save_param_dict(params))



class ModelExecutor(object):
    def run(self, artifact_name, dataset):
        base = os.getcwd() + '/compiled_models/' + artifact_name

        path_lib = base + '_deploy_lib.tar'
        path_graph =  base + '_deploy_graph.json'
        path_params = base + '_deploy_params.params'

        graph = open(path_graph).read()
        lib = tvm.runtime.load_module(path_lib)
        params = bytearray(open(path_params, 'rb').read())

        # if debug:
        #     rt_mod = debug_runtime.create(graph, lib, ctx=tvm.cpu(0))
        #     rt_mod.load_params(params)
        #     rt_mod.run()
        #     return

        rt_mod = runtime.create(graph, lib, ctx=tvm.cpu(0))
        rt_mod.load_params(params)


        acc = AccuracyAggregator()
        for _, record in dataset.items():
            # record[0] is tensor, record[1] is label
            data, label = record
            rt_mod.set_input(**{'data': data})
            rt_mod.run()
            tvm_res = np.squeeze(rt_mod.get_output(0).asnumpy())
            tvm_pred = np.argsort(tvm_res)[-5:][::-1]
            acc.update(label, tvm_pred)

        print(artifact_name, acc.report())


def compile_and_run(mod, params, target, artifact_name, val_dataset, args):
    if args.only_compile == False and args.only_inference == False:
        mc = ModelCompiler()
        mc.compile(mod, params, target, artifact_name)
        me = ModelExecutor()
        me.run(artifact_name, val_dataset)
    elif args.only_compile:
        mc = ModelCompiler()
        mc.compile(mod, params, target, artifact_name)
    elif args.only_inference:
        me = ModelExecutor()
        me.run(artifact_name, val_dataset)
