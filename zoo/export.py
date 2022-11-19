from pathlib import Path
import warnings
import subprocess
import sys

import torch

from . import models
from .util import to_export
from .logger import *

formats = ['ts', 'onnx', 'vmfb', 'tvm']

def export(_, i, log_level):
    logging.basicConfig(level=log_level)

    mod_cls = to_export[i]

    logging.info('='*70)

    name = mod_cls.__name__
    fmtPaths = dict()
    for f in formats:
        root = Path(f)
        root.mkdir(exist_ok=True)
        path = root / (name + '.' + f)
        fmtPaths[f] = path

    INFO(f'constructing module {name}')
    mod = mod_cls()
    mod.eval()

    INFO(f'running module {name}')
    inp = torch.zeros(mod.input_shape)
    result = mod(inp)
    shapes = (result.shape, mod.output_shape)
    assert all(s==shapes[0] for s in shapes), shapes
    # DEBUG(result)

    ### torchscript export
    script_mod = torch.jit.script(mod)
    tvm_input_mod = torch.jit.trace(mod, inp).eval()
    torch.jit.save(script_mod, fmtPaths.get('ts'))
    DONE(f'torchscript: jitted module {name} to {fmtPaths.get("ts")}')

    ### ONNX export
    try:
        with warnings.catch_warnings():
            # ignore warnings about batch size
            warnings.simplefilter("ignore")
            torch.onnx.export(mod, inp, fmtPaths.get('ts'), verbose=False)
        DONE(f'onnx: exported module {name} to {fmtPaths.get("onnx")}')

        ### tflite export
        python = sys.executable.replace('zoo', 'zoo-tf')
        result = subprocess.run([python, 'zoo/export_tf.py', fmtPaths.get("onnx")],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

        DEBUG(result.stdout)
        DEBUG(result.stderr)

        DONE(f'tflite: exported {name}')

    except torch.onnx.errors.UnsupportedOperatorError as e:
        FAIL(f'onnx: exporting {name} to {fmtPaths.get("onnx")} failed (unsupported operator)')
        DEBUG(e)
        FAIL(f'tflite: exporting {name} not attempted (onnx failed)')


    except subprocess.CalledProcessError as e:
        FAIL(f'tflite: exporting {name} failed')
        DEBUG(e)
        DEBUG(e.stdout)
        DEBUG(e.stderr)
        
    ### IREE export


    try:
        
        import torch_mlir
        import iree.compiler.tools

        tosa_mod = torch_mlir.compile(mod, inp, output_type=torch_mlir.OutputType.TOSA, use_tracing=False, extra_args="-mlir-print-ir-after-all")
        # print(tosa_mod)
        iree_mod = iree.compiler.tools.compile_str(str(tosa_mod), input_type="tosa", target_backends=["llvm-cpu"], extra_args="-mlir-print-ir-after-all") #extra_args="–mlir-print-ir-after-all")
        # INFO(tosa_mod)
        # INFO(tosa_mod.operation.get_asm(large_elements_limit=10))
        with open(fmtPaths.get('vmfb'), 'wb') as f:
            f.write(iree_mod)
        DONE(f'torch_mlir: exported module {name} to {fmtPaths.get("vmfb")}')

    except ImportError:
        FAIL(f'torch_mlir: exporting {name} to {str(fmtPaths.get("vmfb"))} failed (torch-mlir not available)')    
    except torch_mlir.compiler_utils.TorchMlirCompilerError as e:
        FAIL(f'torch_mlir: exporting {name} to {fmtPaths.get("vmfb")} failed (compiler error)')
        DEBUG(e.value)
    except Exception as e:
        FAIL(f'torch_mlir: exporting {name} to {fmtPaths.get("vmfb")} failed')
        DEBUG(e)
       
    ### TVM export

    try:
        import tvm
        from tvm import relay

        input_name = "input0"
        shape_list = [(input_name, inp.size())]

        tvm_mod, params = relay.frontend.from_pytorch(tvm_input_mod, shape_list)

        target = "llvm "

        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(tvm_mod, target=target, params=params)
        lib.export_library(fmtPaths.get("tvm"))

        DONE(f'tvm: exported " + {str(fmtPaths.get("tvm"))}')
    except ImportError:
        FAIL(f'tvm: exporting {name} to {str(fmtPaths.get("tvm"))} failed (tvm not available)') 
    except NotImplementedError as e:
        FAIL(f'Ttvm: exporting {name} failed: {e}')



