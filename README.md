# embedded-model-zoo

<!-- setup without torch-mlir with release pytorch:

`conda create -f environment.yml`
`conda activate zoo`
`pip install -e .`
`python -m zoo` -->

this repo is a python package. running it as a script will define and export various neural network graphs to various formats.

these neural networks are not trained models and are somewhat arbitary in architecture. thus, absolute performance numbers are not meaningful. but they should provide a sense of the relative presence/performance of various operators between formats and runtimes.

# setup with torch-mlir and nightly pytorch on arm64 mac:

`CONDA_SUBDIR=osx-64 conda create -n zoo python pip fire`
`conda activate zoo`
`conda config --env --set subdir osx-64`
`pip install --pre torch-mlir -f https://github.com/llvm/torch-mlir/releases --extra-index-url https://download.pytorch.org/whl/nightly/cpu`

may need this workaround:

https://github.com/llvm/torch-mlir/issues/853#issuecomment-1148237757