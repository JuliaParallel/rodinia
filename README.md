Rodinia Benchmark Suite
=======================

This repository hosts a fork of the Rodinia benchmark suite, version 3.1. The fork features
a clean-up of the code and its build system, as well as additional ports of select
benchmarks to the Julia programming language.

For more information, refer to [the original home
page](http://lava.cs.virginia.edu/wiki/rodinia).


Usage
-----

Execute `make` in any of the suite's main directories, or in any of the benchmark
subfolders. At every level, you can create a `Make.user` overriding specific settings (eg.
`cuda/Make.user` defining `CUDA_DIR`).

Every benchmark should contain a `run` script to start execution. Some benchmarks might also
provide a `verify` script, to generate output and check it against known-good outputs in the
`results/` folder.

Sometimes, a `profile` script is provided as well. This script outputs for every
computational kernel a comma-separated string denoting the name of the benchmark suite, the
name of the kernel, and the execution time in microseconds. This information can then be
parsed by a script, see eg. `tools/profile`.

**NOTE**: the current focus of development is on a subset of benchmarks, mostly limited to
the CUDA and Julia+CUDA suites. If you have interest in other benchmarks, or running other
suites, do check the revision history of said benchmark in one of the actively-maintained
suite subfolders (ie. `cuda/` or `julia_cuda/`).
