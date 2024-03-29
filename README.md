# Consumet: *con*structor of *su*rrogates and *met*amodels

This is a tool for creating surrogate models from a user-provided
black box, via penalized regression methods and adaptive sampling.
It uses the same sampling algorithm as the proprietary black-box
modeling tool [Alamo](http://archimedes.cheme.cmu.edu/?q=alamo),
i.e. error-maximization sampling with derivative-free optimization.
However, in constrast to e.g. Alamo, **Consumet and all of its
dependencies are completely free**. In other words, you can use
it for any purpose without having to purchase any license.

Please see the [user manual](doc/manual.pdf) for information about
how to use Consumet. For details about the technical implementation,
see [this research paper](https://dx.doi.org/10.1002/aic.17357).

The surrogate modeling tool is available as free and open-source
software under the [MIT license](LICENSE.md). This is a permissive
license that permits you to use the software for any purpose, as
long as you just give credit where appropriate. However, outside
of any legal obligations, the authors at SINTEF Energy Research
kindly request that any useful modifications you make to the code
be contributed back to us, so we can improve the tool over time.

Most of the code was developed by SINTEF Energy Research as part of
the ELEGANCY project. The exception is `src/sobol.py`; this is also
covered by an MIT license, and the authors are listed in the source.
