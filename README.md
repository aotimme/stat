stat
====

basic statistical functions in golang


Distributions
-------------

Many univariate and multivariate distributions, implemented as structs, with the
following methods

* `NewDistribution`
* `Sample`
* `Pdf`
* `LogPdf`

Note that these are all still WIP, either done when I have the chance or when I
to implement a new function.


N.B.
----

Many multivariate distributions rely on a good matrix library for tools like
LU decomposition, eigen decomposition, and inverses. The matrix library used
here is [skelterjohn/go.matrix](https://github.com/skelterjohn/go.matrix).
This may not be the fastest library, as it is all implemented in golang, rather
than relying on BLAS/LAPACK implementations (see
[gonum/matrix/mat64](https://github.com/gonum/matrix) as an example). However,
it is the most complete and straightforward matrix library for golang I have
thus far come across.
