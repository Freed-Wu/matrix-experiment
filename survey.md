---
title: A Survey on RIP
author: Wu Zhenyu (SA21006096)
institute: USTC
bibliography: refs/main.bib
---

## RIP

In linear algebra, the restricted isometry property (RIP) characterizes
matrices which are nearly orthonormal, at least when operating on sparse
vectors. The concept was introduced by Emmanuel Candes and Terence Tao[@1542412]
and is used to prove many theorems in the field of compressed sensing. [@20124]
There are no known large matrices with bounded restricted isometry constants
(computing these constants is strongly NP-hard,[@6658871] and is hard to
approximate as well[@DBLP:journals/corr/NatarajanW14]), but many random
matrices have been shown to remain bounded. In particular, it has been shown
that with exponentially high probability, random Gaussian, Bernoulli, and
partial Fourier matrices satisfy the RIP with number of measurements nearly
linear in the sparsity level.[@5658507] The current smallest upper bounds for any
large rectangular matrices are for those of Gaussian
matrices.[@DBLP:journals/corr/abs-1003-3299] Web forms to evaluate bounds for
the Gaussian ensemble are available at the Edinburgh Compressed Sensing RIC
page.

## Sparse Reconstruction

 Compressed sensing (also known as compressive sensing, compressive sampling,
 or sparse sampling) is a signal processing technique for efficiently acquiring
 and reconstructing a signal, by finding solutions to underdetermined linear
 systems. This is based on the principle that, through optimization, the
 sparsity of a signal can be exploited to recover it from far fewer samples
 than required by the Nyquist–Shannon sampling theorem. There are two
 conditions under which recovery is possible.[@Erlich2009DNASH] The first one
 is sparsity, which requires the signal to be sparse in some domain. The second
 one is incoherence, which is applied through the isometric property, which is
 sufficient for sparse signals.[@20132]

### Heuristic Solution

In order to choose a solution to such a system, one must impose extra
constraints or conditions (such as smoothness) as appropriate. In compressed
sensing, one adds the constraint of sparsity, allowing only solutions which
have a small number of nonzero coefficients. Not all underdetermined systems of
linear equations have a sparse solution. However, if there is a unique sparse
solution to the underdetermined system, then the compressed sensing framework
allows the recovery of that solution.

Compressed sensing takes advantage of the redundancy in many interesting
signals—they are not pure noise. In particular, many signals are sparse, that
is, they contain many coefficients close to or equal to zero, when represented
in some domain.[@4472240] This is the same insight used in many forms of lossy
compression.

Compressed sensing typically starts with taking a weighted linear combination
of samples also called compressive measurements in a basis different from the
basis in which the signal is known to be sparse. The results found by Emmanuel
Candès, Justin Romberg, Terence Tao and David Donoho, showed that the number of
these compressive measurements can be small and still contain nearly all the
useful information. Therefore, the task of converting the image back into the
intended domain involves solving an underdetermined matrix equation since the
number of compressive measurements taken is smaller than the number of pixels
in the full image. However, adding the constraint that the initial signal is
sparse enables one to solve this underdetermined system of linear equations.

The least-squares solution to such problems is to minimize the $L_2$ norm—that
is, minimize the amount of energy in the system. This is usually simple
mathematically (involving only a matrix multiplication by the pseudo-inverse of
the basis sampled in). However, this leads to poor results for many practical
applications, for which the unknown coefficients have nonzero energy.

To enforce the sparsity constraint when solving for the underdetermined system
of linear equations, one can minimize the number of nonzero components of the
solution. The function counting the number of non-zero components of a vector
was called the $L_0$ "norm" by David Donoho.

Candes et al. proved that for many problems it is probable that the $L_1$ norm
is equivalent to the $L_0$ norm, in a technical sense: This equivalence result
allows one to solve the $L_1$ problem, which is easier than the $L_0$ problem.
Finding the candidate with the smallest $L_1$ norm can be expressed relatively
easily as a linear program, for which efficient solution methods already exist.
When measurements may contain a finite amount of noise, basis pursuit denoising
is preferred over linear programming, since it preserves sparsity in the face
of noise and can be solved faster than an exact linear program.
