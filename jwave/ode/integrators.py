from typing import Tuple

from diffrax import AbstractSolver, LocalLinearInterpolation
from diffrax.custom_types import Bool, DenseInfo, PyTree, Scalar
from diffrax.solution import RESULTS
from diffrax.term import AbstractTerm

_ErrorEstimate = None
_SolverState = None


class SemiImplicitEulerCorrected(AbstractSolver):
    """Semi-implicit Euler's method.
    Symplectic method with correction factors. Does not support adaptive step sizing. Uses 1st order local
    linear interpolation for dense/ts output.

    The step is given by

    u' = a(a*u + dt*f(t,p,args'))
    p' = b(b*u + dt*g(t,u',args'))

    where
    a, b, args' = args

    When used to solve the wave equation with a k-space correction,
    this solver is equivalent to the
    [`kSpaceFirstOrderXD` solvers of k-Wave](http://www.k-wave.org/documentation/kspaceFirstOrder3D.php).
    """
    term_structure = (AbstractTerm, AbstractTerm)
    interpolation_cls = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def init(
        self,
        terms: Tuple[AbstractTerm, AbstractTerm],
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: Tuple[PyTree, PyTree, PyTree],
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: Tuple[AbstractTerm, AbstractTerm],
        t0: Scalar,
        t1: Scalar,
        y0: Tuple[PyTree, PyTree],
        args: Tuple[PyTree, PyTree, PyTree],
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> Tuple[Tuple[PyTree, PyTree], _ErrorEstimate, DenseInfo, _SolverState,
               RESULTS]:
        del solver_state, made_jump

        term_1, term_2 = terms
        y0_1, y0_2 = y0
        a, b, args_vf = args

        # Prepare for pytree product
        control1 = term_1.contr(t0, t1)
        control2 = term_2.contr(t0, t1)
        y1_1 = (a * (a * (y0_1) + term_1.vf_prod(t0, y0_2, args_vf, control1)))
        y1_2 = (b * (b * (y0_2) + term_2.vf_prod(t0, y1_1, args_vf, control2)))

        y1 = (y1_1, y1_2)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: Tuple[AbstractTerm, AbstractTerm],
        t0: Scalar,
        y0: Tuple[PyTree, PyTree],
        args: Tuple[PyTree, PyTree, PyTree],
    ):
        """Not sure I understand what this method is for, in this context"""
        term_1, term_2 = terms
        y0_1, y0_2 = y0
        _, _, args_vf = args

        f1 = term_1.func(t0, y0_2, args_vf)
        f2 = term_2.func(t0, y0_1, args_vf)
        return (f1, f2)
