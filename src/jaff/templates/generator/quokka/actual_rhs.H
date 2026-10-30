#ifndef actual_rhs_H
#define actual_rhs_H
/// \file actual_rhs.H
/// \brief The right-hand side and Jacobian of a jaff-generated network, on
/// Quokka's burn state.

#include <cmath>

#include <AMReX_Array.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include "microphysics/integrator_layout.hpp"

// Network symbols mapped onto the burn state, shared by the RHS and the
// Jacobian. jaff's energy row is the volumetric dE/dt; Quokka integrates the
// specific internal energy e = E / rho, so the functions below divide that row
// by state.rho.
$$JAFF DEFINE quokka {
    REPLACE {
        "radeden\[\s*(\d+)\s*\]" -> "state.rn[2*\1]"
        "photden\[\s*(\d+)\s*\]" -> "state.rn[2*\1]"
        "rflux\[\s*(\d+)\s*\]" -> "state.rn[2*\1+1]"
        "nden\[\s*(\d+)\s*\]" -> "state.xn[\1]"
        "tgas" -> "state.T"
        "c_hat" -> "state.c_hat"
    }
    THERMAL { MODE -> dedt  ENERGY -> volumetric }
    RADIATION { ENABLED -> True }
}

template <typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_INLINE void actual_rhs(
    const BurnT &state,
    amrex::Array1D<amrex::Real, 1,
                   quokka::microphysics::IntegratorLayout<
                       typename BurnT::problem_type, BurnT>::neqs> &ydot) {
  $$JAFF REPEAT idx, rhs, cse IN rhses { USE quokka }
  const amrex::Real cse$idx$ = $cse$;
  ydot($idx+1$) = $rhs$;
  $$JAFF END
  $$JAFF SUB nspec
  ydot($nspec+1$) /= state.rho;
  $$JAFF END
}

template <typename BurnT, class MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_INLINE void actual_jac(const BurnT &state,
                                                   MatrixType &jac) {
  using Layout = quokka::microphysics::IntegratorLayout<
      typename BurnT::problem_type, BurnT>;

  $$JAFF REPEAT idx, expr, cse IN jacobian { USE quokka }
  const amrex::Real cse$idx$ = $cse$;
  jac($idx+1$, $idx+1$) = $expr$;
  $$JAFF END
  // jaff's energy row and column are those of E = rho e: the row is d(dE/dt)
  // and the column d/dE. Change them to e, with rho fixed during the burn: the
  // row divides by rho, the column multiplies by rho, and d(de/dt)/de =
  // d(dE/dt)/dE is unchanged.
  $$JAFF SUB nspec
  for (int i = 1; i <= Layout::neqs; i++) {
    if (i != $nspec+1$) {
      jac($nspec+1$, i) /= state.rho;
      jac(i, $nspec+1$) *= state.rho;
    }
  }
  $$JAFF END
}

#endif
