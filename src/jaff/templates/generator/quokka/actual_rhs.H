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

template <typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_INLINE void actual_rhs(
    const BurnT &state,
    amrex::Array1D<amrex::Real, 1,
                   quokka::microphysics::IntegratorLayout<
                       typename BurnT::problem_type, BurnT>::neqs> &ydot) {
  // $JAFF REPEAT idx, rhs, cse IN rhses $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[2*\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[2*\1] REPLACE rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T REPLACE c_hat state.c_hat  RADIATION True]$
  const amrex::Real cse$idx$ = $cse$;
  ydot($idx+1$) = $rhs$;
  // $JAFF END
  // $JAFF SUB nspec
  ydot($nspec+1$) /= state.rho;
  // $JAFF END
}

template <typename BurnT, class MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_INLINE void actual_jac(const BurnT &state,
                                                   MatrixType &jac) {
  using Layout = quokka::microphysics::IntegratorLayout<
      typename BurnT::problem_type, BurnT>;

  // $JAFF REPEAT idx, expr, cse IN jacobian $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[2*\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[2*\1] REPLACE rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T REPLACE c_hat state.c_hat RADIATION True USE_DEDT True]$
  const amrex::Real cse$idx$ = $cse$;
  jac($idx+1$, $idx+1$) = $expr$;
  // $JAFF END
  // $JAFF SUB nspec
  for (int i = 1; i <= Layout::neqs; i++) {
    jac($nspec+1$, i) /= state.rho;
  }
  // $JAFF END
}

#endif
