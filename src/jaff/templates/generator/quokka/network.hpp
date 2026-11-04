#ifndef JAFF_NETWORK_HPP_
#define JAFF_NETWORK_HPP_
/// \file network.hpp
/// \brief A jaff-generated network: its constants, runtime parameters,
/// right-hand side and Jacobian. A Quokka problem names the network type with
/// Microphysics_Traits::Network.

#include <cmath> // INFINITY, for an open top band edge
#include <string_view>

#include "AMReX_Array.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "microphysics/integrator_layout.hpp"

/// The jaff network's own namespace, named after the network, so that several
/// jaff networks can be linked into one executable.
$$JAFF SUB label
namespace quokka::networks::$label$ {
$$JAFF END

// The runtime parameters of Network, read from the `network` prefix. They are
// defined in network.cpp, not inline here: HIP cannot register inline managed
// variables.

/// floor of the species and radiation after a burn, and of the mass scalars
extern AMREX_GPU_MANAGED amrex::Real small_x; // NOLINT

// Network symbols mapped onto the burn state, shared by the RHS and the
// Jacobian. jaff's energy row is the volumetric dE/dt; Quokka integrates the
// specific internal energy e = E / rho, so the functions below divide that row
// by state.rho. jaff emits a density and a flux moment per band, interleaved in
// state.rn, so the problem must integrate the photon flux.
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

$$JAFF SUB label
/// The jaff network $label$.
$$JAFF END
struct Network {
  $$JAFF SUB nspec
  static constexpr int NumSpec = $nspec$;
  $$JAFF END
  /// indices of the species in xn
  enum Species : int { // NOLINT(cppcoreguidelines-use-enum-class)
    $$JAFF REPEAT idx, specie_with_normalized_sign IN species_with_normalized_sign
    $specie_with_normalized_sign$ = $idx$,
    $$JAFF END
  };

  /// mass of each species [g]
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto SpeciesMass()
      -> amrex::GpuArray<amrex::Real, NumSpec> {
    $$JAFF REPEAT specie_mass IN specie_masses
    return {{$specie_mass$}};
    $$JAFF END
  }
  /// adiabatic index of each species, from the network's [network.eos]
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto SpeciesGamma()
      -> amrex::GpuArray<amrex::Real, NumSpec> {
    $$JAFF REPEAT specie_gamma IN specie_gammas
    return {{$specie_gamma$}};
    $$JAFF END
  }
  /// the equation of state the network assumes, as jaff names it (the type of
  /// [network.eos]); Quokka maps it to one of its EOSs
  $$JAFF SUB eos_type
  static constexpr std::string_view EOSType = "$eos_type$";
  $$JAFF END

  $$JAFF SUB nbands
  static constexpr int NumChemBands = $nbands$;
  $$JAFF END
  /// edges of the chemical bands [eV]
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr auto ChemBands()
      -> amrex::GpuArray<double, NumChemBands + 1> {
    $$JAFF REPEAT radiation_boundary IN radiation_boundaries
    return {{$radiation_boundary$}};
    $$JAFF END
  }
  /// index alpha of the photon-number spectrum n(E) ~ E^(alpha-2) within a
  /// chemical band; Quokka takes one alpha for all bands
  $$JAFF GET profile_index IN 0
  static constexpr double ChemBandsPowerLawIndex = $profile_index$;
  $$JAFF END
  $$JAFF REPEAT idx, profile_index IN profile_indices
  static_assert($profile_index$ == ChemBandsPowerLawIndex, "band $idx$: Quokka takes one profile index for all chemical bands");
  $$JAFF END

  static constexpr int NumThermalBands = 0;

  /// floor of the species and radiation after a burn, and of the mass scalars
  /// (small_x)
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto SmallX() -> amrex::Real {
    return small_x;
  }

  /// Read the runtime parameters.
  static void ReadParameters();

  /// The right-hand side on Quokka's burn state: d/dt of the species, the
  /// specific internal energy and the radiation.
  template <typename BurnT>
  AMREX_GPU_HOST_DEVICE static void actual_rhs(
      const BurnT &state,
      amrex::GpuArray<amrex::Real,
                      quokka::microphysics::IntegratorLayout<
                          typename BurnT::problem_type, BurnT>::neqs> &ydot) {
    static_assert(
        BurnT::traits::integrate_rad_flux,
        "a jaff network needs Microphysics_Traits::integrate_rad_flux");
    $$JAFF REPEAT idx, rhs, cse IN rhses { USE quokka }
    const amrex::Real cse$idx$ = $cse$;
    ydot[$idx$] = $rhs$;
    $$JAFF END
    $$JAFF SUB nspec
    ydot[$nspec$] /= state.rho;
    $$JAFF END
  }

  /// The Jacobian of actual_rhs.
  template <typename BurnT, class MatrixType>
  AMREX_GPU_HOST_DEVICE static void actual_jac(const BurnT &state,
                                               MatrixType &jac) {
    using Layout = quokka::microphysics::IntegratorLayout<
        typename BurnT::problem_type, BurnT>;
    static_assert(
        BurnT::traits::integrate_rad_flux,
        "a jaff network needs Microphysics_Traits::integrate_rad_flux");

    $$JAFF REPEAT idx, expr, cse IN jacobian { USE quokka }
    const amrex::Real cse$idx$ = $cse$;
    jac($idx$, $idx$) = $expr$;
    $$JAFF END
    // jaff's energy row and column are those of E = rho e: the row is d(dE/dt)
    // and the column d/dE. Change them to e, with rho fixed during the burn:
    // the row divides by rho, the column multiplies by rho, and d(de/dt)/de =
    // d(dE/dt)/dE is unchanged.
    $$JAFF SUB nspec
    for (int i = 0; i < Layout::neqs; i++) {
      if (i != $nspec$) {
        jac($nspec$, i) /= state.rho;
        jac(i, $nspec$) *= state.rho;
      }
    }
    $$JAFF END
  }
};

$$JAFF SUB label
} // namespace quokka::networks::$label$
$$JAFF END

#endif // JAFF_NETWORK_HPP_
