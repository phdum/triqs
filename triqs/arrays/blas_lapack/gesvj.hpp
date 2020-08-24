/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2012-2017 by O. Parcollet
 * Copyright (C) 2018 by Simons Foundation
 *   author : O. Parcollet, P. Dumitrescu, N. Wentzell
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once

#include <complex>
#include "f77/cxx_interface.hpp"
#include "tools.hpp"
#include "qcache.hpp"

namespace triqs::arrays::lapack {
  using namespace blas_lapack_tools;

  /**
  * Calls gesvd on a matrix or view
  * Takes care of making temporary copies if necessary
  */
  template <typename MTA, typename VCS, typename MTU, typename MTVT>
  typename std::enable_if_t<is_blas_lapack_type<typename MTA::value_type>::value && is_blas_lapack_type<typename VCS::value_type>::value
                               && is_blas_lapack_type<typename MTU::value_type>::value && is_blas_lapack_type<typename MTVT::value_type>::value,
                            int>
  gesvj(MTA const &A, VCS &S, MTU &U, MTVT &VT) {
    int info;

    int m = get_n_rows(A);
    int n = get_n_cols(A);

    if (!A.memory_layout_is_fortran()) {
      TRIQS_RUNTIME_ERROR << "Error in gesvj : Only Fortran Layout Supported: A";
    }
    if (!U.memory_layout_is_fortran()) {
      TRIQS_RUNTIME_ERROR << "Error in gesvj : Only Fortran Layout Supported: U";
    }
    if (!VT.memory_layout_is_fortran()) {
      TRIQS_RUNTIME_ERROR << "Error in gesvj : Only Fortran Layout Supported: VT";
    }

    // Copy A, since it is altered by gesvj
    typename MTA::regular_type _A{A, FORTRAN_LAYOUT};

    reflexive_qcache<MTA> Ca(_A);
    // reflexive_qcache<VCS> Cs(S);
    reflexive_qcache<MTU> Cu(U);
    reflexive_qcache<MTVT> Cvt(VT);

    if constexpr (std::is_same<typename MTA::value_type, double>::value) {

      int lwork = std::max(6, m + n);
      arrays::vector<typename MTA::value_type> work(lwork);

      f77::gesvj('G', 'U', 'V', m, n, Ca().data_start(), get_ld(Ca()), S.data_start(), 0, Cvt().data_start(), get_ld(Cvt()), work.data_start(), lwork,
                 info);

      // std::cerr << "work: " << work << std::endl;

    } else if constexpr (std::is_same<typename MTA::value_type, std::complex<double>>::value) {

      int lwork = std::max(6, m + n);
      arrays::vector<std::complex<double>> cwork(lwork);

      int lrwork = std::max(6, m + n);
      arrays::vector<double> rwork(lrwork);

      f77::gesvj('G', 'U', 'V', m, n, Ca().data_start(), get_ld(Ca()), S.data_start(), 0, Cvt().data_start(), get_ld(Cvt()), cwork.data_start(),
                 lwork, rwork.data_start(), lrwork, info);

      // std::cerr << "cwork: " << cwork << std::endl;
      // std::cerr << "rwork: " << rwork << std::endl;

    } else {
      TRIQS_RUNTIME_ERROR << "Error in gesvj : only implemented for value_type double and std::complex<double>";
    }

    Cu() = Ca(); // fix dimensionality

    if (info) { TRIQS_RUNTIME_ERROR << "Error in gesvj : info = " << info; }
    return info;
  }
} // namespace triqs::arrays::lapack
