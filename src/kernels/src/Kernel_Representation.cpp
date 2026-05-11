#include "psnd/Kernel_Representation.h"

#include "psnd/Kernel_NAForce.h"
#include "psnd/hash_fnv1a.h"
#include "psnd/linalg.h"
// #include "psnd/linalg_tpl.h"
#include "psnd/debug_utils.h"
#include "psnd/macro_utils.h"
#include "psnd/vars_list.h"

namespace PROJECT_NS {

const std::string Kernel_Representation::getName() { return "Kernel_Representation"; }

int Kernel_Representation::getType() const { return utils::hash(FUNCTION_NAME); }

// Real ADT (orthogonal T from diagonalizing real-symmetric V):  V T = T diag(eig), T^T T = I.
//   Stype = H (Hilbert, vector ψ, fdim x 1):
//     dia -> adi:  ψ_adi = T^T ψ_dia
//     adi -> dia:  ψ_dia = T   ψ_adi
//   Stype = L (Liouville, matrix A, fdim x fdim):
//     dia -> adi:  A_adi = T^T A_dia T
//     adi -> dia:  A_dia = T   A_adi T^T
int Kernel_Representation::transform(psnd_complex* A, psnd_real* T, int fdim,  //
                                     RepresentationPolicy::_type from, RepresentationPolicy::_type to,
                                     SpacePolicy::_type Stype) {
    if (from == to) return 0;
    int lda = (Stype == SpacePolicy::L) ? fdim : 1;
    if (from == RepresentationPolicy::Diabatic && to == RepresentationPolicy::Adiabatic) {
        // A <- T^T A     ; then (L only) A <- A T   =>  A_adi = T^T A_dia T
        ARRAY_MATMUL_TRANS1(A, T, A, fdim, fdim, lda);
        if (Stype == SpacePolicy::L) ARRAY_MATMUL(A, A, T, fdim, fdim, fdim);
    }
    if (from == RepresentationPolicy::Adiabatic && to == RepresentationPolicy::Diabatic) {
        // A <- T   A     ; then (L only) A <- A T^T =>  A_dia = T A_adi T^T
        ARRAY_MATMUL(A, T, A, fdim, fdim, lda);
        if (Stype == SpacePolicy::L) ARRAY_MATMUL_TRANS2(A, A, T, fdim, fdim, fdim);
    }
    return 0;
}

// Complex ADT. Two unitary matrices live here:
//   - Real T  (orthogonal): diabatic <-> adiabatic, same as the overload above.
//   - Complex Tc (unitary): general_soc <-> adiabatic; Tc diagonalizes Hermitian Vc, Tc^H Tc = I.
// MATMUL_TRANS1/TRANS2 use Eigen .adjoint(), so for complex T they are Hermitian conjugates (T^H),
// for real T they reduce to plain transpose (T^T).
//   Stype = H:  ψ_adi = Tc^H ψ_soc ;  ψ_soc = Tc   ψ_adi
//   Stype = L:  A_adi = Tc^H A_soc Tc ;  A_soc = Tc A_adi Tc^H
int Kernel_Representation::transform(psnd_complex* A, psnd_complex* T, int fdim,  //
                                     RepresentationPolicy::_type from, RepresentationPolicy::_type to,
                                     SpacePolicy::_type Stype) {
    if (from == to) return 0;
    int lda = (Stype == SpacePolicy::L) ? fdim : 1;
    if (from == RepresentationPolicy::Diabatic && to == RepresentationPolicy::Adiabatic) {
        // A_adi = T^H A_dia T  (T real here, so T^H = T^T)
        ARRAY_MATMUL_TRANS1(A, T, A, fdim, fdim, lda);
        if (Stype == SpacePolicy::L) ARRAY_MATMUL(A, A, T, fdim, fdim, fdim);
    }
    if (from == RepresentationPolicy::Adiabatic && to == RepresentationPolicy::Diabatic) {
        // A_dia = T A_adi T^H
        ARRAY_MATMUL(A, T, A, fdim, fdim, lda);
        if (Stype == SpacePolicy::L) ARRAY_MATMUL_TRANS2(A, A, T, fdim, fdim, fdim);
    }
    // General_soc uses a complex ADT matrix (Tc); caller must pass Tc as T.
    if (from == RepresentationPolicy::General_soc && to == RepresentationPolicy::Adiabatic) {
        // A_adi = Tc^H A_soc Tc
        ARRAY_MATMUL_TRANS1(A, T, A, fdim, fdim, lda);
        if (Stype == SpacePolicy::L) ARRAY_MATMUL(A, A, T, fdim, fdim, fdim);
    }
    if (from == RepresentationPolicy::Adiabatic && to == RepresentationPolicy::General_soc) {
        // A_soc = Tc A_adi Tc^H
        ARRAY_MATMUL(A, T, A, fdim, fdim, lda);
        if (Stype == SpacePolicy::L) ARRAY_MATMUL_TRANS2(A, A, T, fdim, fdim, fdim);
    }
    // TODO: General_soc <-> Diabatic not implemented; caller should avoid this combination
    if (from == RepresentationPolicy::General_soc && to == RepresentationPolicy::Diabatic) return 0;
    if (from == RepresentationPolicy::Diabatic && to == RepresentationPolicy::General_soc) return 0;
    return 0;
}

void Kernel_Representation::setInputParam_impl(std::shared_ptr<Param> PM) {
    std::string rep_string = _param->get_string({"solver.representation_flag"}, LOC(), "Diabatic");
    representation_type    = RepresentationPolicy::_from(rep_string);
    inp_repr_type    = RepresentationPolicy::_from(_param->get_string({"solver.inp_repr_flag"}, LOC(), rep_string));
    ele_repr_type    = RepresentationPolicy::_from(_param->get_string({"solver.ele_repr_flag"}, LOC(), rep_string));
    nuc_repr_type    = RepresentationPolicy::_from(_param->get_string({"solver.nuc_repr_flag"}, LOC(), rep_string));
    tcf_repr_type    = RepresentationPolicy::_from(_param->get_string({"solver.tcf_repr_flag"}, LOC(), rep_string));
    phase_correction = _param->get_bool({"solver.phase_correction"}, LOC(), false);
    basis_switch     = _param->get_bool({"solver.basis_switch"}, LOC(), false);
}

void Kernel_Representation::setInputDataSet_impl(std::shared_ptr<DataSet> DS) {
    V  = DS->def(DATA::model::V);
    dV = DS->def(DATA::model::dV);
    // ddV = DS->def(DATA::model::ddV);
    eig  = DS->def(DATA::model::rep::eig);
    E    = DS->def(DATA::model::rep::E);
    T    = DS->def(DATA::model::rep::T);
    Told = DS->def(DATA::model::rep::Told);
    dE   = DS->def(DATA::model::rep::dE);
    // ddE = DS->def(DATA::model::rep::ddE);
    lam = DS->def(DATA::model::rep::lam);
    R   = DS->def(DATA::model::rep::R);
    H   = DS->def(DATA::model::rep::H);

    m       = DS->def(DATA::integrator::m);
    p       = DS->def(DATA::integrator::p);
    occ_nuc = DS->def(DATA::integrator::occ_nuc);
    rho_ele = DS->def(DATA::integrator::rho_ele);

    TtTold = DS->def(DATA::integrator::tmp::TtTold);
    ve     = DS->def(DATA::integrator::tmp::ve);
    vedE   = DS->def(DATA::integrator::tmp::vedE);

    nac = DS->def(DATA::model::rep::nac);

    // --- General_soc bindings ---
    Vc             = DS->def(DATA::model::Vc);
    dVc            = DS->def(DATA::model::dVc);
    Ec             = DS->def(DATA::model::rep::Ec);
    Tc             = DS->def(DATA::model::rep::Tc);
    Toldc          = DS->def(DATA::model::rep::Toldc);
    dEc            = DS->def(DATA::model::rep::dEc);
    vedEc          = DS->def(DATA::integrator::tmp::vedEc);
    TtToldc        = DS->def(DATA::integrator::tmp::TtToldc);
    venac          = DS->def(DATA::integrator::tmp::venac);
    commutator_d_V = DS->def(DATA::integrator::tmp::commutator_d_V);
}

Status& Kernel_Representation::initializeKernel_impl(Status& stat) { return stat; }

Status& Kernel_Representation::executeKernel_impl(Status& stat) {
    if (Dimension::F <= 1) return stat;

    for (int iP = 0; iP < Dimension::P; ++iP) {
        auto V       = this->V.subspan(iP * Dimension::FF, Dimension::FF);
        auto eig     = this->eig.subspan(iP * Dimension::F, Dimension::F);
        auto E       = this->E.subspan(iP * Dimension::FF, Dimension::FF);
        auto T       = this->T.subspan(iP * Dimension::FF, Dimension::FF);
        auto Told    = this->Told.subspan(iP * Dimension::FF, Dimension::FF);
        auto dV      = this->dV.subspan(iP * Dimension::NFF, Dimension::NFF);
        auto dE      = this->dE.subspan(iP * Dimension::NFF, Dimension::NFF);
        auto lam     = this->lam.subspan(iP * Dimension::F, Dimension::F);
        auto R       = this->R.subspan(iP * Dimension::FF, Dimension::FF);
        auto H       = this->H.subspan(iP * Dimension::FF, Dimension::FF);
        auto p       = this->p.subspan(iP * Dimension::N, Dimension::N);
        auto m       = this->m.subspan(iP * Dimension::N, Dimension::N);
        auto occ_nuc = this->occ_nuc.subspan(iP, 1);
        auto rho_ele = this->rho_ele.subspan(iP * Dimension::FF, Dimension::FF);
        // --- General_soc subspans (only used in that case) ---
        auto Vc    = this->Vc.subspan(iP * Dimension::FF, Dimension::FF);
        auto dVc   = this->dVc.subspan(iP * Dimension::NFF, Dimension::NFF);
        auto Ec    = this->Ec.subspan(iP * Dimension::FF, Dimension::FF);
        auto Tc    = this->Tc.subspan(iP * Dimension::FF, Dimension::FF);
        auto Toldc = this->Toldc.subspan(iP * Dimension::FF, Dimension::FF);
        auto dEc   = this->dEc.subspan(iP * Dimension::NFF, Dimension::NFF);

        // -----------------------------------------------------------------
        // Per-step representation update. For each bead iP we (a) refresh the
        // ADT (T or Tc) by diagonalising the model potential, (b) build the
        // diagonal energy matrix E, and (c) assemble the effective propagator
        // Hamiltonian H so Kernel_Update_U can do U = exp(-i H dt) via the
        // factorisation R diag(lam) R^H = H produced at the end of each case.
        // -----------------------------------------------------------------
        switch (representation_type) {
            case RepresentationPolicy::Diabatic: {
                // Diabatic propagation: dynamics live in the diabatic basis,
                // so the propagator is just H = V. T and eig are still solved
                // (V T = T diag(eig)) so observables can be rotated to
                // adiabatic on demand, but no temporal sign/permutation fix
                // is applied — T may flip arbitrarily between steps.
                // Outputs: T, eig, E = diag(eig), H = V.
                EigenSolve(eig.data(), T.data(), V.data(), Dimension::F);
                for (int i = 0, ik = 0; i < Dimension::F; ++i)
                    for (int k = 0; k < Dimension::F; ++k, ++ik) E[ik] = (i == k) ? eig[i] : 0.0e0;
                for (int ik = 0; ik < Dimension::FF; ++ik) H[ik] = V[ik];
                break;
            }
            case RepresentationPolicy::Diabatic_NAF2: {
                // Diabatic propagation but the non-adiabatic force is taken
                // in the (smoothed) adiabatic basis, so T must stay continuous
                // across steps. Same V-eigen step as Diabatic, plus:
                //   TtTold = T^T Told  ->  rounded to a signed permutation
                //   T <- T * TtTold    (and eig reordered if basis_switch)
                // basis_switch=false  : per-column sign match only.
                // basis_switch=true   : full crossing/permutation reordering.
                // Outputs: T, Told, eig, E = diag(eig), H = V.
                for (int i = 0; i < Dimension::FF; ++i) Told[i] = T[i];    // backup old T matrix
                EigenSolve(eig.data(), T.data(), V.data(), Dimension::F);  // solve new eigen problem
                for (int i = 0, ik = 0; i < Dimension::F; ++i)
                    for (int k = 0; k < Dimension::F; ++k, ++ik) E[ik] = (i == k) ? eig[i] : 0.0e0;
                for (int ik = 0; ik < Dimension::FF; ++ik) H[ik] = V[ik];
                if (do_refer && !stat.first_step) {
                    // calculate permutation matrix = rountint(T^ * Told)
                    ARRAY_MATMUL_TRANS1(TtTold.data(), T.data(), Told.data(),  //
                                        Dimension::F, Dimension::F, Dimension::F);

                    if (!basis_switch) {
                        for (int i = 0, ik = 0; i < Dimension::F; ++i) {
                            for (int k = 0; k < Dimension::F; ++k, ++ik) {
                                TtTold[ik] = (i == k) ? copysign(1.0f, TtTold[ik]) : 0;
                            }
                        }
                    } else {
                        double vset = 0.1 * std::sqrt(1.0e0 / Dimension::F);
                        for (int i = 0; i < Dimension::F; ++i) {
                            double maxnorm = 0;
                            int    csr1 = 0, csr2 = 0, csr12 = 0;
                            for (int k1 = 0, k1k2 = 0; k1 < Dimension::F; ++k1) {
                                for (int k2 = 0; k2 < Dimension::F; ++k2, ++k1k2) {
                                    // vmax must be larger than sqrt(1/fdim)
                                    if (std::abs(TtTold[k1k2]) > maxnorm) {
                                        maxnorm = std::abs(TtTold[k1k2]);
                                        csr1 = k1, csr2 = k2, csr12 = k1k2;
                                    }
                                }
                            }
                            double vsign = copysign(1.0f, TtTold[csr12]);
                            for (int k2 = 0, k1k2 = csr1 * Dimension::F;  //
                                    k2 < Dimension::F;                       //
                                    ++k2, ++k1k2) {
                                TtTold[k1k2] = 0;
                            }
                            for (int k1 = 0, k1k2 = csr2; k1 < Dimension::F; ++k1, k1k2 += Dimension::F) {
                                TtTold[k1k2] = 0;
                            }
                            TtTold[csr12] = vsign * vset;
                        }
                        for (int i = 0; i < Dimension::FF; ++i) TtTold[i] = round(TtTold[i] / vset);
                    }
                    // adjust order of eigenvectors & eigenvalues
                    ARRAY_MATMUL(T.data(), T.data(), TtTold.data(),  //
                                    Dimension::F, Dimension::F, Dimension::F);
                    if (basis_switch) {
                        for (int i = 0; i < Dimension::FF; ++i) TtTold[i] = std::abs(TtTold[i]);
                        ARRAY_MATMUL(eig.data(), eig.data(), TtTold.data(), 1, Dimension::F, Dimension::F);
                    }
                }
                break;
            }
            case RepresentationPolicy::Adiabatic: {
                // Adiabatic propagation. Two routes for (T, eig, dE, nac):
                //   !onthefly : diagonalise model V here, build dE = T^T dV T
                //               (BATH_FORCE_BILINEAR exploits bath structure).
                //               Continuity of T is enforced via the same
                //               TtTold = T^T Told sign/permutation fix
                //   onthefly  : eig, dE, nac come straight from the QM model;
                //               T is left as the identity (every QM model
                //               sets T = I once in setInputDataSet_impl and
                //               never touches it again), because on-the-fly
                //               backends already work in their own quasi-
                //               adiabatic basis. Adiabatic-state continuity
                //               is then carried by the model's NAC sign
                //               tracking (e.g. nac_prev), not by T.
                //
                // Effective Hamiltonian for U = exp(-i H dt):
                //   E = diag(eig - Emean)              (Emean shift = trace gauge)
                //   off-diag, !onthefly :  H_ij = -i (p/m)·dE_ij / (eig_j - eig_i)
                //                          (uses nacv_ij = dE_ij/(E_j - E_i))
                //   off-diag,  onthefly :  H_ij = -i sum_k (p_k/m_k) nac_k_ij
                //   phase_correction (opt.): replace diagonal by
                //                          -2 Ekin sqrt(1 + (Epes - eig_i)/Ekin)
                // H is then diagonalised: R diag(lam) R^H = H.
                // Outputs: T, eig, E, dE (when !onthefly), H, lam, R.
                if (!onthefly) {
                    for (int i = 0; i < Dimension::FF; ++i) Told[i] = T[i];    // backup old T matrix
                    EigenSolve(eig.data(), T.data(), V.data(), Dimension::F);  // solve new eigen problem

                    if (do_refer && !stat.first_step) {
                        // calculate permutation matrix = rountint(T^ * Told)
                        ARRAY_MATMUL_TRANS1(TtTold.data(), T.data(), Told.data(),  //
                                            Dimension::F, Dimension::F, Dimension::F);

                        if (!basis_switch) {
                            for (int i = 0, ik = 0; i < Dimension::F; ++i) {
                                for (int k = 0; k < Dimension::F; ++k, ++ik) {
                                    TtTold[ik] = (i == k) ? copysign(1.0, TtTold[ik]) : 0;
                                }
                            }
                        } else {
                            double vset = 0.1 * std::sqrt(1.0e0 / Dimension::F);
                            for (int i = 0; i < Dimension::F; ++i) {
                                double maxnorm = 0;
                                int    csr1 = 0, csr2 = 0, csr12 = 0;
                                for (int k1 = 0, k1k2 = 0; k1 < Dimension::F; ++k1) {
                                    for (int k2 = 0; k2 < Dimension::F; ++k2, ++k1k2) {
                                        // vmax must be larger than sqrt(1/fdim)
                                        if (std::abs(TtTold[k1k2]) > maxnorm) {
                                            maxnorm = std::abs(TtTold[k1k2]);
                                            csr1 = k1, csr2 = k2, csr12 = k1k2;
                                        }
                                    }
                                }
                                double vsign = copysign(1.0, TtTold[csr12]);
                                for (int k2 = 0, k1k2 = csr1 * Dimension::F;  //
                                     k2 < Dimension::F;                       //
                                     ++k2, ++k1k2) {
                                    TtTold[k1k2] = 0;
                                }
                                for (int k1 = 0, k1k2 = csr2; k1 < Dimension::F; ++k1, k1k2 += Dimension::F) {
                                    TtTold[k1k2] = 0;
                                }
                                TtTold[csr12] = vsign * vset;
                            }
                            for (int i = 0; i < Dimension::FF; ++i) TtTold[i] = round(TtTold[i] / vset);
                        }
                        // adjust order of eigenvectors & eigenvalues
                        ARRAY_MATMUL(T.data(), T.data(), TtTold.data(),  //
                                     Dimension::F, Dimension::F, Dimension::F);
                        if (basis_switch) {
                            for (int i = 0; i < Dimension::FF; ++i) TtTold[i] = std::abs(TtTold[i]);
                            ARRAY_MATMUL(eig.data(), eig.data(), TtTold.data(), 1, Dimension::F, Dimension::F);
                        }
                    }

                    if (FORCE_OPT::BATH_FORCE_BILINEAR) {
                        int& B   = FORCE_OPT::nbath;
                        int& J   = FORCE_OPT::Nb;
                        int  JFF = J * Dimension::FF;
                        for (int b = 0, bb = 0; b < B; ++b, bb += Dimension::Fadd1) {
                            auto dVb0 = dV.subspan(b * JFF, JFF);
                            auto dEb0 = dE.subspan(b * JFF, JFF);
                            ARRAY_MATMUL3_TRANS1(dEb0.data(), T.data(), dVb0.data(), T.data(),  //
                                                 Dimension::F, Dimension::F, Dimension::F, Dimension::F);
                            for (int j = 0, jik = 0, jbb = bb; j < J; ++j, jbb += Dimension::FF) {
                                double scale = dVb0[jbb] / dVb0[bb];
                                for (int ik = 0; ik < Dimension::FF; ++ik, ++jik) { dEb0[jik] = dEb0[ik] * scale; }
                            }
                        }
                    } else {
                        // Adiabatic gradient: for every nuclear DOF k,
                        //     dE_k = T^T · dV_k · T            (k = 0 .. N-1)
                        // Storage of dV / dE is the row-major 3-D layout
                        // (N, F, F), flat index = k*FF + i*F + j.
                        //
                        // Instead of N small triple products, we batch all k
                        // into two BLAS calls by reshaping the buffer; the i
                        // axis is the one that has to slide between "row" and
                        // "column" position, hence the two transposes.
                        //
                        //   step 1  view (NF, F):  rows = k*F + i, cols = j
                        //           dE = dV · T          -> dV_k · T  for every k
                        //   step 2  TRANSPOSE (N, FF) -> (FF, N)
                        //           moves the k axis to the right, exposing i
                        //           as the leading row of an (F, NF) view
                        //   step 3  view (F, NF):  rows = i, cols = j'*N + k
                        //           dE = T^T · dE       -> (T^T dV_k T) for every k
                        //   step 4  TRANSPOSE (FF, N) -> (N, FF)
                        //           restores the canonical (N, F, F) layout
                        //           expected by Kernel_NAForce / Update_p.
                        //
                        // Why batched and not a per-k loop with MATMUL3_TRANS1:
                        // for the typical PSiNad regime (F ≈ 4-15) the two
                        // big matmuls amortise Eigen's per-call overhead and
                        // the 2*N*F^2 transpose traffic still fits in L2; per
                        // /tmp/bench_dE.cpp the crossover is around F ≈ 15 -
                        // beyond that, switch to the per-k MATMUL3_TRANS1
                        // form (cf. the BATH_FORCE_BILINEAR branch above).
                        ARRAY_MATMUL(dE.data(), dV.data(), T.data(), Dimension::NF, Dimension::F, Dimension::F);
                        ARRAY_TRANSPOSE(dE.data(), Dimension::N, Dimension::FF);
                        ARRAY_MATMUL_TRANS1(dE.data(), T.data(), dE.data(), Dimension::F, Dimension::F, Dimension::NF);
                        ARRAY_TRANSPOSE(dE.data(), Dimension::FF, Dimension::N);
                    }
                }
                for (int i = 0, ik = 0; i < Dimension::F; ++i)
                    for (int k = 0; k < Dimension::F; ++k, ++ik) E[ik] = (i == k) ? eig[i] : 0.0e0;

                // calc H = E - im * nacv * p / m, and note here nacv_{ij} = dEij / (Ej - Ei)
                for (int i = 0; i < Dimension::N; ++i) ve[i] = p[i] / m[i];
                ARRAY_MATMUL(vedE.data(), ve.data(), dE.data(), 1, Dimension::N, Dimension::FF);

                double Emean = 0.0e0;
                for (int i = 0; i < Dimension::F; ++i) Emean += eig[i];
                Emean /= Dimension::F;

                if (!onthefly){
                    for (int i = 0, ij = 0; i < Dimension::F; ++i) {
                        for (int j = 0; j < Dimension::F; ++j, ++ij) {  //
                            H[ij] = ((i == j) ? eig[i] - Emean : -phys::math::im * vedE[ij] / (eig[j] - eig[i]));
                        }
                    }
                } else {
                    // 先全部计算，然后修正对角线
                    for (int ij = 0; ij < Dimension::FF; ++ij) {
                        H[ij] = 0.0e0;
                    }

                    for (int k = 0; k < Dimension::N; ++k) {
                        psnd_complex factor = -phys::math::im * ve[k];
                        for (int ij = 0; ij < Dimension::FF; ++ij) {
                            H[ij] += factor * nac[k * Dimension::FF + ij];
                        }
                    }

                    // 修正对角线
                    for (int i = 0, ii = 0; i < Dimension::F; ++i, ii += Dimension::Fadd1) {
                        H[ii] = eig[i] - Emean;
                    }
                }

                if (phase_correction) {
                    psnd_real Ekin = 0;
                    for (int j = 0; j < Dimension::N; ++j) Ekin += 0.5f * p[j] * p[j] / m[j];
                    double Epes = 0.0;
                    if (Kernel_NAForce::NAForce_type == NAForcePolicy::BO) {
                        Epes = eig[occ_nuc[0]];
                    } else {
                        for (int i = 0, ii = 0; i < Dimension::F; ++i, ii += Dimension::Fadd1) {
                            Epes += std::real(rho_ele[ii]) * eig[i];
                        }
                    }
                    for (int i = 0, ii = 0; i < Dimension::F; ++i, ii += Dimension::Fadd1) {
                        H[ii] = -2 * Ekin * sqrt(std::max<double>(1.0 + (Epes - eig[i]) / Ekin, 0.0));
                    }
                }
                EigenSolve(lam.data(), R.data(), H.data(), Dimension::F);  // R*L*R^ = H
                break;
            }
            case RepresentationPolicy::General_soc: {
                // General complex (SOC) representation: Vc is Hermitian, its
                // eigenvectors Tc are complex unitary. Vc = Tc diag(eig) Tc^H.
                //   1. Toldc <- Tc ; Hermitian eigen on Vc -> (eig, Tc).
                //      Continuity fix uses the *complex* TtToldc = Tc^H Toldc
                //      (real part rounded to a signed permutation).
                //   2. dEc_j = dVc_j + [nac_j, Vc].   For a unitary ADT,
                //      d(Tc^H Vc Tc) = Tc^H (dVc + [nac, Vc]) Tc, so the
                //      bracketed quantity is the soc-basis gradient stored
                //      for the force routines.
                //   3. Ec = diag(eig).
                // Effective propagator is built directly in the soc basis
                // (not the adiabatic basis):
                //   venac_ik = sum_j (p_j/m_j) · nac_j_ik         (real)
                //   H        = Vc - i · venac
                //   R diag(lam) R^H = H
                // Outputs: Tc, Toldc, eig, Ec, dEc, venac, H, lam, R.
                for (int i = 0; i < Dimension::FF; ++i) Toldc[i] = Tc[i];    // backup old Tc
                EigenSolve(eig.data(), Tc.data(), Vc.data(), Dimension::F);  // Hermitian eigen，Vc = Tc diag(eig) Tc^dag

                if (do_refer && !stat.first_step) {
                    // TtToldc = Tc^H * Toldc (complex)
                    ARRAY_MATMUL_TRANS1(TtToldc.data(), Tc.data(), Toldc.data(),  //
                                        Dimension::F, Dimension::F, Dimension::F);
                    if (!basis_switch) {
                        for (int i = 0, ik = 0; i < Dimension::F; ++i) {
                            for (int k = 0; k < Dimension::F; ++k, ++ik) {
                                TtToldc[ik] = (i == k) ? psnd_complex(std::copysign(1.0, TtToldc[ik].real()), 0.0)
                                                       : psnd_complex(0.0, 0.0);
                            }
                        }
                    } else {
                        double vset = 0.1 * std::sqrt(1.0e0 / Dimension::F);
                        for (int i = 0; i < Dimension::F; ++i) {
                            double maxnorm = 0;
                            int    csr1 = 0, csr2 = 0, csr12 = 0;
                            for (int k1 = 0, k1k2 = 0; k1 < Dimension::F; ++k1) {
                                for (int k2 = 0; k2 < Dimension::F; ++k2, ++k1k2) {
                                    if (std::abs(TtToldc[k1k2]) > maxnorm) {
                                        maxnorm = std::abs(TtToldc[k1k2]);
                                        csr1 = k1, csr2 = k2, csr12 = k1k2;
                                    }
                                }
                            }
                            double vsign = std::copysign(1.0, TtToldc[csr12].real());
                            for (int k2 = 0, k1k2 = csr1 * Dimension::F; k2 < Dimension::F; ++k2, ++k1k2)
                                TtToldc[k1k2] = psnd_complex(0.0, 0.0);
                            for (int k1 = 0, k1k2 = csr2; k1 < Dimension::F; ++k1, k1k2 += Dimension::F)
                                TtToldc[k1k2] = psnd_complex(0.0, 0.0);
                            TtToldc[csr12] = psnd_complex(vsign * vset, 0.0);
                        }
                        for (int i = 0; i < Dimension::FF; ++i)
                            TtToldc[i] = psnd_complex(std::round(TtToldc[i].real() / vset), 0.0);
                    }
                    ARRAY_MATMUL(Tc.data(), Tc.data(), TtToldc.data(),  //
                                 Dimension::F, Dimension::F, Dimension::F);
                    if (basis_switch) {
                        std::vector<double> TtToldc_abs(Dimension::FF);
                        for (int i = 0; i < Dimension::FF; ++i) TtToldc_abs[i] = std::abs(TtToldc[i]);
                        ARRAY_MATMUL(eig.data(), eig.data(), TtToldc_abs.data(), 1, Dimension::F, Dimension::F);
                    }
                }

                // dEc_j = dVc_j + [nac_j, Vc]
                for (int j = 0, jFF = 0; j < Dimension::N; ++j, jFF += Dimension::FF) {
                    auto nacj = nac.subspan(jFF, Dimension::FF);
                    auto dVcj = dVc.subspan(jFF, Dimension::FF);
                    ARRAY_COMMUNTATOR(commutator_d_V.data(), nacj.data(), Vc.data(), Dimension::F);
                    for (int i = 0; i < Dimension::FF; ++i) dEc[jFF + i] = dVcj[i] + commutator_d_V[i];
                }

                // Ec = diag(eig)
                for (int i = 0, ik = 0; i < Dimension::F; ++i)
                    for (int k = 0; k < Dimension::F; ++k, ++ik)
                        Ec[ik] = (i == k) ? psnd_complex(eig[i], 0.0) : psnd_complex(0.0, 0.0);

                // venac_{ik} = sum_j (p_j/m_j) * nac_j_{ik}   (real)
                for (int i = 0; i < Dimension::N; ++i) ve[i] = p[i] / m[i];
                for (int i = 0; i < Dimension::FF; ++i) {
                    venac[i] = 0.0e0;
                    for (int j = 0; j < Dimension::N; ++j) venac[i] += ve[j] * nac[j * Dimension::FF + i];
                }

                // H = Vc - i * venac
                for (int i = 0; i < Dimension::FF; ++i) H[i] = Vc[i] - phys::math::im * venac[i];
                EigenSolve(lam.data(), R.data(), H.data(), Dimension::F);  // R*L*R^ = H
                break;
            }
            case RepresentationPolicy::Force:
            case RepresentationPolicy::Density:
            default:
                // Pure-force / density-only schemes do not need a basis
                // transform here; the relevant kernels handle their own state.
                break;
        }
    }

    return stat;
}

RepresentationPolicy::_type Kernel_Representation::representation_type;
RepresentationPolicy::_type Kernel_Representation::inp_repr_type;
RepresentationPolicy::_type Kernel_Representation::ele_repr_type;
RepresentationPolicy::_type Kernel_Representation::nuc_repr_type;
RepresentationPolicy::_type Kernel_Representation::tcf_repr_type;
bool                        Kernel_Representation::onthefly;

};  // namespace PROJECT_NS
