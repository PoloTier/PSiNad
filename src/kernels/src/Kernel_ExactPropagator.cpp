/*
Exact propagator of effective nonadiabatic force for NaF simulations
Author: Baihua Wu, Haocheng Lu
Date: 2025-10-29
Wu, Li, He, Cheng, Ren, Liu, J. Chem. Theory Comput. 2025, 21, 8, 3775-3813.
*/

#include "psnd/Kernel_ExactPropagator.h"

#include "psnd/Kernel_Elec_Utils.h"
#include "psnd/Kernel_Representation.h"
#include "psnd/hash_fnv1a.h"
#include "psnd/linalg.h"
#include "psnd/macro_utils.h"
#include "psnd/vars_list.h"

namespace PROJECT_NS {

const std::string Kernel_ExactPropagator::getName() { return "Kernel_ExactPropagator"; }

int Kernel_ExactPropagator::getType() const { return utils::hash("Kernel_ExactPropagator"); }

void Kernel_ExactPropagator::setInputParam_impl(std::shared_ptr<Param> PM) {
    NAForce_type                   = NAForcePolicy::_dict.at(  //
        _param->get_string({"solver.naforce"}, LOC(), "EHR"));
}

void Kernel_ExactPropagator::setInputDataSet_impl(std::shared_ptr<DataSet> DS) {
    f    = DS->def(DATA::integrator::f);
    fadd = DS->def(DATA::integrator::fadd);
    p    = DS->def(DATA::integrator::p);
    // ve   = DS->def(DATA::integrator::ve);
    minv = DS->def(DATA::integrator::minv);

    // mask    = DS->def(DATA::integrator::forceeval::mask);
    // dmask   = DS->def(DATA::integrator::forceeval::dmask);
    T       = DS->def(DATA::model::rep::T);
    grad    = DS->def(DATA::model::grad);
    dV      = DS->def(DATA::model::dV);
    dE      = DS->def(DATA::model::rep::dE);
    nac     = DS->def(DATA::model::rep::nac);
    eig     = DS->def(DATA::model::rep::eig);
    rho_nuc = DS->def(DATA::integrator::rho_nuc);
    c       = DS->def(DATA::integrator::c);
    Ekin    = DS->def(DATA::integrator::Ekin);
    dt      = DS->def(DATA::control::dt);

    // 分配 B_vec - Dim: [N]
    // B_vec = DS->def(VARIABLE<psnd_real>("integrator.B_vec", &Dimension::shape_N, "@"));

    // 分配其他必需的变量
    alpha = DS->def(DATA::integrator::alpha);
    fproj = DS->def(DATA::integrator::tmp::fproj);
    ftmp  = DS->def(DATA::integrator::tmp::ftmp);
    wrho  = DS->def(DATA::integrator::tmp::wrho);

    // 分配缺少的变量
    occ_nuc = DS->def(DATA::integrator::occ_nuc);
    rho_ele = DS->def(DATA::integrator::rho_ele);
    m       = DS->def(DATA::integrator::m);
    V       = DS->def(DATA::model::V);
    // vpes    = DS->def(DATA::model::vpes);
    // Epot    = DS->def(DATA::integrator::Epot);

    switch (Kernel_Representation::nuc_repr_type) {
        case RepresentationPolicy::Diabatic:
            EMat     = DS->def(DATA::model::V);
            ForceMat = DS->def(DATA::model::dV);
            break;
        case RepresentationPolicy::Adiabatic:
            EMat     = DS->def(DATA::model::rep::E);
            ForceMat = DS->def(DATA::model::rep::dE);
            break;
        case RepresentationPolicy::General_soc:
            EMatc     = DS->def(DATA::model::rep::Ec);
            ForceMatc = DS->def(DATA::model::rep::dEc);
            break;
    }
    Tc = DS->def(DATA::model::rep::Tc);
}

Status& Kernel_ExactPropagator::initializeKernel_impl(Status& stat) { return stat; }

Status& Kernel_ExactPropagator::executeKernel_impl(Status& stat) {
    if (stat.frozen) return stat;
    if (Kernel_ExactPropagator::NAForce_type != NAForcePolicy::NAFEXACT) return stat;
    if (Dimension::N == 1) return stat; // 仅适用于多维情况

    for (int iP = 0; iP < Dimension::P; ++iP) {
        auto occ_nuc  = this->occ_nuc.subspan(iP, 1);
        auto rho_ele  = this->rho_ele.subspan(iP * Dimension::FF, Dimension::FF);
        auto rho_nuc  = this->rho_nuc.subspan(iP * Dimension::FF, Dimension::FF);
        auto f        = this->f.subspan(iP * Dimension::N, Dimension::N);
        auto p        = this->p.subspan(iP * Dimension::N, Dimension::N);
        auto m        = this->m.subspan(iP * Dimension::N, Dimension::N);
        auto fadd     = this->fadd.subspan(iP * Dimension::N, Dimension::N);
        auto fproj    = this->fproj.subspan(iP * Dimension::N, Dimension::N);
        auto grad     = this->grad.subspan(iP * Dimension::N, Dimension::N);
        auto ForceMat = this->ForceMat.subspan(iP * Dimension::NFF, Dimension::NFF);
        auto EMat     = this->EMat.subspan(iP * Dimension::FF, Dimension::FF);
        auto T        = this->T.subspan(iP * Dimension::FF, Dimension::FF);
        auto V        = this->V.subspan(iP * Dimension::FF, Dimension::FF);
        auto vpes     = this->vpes.subspan(iP, 1);
        auto alpha    = this->alpha.subspan(iP, 1);

        // --- Compute fproj: off-diagonal force projection ---
        // See docs/dev/general_soc_representation.md for the occ/force bridging logic.
        //
        // NAFEXACT splits the total force into a BO diagonal part (handled by
        // Kernel_NAForce) and an off-diagonal nonadiabatic part (fproj, handled
        // here).  fproj drives the exact analytical momentum propagation below.
        //
        // The "off-diagonal" part must be extracted in the Adiabatic basis,
        // because the BO / NAF splitting is only physically meaningful there:
        //   fproj_j = Re sum_{i!=k} rho_A[i,k] * dE_A[k,i]_j
        //
        // - Adiabatic repr:   rho is already in the Adiabatic basis, so we can
        //                     use TRACE2_OFFD directly.
        // - General_soc repr: rho is in the diabatic-like basis.  We transform
        //                     to Adiabatic, zero out the diagonal (-> rho_Q2),
        //                     transform rho_Q2 back, and use the full TRACE2.
        //                     This is equivalent to the Adiabatic off-diagonal
        //                     trace thanks to the cyclic property of the trace.
        //
        bool is_general_soc = (Kernel_Representation::nuc_repr_type == RepresentationPolicy::General_soc);
        if (is_general_soc) {
            auto Tc        = this->Tc.subspan(iP * Dimension::FF, Dimension::FF);
            auto ForceMatc = this->ForceMatc.subspan(iP * Dimension::NFF, Dimension::NFF);

            // Transform rho_nuc: General_soc → Adiabatic via Tc
            Kernel_Representation::transform(rho_nuc.data(), Tc.data(), Dimension::F,
                                             Kernel_Representation::inp_repr_type,
                                             RepresentationPolicy::Adiabatic,
                                             SpacePolicy::L);
            // Extract off-diagonal part of rho_nuc in the Adiabatic basis (zero diagonal)
            psnd_complex rho_Q2[Dimension::FF];
            for (int i = 0; i < Dimension::F; ++i) {
                for (int j = 0; j < Dimension::F; ++j) {
                    rho_Q2[i * Dimension::F + j] = rho_nuc[i * Dimension::F + j];
                    if (i == j) rho_Q2[i * Dimension::F + j] = 0.0;
                }
            }
            // Transform rho_Q2 back: Adiabatic → General_soc
            Kernel_Representation::transform(rho_Q2, Tc.data(), Dimension::F,
                                             RepresentationPolicy::Adiabatic,
                                             RepresentationPolicy::General_soc,
                                             SpacePolicy::L);
            // Restore rho_nuc: Adiabatic → General_soc
            Kernel_Representation::transform(rho_nuc.data(), Tc.data(), Dimension::F,
                                             RepresentationPolicy::Adiabatic,
                                             RepresentationPolicy::General_soc,
                                             SpacePolicy::L);
            // fproj = Re Tr(rho_Q2 * dEc_j): full trace because rho_Q2 already
            // contains only the Adiabatic off-diagonal contribution
            for (int j = 0, jFF = 0; j < Dimension::N; ++j, jFF += Dimension::FF) {
                auto dVcj = ForceMatc.subspan(jFF, Dimension::FF);
                fproj[j] = std::real(ARRAY_TRACE2(rho_Q2, dVcj.data(), Dimension::F, Dimension::F));
            }
        } else if (Kernel_Representation::nuc_repr_type == RepresentationPolicy::Adiabatic) {
            // Already in Adiabatic basis: TRACE2_OFFD gives off-diagonal directly
            Kernel_Representation::transform(rho_nuc.data(), T.data(), Dimension::F,
                                             Kernel_Representation::inp_repr_type,
                                             Kernel_Representation::nuc_repr_type,
                                             SpacePolicy::L);
            for (int j = 0, jFF = 0; j < Dimension::N; ++j, jFF += Dimension::FF) {
                auto dVj = ForceMat.subspan(jFF, Dimension::FF);
                fproj[j] = std::real(ARRAY_TRACE2_OFFD(rho_nuc.data(), dVj.data(), Dimension::F, Dimension::F));
            }
        } else {
            // Diabatic or other
            Kernel_Representation::transform(rho_nuc.data(), T.data(), Dimension::F,
                                             Kernel_Representation::inp_repr_type,
                                             Kernel_Representation::nuc_repr_type,
                                             SpacePolicy::L);
            for (int j = 0, jFF = 0; j < Dimension::N; ++j, jFF += Dimension::FF) {
                auto dVj = ForceMat.subspan(jFF, Dimension::FF);
                fproj[j] = std::real(ARRAY_TRACE2_OFFD(rho_nuc.data(), dVj.data(), Dimension::F, Dimension::F));
            }
        }
        psnd_real B_vec[Dimension::N]; // 局部变量 B_vec
        for (int j = 0; j < Dimension::N; ++j) {
            B_vec[j] = fproj[j] / sqrt(m[j]);
        }

        psnd_real norm_B;
        norm_B = 0.0;
        for (int j = 0; j < Dimension::N; ++j) {
            norm_B += B_vec[j] * B_vec[j];
        }
        norm_B = sqrt(norm_B);

        // e_pall: unit vector along B
        psnd_real e_pall[Dimension::N];
        for (int j = 0; j < Dimension::N; ++j) {
            e_pall[j] =  B_vec[j] / norm_B;
        }

        psnd_real alpha_pall;
        alpha_pall = 0.0;
        for (int j = 0; j < Dimension::N; ++j) {
            alpha_pall += p[j] * e_pall[j] / sqrt(m[j]) ;
        }

        psnd_real pi_vert_vec[Dimension::N];
        for (int j = 0; j < Dimension::N; ++j) {
            pi_vert_vec[j] = p[j] / sqrt(m[j]) - alpha_pall * e_pall[j];
        }

        psnd_real Ekin;
        Ekin = 0.0;
        for (int j = 0; j < Dimension::N; ++j) {
            Ekin += 0.5 * p[j] * p[j] / m[j];
        }

        if ( norm_B * dt[0] * scale / sqrt(2.0 * Ekin) < 1e-20 ) { 
            // B is too small, use Taylor expansion
            psnd_real temp_a;
            temp_a = 0;
            for (int j = 0; j < Dimension::N; ++j) {
                temp_a += B_vec[j] * p[j] / sqrt(m[j]);
            }
            for (int j = 0; j < Dimension::N; ++j) {
                p[j] = (1.0 + dt[0] * scale * temp_a / (2.0 * Ekin)) * p[j] - dt[0] * scale * sqrt(m[j]) * B_vec[j];
            }
        } else if ( norm_B * dt[0] * scale / sqrt(2.0 * Ekin) > 100 && abs(1.0 - alpha_pall/sqrt(2 * Ekin)) < 1e-15 ) {
            // B and M^{-1/2}P are nearly parallel or Ekin is too small, skip exact propagation
            // self-adaptive time-step strategy will be implemented later
            return stat; 
        } else {
            // normal case, use exact propagation formula
            psnd_real c1, c2;
            c1 = (sqrt(2.0 * Ekin) * ( (alpha_pall - sqrt(2.0 * Ekin)) + (alpha_pall + sqrt(2.0 * Ekin)) * exp(- 2.0 * norm_B * dt[0] * scale / sqrt(2.0 * Ekin))))
                / (sqrt(2.0 * Ekin) - alpha_pall  + (alpha_pall + sqrt(2.0 * Ekin)) * exp(- 2.0 * norm_B * dt[0] * scale / sqrt(2.0 * Ekin)));

            c2 = (sqrt(2.0 * Ekin) * 2.0 * exp(- norm_B * dt[0] * scale / sqrt(2.0 * Ekin)))
                / (sqrt(2.0 * Ekin) - alpha_pall  + (alpha_pall + sqrt(2.0 * Ekin)) * exp(- 2.0 * norm_B * dt[0] * scale / sqrt(2.0 * Ekin)));

            
            for (int j = 0; j < Dimension::N; ++j) {
                p[j] = c1 * sqrt(m[j]) * e_pall[j] + c2 * sqrt(m[j]) * pi_vert_vec[j];
            }
        };

        if (is_general_soc) {
            auto Tc = this->Tc.subspan(iP * Dimension::FF, Dimension::FF);
            Kernel_Representation::transform(rho_nuc.data(), Tc.data(), Dimension::F,  //
                                             RepresentationPolicy::General_soc,
                                             Kernel_Representation::inp_repr_type,  //
                                             SpacePolicy::L);
        } else {
            Kernel_Representation::transform(rho_nuc.data(), T.data(), Dimension::F,  //
                                        Kernel_Representation::nuc_repr_type,    //
                                        Kernel_Representation::inp_repr_type,    //
                                        SpacePolicy::L);
        }
        // std::cout << "[Kernel_ExactPropagator] scale: " << scale << "\n";
        // std::cout << "occ_nuc after exact propagator: " << occ_nuc[0] << std::endl;

        // std::cout << "[Kernel_ExactPropagator] After exact propagator: rho_nuc: " <<
        //     rho_nuc[0] << ", " << rho_nuc[1] << ", " << rho_nuc[2] << ", " << rho_nuc[3] << "\n";
        // // check if Ekin conserved
        // psnd_real Ekin2 = 0.0;
        // for (int j = 0; j < Dimension::N; ++j) {
        //     Ekin2 += 0.5 * p[j] * p[j] / m[j];    
        // }
        // std::cout << "Ekin before and after exact propagator: " << Ekin << " , " << Ekin2 << std::endl;
    }

    return stat;
}

}  // namespace PROJECT_NS
