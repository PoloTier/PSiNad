#include "psnd/Kernel_Elec_Switch.h"

#include <algorithm>

#include "psnd/Kernel_Elec_Utils.h"
#include "psnd/Kernel_NAForce.h"
#include "psnd/Kernel_Random.h"
#include "psnd/Kernel_Representation.h"
#include "psnd/debug_utils.h"
#include "psnd/hash_fnv1a.h"
#include "psnd/linalg.h"
#include "psnd/macro_utils.h"
#include "psnd/vars_list.h"

namespace PROJECT_NS {

const std::string Kernel_Elec_Switch::getName() { return "Kernel_Elec_Switch"; }

int Kernel_Elec_Switch::getType() const { return utils::hash(FUNCTION_NAME); }

void Kernel_Elec_Switch::setInputParam_impl(std::shared_ptr<Param> PM) {
    hopping_choose_type    = 0;
    hopping_direction_type = 0;
    reflect                = true;
    hopping_choose_type    = _param->get_int({"solver.hopping_choose_type"}, LOC(), hopping_choose_type);
    hopping_direction_type = _param->get_int({"solver.hopping_direction_type"}, LOC(), hopping_direction_type);
    reflect                = _param->get_bool({"solver.reflect"}, LOC(), reflect);
}

void Kernel_Elec_Switch::setInputDataSet_impl(std::shared_ptr<DataSet> DS) {
    dt_ptr    = DS->def(DATA::control::dt);
    Epot      = DS->def(DATA::integrator::Epot);
    p         = DS->def(DATA::integrator::p);
    m         = DS->def(DATA::integrator::m);
    vpes      = DS->def(DATA::model::vpes);
    T         = DS->def(DATA::model::rep::T);
    H         = DS->def(DATA::model::rep::H);
    occ_nuc            = DS->def(DATA::integrator::occ_nuc);
    occ_nuc_pre_switch = DS->def(DATA::integrator::occ_nuc_pre_switch);
    rho_ele   = DS->def(DATA::integrator::rho_ele);
    rho_nuc   = DS->def(DATA::integrator::rho_nuc);
    direction = DS->def(DATA::integrator::tmp::direction);

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
    // Always bind Tc for General_soc path (no-op when not used)
    Tc = DS->def(DATA::model::rep::Tc);
}

Status& Kernel_Elec_Switch::initializeKernel_impl(Status& stat) {
    if (_param->get_string({"load", "solver.load"}, LOC(), "").find(":resume") != std::string::npos) return stat;
    double dt_bak = dt_ptr[0];
    dt_ptr[0]     = 0.0e0;  // @make no dynamics here
    executeKernel_impl(stat);
    dt_ptr[0] = dt_bak;
    return stat;
}

Status& Kernel_Elec_Switch::executeKernel_impl(Status& stat) {
    // Snapshot occ_nuc before any hop logic runs. Kernel_Hop_Replan compares
    // occ_nuc against occ_nuc_pre_switch to detect a real hop in this step.
    // This kernel is the only writer of occ_nuc in the chain, so the snapshot
    // lives here. P is the PIMD bead count (= 1 for nonadiabatic dynamics);
    // using it matches the main loop below.
    for (int iP = 0; iP < Dimension::P; ++iP) occ_nuc_pre_switch[iP] = occ_nuc[iP];

    for (int iP = 0; iP < Dimension::P; ++iP) {
        auto occ_nuc  = this->occ_nuc.subspan(iP, 1);
        auto rho_ele  = this->rho_ele.subspan(iP * Dimension::FF, Dimension::FF);
        auto rho_nuc  = this->rho_nuc.subspan(iP * Dimension::FF, Dimension::FF);
        auto T        = this->T.subspan(iP * Dimension::FF, Dimension::FF);
        auto Epot     = this->Epot.subspan(iP, 1);
        auto vpes     = this->vpes.subspan(iP, 1);
        auto p        = this->p.subspan(iP * Dimension::N, Dimension::N);
        auto m        = this->m.subspan(iP * Dimension::N, Dimension::N);
        auto H        = this->H.subspan(iP * Dimension::FF, Dimension::FF);  // ????
        auto EMat     = this->EMat.subspan(iP * Dimension::FF, Dimension::FF);
        auto ForceMat = this->ForceMat.subspan(iP * Dimension::NFF, Dimension::NFF);

        // --- General_soc short-circuit (complex SOC path) ---
        // See docs/dev/general_soc_representation.md for the occ/force bridging logic.
        if (Kernel_Representation::nuc_repr_type == RepresentationPolicy::General_soc) {
            auto Tc        = this->Tc.subspan(iP * Dimension::FF, Dimension::FF);
            auto EMatc     = this->EMatc.subspan(iP * Dimension::FF, Dimension::FF);
            auto ForceMatc = this->ForceMatc.subspan(iP * Dimension::NFF, Dimension::NFF);

            Kernel_Representation::transform(rho_ele.data(), Tc.data(), Dimension::F,  //
                                             Kernel_Representation::inp_repr_type,
                                             RepresentationPolicy::General_soc,  //
                                             SpacePolicy::L);
            Kernel_Representation::transform(rho_nuc.data(), Tc.data(), Dimension::F,  //
                                             Kernel_Representation::inp_repr_type,
                                             RepresentationPolicy::General_soc,  //
                                             SpacePolicy::L);

            // Force types that depend on occ_nuc (BO, NAF, NAFEXACT, etc.) require
            // the hopping target to be chosen in the Adiabatic basis, so that
            // occ_nuc correctly indexes adiabatic surfaces.
            // EHR is purely mean-field (f = Tr(rho*dV)) and does not use occ_nuc.
            bool need_adia_hop = (Kernel_NAForce::NAForce_type == NAForcePolicy::BO ||
                                  Kernel_NAForce::NAForce_type == NAForcePolicy::NAF ||
                                  Kernel_NAForce::NAForce_type == NAForcePolicy::NAFEXACT ||
                                  Kernel_NAForce::NAForce_type == NAForcePolicy::NAF2 ||
                                  Kernel_NAForce::NAForce_type == NAForcePolicy::BOSD ||
                                  Kernel_NAForce::NAForce_type == NAForcePolicy::NAFSD);
            if (need_adia_hop) {
                Kernel_Representation::transform(rho_ele.data(), Tc.data(), Dimension::F,
                                                 RepresentationPolicy::General_soc,
                                                 RepresentationPolicy::Adiabatic,
                                                 SpacePolicy::L);
                Kernel_Representation::transform(rho_nuc.data(), Tc.data(), Dimension::F,
                                                 RepresentationPolicy::General_soc,
                                                 RepresentationPolicy::Adiabatic,
                                                 SpacePolicy::L);
            }

            psnd_int  from = occ_nuc[0], to = occ_nuc[0];
            psnd_real Efrom, Eto;
            Efrom = elec_utils::calc_ElectricalEnergy(EMatc.data(), rho_nuc.data(), occ_nuc[0]);

            switch (hopping_choose_type) {
                case 0: to = elec_utils::max_choose(rho_ele.data()); break;
                case 1: to = elec_utils::max_choose(rho_nuc.data()); break;
                case 2: to = elec_utils::hopping_choose(rho_ele.data(), H.data(), occ_nuc[0], dt_ptr[0]); break;
                case 3: to = elec_utils::pop_choose(rho_ele.data()); break;
                case 4: to = elec_utils::pop_choose(rho_nuc.data()); break;
                case 5: to = elec_utils::pop_neg_choose(rho_nuc.data()); break;
                default: break;
            }
            Eto = elec_utils::calc_ElectricalEnergy(EMatc.data(), rho_nuc.data(), to);

            switch (hopping_direction_type) {
                case 0:
                    elec_utils::hopping_direction(direction.data(), EMatc.data(), ForceMatc.data(), rho_ele.data(),
                                                  occ_nuc[0], to);
                    break;
                case 1:
                    elec_utils::hopping_direction(direction.data(), ForceMatc.data(), occ_nuc[0], to);
                    break;
                case 2:
                    for (int j = 0; j < Dimension::N; ++j) direction[j] = p[j];
                    break;
                case 3:
                    for (int j = 0; j < Dimension::N; ++j)
                        direction[j] = std::real(ForceMatc[j * Dimension::FF + to * Dimension::Fadd1] -
                                                 ForceMatc[j * Dimension::FF + occ_nuc[0] * Dimension::Fadd1]);
                    break;
            }

            if (occ_nuc[0] != to) {
                occ_nuc[0] = elec_utils::hopping_impulse(direction.data(), p.data(), m.data(),  //
                                                         Efrom, Eto, occ_nuc[0], to, reflect);
            }
            Epot[0] = vpes[0] + ((occ_nuc[0] == to) ? Eto : Efrom);

            // Reverse the NAFEXACT Adiabatic transform
            if (need_adia_hop) {
                Kernel_Representation::transform(rho_ele.data(), Tc.data(), Dimension::F,
                                                 RepresentationPolicy::Adiabatic,
                                                 RepresentationPolicy::General_soc,
                                                 SpacePolicy::L);
                Kernel_Representation::transform(rho_nuc.data(), Tc.data(), Dimension::F,
                                                 RepresentationPolicy::Adiabatic,
                                                 RepresentationPolicy::General_soc,
                                                 SpacePolicy::L);
            }

            Kernel_Representation::transform(rho_ele.data(), Tc.data(), Dimension::F,  //
                                             RepresentationPolicy::General_soc,
                                             Kernel_Representation::inp_repr_type,  //
                                             SpacePolicy::L);
            Kernel_Representation::transform(rho_nuc.data(), Tc.data(), Dimension::F,  //
                                             RepresentationPolicy::General_soc,
                                             Kernel_Representation::inp_repr_type,  //
                                             SpacePolicy::L);
            continue;
        }
        // --- end General_soc short-circuit ---

        //////////////////////////////////////////////////////////////////////
        // switching is taken on nuc_repr_type representation
        // & Emat, ForceMat is aligned with nuc_repr_type representation
        // occ_nuc[0] = occ_nuc[0]; /// occ_nuc is defined in nuc_repr_type representation
        Kernel_Representation::transform(rho_ele.data(), T.data(), Dimension::F,  //
                                         Kernel_Representation::inp_repr_type,    //
                                         Kernel_Representation::nuc_repr_type,    //
                                         SpacePolicy::L);
        Kernel_Representation::transform(rho_nuc.data(), T.data(), Dimension::F,  //
                                         Kernel_Representation::inp_repr_type,    //
                                         Kernel_Representation::nuc_repr_type,    //
                                         SpacePolicy::L);

        /// step 1: generate new state
        psnd_int  from, to;
        psnd_real Efrom, Eto;
        Efrom = elec_utils::calc_ElectricalEnergy(EMat.data(), rho_nuc.data(),
                                                  occ_nuc[0]);  // occ_nuc defined in nuc_repr_type

        // std::cout << LOC() << "Current state: " << occ_nuc[0] << ", Efrom=" << Efrom << "\n" << std::endl;

        switch (hopping_choose_type) {
            case 0: {  // from the max elecment of rho_ele
                to = elec_utils::max_choose(rho_ele.data());
                break;
            }
            case 1: {  // from the max elecment of rho_nuc = rho_ele - Gamma
                to = elec_utils::max_choose(rho_nuc.data());
                break;
            }
            case 2: {  // from hopping rate
                // @bug if nuc_repr_type = Diabatic, then hopping_choose(rho_ele, V, occ_nuc[0], dt_ptr[0]);
                to = elec_utils::hopping_choose(rho_ele.data(), H.data(), occ_nuc[0], dt_ptr[0]);
                break;
            }
            case 3: {  // from the probability of rho_ele
                to = elec_utils::pop_choose(rho_ele.data());
                break;
            }
            case 4: {  // from the probability of rho_nuc (cutoff of negativity)
                to = elec_utils::pop_choose(rho_nuc.data());
                break;
            }
            case 5: {  // from the probability of rho_nuc (absolute of negativity)
                to = elec_utils::pop_neg_choose(rho_nuc.data());
                break;
            }
            case 12: {  // adiabatic Landau-Zener

                // for (int i = 0, ik = 0; i < Dimension::F; ++i) {
                //     for (int k = 0; k < Dimension::F; ++k, ++ik)
                //         DeltaE[ik] = EMat[i * Dimension::Fadd1] - EMat[k * Dimension::Fadd1];
                // }

                // for (int to = 0, to_occ = occ_nuc[0]; to < Dimension::F; ++to, to_occ += Dimension::F) {
                //     if (std::abs(DeltaE_last[to_occ]) < std::abs(DeltaE_llast[to_occ]) &&
                //         std::abs(DeltaE_last[to_occ]) < std::abs(DeltaE_llast[to_occ])) {
                //         double DE3 = DeltaE_last[to_occ] * DeltaE_last[to_occ] * DeltaE_last[to_occ];
                //         double DDE = (DeltaE[to_occ] - 2 * DeltaE_last[to_occ] + DeltaE_llast[to_occ])  //
                //                      / (dt_ptr[0] * dt_ptr[0]);
                //         if (DE3 / DDE >= 0.0e0) {  //
                //             double prob = std::exp(-phys::math::halfpi * std::sqrt(DE3 / DDE));
                //             double randu;
                //             Kernel_Random::rand_uniform(&randu);
                //             if (randu < prob) occ_nuc[0] = to;
                //             break;
                //         }
                //     }
                // }
                // for (int ik = 0; ik < Dimension::FF; ++ik) DeltaE_llast[ik] = DeltaE_last[ik];
                // for (int ik = 0; ik < Dimension::FF; ++ik) DeltaE_last[ik] = DeltaE[ik];
                break;
            }
        }
        Eto = elec_utils::calc_ElectricalEnergy(EMat.data(), rho_nuc.data(), to);

        // std::cout << LOC() << occ_nuc[0] << " -> " << to << ": Efrom=" << Efrom << ",  Eto=" << Eto << "\n" << std::endl;
 
        /// step 2: determine a direction
        switch (hopping_direction_type) {
            case 0: {  // along density weighted nacv
                elec_utils::hopping_direction(direction.data(), EMat.data(), ForceMat.data(), rho_ele.data(),
                                              occ_nuc[0], to);
                break;
            }
            case 1: {  // along nacv
                elec_utils::hopping_direction(direction.data(), ForceMat.data(), occ_nuc[0], to);
                break;
            }
            case 2: {  // along p
                for (int j = 0; j < Dimension::N; ++j) direction[j] = p[j];
                break;
            }
            case 3: {  // along difference of force
                for (int j = 0; j < Dimension::N; ++j)
                    direction[j] = ForceMat[j * Dimension::FF + to * Dimension::Fadd1] -
                                   ForceMat[j * Dimension::FF + occ_nuc[0] * Dimension::Fadd1];
                break;
            }
        }

        // step 3: try hop and adjust momentum
        if (occ_nuc[0] != to) {
            occ_nuc[0] =
                elec_utils::hopping_impulse(direction.data(), p.data(), m.data(), Efrom, Eto, occ_nuc[0], to, reflect);
        }
        Epot[0] = vpes[0] + ((occ_nuc[0] == to) ? Eto : Efrom);

        Kernel_Representation::transform(rho_ele.data(), T.data(), Dimension::F,  //
                                         Kernel_Representation::nuc_repr_type,    //
                                         Kernel_Representation::inp_repr_type,    //
                                         SpacePolicy::L);
        Kernel_Representation::transform(rho_nuc.data(), T.data(), Dimension::F,  //
                                         Kernel_Representation::nuc_repr_type,    //
                                         Kernel_Representation::inp_repr_type,    //
                                         SpacePolicy::L);
    }
    return stat;
}
};  // namespace PROJECT_NS
