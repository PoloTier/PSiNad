#include "psnd/Kernel.h"
#include "psnd/Kernel_Conserve.h"
#include "psnd/Kernel_Dump_DataSet.h"
#include "psnd/Kernel_Elec_Functions.h"
#include "psnd/Kernel_Elec_Switch.h"
#include "psnd/Kernel_Hop_Replan.h"
#include "psnd/Kernel_Load_DataSet.h"
#include "psnd/Kernel_NAForce.h"
#include "psnd/Kernel_Prioritization.h"
#include "psnd/Kernel_Random.h"
#include "psnd/Kernel_Read_Dimensions.h"
#include "psnd/Kernel_Recorder.h"
#include "psnd/Kernel_Representation.h"
#include "psnd/Kernel_Update.h"
#include "psnd/Model.h"
#include "psnd/Solver.h"
#include "psnd/Kernel_ExactPropagator.h"

namespace PROJECT_NS {

/*
 * NAD (Nonadiabatic Dynamics) Solver -- BAOAB splitting integrator
 *
 * Dynamics loop: each iteration = kcond (observe) + kinte (integrate)
 *
 *   kcond:  kfuncs -> krecd
 *     kfuncs  (Kernel_Elec_Functions) -- compute observables K0, KTWD, etc. from rho_ele
 *     krecd   (Kernel_Recorder)       -- record observables to output files
 *
 *   kinte:  BAOAB integrator chain
 *     ku_p(0.5)   -- B: half-step momentum update,  p -= f * dt/2
 *     kexact(0.5) -- exact propagator for NAFEXACT off-diagonal force projection
 *     krepr       -- representation: eigensolve V->eig/T, build H, eigensolve H->lam/R
 *     ku_U(0.5)   -- update propagator U from lam/R, propagate c/rho_ele/rho_nuc
 *     ku_x(0.5)   -- A: half-step position update,  x += p/m * dt/2
 *     ku_x(0.5)   -- A: half-step position update,  x += p/m * dt/2
 *     kmodel      -- evaluate model potential V(x), gradients dV(x)
 *     krepr       -- representation: eigensolve at new x
 *     ku_U(0.5)   -- update propagator U, propagate c/rho_ele/rho_nuc
 *     kswitch     -- surface hopping: decide hop, update occ_nuc
 *     knaf        -- compute force f on the (possibly new) active surface
 *     kexact(0.5) -- exact propagator for NAFEXACT off-diagonal force projection
 *     ku_p(0.5)   -- B: half-step momentum update,  p -= f * dt/2
 *     kconserve   -- enforce energy conservation after hop
 */
std::shared_ptr<Solver> NAD_Kernel(std::shared_ptr<Model> kmodel, std::string NAD_Kernel_name) {
    bool take_ownership_false = false;

    // Root Kernel
    std::shared_ptr<Kernel> ker(new Kernel(NAD_Kernel_name));

    /// Integrator Kernel
    std::shared_ptr<Kernel> kinte(new Kernel("BAOAB_Integrator"));

    std::shared_ptr<Kernel_Representation> krepr(new Kernel_Representation());
    std::shared_ptr<Kernel_Elec_Switch>    kswitch(new Kernel_Elec_Switch());
    std::shared_ptr<Kernel_NAForce>        knaf(new Kernel_NAForce());
    std::shared_ptr<Kernel_Elec_Functions> kfuncs(new Kernel_Elec_Functions());

    std::shared_ptr<Kernel_Update_p> ku_p(new Kernel_Update_p(0.5));
    std::shared_ptr<Kernel_Update_x> ku_x(new Kernel_Update_x(0.5));
    std::shared_ptr<Kernel_Update_U> ku_U(new Kernel_Update_U(0.5));
    std::shared_ptr<Kernel_ExactPropagator> kexact(new Kernel_ExactPropagator(0.5));
    std::shared_ptr<Kernel_Hop_Replan>      kreplan(new Kernel_Hop_Replan(kmodel, krepr));

    /// Result & Sampling & TCF
    std::shared_ptr<Kernel_Recorder> krecd(new Kernel_Recorder());

    // BAOAB integration chain (see block comment above for details)
    kinte->appendChild(ku_p);       // B: p -= f * dt/2
    kinte->appendChild(kexact);     // NAFEXACT off-diagonal exact propagation (first half)
    kinte->appendChild(krepr);      // O: eigensolve V -> eig/T/Tc, build H -> lam/R
    kinte->appendChild(ku_U);       // O: propagate c, rho_ele, rho_nuc via U (first half)
    kinte->appendChild(ku_x);       // A: x += p/m * dt/2
    kinte->appendChild(ku_x);       // A: x += p/m * dt/2
    kinte->appendChild(kmodel);     // evaluate V(x), dV(x) at new position
    kinte->appendChild(krepr);      // O: eigensolve at new x
    kinte->appendChild(ku_U);       // O: propagate c, rho_ele, rho_nuc via U (second half)
    kinte->appendChild(kswitch);    // surface hopping: decide hop, update occ_nuc
    // If state detection is on and a real hop just happened, re-runs kmodel
    // + krepr so the new occupied state's gradient is written. Otherwise
    // a no-op. See Kernel_Hop_Replan.h for the rationale.
    kinte->appendChild(kreplan);
    kinte->appendChild(knaf);       // compute force on active surface (uses updated occ_nuc)
    kinte->appendChild(kexact);     // NAFEXACT off-diagonal exact propagation (second half)
    kinte->appendChild(ku_p);       // B: p -= f * dt/2
    kinte->appendChild(std::shared_ptr<Kernel_Conserve>(new Kernel_Conserve()));  // energy conservation

    std::shared_ptr<Kernel_Iterative>   kiter(new Kernel_Iterative());
    std::shared_ptr<Kernel_Conditional> kcond(new Kernel_Conditional());
    kcond->appendChild(kfuncs);
    kcond->appendChild(krecd);
    kiter->appendChild(kcond);  // stacked in iteration
    kiter->appendChild(kinte);  // stacked in iteration

    // /// CMM kernel
    ker->appendChild(std::shared_ptr<Kernel_Load_DataSet>(new Kernel_Load_DataSet()))
        .appendChild(std::shared_ptr<Kernel_Random>(new Kernel_Random()))
        .appendChild(std::shared_ptr<Kernel_Read_Dimensions>(new Kernel_Read_Dimensions()))
        .appendChild(std::shared_ptr<Kernel_Prioritization>(new Kernel_Prioritization({kmodel, kinte}, 1)))
        // .appendChild(std::shared_ptr<Kernel_Prioritization>(  //
        //     new Kernel_Prioritization({kmodel, krepr, kfuncs, ku_U, kswitch, knaf, krecd}, 2)))
        .appendChild(std::shared_ptr<Kernel_Prioritization>(  //
            new Kernel_Prioritization({ku_U, kswitch, krecd}, 2)))
        .appendChild(kiter)
        .appendChild(std::shared_ptr<Kernel_Dump_DataSet>(new Kernel_Dump_DataSet()));

    std::shared_ptr<Solver> sol(new Solver(ker));
    return sol;
}

};  // namespace PROJECT_NS
