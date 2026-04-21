#ifndef Kernel_Hop_Replan_H
#define Kernel_Hop_Replan_H

#include "psnd/Kernel.h"

namespace PROJECT_NS {

/**
 * @brief Conditionally re-invoke kmodel + krepr after a surface hop.
 *
 * Background: when model.use_state_detection = true, the QM call that
 * precedes kswitch only computes gradients for the currently occupied
 * state (and a small set of energy-coupled pairs). If kswitch then hops
 * to a state outside that set, downstream Kernel_NAForce reads a
 * ForceMat diagonal element that QM never wrote -- the force collapses
 * to ~0, the next Kernel_Update_p half-step under-drives momentum, and
 * Kernel_Conserve may silently paper over the drift.
 *
 * What this kernel does:
 *   1. Read use_state_detection once. If false, short-circuit to a
 *      no-op on every step (zero overhead).
 *   2. If true, compare occ_nuc to the occ_nuc_pre_switch snapshot that
 *      Kernel_Elec_Switch took at the start of its execute. If any
 *      trajectory hopped, re-run kmodel and krepr so state detection
 *      writes the correct gradient for the new occupied state.
 *
 * ku_U does not need to be re-run: geometry is unchanged between the
 * two kmodel calls, so V is unchanged, T/eig are bit-identical, and the
 * propagator U is the same. Only dE changes because dV was refreshed.
 */
class Kernel_Hop_Replan final : public Kernel {
   public:
    Kernel_Hop_Replan(std::shared_ptr<Kernel> kmodel, std::shared_ptr<Kernel> krepr);

    virtual const std::string getName();

    virtual int getType() const;

   private:
    std::shared_ptr<Kernel> _kmodel;
    std::shared_ptr<Kernel> _krepr;
    bool                    use_state_detection = false;

    span<psnd_int> occ_nuc;
    span<psnd_int> occ_nuc_pre_switch;

    virtual void setInputParam_impl(std::shared_ptr<Param> PM);

    virtual void setInputDataSet_impl(std::shared_ptr<DataSet> DS);

    virtual Status& executeKernel_impl(Status& stat);
};

};  // namespace PROJECT_NS

#endif  // Kernel_Hop_Replan_H
