#ifndef PSND_UTIL_QM_H
#define PSND_UTIL_QM_H

namespace PROJECT_NS {
namespace util_qm {

// Enforce sign continuity of the NAC tensor between two consecutive QM
// calls. For every (i != j) state pair, the Bohr-frame vector
// nac[:, i, j] (length N) is compared against nac_prev[:, i, j]:
//   - obtuse angle (cos < 0)         -> flip nac[:, i, j]
//   - one of the norms blows up      -> copysign each component
// On exit, nac_prev is overwritten with the (possibly sign-corrected) nac.
//
// nac and nac_prev are flat arrays of length N*F*F laid out as [k, i, j]
// with stride (FF, F, 1). Caller must guard the first step (no prev frame
// to compare against).
void track_nac_sign(double* nac, double* nac_prev, int F, int N);

}  // namespace util_qm
}  // namespace PROJECT_NS

#endif  // PSND_UTIL_QM_H
