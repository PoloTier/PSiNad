#include "psnd/util_qm.h"

#include <cmath>

namespace PROJECT_NS {
namespace util_qm {

void track_nac_sign(double* nac, double* nac_prev, int F, int N) {
    const int FF = F * F;
    for (int i = 0; i < F; ++i) {
        for (int j = 0; j < F; ++j) {
            if (i == j) continue;

            const double norm_eps = 1e-13;
            double       norm_old = 0.0;
            double       norm_new = 0.0;
            double       cosangle = 0.0;
            const int    IJ       = i * F + j;
            for (int k = 0, idx = IJ; k < N; ++k, idx += FF) {
                norm_old += nac_prev[idx] * nac_prev[idx];
                norm_new += nac[idx] * nac[idx];
                cosangle += nac_prev[idx] * nac[idx];
            }
            norm_old = std::sqrt(norm_old);
            norm_new = std::sqrt(norm_new);
            if (norm_old < norm_eps || norm_new < norm_eps) {
                cosangle = 1.0;
            } else {
                cosangle = cosangle / (norm_old * norm_new);
            }

            if (norm_new > 1e13 || norm_old > 1e13) {
                for (int k = 0; k < N; ++k) {
                    nac[k * FF + IJ] = std::copysign(nac[k * FF + IJ], nac_prev[k * FF + IJ]);
                }
            } else if (cosangle < 0) {
                for (int k = 0; k < N; ++k) { nac[k * FF + IJ] *= -1; }
            }
        }
    }
    for (int i = 0; i < N * FF; ++i) nac_prev[i] = nac[i];
}

}  // namespace util_qm
}  // namespace PROJECT_NS
