// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace amgx
{
// Tracks completed iterations, never intermediate BiCGStab s residuals.
// A positive decision requests an independent b-A*x check, not convergence.
class ResidualDivergenceGuard
{
public:
    ResidualDivergenceGuard(double factor, int patience, int grace,
                            double initial, double roundoff)
        : factor_(factor), patience_(patience), grace_(grace), count_(0),
          best_(initial), floor_(std::max(std::numeric_limits<double>::min(),
                                        64.0 * roundoff * std::abs(initial))) {}

    bool enabled() const { return factor_ > 1.0; }

    bool excessive(double residual) const
    {
        return !std::isfinite(residual)
               || residual / std::max(best_, floor_) > factor_;
    }

    bool observe(double residual, int completed_iterations)
    {
        if (!enabled()) return false;
        if (!std::isfinite(residual)) return true;
        best_ = std::min(best_, residual);
        if (completed_iterations <= grace_ || !excessive(residual)) count_ = 0;
        else if (count_ < patience_) ++count_;
        return count_ >= patience_;
    }

    void unconfirmed() { count_ = 0; }
    double best() const { return best_; }

private:
    double factor_;
    int patience_, grace_, count_;
    double best_, floor_;
};
} // namespace amgx
