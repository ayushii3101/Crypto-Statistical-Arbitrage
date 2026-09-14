#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <algorithm>


namespace SpreadEngine {

// ------------------------------------------------------------------ //
//  SpreadCalculator                                                    //
//                                                                      //
//  Core computation engine for real-time spread and z-score.          //
//                                                                      //
//  Design: stateful class rather than free functions.                 //
//  Why? The z-score requires a rolling mean and std.                  //
//  These must be maintained across ticks — they are STATE.            //
//  A stateless function would require passing the entire              //
//  price history on every tick call, which defeats the                //
//  purpose of using C++ for speed.                                    //
//                                                                      //
//  Instead: construct once, call update() on each tick.               //
//  The class maintains its own rolling buffer internally.             //
// ------------------------------------------------------------------ //

class SpreadCalculator {
public:

    // ---------------------------------------------------------------- //
    //  Constructor                                                       //
    //                                                                    //
    //  Parameters:                                                       //
    //  coint_vector   : hedge ratios from Johansen estimation           //
    //                   e.g. {1.0, -0.6, -0.3} for BTC/ETH/SOL        //
    //  window_size    : rolling window for z-score computation          //
    //                   (30 days × 1440 min/day = 43,200 rows)         //
    //                                                                    //
    //  const reference (&): pass vector by reference to avoid          //
    //  copying — copying a vector of N floats costs O(N) time.         //
    //  const means the function promises not to modify it.             //
    // ---------------------------------------------------------------- //
    SpreadCalculator(
        const std::vector<double>& coint_vector,
        int window_size
    );

    // ---------------------------------------------------------------- //
    //  update()                                                          //
    //                                                                    //
    //  Called on every new tick. Takes current log prices for           //
    //  all assets and returns the current z-score.                      //
    //                                                                    //
    //  This is the HOT PATH — the function called thousands of          //
    //  times per second. Every microsecond saved here matters.          //
    //                                                                    //
    //  Returns: z-score at current tick                                 //
    //  Side effect: updates internal rolling buffer                     //
    // ---------------------------------------------------------------- //
    double update(const std::vector<double>& log_prices);

    // ---------------------------------------------------------------- //
    //  Getters — read current internal state                            //
    //  Called by Python after update() for signal generation           //
    // ---------------------------------------------------------------- //
    double get_spread()       const;
    double get_zscore()       const;
    double get_rolling_mean() const;
    double get_rolling_std()  const;
    int    get_buffer_size()  const;
    bool   is_ready()         const;    // true once buffer is full

    // ---------------------------------------------------------------- //
    //  reset()                                                           //
    //                                                                    //
    //  Called by Python when BOCPD detects a structural break.          //
    //  Clears the rolling buffer so the z-score is recomputed           //
    //  from scratch after a regime change.                              //
    //                                                                    //
    //  Without this, the rolling mean and std from the old             //
    //  regime contaminate z-scores in the new regime.                  //
    // ---------------------------------------------------------------- //
    void reset();

    // ---------------------------------------------------------------- //
    //  batch_compute()                                                   //
    //                                                                    //
    //  Process a batch of historical prices at once.                    //
    //  Used during backtesting warm-up to populate the rolling         //
    //  buffer without calling update() row by row from Python.         //
    //                                                                    //
    //  Input:  matrix of log prices, shape (n_rows × n_assets)         //
    //  Output: vector of z-scores, one per row                         //
    //                                                                    //
    //  This is what makes backtesting fast: instead of                 //
    //  Python looping over 43,200 warmup rows calling update(),        //
    //  we pass the entire matrix to C++ in one call.                   //
    // ---------------------------------------------------------------- //
    std::vector<double> batch_compute(
        const std::vector<std::vector<double>>& price_matrix
    );

private:

    // ---------------------------------------------------------------- //
    //  Private members — internal state                                 //
    //                                                                    //
    //  These are invisible to Python. The pybind11 bindings only       //
    //  expose the public methods above.                                 //
    // ---------------------------------------------------------------- //

    std::vector<double> coint_vector_;   // hedge ratios
    int                 window_size_;    // rolling window length
    std::vector<double> spread_buffer_;  // circular buffer of spreads

    // Current computed values — updated by update()
    double current_spread_;
    double current_zscore_;
    double rolling_mean_;
    double rolling_std_;

    // Buffer management
    int    buffer_head_;      // index of oldest element (circular buffer)
    int    buffer_count_;     // how many elements currently in buffer
    double buffer_sum_;       // running sum for O(1) mean computation
    double buffer_sum_sq_;    // running sum of squares for O(1) std

    // ---------------------------------------------------------------- //
    //  Private methods                                                   //
    // ---------------------------------------------------------------- //

    // Compute spread from log prices using dot product
    double _compute_spread(const std::vector<double>& log_prices) const;

    // Update rolling statistics with new spread value
    void _update_rolling_stats(double new_spread);

    // Compute z-score from current rolling stats
    double _compute_zscore(double spread) const;

    // Validate input dimensions match coint_vector
    void _validate_input(const std::vector<double>& log_prices) const;
};


// ------------------------------------------------------------------ //
//  Free functions — utilities not requiring state                      //
// ------------------------------------------------------------------ //

// Compute log price from raw price
// Inline: compiler replaces the function call with the
// computation directly — zero function call overhead
inline double log_price(double price) {
    if (price <= 0.0) {
        throw std::invalid_argument(
            "Price must be positive for log computation"
        );
    }
    return std::log(price);
}

// Compute a vector of log prices from raw prices
std::vector<double> log_prices(const std::vector<double>& raw_prices);

// Dot product of two equal-length vectors
// Used for spread = prices @ coint_vector
double dot_product(
    const std::vector<double>& a,
    const std::vector<double>& b
);

} // namespace SpreadEngine
