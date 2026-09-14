// This file has one job:
// Expose the C++ SpreadEngine to Python via pybind11.
//
// pybind11 works by generating a Python extension module
// at compile time. When Python does:
//     import spread_engine
// it loads a compiled .so (Linux/Mac) or .pyd (Windows)
// file that pybind11 produced from this file.
//
// Everything in this file is a declaration of what Python
// can see. Nothing computational happens here — this is
// pure interface definition.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>         // automatic list↔vector conversion
#include <pybind11/numpy.h>       // NumPy array support
#include "spread_engine.h"

// pybind11 namespace alias — standard convention
namespace py = pybind11;

using namespace SpreadEngine;


// ------------------------------------------------------------------ //
//  Helper: NumPy array → std::vector<double>                          //
//                                                                      //
//  Used in the NumPy-optimised update path.                           //
//  Takes advantage of contiguous memory layout                        //
//  for zero-copy conversion where possible.                           //
// ------------------------------------------------------------------ //

std::vector<double> numpy_to_vector(
    py::array_t<double> arr
) {
    // request() gives us raw buffer info:
    // pointer to data, shape, strides, itemsize
    py::buffer_info buf = arr.request();

    if (buf.ndim != 1) {
        throw std::invalid_argument(
            "numpy_to_vector: expected 1D array, got " +
            std::to_string(buf.ndim) + "D"
        );
    }

    // ptr is a raw C++ pointer to the NumPy data buffer
    // No element-by-element Python object unpacking —
    // just a direct pointer to contiguous memory
    double* ptr = static_cast<double*>(buf.ptr);

    // Construct vector from pointer range
    // This is a single memcpy under the hood
    return std::vector<double>(ptr, ptr + buf.shape[0]);
}


std::vector<std::vector<double>> numpy_matrix_to_vector(
    py::array_t<double> arr
) {
    // Convert 2D NumPy array to vector of vectors
    // Used by batch_compute() for the warmup phase
    //
    // arr shape: (n_rows, n_assets) = (43200, 3)
    // Output:    vector of n_rows vectors, each of n_assets

    py::buffer_info buf = arr.request();

    if (buf.ndim != 2) {
        throw std::invalid_argument(
            "Expected 2D array for batch compute"
        );
    }

    int n_rows   = buf.shape[0];
    int n_cols   = buf.shape[1];
    double* ptr  = static_cast<double*>(buf.ptr);

    std::vector<std::vector<double>> result;
    result.reserve(n_rows);

    for (int i = 0; i < n_rows; ++i) {
        // Each row starts at ptr + i*n_cols
        // Construct vector from that pointer range
        result.emplace_back(
            ptr + i * n_cols,
            ptr + i * n_cols + n_cols
        );
    }

    return result;
}


// ------------------------------------------------------------------ //
//  PYBIND11_MODULE — the module definition                            //
//                                                                      //
//  This macro generates the Python module entry point.                //
//  "spread_engine" becomes the Python import name.                    //
//  "m" is the module object we attach everything to.                  //
// ------------------------------------------------------------------ //

PYBIND11_MODULE(spread_engine, m) {

    // Module docstring — visible in Python via:
    //     import spread_engine; help(spread_engine)
    m.doc() = R"pbdoc(
        spread_engine: C++ real-time spread and z-score computation.

        Provides sub-millisecond spread computation and rolling
        z-score calculation for tick-level statistical arbitrage
        on cryptocurrency perpetual futures.

        Main class: SpreadCalculator
        Usage:
            import spread_engine as se
            calc = se.SpreadCalculator(
                coint_vector=[1.0, -0.6, -0.3],
                window_size=43200
            )
            zscore = calc.update([10.5, 7.2, 4.1])
    )pbdoc";


    // ---------------------------------------------------------------- //
    //  Expose SpreadCalculator class                                    //
    // ---------------------------------------------------------------- //

    py::class_<SpreadCalculator>(m, "SpreadCalculator",
        R"pbdoc(
        Real-time spread calculator with O(1) rolling z-score.

        Maintains a circular buffer of spread values and updates
        rolling mean and standard deviation in O(1) per tick
        using running sum arithmetic.

        Parameters
        ----------
        coint_vector : list of float
            Hedge ratios from Johansen cointegration.
            e.g. [1.0, -0.6, -0.3] for BTC/ETH/SOL
        window_size : int
            Rolling window size in ticks.
            30 days × 1440 min/day = 43200 for minute data.
        )pbdoc"
    )

    // ── Constructor ────────────────────────────────────────────────
    .def(
        py::init<const std::vector<double>&, int>(),
        py::arg("coint_vector"),
        py::arg("window_size"),
        "Construct SpreadCalculator with hedge ratios and window."
    )

    // ── update() — primary tick interface ─────────────────────────
    //
    // Two overloads exposed to Python:
    // 1. Accepts Python list  → convenient for single ticks
    // 2. Accepts NumPy array  → faster for batched calls
    //
    // Python will call whichever matches the argument type.

    .def(
        "update",
        // Overload 1: Python list → std::vector (auto-converted)
        [](SpreadCalculator& self,
           const std::vector<double>& log_prices) {
            return self.update(log_prices);
        },
        py::arg("log_prices"),
        R"pbdoc(
        Process one tick and return the current z-score.

        Parameters
        ----------
        log_prices : list of float
            Log prices for all assets in cointegrating vector order.
            e.g. [log(BTC_price), log(ETH_price), log(SOL_price)]

        Returns
        -------
        float
            Current z-score. Returns 0.0 during warmup period
            (before buffer is full). Check is_ready() first.

        Raises
        ------
        ValueError
            If log_prices dimension != coint_vector dimension,
            or if any price is non-finite (NaN or Inf).
        )pbdoc"
    )

    .def(
        "update_numpy",
        // Overload 2: NumPy array → faster buffer access
        [](SpreadCalculator& self,
           py::array_t<double> log_prices) {
            auto vec = numpy_to_vector(log_prices);
            return self.update(vec);
        },
        py::arg("log_prices"),
        R"pbdoc(
        NumPy-optimised update. Preferred for high-frequency calls.

        Accepts a 1D NumPy array of float64.
        Uses direct buffer access — avoids per-element
        Python object unpacking overhead.
        )pbdoc"
    )

    // ── batch_compute() ───────────────────────────────────────────
    .def(
        "batch_compute",
        // List of lists version
        [](SpreadCalculator& self,
           const std::vector<std::vector<double>>& price_matrix) {
            return self.batch_compute(price_matrix);
        },
        py::arg("price_matrix"),
        R"pbdoc(
        Process a batch of historical prices.

        Used during backtest warmup to populate the rolling buffer
        without calling update() row by row from Python.

        Parameters
        ----------
        price_matrix : list of list of float
            Shape: (n_rows, n_assets)
            Each row is one timestep's log prices.

        Returns
        -------
        list of float
            Z-score for each row. Rows before buffer is full
            return 0.0.
        )pbdoc"
    )

    .def(
        "batch_compute_numpy",
        // NumPy matrix version — preferred for large batches
        [](SpreadCalculator& self,
           py::array_t<double> price_matrix) {
            auto mat = numpy_matrix_to_vector(price_matrix);
            return self.batch_compute(mat);
        },
        py::arg("price_matrix"),
        R"pbdoc(
        NumPy-optimised batch compute.

        Accepts a 2D NumPy array of shape (n_rows, n_assets).
        Uses direct buffer access for the conversion —
        significantly faster than list of lists for large batches
        (e.g. 43200 × 3 warmup matrix).
        )pbdoc"
    )

    // ── reset() ───────────────────────────────────────────────────
    .def(
        "reset",
        &SpreadCalculator::reset,
        R"pbdoc(
        Reset internal state after a structural break.

        Call this when BOCPD detects a changepoint.
        Clears the rolling buffer so z-scores are recomputed
        from scratch — prevents old-regime statistics from
        contaminating new-regime signals.

        O(1) operation. No memory allocation or deallocation.
        )pbdoc"
    )

    // ── Getters ───────────────────────────────────────────────────
    .def(
        "get_spread",
        &SpreadCalculator::get_spread,
        "Return the spread value at the most recent tick."
    )
    .def(
        "get_zscore",
        &SpreadCalculator::get_zscore,
        "Return the z-score at the most recent tick."
    )
    .def(
        "get_rolling_mean",
        &SpreadCalculator::get_rolling_mean,
        "Return the current rolling mean of the spread buffer."
    )
    .def(
        "get_rolling_std",
        &SpreadCalculator::get_rolling_std,
        "Return the current rolling std of the spread buffer."
    )
    .def(
        "get_buffer_size",
        &SpreadCalculator::get_buffer_size,
        "Return the number of elements currently in the buffer."
    )
    .def(
        "is_ready",
        &SpreadCalculator::is_ready,
        R"pbdoc(
        Return True when the rolling buffer is full.

        Z-scores before is_ready() are computed on a partial
        window and are statistically unreliable.
        The signal generator must check this before trading.
        )pbdoc"
    )

    // ── Python-friendly repr ──────────────────────────────────────
    // Defines what you see when you print() the object in Python
    .def(
        "__repr__",
        [](const SpreadCalculator& self) {
            return
                "SpreadCalculator("
                "n_assets=" +
                std::to_string(self.get_buffer_size()) +
                ", buffer_ready=" +
                (self.is_ready() ? "True" : "False") +
                ", current_zscore=" +
                std::to_string(self.get_zscore()) +
                ")";
        }
    );


    // ---------------------------------------------------------------- //
    //  Expose free functions                                            //
    // ---------------------------------------------------------------- //

    m.def(
        "log_price",
        &log_price,
        py::arg("price"),
        R"pbdoc(
        Compute log of a single price.

        Parameters
        ----------
        price : float
            Raw price. Must be positive.

        Returns
        -------
        float
            Natural log of the price.
        )pbdoc"
    );

    m.def(
        "log_prices",
        &log_prices,
        py::arg("raw_prices"),
        R"pbdoc(
        Compute log prices for a list of raw prices.

        Parameters
        ----------
        raw_prices : list of float
            Raw prices for all assets.

        Returns
        -------
        list of float
            Natural log of each price.
        )pbdoc"
    );

    m.def(
        "dot_product",
        &dot_product,
        py::arg("a"),
        py::arg("b"),
        R"pbdoc(
        Compute dot product of two equal-length vectors.

        Used to verify spread computation matches Python
        implementation during testing.
        )pbdoc"
    );


    // ---------------------------------------------------------------- //
    //  Version info — accessible from Python                           //
    // ---------------------------------------------------------------- //

    m.attr("__version__") = "0.1.0";
    m.attr("__author__")  = "crypto_stat_arb";
}
