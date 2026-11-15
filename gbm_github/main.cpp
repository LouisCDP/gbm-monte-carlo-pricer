// GBM Monte Carlo option pricer
// - Exact GBM sampling (no time-stepping error for European payoffs)
// - Streaming simulation: paths are never stored, memory use is O(threads)
// - Antithetic variates, multithreaded, one independent RNG stream per thread
// - Greeks: pathwise delta/vega, likelihood-ratio gamma
// - Benchmarked against closed-form Black-Scholes
// - Path-dependent example: arithmetic Asian call (no closed form)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

constexpr double kPi = 3.14159265358979323846;

struct Option {
    double S0, K, r, sigma, T;
    bool call;
};

struct Estimate {
    double price, std_err, delta, gamma, vega;
};

// ---------------------------------------------------------------- Black-Scholes
namespace bs {

inline double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }
inline double norm_pdf(double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * kPi); }

Estimate price(const Option& o) {
    const double sqT = std::sqrt(o.T);
    const double d1 = (std::log(o.S0 / o.K) + (o.r + 0.5 * o.sigma * o.sigma) * o.T) / (o.sigma * sqT);
    const double d2 = d1 - o.sigma * sqT;
    const double disc = std::exp(-o.r * o.T);
    Estimate e{};
    if (o.call) {
        e.price = o.S0 * norm_cdf(d1) - o.K * disc * norm_cdf(d2);
        e.delta = norm_cdf(d1);
    } else {
        e.price = o.K * disc * norm_cdf(-d2) - o.S0 * norm_cdf(-d1);
        e.delta = norm_cdf(d1) - 1.0;
    }
    e.gamma = norm_pdf(d1) / (o.S0 * o.sigma * sqT);
    e.vega = o.S0 * norm_pdf(d1) * sqT;
    e.std_err = 0.0;
    return e;
}

}  // namespace bs

// ---------------------------------------------------------------- Monte Carlo
// alignas(64): each thread writes to its own cache line -> no false sharing.
struct alignas(64) Accum {
    double sum = 0, sum2 = 0, delta = 0, gamma = 0, vega = 0;
};

// splitmix64: decorrelates per-thread seeds derived from a single master seed.
static uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Runs fn(rng, pairs_for_this_thread, accum) on `threads` threads.
template <typename Fn>
std::vector<Accum> run_parallel(unsigned threads, uint64_t pairs, uint64_t seed, Fn fn) {
    std::vector<Accum> acc(threads);
    std::vector<std::thread> pool;
    uint64_t state = seed;
    const uint64_t base = pairs / threads, rem = pairs % threads;
    for (unsigned t = 0; t < threads; ++t) {
        const uint64_t n = base + (t < rem ? 1 : 0);
        const uint64_t s = splitmix64(state);
        pool.emplace_back([=, &acc, &fn] {
            std::mt19937_64 rng(s);
            fn(rng, n, acc[t]);
        });
    }
    for (auto& th : pool) th.join();
    return acc;
}

// European option: one normal draw per antithetic pair, exact GBM terminal value.
Estimate mc_european(const Option& o, uint64_t n_paths, unsigned threads, uint64_t seed) {
    const uint64_t pairs = n_paths / 2;
    const double sqT = std::sqrt(o.T);
    const double drift = (o.r - 0.5 * o.sigma * o.sigma) * o.T;
    const double volT = o.sigma * sqT;
    const double disc = std::exp(-o.r * o.T);
    const double sgn = o.call ? 1.0 : -1.0;
    const double inv_s2T = 1.0 / (o.sigma * o.sigma * o.T);
    const double inv_sT = 1.0 / (o.sigma * sqT);

    auto worker = [&](std::mt19937_64& rng, uint64_t n, Accum& a) {
        std::normal_distribution<double> N01;
        for (uint64_t i = 0; i < n; ++i) {
            const double z = N01(rng);
            double pair_payoff = 0.0;
            for (int s = 1; s >= -1; s -= 2) {  // z and -z (antithetic)
                const double zz = s * z;
                const double ST = o.S0 * std::exp(drift + volT * zz);
                const double payoff = std::max(sgn * (ST - o.K), 0.0);
                pair_payoff += payoff;
                if (payoff > 0.0) {
                    a.delta += sgn * ST / o.S0;                         // pathwise delta
                    a.vega += sgn * ST * (sqT * zz - o.sigma * o.T);    // pathwise vega
                }
                // likelihood-ratio gamma (payoff has no smooth second derivative)
                a.gamma += payoff * ((zz * zz - 1.0) * inv_s2T - zz * inv_sT) / (o.S0 * o.S0);
            }
            const double pv = 0.5 * pair_payoff;
            a.sum += pv;
            a.sum2 += pv * pv;
        }
    };

    const auto acc = run_parallel(threads, pairs, seed, worker);
    Accum tot;
    for (const auto& a : acc) {
        tot.sum += a.sum; tot.sum2 += a.sum2;
        tot.delta += a.delta; tot.gamma += a.gamma; tot.vega += a.vega;
    }
    const double N = static_cast<double>(pairs);
    const double mean = tot.sum / N;
    const double var = std::max(tot.sum2 / N - mean * mean, 0.0);
    Estimate e;
    e.price = disc * mean;
    e.std_err = disc * std::sqrt(var / N);
    e.delta = disc * tot.delta / (2.0 * N);
    e.gamma = disc * tot.gamma / (2.0 * N);
    e.vega = disc * tot.vega / (2.0 * N);
    return e;
}

// Arithmetic-average Asian call: genuinely path-dependent, so we step through time.
// A single per-thread buffer of normals is reused for every path (no per-path allocation).
Estimate mc_asian_call(const Option& o, int steps, uint64_t n_paths, unsigned threads, uint64_t seed) {
    const uint64_t pairs = n_paths / 2;
    const double dt = o.T / steps;
    const double drift = (o.r - 0.5 * o.sigma * o.sigma) * dt;
    const double vol = o.sigma * std::sqrt(dt);
    const double disc = std::exp(-o.r * o.T);

    auto worker = [&](std::mt19937_64& rng, uint64_t n, Accum& a) {
        std::normal_distribution<double> N01;
        std::vector<double> z(steps);
        for (uint64_t i = 0; i < n; ++i) {
            for (int k = 0; k < steps; ++k) z[k] = N01(rng);
            double pv = 0.0;
            for (int s = 1; s >= -1; s -= 2) {
                double S = o.S0, avg = 0.0;
                for (int k = 0; k < steps; ++k) {
                    S *= std::exp(drift + vol * s * z[k]);
                    avg += S;
                }
                pv += std::max(avg / steps - o.K, 0.0);
            }
            pv *= 0.5;
            a.sum += pv;
            a.sum2 += pv * pv;
        }
    };

    const auto acc = run_parallel(threads, pairs, seed, worker);
    double sum = 0, sum2 = 0;
    for (const auto& a : acc) { sum += a.sum; sum2 += a.sum2; }
    const double N = static_cast<double>(pairs);
    const double mean = sum / N;
    const double var = std::max(sum2 / N - mean * mean, 0.0);
    Estimate e{};
    e.price = disc * mean;
    e.std_err = disc * std::sqrt(var / N);
    return e;
}

// ---------------------------------------------------------------- Reporting
static void print_row(const char* label, const Estimate& mc, const Estimate& ref) {
    std::printf("%-6s MC %9.5f (+/-%.5f)  BS %9.5f | delta %7.4f/%7.4f | gamma %7.5f/%7.5f | vega %8.4f/%8.4f\n",
                label, mc.price, 1.96 * mc.std_err, ref.price,
                mc.delta, ref.delta, mc.gamma, ref.gamma, mc.vega, ref.vega);
}

int main(int argc, char** argv) {
    const uint64_t n_paths = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1'000'000ULL;
    unsigned threads = argc > 2 ? static_cast<unsigned>(std::atoi(argv[2]))
                                : std::max(1u, std::thread::hardware_concurrency());
    const uint64_t seed = 42;

    std::printf("Paths: %llu | threads: %u (95%% CI shown as +/-; Greeks printed MC/BS)\n\n",
                static_cast<unsigned long long>(n_paths), threads);

    // Scenario 1: base case, call and put
    const Option base{100.0, 100.0, 0.03, 0.20, 1.0, true};
    for (bool call : {true, false}) {
        Option o = base; o.call = call;
        print_row(call ? "Call" : "Put", mc_european(o, n_paths, threads, seed), bs::price(o));
    }

    // Scenario 2: volatility sensitivity
    std::printf("\nVolatility sensitivity (ATM call)\n");
    for (double sigma : {0.10, 0.20, 0.30, 0.40, 0.60}) {
        Option o = base; o.sigma = sigma;
        char label[16]; std::snprintf(label, sizeof label, "s=%.2f", sigma);
        print_row(label, mc_european(o, n_paths, threads, seed), bs::price(o));
    }

    // Scenario 3: moneyness
    std::printf("\nMoneyness (call, sigma=0.20)\n");
    for (double K : {80.0, 90.0, 100.0, 110.0, 120.0}) {
        Option o = base; o.K = K;
        char label[16]; std::snprintf(label, sizeof label, "K=%.0f", K);
        print_row(label, mc_european(o, n_paths, threads, seed), bs::price(o));
    }

    // Path-dependent example
    const Estimate asian = mc_asian_call(base, 252, n_paths, threads, seed);
    std::printf("\nArithmetic Asian call (252 steps): %.5f (+/-%.5f)\n", asian.price, 1.96 * asian.std_err);

    // Timing
    const auto t0 = std::chrono::steady_clock::now();
    mc_european(base, n_paths, threads, seed);
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("European pricing + Greeks, %llu paths: %.1f ms\n",
                static_cast<unsigned long long>(n_paths),
                std::chrono::duration<double, std::milli>(t1 - t0).count());
    return 0;
}
