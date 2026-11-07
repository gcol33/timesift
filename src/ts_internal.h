#ifndef TIMESIFT_TS_INTERNAL_H
#define TIMESIFT_TS_INTERNAL_H

// What the fits share and neither language sees: the guard on the numbers a fit is handed, and the
// pool that runs independent fits at once.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "ts_core.h"

namespace timesift {
namespace detail {

inline void check_finite(const double* v, std::size_t n, const char* who, const char* what) {
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(v[i])) {
      throw Error(std::string(who) + " is fitted on finite values, and the " + what +
                  " holds one that is not, at position " + std::to_string(i + 1) + ".");
    }
  }
}

// What a task threw, held as its kind and its message so that the thread which collects it throws
// a new exception of the same kind. An exception object is never carried from one thread to
// another: under libc++ 23 with libc++abi, an exception passed through `std::exception_ptr` and
// `std::rethrow_exception()` no longer matches a handler for `std::exception`, so R's wrapper,
// which catches by that base, reports it without its message.
class Failure {
 public:
  // Records the exception being handled; called only from inside a `catch (...)` block.
  void record() noexcept {
    try {
      throw;
    } catch (const Error& e) {
      set(Kind::error, e.what());
    } catch (const std::invalid_argument& e) {
      set(Kind::invalid_argument, e.what());
    } catch (const std::bad_alloc&) {
      set(Kind::bad_alloc, "");
    } catch (const std::exception& e) {
      set(Kind::runtime_error, e.what());
    } catch (...) {
      set(Kind::error, "a fit stopped on an exception that carries no message.");
    }
  }

  explicit operator bool() const noexcept { return kind_ != Kind::none; }

  [[noreturn]] void raise() const {
    switch (kind_) {
      case Kind::invalid_argument:
        throw std::invalid_argument(message_);
      case Kind::bad_alloc:
        throw std::bad_alloc();
      case Kind::runtime_error:
        throw std::runtime_error(message_);
      case Kind::none:
      case Kind::error:
        break;
    }
    throw Error(message_);
  }

 private:
  enum class Kind { none, error, invalid_argument, bad_alloc, runtime_error };

  void set(Kind kind, const char* message) noexcept {
    kind_ = kind;
    try {
      message_ = message;
    } catch (...) {
      kind_ = Kind::bad_alloc;
    }
  }

  Kind kind_ = Kind::none;
  std::string message_;
};

// Runs `task(0)` to `task(count - 1)`, on up to `workers` threads where more than one is asked
// for. Each task is independent and writes its own slot, so the result does not depend on how
// many ran at once; the first failure in task order is raised once all have finished.
template <typename Task>
void run_tasks(std::size_t count, int workers, Task task) {
  std::vector<Failure> failed(count);
  auto guarded = [&](std::size_t i) {
    try {
      task(i);
    } catch (...) {
      failed[i].record();
    }
  };
  if (workers <= 1 || count <= 1) {
    for (std::size_t i = 0; i < count; ++i) guarded(i);
  } else {
    std::atomic<std::size_t> next{0};
    auto take = [&]() {
      for (;;) {
        const std::size_t i = next.fetch_add(1);
        if (i >= count) return;
        guarded(i);
      }
    };
    std::vector<std::thread> pool;
    const std::size_t spare = std::min<std::size_t>(static_cast<std::size_t>(workers) - 1, count);
    pool.reserve(spare);
    // A machine that will not give another thread is a reason to run on fewer, not to fail.
    for (std::size_t t = 0; t < spare; ++t) {
      try {
        pool.emplace_back(take);
      } catch (const std::system_error&) {
        break;
      }
    }
    take();
    for (std::thread& t : pool) t.join();
  }
  for (const Failure& f : failed) {
    if (f) f.raise();
  }
}

// Response `s`'s column of a fold map [n, r] of 0-based indices, and its count of folds; null and 0
// where no map was given.
inline const std::int32_t* response_fold(const std::int32_t* fold, std::size_t n, std::size_t s) {
  return fold == nullptr ? nullptr : fold + s * n;
}

inline std::int32_t response_fold_count(const std::int32_t* fold, const std::int32_t* n_fold,
                                        std::size_t s) {
  return fold == nullptr ? 0 : n_fold[s];
}

// One fit per response, `fit(s, inner)` for `s` from 0 to `count - 1`. With several responses the
// workers go across them and each fit is handed one; with one response, that fit is handed every
// worker for its own work. A fit is the one it gets alone either way, so what comes back does not
// depend on how the workers were spent.
template <typename Fit>
auto fit_responses(std::size_t count, int workers, Fit fit)
    -> std::vector<decltype(fit(std::size_t{0}, 1))> {
  std::vector<decltype(fit(std::size_t{0}, 1))> out(count);
  const int inner = count == 1 ? workers : 1;
  run_tasks(count, workers, [&](std::size_t s) { out[s] = fit(s, inner); });
  return out;
}

}  // namespace detail
}  // namespace timesift

#endif  // TIMESIFT_TS_INTERNAL_H
