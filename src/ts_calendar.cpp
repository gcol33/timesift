#include "ts_core.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace timesift {

namespace {

constexpr seconds kDay = 86400;

struct NameMap {
  const char* name;
  Grain grain;
};

const NameMap kGrains[] = {
  {"native", Grain::native}, {"halfday", Grain::halfday}, {"day", Grain::day},
  {"week", Grain::week}, {"month", Grain::month}, {"season", Grain::season},
  {"year", Grain::year}, {"custom", Grain::custom}
};

struct StatMap {
  const char* name;
  Stat stat;
};

const StatMap kStats[] = {
  {"mean", Stat::mean}, {"min", Stat::min}, {"max", Stat::max},
  {"cold_day", Stat::cold_day}, {"warm_day", Stat::warm_day},
  {"mean_daily_min", Stat::mean_daily_min}, {"mean_daily_max", Stat::mean_daily_max}
};

// How many whole months a day sits past the year_start anniversary, counted from year 0. Flooring
// this by three or by twelve is what puts a seasonal or a hydrological-year bin in phase with the
// anniversary rather than with January.
std::int64_t offset_months(std::int64_t day_number, YearStart ys) noexcept {
  std::int64_t y;
  unsigned m, d;
  civil_from_days(day_number, y, m, d);
  return y * 12 + (static_cast<std::int64_t>(m) - 1) - (ys.month - 1) -
         (static_cast<int>(d) < ys.day ? 1 : 0);
}

seconds anniversary(std::int64_t offset, YearStart ys) noexcept {
  const std::int64_t absolute = offset + (ys.month - 1);
  const std::int64_t y = floor_div(absolute, 12);
  const unsigned m = static_cast<unsigned>(floor_mod(absolute, 12)) + 1;
  return days_from_civil(y, m, static_cast<unsigned>(ys.day)) * kDay;
}

}  // namespace

std::int64_t days_from_civil(std::int64_t y, unsigned m, unsigned d) noexcept {
  y -= m <= 2;
  const std::int64_t era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = static_cast<unsigned>(y - era * 400);
  const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + static_cast<std::int64_t>(doe) - 719468;
}

void civil_from_days(std::int64_t z, std::int64_t& y, unsigned& m, unsigned& d) noexcept {
  z += 719468;
  const std::int64_t era = (z >= 0 ? z : z - 146096) / 146097;
  const unsigned doe = static_cast<unsigned>(z - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const std::int64_t yy = static_cast<std::int64_t>(yoe) + era * 400;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153;
  d = doy - (153 * mp + 2) / 5 + 1;
  m = mp + (mp < 10 ? 3 : -9);
  y = yy + (m <= 2);
}

std::string iso8601(seconds t) {
  const std::int64_t day_number = floor_div(t, kDay);
  const std::int64_t rest = t - day_number * kDay;
  std::int64_t y;
  unsigned m, d;
  civil_from_days(day_number, y, m, d);
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%04lld-%02u-%02uT%02lld:%02lld:%02lld",
                static_cast<long long>(y), m, d,
                static_cast<long long>(rest / 3600),
                static_cast<long long>((rest / 60) % 60),
                static_cast<long long>(rest % 60));
  return std::string(buffer);
}

Grain grain_from_name(const std::string& name) {
  for (const NameMap& entry : kGrains) {
    if (name == entry.name) return entry.grain;
  }
  throw Error("unknown grain: " + name + ".");
}

const char* grain_name(Grain w) {
  for (const NameMap& entry : kGrains) {
    if (w == entry.grain) return entry.name;
  }
  return "unknown";
}

Stat stat_from_name(const std::string& name) {
  for (const StatMap& entry : kStats) {
    if (name == entry.name) return entry.stat;
  }
  throw Error("unknown statistic: " + name + ".");
}

const char* stat_name(Stat s) {
  for (const StatMap& entry : kStats) {
    if (s == entry.stat) return entry.name;
  }
  return "unknown";
}

bool is_day_level(Stat s) {
  return s == Stat::cold_day || s == Stat::warm_day ||
         s == Stat::mean_daily_min || s == Stat::mean_daily_max;
}

// The granularity a grain's bin boundaries land on. Bin membership is a function of this slot
// alone, which is what lets the calendar be read once per slot of the record rather than once per
// reading: three years hourly is 26,304 slots against 23.5 million readings.
seconds grain_granularity(Grain w) {
  switch (w) {
    case Grain::native: return 1;
    case Grain::halfday: return 43200;
    default: return kDay;
  }
}

// The start of the bin holding a whole slot, given the slot index at the grain's granularity.
seconds slot_bin_start(std::int64_t slot, Grain w, YearStart ys) noexcept {
  switch (w) {
    case Grain::native:
      // The `native` grain is the record unreduced: a reading is its own bin.
      return slot;
    case Grain::halfday:
      return slot * 43200;
    case Grain::day:
      return slot * kDay;
    case Grain::week:
      // 1970-01-01 was a Thursday, so the slot index plus three is zero on a Monday.
      return (slot - floor_mod(slot + 3, 7)) * kDay;
    case Grain::month: {
      std::int64_t y;
      unsigned m, d;
      civil_from_days(slot, y, m, d);
      return days_from_civil(y, m, 1) * kDay;
    }
    case Grain::season:
      return anniversary(floor_div(offset_months(slot, ys), 3) * 3, ys);
    case Grain::year:
      return anniversary(floor_div(offset_months(slot, ys), 12) * 12, ys);
    default:
      return slot * kDay;
  }
}

void bin_starts(const seconds* when, std::size_t n, Grain w, YearStart ys, seconds* out) {
  if (w == Grain::custom) {
    throw Error("a supplied calendar declares its own bins; bin_starts() does not apply.");
  }
  const seconds g = grain_granularity(w);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = slot_bin_start(floor_div(when[i], g), w, ys);
  }
}

void bin_nexts(const seconds* bin_start, std::size_t n, Grain w, YearStart ys, seconds* out) {
  for (std::size_t i = 0; i < n; ++i) {
    const seconds t = bin_start[i];
    switch (w) {
      case Grain::native:
        throw Error("the native grain's bin is the reading itself; "
                    "bin_nexts() does not apply.");
      case Grain::halfday: out[i] = t + 43200; break;
      case Grain::day: out[i] = t + kDay; break;
      case Grain::week: out[i] = t + 7 * kDay; break;
      case Grain::month: {
        std::int64_t y;
        unsigned m, d;
        civil_from_days(floor_div(t, kDay), y, m, d);
        out[i] = days_from_civil(m == 12 ? y + 1 : y, m == 12 ? 1 : m + 1, 1) * kDay;
        break;
      }
      case Grain::season:
        out[i] = anniversary(offset_months(floor_div(t, kDay), ys) + 3, ys);
        break;
      case Grain::year:
        out[i] = anniversary(offset_months(floor_div(t, kDay), ys) + 12, ys);
        break;
      default:
        throw Error("a supplied calendar declares its own bins; bin_nexts() does not apply.");
    }
  }
}

// Where in the year a bin sits. The midpoint of the record the bin holds is read on the Gregorian
// calendar in UTC, and its position in the year that midpoint falls in becomes an angle: a bin is
// a span of the calendar, and the turn of the year is where the fraction itself jumps and the sine
// and the cosine do not.
//
// The instants are read in UTC rather than on the clock the bins were placed on, because the phase
// is a place on the orbit rather than a reading of a clock, and a zone moves it by its offset:
// under a day on a cycle of a year, the same shift for every bin of the record.
void year_fraction(const seconds* bin_start, const seconds* bin_end, std::size_t n, double* out) {
  for (std::size_t i = 0; i < n; ++i) {
    // A bin spanning an odd number of seconds has its midpoint on a half second, and the second it
    // began is the one it is read at. Every calendar grain spans whole hours; a supplied calendar
    // need not.
    const seconds mid = bin_start[i] + floor_div(bin_end[i] - bin_start[i], 2);
    std::int64_t y;
    unsigned m, d;
    civil_from_days(floor_div(mid, kDay), y, m, d);
    const seconds opens = days_from_civil(y, 1, 1) * kDay;
    const seconds length = days_from_civil(y + 1, 1, 1) * kDay - opens;
    out[i] = static_cast<double>(mid - opens) / static_cast<double>(length);
  }
}

void year_phase(const seconds* bin_start, const seconds* bin_end, std::size_t n, double* year_sin,
                double* year_cos) {
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  std::vector<double> frac(n);
  year_fraction(bin_start, bin_end, n, frac.data());
  for (std::size_t i = 0; i < n; ++i) {
    year_sin[i] = std::sin(kTwoPi * frac[i]);
    year_cos[i] = std::cos(kTwoPi * frac[i]);
  }
}

}  // namespace timesift
