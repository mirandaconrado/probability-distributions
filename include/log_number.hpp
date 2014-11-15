#ifndef __PROBABILITY_DISTRIBUTIONS__LOG_NUMBER_HPP__
#define __PROBABILITY_DISTRIBUTIONS__LOG_NUMBER_HPP__

#include <cassert>
#include <cmath>

namespace ProbabilityDistributions {
  class LogNumber {
    public:
      LogNumber(): is_zero(true), is_negative(false), val(0) { }
      LogNumber(LogNumber const& other): is_zero(other.is_zero),
      is_negative(other.is_negative), val(other.val) { }
      LogNumber(double num): is_zero(num == 0), is_negative(num < 0) {
        if (is_zero)
          val = 0;
        else {
          if (is_negative)
            num = -num;
          val = std::log(num);
        }
      }

      LogNumber& operator=(LogNumber const& other) {
        is_zero = other.is_zero;
        is_negative = other.is_negative;
        val = other.val;
        return *this;
      }
      LogNumber& operator=(double num) {
        is_zero = (num == 0);
        is_negative = (num < 0);
        if (is_zero)
          val = 0;
        else {
          if (is_negative)
            num = -num;
          val = std::log(num);
        }
        return *this;
      }

      LogNumber operator-() const {
        LogNumber ret(*this);
        ret.is_negative = !ret.is_negative;
        return ret;
      }

      LogNumber& operator+=(LogNumber const& other) {
        if (is_zero) {
          *this = other;
          return *this;
        }
        if (other.is_zero)
          return *this;

        if (is_negative == other.is_negative)
          make_sum(val, other.val);
        else {
          if (val == other.val) {
            is_zero = true;
            is_negative = false;
            val = 0;
          }
          else if (val > other.val)
            make_sub(val, other.val);
          else {
            is_negative = !is_negative;
            make_sub(other.val, val);
          }
        }

        return *this;
      }

      LogNumber& operator-=(LogNumber const& other) {
        return *this += -other;
      }

      LogNumber& operator*=(LogNumber const& other) {
        is_zero = is_zero || other.is_zero;
        if (is_zero) {
          is_negative = false;
          val = 0;
        }
        else {
          is_negative = (is_negative ^ other.is_negative);
          val += other.val;
        }

        return *this;
      }

      LogNumber& operator/=(LogNumber const& other) {
        assert(!other.is_zero);
        if (!is_zero) {
          is_negative = (is_negative ^ other.is_negative);
          val -= other.val;
        }

        return *this;
      }

      friend ProbabilityDistributions::LogNumber
        operator+(ProbabilityDistributions::LogNumber const& v1, double v2) {
          ProbabilityDistributions::LogNumber ret(v1);
          ret += v2;
          return ret;
        }
      friend ProbabilityDistributions::LogNumber
        operator+(double v1, ProbabilityDistributions::LogNumber const& v2) {
          ProbabilityDistributions::LogNumber ret(v1);
          ret += v2;
          return ret;
        }
      friend ProbabilityDistributions::LogNumber
        operator+(ProbabilityDistributions::LogNumber const& v1,
            ProbabilityDistributions::LogNumber const& v2) {
          ProbabilityDistributions::LogNumber ret(v1);
          ret += v2;
          return ret;
        }

      friend ProbabilityDistributions::LogNumber
        operator-(ProbabilityDistributions::LogNumber const& v1, double v2) {
          ProbabilityDistributions::LogNumber ret(v1);
          ret -= v2;
          return ret;
        }
      friend ProbabilityDistributions::LogNumber
        operator-(double v1, ProbabilityDistributions::LogNumber const& v2) {
          ProbabilityDistributions::LogNumber ret(v1);
          ret -= v2;
          return ret;
        }
      friend ProbabilityDistributions::LogNumber
        operator-(ProbabilityDistributions::LogNumber const& v1,
            ProbabilityDistributions::LogNumber const& v2) {
          ProbabilityDistributions::LogNumber ret(v1);
          ret -= v2;
          return ret;
        }

      friend ProbabilityDistributions::LogNumber
        operator*(ProbabilityDistributions::LogNumber const& v1, double v2) {
          ProbabilityDistributions::LogNumber ret(v1);
          ret *= v2;
          return ret;
        }
      friend ProbabilityDistributions::LogNumber
        operator*(double v1, ProbabilityDistributions::LogNumber const& v2) {
          ProbabilityDistributions::LogNumber ret(v1);
          ret *= v2;
          return ret;
        }
      friend ProbabilityDistributions::LogNumber
        operator*(ProbabilityDistributions::LogNumber const& v1,
            ProbabilityDistributions::LogNumber const& v2) {
          ProbabilityDistributions::LogNumber ret(v1);
          ret *= v2;
          return ret;
        }

      friend ProbabilityDistributions::LogNumber
        operator/(ProbabilityDistributions::LogNumber const& v1, double v2) {
          ProbabilityDistributions::LogNumber ret(v1);
          ret /= v2;
          return ret;
        }
      friend ProbabilityDistributions::LogNumber
        operator/(double v1, ProbabilityDistributions::LogNumber const& v2) {
          ProbabilityDistributions::LogNumber ret(v1);
          ret /= v2;
          return ret;
        }
      friend ProbabilityDistributions::LogNumber
        operator/(ProbabilityDistributions::LogNumber const& v1,
            ProbabilityDistributions::LogNumber const& v2) {
          ProbabilityDistributions::LogNumber ret(v1);
          ret /= v2;
          return ret;
        }

      void from_log(double v) {
        is_zero = false;
        is_negative = false;
        val = v;
      }

      double to_double() const {
        if (is_zero)
          return 0;
        if (is_negative)
          return -std::exp(val);
        return std::exp(val);
      }

      double get_val() const { return val; }

    private:
      void make_sum(double v1, double v2) {
        if (v2 > v1) {
          double temp = v1;
          v1 = v2;
          v2 = temp;
        }

        val = v1 + std::log(1 + std::exp(v2-v1));
      }
      void make_sub(double v1, double v2) {
        assert(v1 >= v2);
        val = v1 + std::log(1 - std::exp(v2-v1));
      }

      bool is_zero, is_negative;
      double val;
  };
};

/*ProbabilityDistributions::LogNumber
operator+(ProbabilityDistributions::LogNumber const& v1, double v2) {
  ProbabilityDistributions::LogNumber ret(v1);
  ret += v2;
  return ret;
}
ProbabilityDistributions::LogNumber
operator+(double v1, ProbabilityDistributions::LogNumber const& v2) {
  ProbabilityDistributions::LogNumber ret(v1);
  ret += v2;
  return ret;
}
ProbabilityDistributions::LogNumber
operator+(ProbabilityDistributions::LogNumber const& v1,
    ProbabilityDistributions::LogNumber const& v2) {
  ProbabilityDistributions::LogNumber ret(v1);
  ret += v2;
  return ret;
}

ProbabilityDistributions::LogNumber
operator-(ProbabilityDistributions::LogNumber const& v1, double v2) {
  ProbabilityDistributions::LogNumber ret(v1);
  ret -= v2;
  return ret;
}
ProbabilityDistributions::LogNumber
operator-(double v1, ProbabilityDistributions::LogNumber const& v2) {
  ProbabilityDistributions::LogNumber ret(v1);
  ret -= v2;
  return ret;
}
ProbabilityDistributions::LogNumber
operator-(ProbabilityDistributions::LogNumber const& v1,
    ProbabilityDistributions::LogNumber const& v2) {
  ProbabilityDistributions::LogNumber ret(v1);
  ret -= v2;
  return ret;
}

ProbabilityDistributions::LogNumber
operator*(ProbabilityDistributions::LogNumber const& v1, double v2) {
  ProbabilityDistributions::LogNumber ret(v1);
  ret *= v2;
  return ret;
}
ProbabilityDistributions::LogNumber
operator*(double v1, ProbabilityDistributions::LogNumber const& v2) {
  ProbabilityDistributions::LogNumber ret(v1);
  ret *= v2;
  return ret;
}
ProbabilityDistributions::LogNumber
operator*(ProbabilityDistributions::LogNumber const& v1,
    ProbabilityDistributions::LogNumber const& v2) {
  ProbabilityDistributions::LogNumber ret(v1);
  ret *= v2;
  return ret;
}

ProbabilityDistributions::LogNumber
operator/(ProbabilityDistributions::LogNumber const& v1, double v2) {
  ProbabilityDistributions::LogNumber ret(v1);
  ret /= v2;
  return ret;
}
ProbabilityDistributions::LogNumber
operator/(double v1, ProbabilityDistributions::LogNumber const& v2) {
  ProbabilityDistributions::LogNumber ret(v1);
  ret /= v2;
  return ret;
}
ProbabilityDistributions::LogNumber
operator/(ProbabilityDistributions::LogNumber const& v1,
    ProbabilityDistributions::LogNumber const& v2) {
  ProbabilityDistributions::LogNumber ret(v1);
  ret /= v2;
  return ret;
}*/

#endif
