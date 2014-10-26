#ifndef __PROBABILITY_DISTRIBUTIONS__DISTRIBUTION_HPP__
#define __PROBABILITY_DISTRIBUTIONS__DISTRIBUTION_HPP__

#include "array.hpp"
#include "const_array.hpp"

namespace ProbabilityDistributions {
  namespace MA = MultidimensionalArray;

  template <class T>
  class Distribution {
    public:
      virtual T log_likelihood(MA::ConstArray<T> const& data) const;

      virtual T log_likelihood(MA::ConstArray<T> const& data,
          MA::ConstArray<T> const& weight) const = 0;

      virtual void MLE(MA::ConstArray<T> const& data,
          std::vector<size_t> const& indexes = std::vector<size_t>());

      virtual void MLE(MA::ConstArray<T> const& data,
          MA::ConstArray<T> const& weight,
          std::vector<size_t> const& indexes = std::vector<size_t>()) = 0;

      virtual bool require_sorted() const { return false; }

      static std::vector<size_t> sort_data(MA::ConstArray<T> const& data);
  };
};

#include "distribution_impl.hpp"

#endif
