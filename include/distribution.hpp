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

      virtual void MLE(MA::ConstArray<T> const& data);

      virtual void MLE(MA::ConstArray<T> const& data,
          MA::ConstArray<T> const& weight) = 0;
  };
};

#include "distribution_impl.hpp"

#endif
