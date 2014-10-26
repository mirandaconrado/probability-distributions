#ifndef __PROBABILITY_DISTRIBUTIONS__DISTRIBUTION_HPP__
#define __PROBABILITY_DISTRIBUTIONS__DISTRIBUTION_HPP__

#include "array.hpp"
#include "const_array.hpp"

namespace ProbabilityDistributions {
  namespace MA = MultidimensionalArray;

  template <class D, class W = D, class T = W>
  class Distribution {
    public:
      typedef D data_type;
      typedef W weight_type;
      typedef T float_type;

      virtual T log_likelihood(MA::ConstArray<D> const& data) const;

      virtual T log_likelihood(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight) const = 0;

      virtual void MLE(MA::ConstArray<D> const& data,
          std::vector<size_t> const& indexes = std::vector<size_t>());

      virtual void MLE(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight,
          std::vector<size_t> const& indexes = std::vector<size_t>()) = 0;

      virtual bool require_sorted() const { return false; }

      static std::vector<size_t> sort_data(MA::ConstArray<D> const& data);
  };
};

#include "distribution_impl.hpp"

#endif
