#ifndef __PROBABILITY_DISTRIBUTIONS__DISCRETE_HPP__
#define __PROBABILITY_DISTRIBUTIONS__DISCRETE_HPP__

#include "distribution.hpp"

#include <boost/random/discrete_distribution.hpp>

namespace ProbabilityDistributions {
  template <class T>
  class Discrete: public Distribution<T> {
    public:
      Discrete(unsigned int K);
      Discrete(std::vector<T> const& p);

      void set_p(std::vector<T> const& p) { p_ = p; normalize(); }
      std::vector<T> const& get_p() const { return p_; }
      unsigned int get_number_of_classes() const { return p_.size(); }

      template <class RNG>
      void sample(MA::Array<T>& samples, size_t n_samples, RNG& rng) const;

      using Distribution<T>::log_likelihood;
      T log_likelihood(MA::ConstArray<T> const& data,
          MA::ConstArray<T> const& weight) const;

      using Distribution<T>::MLE;
      void MLE(MA::ConstArray<T> const& data, MA::ConstArray<T> const& weight);

      void sample_to_index(MA::Array<unsigned int>& indexes,
          MA::ConstArray<T> const& samples) const;

      void index_to_sample(MA::Array<T>& samples,
          MA::ConstArray<unsigned int> const& indexes) const;

    private:
      void check_data_and_weight(MA::ConstArray<T> const& data,
          MA::ConstArray<T> const& weight) const;
      void normalize();

      typedef boost::random::discrete_distribution<unsigned int, T> DD;

      std::vector<T> p_;
  };
};

#include "discrete_impl.hpp"

#endif
