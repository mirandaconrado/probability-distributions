#ifndef __PROBABILITY_DISTRIBUTIONS__MIXTURE_HPP__
#define __PROBABILITY_DISTRIBUTIONS__MIXTURE_HPP__

#include "distribution.hpp"

#include "binary_combination.hpp"
#include "clean_tuple.hpp"
#include "repeated_tuple.hpp"
#include "sequence.hpp"

#include <tuple>
#include <type_traits>

namespace ProbabilityDistributions {
  template <class D, class W = D, class T = W>
  class Discrete;

  template <unsigned int K, class D, class W = D, class T = W, class... Dists>
  class Mixture {
    public:
      static constexpr unsigned int sample_size = K;

      Mixture(Dists&&... dists);

      void set_stop_condition(T condition) { stop_condition_ = condition; }
      T get_stop_condition() const { return stop_condition_; }

      void set_max_iterations(size_t it) { max_iterations_ = it; }
      size_t get_max_iterations() const { return max_iterations_; }

      template <class RNG>
      void sample(MA::Array<D>& samples, size_t n_samples, RNG& rng) const;

      using Distribution<D,W,T>::log_likelihood;
      T log_likelihood(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight) const;
      T log_likelihood(MA::ConstArray<D> const& data,
          std::vector<unsigned int> const& labels) const;

      using Distribution<D,W,T>::MLE;
      void MLE(MA::ConstArray<D> const& data, MA::ConstArray<W> const& weight,
          std::vector<size_t> const& indexes = std::vector<size_t>());

    private:
      void check_data_and_weight(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight) const;

      template <size_t... S>
      void create_component_pointers(CompileUtils::sequence<S...>);

      template <class RNG, size_t... S>
      void sample_components(MA::Array<D>& sample, size_t component_id,
          RNG& rng, CompileUtils::sequence<S...>) const;

      template <size_t I, class RNG>
      bool sample_component(MA::Array<D>& sample, size_t component_id,
          RNG& rng) const;

      T internal_log_likelihood(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& expected_weight) const;

      void build_expectation(MA::Array<W>& expected_weight,
          MA::ConstArray<W> const& weight, MA::ConstArray<D> const& data) const;

      MA::Array<W> transpose(MA::Array<W> const& original) const;

      typedef typename CompileUtils::clean_tuple<Dists...>::type tuple_type;

      T stop_condition_;
      size_t max_iterations_;
      Discrete<D,W,T> mixture_weights_;
      tuple_type components_;
      std::vector<Distribution<D,W,T>*> components_pointers_;
  };

  template <class D1, class... Dists>
  Mixture<D1::sample_size, typename D1::data_type, typename D1::weight_type,
    typename D1::float_type, D1, Dists...>
  make_mixture(D1&& dist1, Dists&&... dists) {
    const unsigned int n_dists = 1 + sizeof...(Dists);

    static_assert(std::is_same<
        std::tuple<typename D1::data_type, typename Dists::data_type...>,
        CompileUtils::repeated_tuple<n_dists, typename D1::data_type>
        >::value, "All distributions must have the same data type");
    static_assert(std::is_same<
        std::tuple<typename D1::weight_type, typename Dists::weight_type...>,
        CompileUtils::repeated_tuple<n_dists, typename D1::weight_type>
        >::value, "All distributions must have the same weight type");
    static_assert(std::is_same<
        std::tuple<typename D1::float_type, typename Dists::float_type...>,
        CompileUtils::repeated_tuple<n_dists, typename D1::float_type>
        >::value, "All distributions must have the same float type");
    static_assert(CompileUtils::and_<D1::sample_size ==
        Dists::sample_size...>::value,
        "All distributions must have the same sample size");

    return Mixture<D1::sample_size, typename D1::data_type, typename
      D1::weight_type, typename D1::float_type, D1,
      Dists...>(std::forward<D1>(dist1),
          std::forward<Dists>(dists)...);
  }
};

#endif
