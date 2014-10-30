#ifndef __PROBABILITY_DISTRIBUTIONS__MIXTURE_HPP__
#define __PROBABILITY_DISTRIBUTIONS__MIXTURE_HPP__

#include "discrete.hpp"
#include "distribution.hpp"

#include "binary_combination.hpp"
#include "clean_tuple.hpp"
#include "clean_type.hpp"
#include "repeated_tuple.hpp"
#include "sequence.hpp"

#include <tuple>
#include <type_traits>

namespace ProbabilityDistributions {

  template <unsigned int SS, class D, class W = D, class T = W, class... Dists>
  class Mixture: public Distribution<D,W,T> {
    private:
      typedef typename CompileUtils::clean_tuple<Dists...>::type tuple_type;

    public:
      static constexpr unsigned int sample_size = SS;
      static constexpr unsigned int K = sizeof...(Dists);

      Mixture(Dists&&... dists);

      void set_stop_condition(T condition) { stop_condition_ = condition; }
      T get_stop_condition() const { return stop_condition_; }

      void set_max_iterations(size_t it) { max_iterations_ = it; }
      size_t get_max_iterations() const { return max_iterations_; }

      Discrete<K,D,W,T>& get_mixture_weights() { return mixture_weights_; }
      Discrete<K,D,W,T> const& get_mixture_weights() const {
        return mixture_weights_; }

      template <size_t I>
      typename std::tuple_element<I,tuple_type>::type& get_component() {
        return std::get<I>(components_); }
      template <size_t I>
      typename std::tuple_element<I,tuple_type>::type const& get_component() const {
        return std::get<I>(components_); }

      Distribution<D,W,T>* get_component_pointer(size_t i) {
        return components_pointers_[i]; }
      Distribution<D,W,T> const* get_component_pointer(size_t i) const {
        return components_pointers_[i]; }

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


      T stop_condition_;
      size_t max_iterations_;
      Discrete<K,D,W,T> mixture_weights_;
      tuple_type components_;
      std::vector<Distribution<D,W,T>*> components_pointers_;
  };

  template <class D1, class... Dists>
  Mixture<CompileUtils::clean_type<D1>::type::sample_size,
          typename CompileUtils::clean_type<D1>::type::data_type,
          typename CompileUtils::clean_type<D1>::type::weight_type,
          typename CompileUtils::clean_type<D1>::type::float_type, D1, Dists...>
  make_mixture(D1&& dist1, Dists&&... dists) {
    const unsigned int n_dists = 1 + sizeof...(Dists);

    static_assert(std::is_same<
        std::tuple<typename CompileUtils::clean_type<D1>::type::data_type,
                   typename CompileUtils::clean_type<Dists>::type::data_type...>,
        typename CompileUtils::repeated_tuple<n_dists,
          typename CompileUtils::clean_type<D1>::type::data_type>::type
        >::value, "All distributions must have the same data type");
    static_assert(std::is_same<
        std::tuple<typename CompileUtils::clean_type<D1>::type::weight_type,
                   typename CompileUtils::clean_type<Dists>::type::weight_type...>,
        typename CompileUtils::repeated_tuple<n_dists,
          typename CompileUtils::clean_type<D1>::type::weight_type>::type
        >::value, "All distributions must have the same weight type");
    static_assert(std::is_same<
        std::tuple<typename CompileUtils::clean_type<D1>::type::float_type,
                   typename CompileUtils::clean_type<Dists>::type::float_type...>,
        typename CompileUtils::repeated_tuple<n_dists,
          typename CompileUtils::clean_type<D1>::type::float_type>::type
        >::value, "All distributions must have the same float type");
    static_assert(
        CompileUtils::and_<CompileUtils::clean_type<D1>::type::sample_size ==
        CompileUtils::clean_type<Dists>::type::sample_size...>::value,
        "All distributions must have the same sample size");

    return Mixture<CompileUtils::clean_type<D1>::type::sample_size,
           typename CompileUtils::clean_type<D1>::type::data_type,
           typename CompileUtils::clean_type<D1>::type::weight_type,
           typename CompileUtils::clean_type<D1>::type::float_type, D1,
           Dists...>(std::forward<D1>(dist1), std::forward<Dists>(dists)...);
  }
};

#include "mixture_impl.hpp"

#endif
