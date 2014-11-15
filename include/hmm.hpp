#ifndef __PROBABILITY_DISTRIBUTIONS__HMM_HPP__
#define __PROBABILITY_DISTRIBUTIONS__HMM_HPP__

#include "asymmetric_distribution.hpp"
#include "discrete.hpp"
#include "distribution.hpp"
#include "log_number.hpp"

#include "binary_combination.hpp"
#include "clean_tuple.hpp"
#include "clean_type.hpp"
#include "repeated_tuple.hpp"
#include "sequence.hpp"

#include <tuple>
#include <type_traits>

namespace ProbabilityDistributions {
  template <unsigned int SS, class D, class W = D, class T = W, class... Dists>
  class HMM: public Distribution<D,W,T> {
    private:
      typedef typename CompileUtils::clean_tuple<Dists...>::type tuple_type;

    public:
      static constexpr unsigned int sample_size = SS;
      static constexpr unsigned int K = sizeof...(Dists);

      explicit HMM(Dists&&... dists);

      void set_stop_condition(T condition) { stop_condition_ = condition; }
      T get_stop_condition() const { return stop_condition_; }

      void set_max_iterations(size_t it) { max_iterations_ = it; }
      size_t get_max_iterations() const { return max_iterations_; }

      Discrete<K,D,W,T>& get_initial_weights() { return initial_weights_; }
      Discrete<K,D,W,T> const& get_initial_weights() const {
        return initial_weights_; }

      Discrete<K,D,W,T>& get_transition_weights(unsigned int k) {
        assert(k < K); return transition_weights_[k]; }
      Discrete<K,D,W,T> const& get_transition_weights(unsigned int k) const {
        assert(k < K); return transition_weights_[k]; }

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

      using Distribution<D,W,T>::MLE;
      void MLE(MA::ConstArray<D> const& data, MA::ConstArray<W> const& weight,
          std::vector<size_t> const& indexes = std::vector<size_t>());

    private:
      template <size_t... S>
      void MLE_as_mixture(MA::ConstArray<D> const& data, MA::ConstArray<W>
          const& weight, std::vector<size_t> const& indexes,
          CompileUtils::sequence<S...>);

      template <size_t... S>
      void fix_all_asymmetries(CompileUtils::sequence<S...>) {
        asymmetry_status_ = {fix_asymmetry(std::get<S>(components_))...};
      }
      template <class Dist>
      typename std::enable_if<
        !std::is_base_of<AsymmetricDistribution<Dist,D,W,T>,
                         Dist>::value, bool>::type
      fix_asymmetry(Dist& d) { return true; }
      template <class Dist>
      bool fix_asymmetry(AsymmetricDistribution<Dist,D,W,T>& d) {
        bool flag = d.is_p_fixed();
        d.fix_p(true);
        return flag;
      }

      template <size_t... S>
      void free_all_asymmetries(CompileUtils::sequence<S...>) {
        bool ret[] = {free_asymmetry(std::get<S>(components_),
            asymmetry_status_[S])...};
        (void)ret;
      }
      template <class Dist>
      typename std::enable_if<
        !std::is_base_of<AsymmetricDistribution<Dist,D,W,T>,
                         Dist>::value, bool>::type
      free_asymmetry(Dist& d, bool flag) { return true; }
      template <class Dist>
      bool free_asymmetry(AsymmetricDistribution<Dist,D,W,T>& d, bool flag) {
        d.fix_p(flag);
        return true;
      }

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

      void build_expectation(MA::Array<W>& expected_weight,
          MA::ConstArray<W> const& weight, MA::ConstArray<T> const& gamma) const;

      void build_prob_emissons(MA::Array<LogNumber>& prob_emissions,
          MA::ConstArray<D> const& data) const;
      void build_alpha_beta(MA::Array<LogNumber>& alpha,
          MA::Array<LogNumber>& beta,
          MA::ConstArray<LogNumber> const& prob_emissions,
          MA::ConstArray<W> const& weight, MA::ConstArray<D> const& data) const;
      void build_gamma(MA::Array<W>& gamma,
          MA::ConstArray<LogNumber> const& alpha,
          MA::ConstArray<LogNumber> const& beta) const;
      void build_xi(MA::Array<W>& xi, MA::ConstArray<LogNumber> const& alpha,
          MA::ConstArray<LogNumber> const& beta,
          MA::ConstArray<LogNumber> const& prob_emissions,
          MA::ConstArray<W> const& weight) const;
      LogNumber get_adjusted_weight(unsigned int component, unsigned int sample,
          MA::ConstArray<W> const& weight,
          MA::ConstArray<LogNumber> const& prob_emissions) const;

      bool have_asymmetric_;
      std::vector<bool> asymmetry_status_;
      T stop_condition_;
      size_t max_iterations_;
      Discrete<K,D,W,T> initial_weights_, transition_weights_[K];
      tuple_type components_;
      std::vector<Distribution<D,W,T>*> components_pointers_;
  };

  template <class D1, class... Dists>
  HMM<CompileUtils::clean_type<D1>::type::sample_size,
          typename CompileUtils::clean_type<D1>::type::data_type,
          typename CompileUtils::clean_type<D1>::type::weight_type,
          typename CompileUtils::clean_type<D1>::type::float_type, D1, Dists...>
  make_hmm(D1&& dist1, Dists&&... dists) {
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

    typedef Distribution<
      typename CompileUtils::clean_type<D1>::type::data_type,
      typename CompileUtils::clean_type<D1>::type::weight_type,
      typename CompileUtils::clean_type<D1>::type::float_type> base_distribution;

    static_assert(
        CompileUtils::and_<
          std::is_base_of<base_distribution,
                          typename CompileUtils::clean_type<D1>::type>::value,
          std::is_base_of<base_distribution,
                          typename CompileUtils::clean_type<Dists>::type>::value...
          >::value,
          "Distributions must be derived from abstract distribution class");

    return HMM<CompileUtils::clean_type<D1>::type::sample_size,
           typename CompileUtils::clean_type<D1>::type::data_type,
           typename CompileUtils::clean_type<D1>::type::weight_type,
           typename CompileUtils::clean_type<D1>::type::float_type, D1,
           Dists...>(std::forward<D1>(dist1), std::forward<Dists>(dists)...);
  }
};

#include "hmm_impl.hpp"

#endif
