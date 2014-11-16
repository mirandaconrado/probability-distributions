#ifndef __PROBABILITY_DISTRIBUTIONS__HMM_IMPL_HPP__
#define __PROBABILITY_DISTRIBUTIONS__HMM_IMPL_HPP__

#include "hmm.hpp"

#include "mixture.hpp"

namespace ProbabilityDistributions {
  template <unsigned int SS, class D, class W, class T, class... Dists>
  HMM<SS,D,W,T,Dists...>::HMM(Dists&&... dists):
    have_asymmetric_(CompileUtils::or_<
        std::is_base_of<
          AsymmetricDistribution<typename CompileUtils::clean_type<Dists>::type,D,W,T>,
          typename CompileUtils::clean_type<Dists>::type>::value...>::value),
    stop_condition_(1e-4),
    max_iterations_(1000),
    components_(tuple_type(dists...)) {
      create_component_pointers(
          typename CompileUtils::tuple_sequence_generator<tuple_type>::type());
    }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  template <size_t... S>
  void HMM<SS,D,W,T,Dists...>::create_component_pointers(
      CompileUtils::sequence<S...>) {
    components_pointers_ =
      std::vector<Distribution<D,W,T>*>({&std::get<S>(components_)...});
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  template <class RNG>
  void HMM<SS,D,W,T,Dists...>::sample(MA::Array<D>& samples,
      size_t n_samples, RNG& rng) const {
    MA::Size::SizeType size(2);
    size[0] = n_samples;
    size[1] = SS;
    samples.resize(size);

    MA::Array<D> components_samples;
    MA::Array<unsigned int> components_indexes;
    std::vector<unsigned int> indexes(n_samples);

    initial_weights_.sample(components_samples, 1, rng);
    initial_weights_.sample_to_index(components_indexes, components_samples);
    indexes[0] = components_indexes(0);

    for (size_t i = 1; i < n_samples; i++) {
      unsigned int k = indexes[i-1];
      transition_weights_[k].sample(components_samples, 1, rng);
      transition_weights_[k].sample_to_index(components_indexes,
          components_samples);
      indexes[i] = components_indexes(0);
    }

    MA::Slice<D> slice(samples, 0);
    for (size_t i = 0; i < n_samples; i++) {
      MA::Array<D> sample = slice.get_element(i);
      sample_components(sample, indexes[i], rng,
          typename CompileUtils::tuple_sequence_generator<tuple_type>::type());
    }
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  template <class RNG, size_t... S>
  void HMM<SS,D,W,T,Dists...>::sample_components(MA::Array<D>& sample,
      size_t component_id, RNG& rng, CompileUtils::sequence<S...>) const {
    bool vec[] = {sample_component<S>(sample, component_id, rng)...};
    (void)vec;
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  template <size_t I, class RNG>
  bool HMM<SS,D,W,T,Dists...>::sample_component(MA::Array<D>& sample,
      size_t component_id, RNG& rng) const {
    if (I != component_id)
      return false;
    std::get<I>(components_).sample(sample, 1, rng);
    return true;
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  T HMM<SS,D,W,T,Dists...>::log_likelihood(MA::ConstArray<D> const& data,
      MA::ConstArray<W> const& weight) const {
    check_data_and_weight(data, weight);

    MA::Array<LogNumber> prob_emissions;
    build_prob_emissons(prob_emissions, data);

    MA::Array<LogNumber> alpha, beta;
    build_alpha_beta(alpha, beta, prob_emissions, weight, data);

    LogNumber ll = 0;
    size_t n_samples = data.size()[0];
    for (unsigned int i = 0; i < K; i++)
      ll += alpha(i, n_samples-1);

    return ll.get_val();
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  void HMM<SS,D,W,T,Dists...>::build_expectation(
      MA::Array<W>& expected_weight, MA::ConstArray<W> const& weight,
      MA::ConstArray<T> const& gamma) const {
    expected_weight.resize({K, weight.size()[0]});

    for (unsigned int t = 0; t < weight.size()[0]; t++)
      for (unsigned int i = 0; i < K; i++)
        expected_weight(i,t) = gamma(i,t) * weight(t);
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  void HMM<SS,D,W,T,Dists...>::build_prob_emissons(
      MA::Array<LogNumber>& prob_emissions,
      MA::ConstArray<D> const& data) const {
    unsigned int n_samples = data.size()[0];
    prob_emissions.resize({K, n_samples});

    MA::Array<W> null_weight({1});
    null_weight(0) = 1;
    MA::Size sample_size({1, SS});

    MA::ConstSlice<D> data_slice(data, 0);

    for (unsigned int t = 0; t < data_slice.total_left_size(); t++) {
      MA::ConstArray<D> sample = data_slice.get_element(t);
      sample.resize(sample_size);
      for (unsigned int i = 0; i < K; i++)
        prob_emissions(i,t).from_log(
          components_pointers_[i]->log_likelihood(sample, null_weight));
    }
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  void HMM<SS,D,W,T,Dists...>::build_alpha_beta(
      MA::Array<LogNumber>& alpha, MA::Array<LogNumber>& beta,
      MA::ConstArray<LogNumber> const& prob_emissions,
      MA::ConstArray<W> const& weight,
      MA::ConstArray<D> const& data) const {
    unsigned int n_samples = data.size()[0];

    alpha.resize({K, n_samples});
    beta.resize({K, n_samples});

    for (unsigned int i = 0; i < K; i++) {
      alpha(i,0) = initial_weights_.get_p()[i] *
        ((1-weight(0)) + weight(0) * prob_emissions(i,0));
      beta(i,n_samples-1) = 1;
    }

    for (unsigned int t = 1; t < n_samples; t++) {
      for (unsigned int i = 0; i < K; i++) {
        alpha(i,t) = 0;
        for (unsigned int j = 0; j < K; j++)
          alpha(i,t) += alpha(j,t-1) * transition_weights_[j].get_p()[i];
        alpha(i,t) *= get_adjusted_weight(i, t, weight, prob_emissions);
      }
    }

    for (unsigned int t = n_samples-1; t > 0; t--) {
      for (unsigned int i = 0; i < K; i++) {
        beta(i,t-1) = 0;
        for (unsigned int j = 0; j < K; j++)
          beta(i,t-1) += beta(j,t) * transition_weights_[i].get_p()[j] *
            get_adjusted_weight(j, t, weight, prob_emissions);
      }
    }
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  void HMM<SS,D,W,T,Dists...>::build_gamma(
      MA::Array<W>& gamma,
      MA::ConstArray<LogNumber> const& alpha,
      MA::ConstArray<LogNumber> const& beta) const {
    gamma.resize(alpha.size());
    for (unsigned int t = 0; t < alpha.size()[1]; t++) {
      LogNumber sum = 0;
      std::vector<LogNumber> vals(alpha.size()[0]);
      for (unsigned int i = 0; i < alpha.size()[0]; i++) {
        vals[i] = alpha(i,t) * beta(i,t);
        sum += vals[i];
      }
      for (unsigned int i = 0; i < alpha.size()[0]; i++)
        gamma(i,t) = (vals[i] / sum).to_double();
    }
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  void HMM<SS,D,W,T,Dists...>::build_xi(
      MA::Array<W>& xi,
      MA::ConstArray<LogNumber> const& alpha,
      MA::ConstArray<LogNumber> const& beta,
      MA::ConstArray<LogNumber> const& prob_emissions,
      MA::ConstArray<W> const& weight) const {
    xi.resize({K, K, alpha.size()[1]-1});
    for (unsigned int t = 0; t < alpha.size()[1]-1; t++) {
      LogNumber sum = 0;
      for (unsigned int i = 0; i < K; i++)
        sum += alpha(i,t) * beta(i,t);

      for (unsigned int i = 0; i < K; i++)
        for (unsigned int j = 0; j < K; j++)
          xi(i,j,t) = (alpha(i,t) * beta(j,t+1) *
            transition_weights_[i].get_p()[j] *
            get_adjusted_weight(j, t+1, weight, prob_emissions) / sum).to_double();
    }
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  LogNumber HMM<SS,D,W,T,Dists...>::get_adjusted_weight(
      unsigned int component, unsigned int sample,
      MA::ConstArray<W> const& weight,
      MA::ConstArray<LogNumber> const& prob_emissions) const {
    return (1-weight(sample)) +
      weight(sample) * prob_emissions(component, sample);
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  template <size_t... S>
  void HMM<SS,D,W,T,Dists...>::MLE_as_mixture(MA::ConstArray<D> const& data,
      MA::ConstArray<W> const& weight, std::vector<size_t> const& indexes,
      CompileUtils::sequence<S...>) {
    auto mix = make_mixture(std::get<S>(components_)...);
    mix.set_max_iterations(max_iterations_);
    mix.set_stop_condition(stop_condition_);

    auto t1 = std::make_tuple(
        mix.template get_component<S>() = std::get<S>(components_)...);
    (void)t1;
    mix.MLE(data, weight, indexes);
    auto t2 = std::make_tuple(
        std::get<S>(components_) = mix.template get_component<S>()...);
    (void)t2;

    // In tests have shown that copying the weights can get to lower local
    // minima.
    //initial_weights_ = mix.get_mixture_weights();
    //for (unsigned int i = 0; i < K; i++)
    //  transition_weights_[i] = initial_weights_;
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  void HMM<SS,D,W,T,Dists...>::MLE(MA::ConstArray<D> const& data,
      MA::ConstArray<W> const& weight, std::vector<size_t> const& indexes) {
    check_data_and_weight(data, weight);

    size_t n_samples = data.size()[0];

    std::vector<size_t> const* index_pointer = &indexes;

    for (unsigned int i = 0; i < K; i++)
      if (components_pointers_[i]->require_sorted() && indexes.size() == 0) {
        index_pointer = new
          std::vector<size_t>(Distribution<D,W,T>::sort_data(data));
        break;
      }

    MLE_as_mixture(data, weight, *index_pointer,
      typename CompileUtils::tuple_sequence_generator<tuple_type>::type());

    MA::Array<W> expected_weight;

    bool fixed_asymmetries = false;
    if (have_asymmetric_) {
      fixed_asymmetries = true;
      fix_all_asymmetries(
          typename CompileUtils::tuple_sequence_generator<tuple_type>::type());
    }

    while (1) {
      T ll_old = -INFINITY, ll_new = -INFINITY;
      size_t it = 0;

      do {
        MA::Array<LogNumber> prob_emissions;
        build_prob_emissons(prob_emissions, data);

        MA::Array<LogNumber> alpha, beta;
        build_alpha_beta(alpha, beta, prob_emissions, weight, data);

        MA::Array<W> gamma;
        build_gamma(gamma, alpha, beta);

        MA::Array<W> xi;
        build_xi(xi, alpha, beta, prob_emissions, weight);

        build_expectation(expected_weight, weight, gamma);

        ll_old = ll_new;

        MA::Slice<W> weight_slice(expected_weight, 0);

        std::vector<T> initial_weights(K);

        for (unsigned int i = 0; i < K; i++) {
          components_pointers_[i]->MLE(data, weight_slice.get_element(i),
              *index_pointer);

          initial_weights[i] = gamma(i, 0);

          T den = 0;
          std::vector<T> num(K, 0);
          for (unsigned int t = 0; t < n_samples-1; t++) {
            den += gamma(i, t);
            for (unsigned int j = 0; j < K; j++)
              num[j] += xi(i, j, t);
          }

          std::vector<T> transition_weights(K);
          for (unsigned int j = 0; j < K; j++)
            transition_weights[j] = num[j]/den;
          transition_weights_[i].set_p(transition_weights);
        }

        initial_weights_.set_p(initial_weights);

        ll_new = log_likelihood(data, weight);
        //assert(ll_new >= ll_old);
        //assert(ll_new >= ll_old - 1e-8);

        it++;
      } while(std::abs(ll_new - ll_old) > stop_condition_ &&
              it < max_iterations_);

      if (!fixed_asymmetries)
        break;

      fixed_asymmetries = false;
      free_all_asymmetries(
          typename CompileUtils::tuple_sequence_generator<tuple_type>::type());
    }

    if (index_pointer != &indexes)
      delete index_pointer;
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  void HMM<SS,D,W,T,Dists...>::check_data_and_weight(
      MA::ConstArray<D> const& data, MA::ConstArray<W> const& weight) const {
    assert(data.size().size() == 2);
    assert(data.size()[0] > 0);
    assert(data.size()[1] == SS);
    assert(weight.size().size() == 1);
    assert(weight.size()[0] == data.size()[0]);
  }
};

#endif
