#include "../src/gru_model.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace mg = mikuru_gru;

static bool close_enough(double a, double b, double tolerance = 2e-4) {
    return std::abs(a - b) <= tolerance * std::max(1.0, std::abs(b));
}

int main() {
    std::mt19937 rng(1234);
    mg::Model model(8, rng);

    mg::EncodedDialogue sample;
    // <BOS>, <USER>, one prompt token, <ASSISTANT>, one answer token, <EOS>
    sample.tokens = {2, 4, 6, 5, 7, 3};
    sample.response_start = 4;

    const double base_loss = mg::sequence_loss_sum(model, sample);
    if (!std::isfinite(base_loss)) {
        std::cerr << "FAIL: loss is not finite\n";
        return 1;
    }

    mg::Grad grad(model.params.size());
    std::size_t targets = 0;
    mg::accumulate_sequence_gradient(
        model, sample, rng, grad, targets, 0.0);
    if (targets != 2) {
        std::cerr << "FAIL: expected answer and EOS targets\n";
        return 1;
    }

    // Finite-difference check of the output bias gradient.
    const std::size_t out_bias = model.layout.bo + 7;
    const double original = model.params[out_bias];
    constexpr double eps = 1e-5;

    model.params[out_bias] = original + eps;
    const double plus = mg::sequence_loss_sum(model, sample);
    model.params[out_bias] = original - eps;
    const double minus = mg::sequence_loss_sum(model, sample);
    model.params[out_bias] = original;

    const double numerical = (plus - minus) / (2.0 * eps);
    const double analytic = grad.values[out_bias];
    if (!close_enough(numerical, analytic)) {
        std::cerr << "FAIL: output gradient mismatch: numeric="
                  << numerical << " analytic=" << analytic << '\n';
        return 1;
    }
    std::cout << "PASS: output gradient finite-difference check\n";

    // The prompt embedding's gradient tests backpropagation through time.
    const std::size_t embedding_weight =
        model.layout.embed + 6 * mg::EMBED_DIM;
    const double original_embedding = model.params[embedding_weight];

    model.params[embedding_weight] = original_embedding + eps;
    const double emb_plus = mg::sequence_loss_sum(model, sample);
    model.params[embedding_weight] = original_embedding - eps;
    const double emb_minus = mg::sequence_loss_sum(model, sample);
    model.params[embedding_weight] = original_embedding;

    const double emb_numerical = (emb_plus - emb_minus) / (2.0 * eps);
    const double emb_analytic = grad.values[embedding_weight];
    if (!close_enough(emb_numerical, emb_analytic, 1e-3)) {
        std::cerr << "FAIL: recurrent embedding gradient mismatch: numeric="
                  << emb_numerical << " analytic=" << emb_analytic << '\n';
        return 1;
    }
    std::cout << "PASS: recurrent gradient finite-difference check\n";

    // Check the output distribution is finite and sums to one.
    std::vector<double> hidden(mg::HIDDEN_DIM, 0.0);
    const auto step = mg::forward_step(model, 2, hidden);
    std::vector<double> ones(mg::HIDDEN_DIM, 1.0);
    const auto probs = mg::softmax(mg::output_logits(model, step.h, ones));
    double sum = 0.0;
    for (double p : probs) {
        if (!std::isfinite(p) || p < 0.0) {
            std::cerr << "FAIL: invalid probability\n";
            return 1;
        }
        sum += p;
    }
    if (std::abs(sum - 1.0) > 1e-9) {
        std::cerr << "FAIL: probabilities do not sum to one\n";
        return 1;
    }
    std::cout << "PASS: output probabilities sum to one\n";
    std::cout << "All GRU unit tests passed.\n";
    return 0;
}
