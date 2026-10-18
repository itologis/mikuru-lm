#pragma once

#include "../../../../src/dialogue_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace mikuru_gru {

namespace mk = mikuru;

using TokenId = mk::TokenId;

constexpr std::size_t EMBED_DIM = 32;
constexpr std::size_t HIDDEN_DIM = 64;
constexpr double OUTPUT_DROPOUT = 0.20;
constexpr double LEARNING_RATE = 0.001;
constexpr double GRAD_CLIP_NORM = 1.0;
constexpr std::size_t BATCH_SIZE = 16;
constexpr int MAX_EPOCHS = 100;
constexpr int EARLY_STOPPING_PATIENCE = 8;

struct Layout {
    std::size_t vocab;
    std::size_t embed, wz, wr, wn, uz, ur, un;
    std::size_t bz, br, bn, wo, bo, total;

    explicit Layout(std::size_t v) : vocab(v) {
        embed = 0;
        wz = embed + vocab * EMBED_DIM;
        wr = wz + HIDDEN_DIM * EMBED_DIM;
        wn = wr + HIDDEN_DIM * EMBED_DIM;
        uz = wn + HIDDEN_DIM * EMBED_DIM;
        ur = uz + HIDDEN_DIM * HIDDEN_DIM;
        un = ur + HIDDEN_DIM * HIDDEN_DIM;
        bz = un + HIDDEN_DIM * HIDDEN_DIM;
        br = bz + HIDDEN_DIM;
        bn = br + HIDDEN_DIM;
        wo = bn + HIDDEN_DIM;
        bo = wo + vocab * HIDDEN_DIM;
        total = bo + vocab;
    }
};

struct Model {
    std::size_t vocab_size;
    Layout layout;
    std::vector<double> params;

    Model(std::size_t vocab, std::mt19937& rng)
        : vocab_size(vocab), layout(vocab),
          params(layout.total, 0.0) {
        std::uniform_real_distribution<double> emb(-0.08, 0.08);
        const double input_limit =
            std::sqrt(6.0 / (EMBED_DIM + HIDDEN_DIM));
        const double recurrent_limit =
            std::sqrt(6.0 / (2.0 * HIDDEN_DIM));
        const double output_limit =
            std::sqrt(6.0 / (HIDDEN_DIM + vocab_size));

        std::uniform_real_distribution<double> input_dist(
            -input_limit, input_limit);
        std::uniform_real_distribution<double> recurrent_dist(
            -recurrent_limit, recurrent_limit);
        std::uniform_real_distribution<double> output_dist(
            -output_limit, output_limit);

        for (std::size_t i = layout.embed; i < layout.wz; ++i)
            params[i] = emb(rng);
        for (std::size_t i = layout.wz; i < layout.uz; ++i)
            params[i] = input_dist(rng);
        for (std::size_t i = layout.uz; i < layout.bz; ++i)
            params[i] = recurrent_dist(rng);
        for (std::size_t i = layout.wo; i < layout.bo; ++i)
            params[i] = output_dist(rng);

        // With h = z*h_prev + (1-z)*candidate, this initially
        // favors carrying some of the previous state forward.
        for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
            params[layout.bz + j] = 1.0;
    }
};

struct StepCache {
    TokenId token = 0;
    std::vector<double> prev, z, r, candidate, h;
    std::vector<double> output_scale, output_dh;

    StepCache() {
        prev.assign(HIDDEN_DIM, 0.0);
        z.assign(HIDDEN_DIM, 0.0);
        r.assign(HIDDEN_DIM, 0.0);
        candidate.assign(HIDDEN_DIM, 0.0);
        h.assign(HIDDEN_DIM, 0.0);
        output_scale.assign(HIDDEN_DIM, 1.0);
        output_dh.assign(HIDDEN_DIM, 0.0);
    }
};

struct EncodedDialogue {
    std::vector<TokenId> tokens;
    std::size_t response_start = 0;
};

struct Grad {
    std::vector<double> values;

    explicit Grad(std::size_t n) : values(n, 0.0) {}
};

struct Adam {
    std::vector<double> first, second;
    std::size_t step = 0;

    explicit Adam(std::size_t n)
        : first(n, 0.0), second(n, 0.0) {}
};

inline double sigmoid(double x) {
    if (x >= 0.0) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    const double e = std::exp(x);
    return e / (1.0 + e);
}

inline StepCache forward_step(
    const Model& model,
    TokenId token,
    const std::vector<double>& previous
) {
    const auto& p = model.params;
    const auto& l = model.layout;

    StepCache c;
    c.token = token;
    c.prev = previous;

    if (token >= model.vocab_size) token = 1;
    const std::size_t xbase = l.embed + token * EMBED_DIM;

    std::vector<double> x(EMBED_DIM);
    for (std::size_t k = 0; k < EMBED_DIM; ++k)
        x[k] = p[xbase + k];

    for (std::size_t j = 0; j < HIDDEN_DIM; ++j) {
        double az = p[l.bz + j];
        double ar = p[l.br + j];

        for (std::size_t k = 0; k < EMBED_DIM; ++k) {
            az += p[l.wz + j * EMBED_DIM + k] * x[k];
            ar += p[l.wr + j * EMBED_DIM + k] * x[k];
        }
        for (std::size_t k = 0; k < HIDDEN_DIM; ++k) {
            az += p[l.uz + j * HIDDEN_DIM + k] * previous[k];
            ar += p[l.ur + j * HIDDEN_DIM + k] * previous[k];
        }
        c.z[j] = sigmoid(az);
        c.r[j] = sigmoid(ar);
    }

    for (std::size_t j = 0; j < HIDDEN_DIM; ++j) {
        double an = p[l.bn + j];
        for (std::size_t k = 0; k < EMBED_DIM; ++k)
            an += p[l.wn + j * EMBED_DIM + k] * x[k];

        for (std::size_t k = 0; k < HIDDEN_DIM; ++k) {
            an += p[l.un + j * HIDDEN_DIM + k]
                * c.r[k] * previous[k];
        }
        c.candidate[j] = std::tanh(an);
        c.h[j] = c.z[j] * previous[j]
               + (1.0 - c.z[j]) * c.candidate[j];
    }
    return c;
}

inline std::vector<double> output_logits(
    const Model& model,
    const std::vector<double>& h,
    const std::vector<double>& scale
) {
    const auto& p = model.params;
    const auto& l = model.layout;
    std::vector<double> logits(model.vocab_size);

    for (std::size_t v = 0; v < model.vocab_size; ++v) {
        double sum = p[l.bo + v];
        for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
            sum += p[l.wo + v * HIDDEN_DIM + j]
                 * h[j] * scale[j];
        logits[v] = sum;
    }
    return logits;
}

inline std::vector<double> softmax(const std::vector<double>& logits) {
    const double maximum =
        *std::max_element(logits.begin(), logits.end());
    std::vector<double> probs(logits.size());
    double total = 0.0;

    for (std::size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - maximum);
        total += probs[i];
    }

    if (!(total > 0.0) || !std::isfinite(total)) {
        std::fill(probs.begin(), probs.end(),
                  1.0 / static_cast<double>(probs.size()));
    } else {
        for (double& x : probs) x /= total;
    }
    return probs;
}

inline EncodedDialogue encode_dialogue(
    const mk::Dialogue& dialogue,
    const std::unordered_map<std::string, TokenId>& ids
) {
    EncodedDialogue seq;
    seq.tokens.push_back(ids.at("<BOS>"));
    seq.tokens.push_back(ids.at("<USER>"));

    const auto user = mk::encode_tokens(dialogue.user, ids);
    seq.tokens.insert(seq.tokens.end(), user.begin(), user.end());

    seq.tokens.push_back(ids.at("<ASSISTANT>"));
    seq.response_start = seq.tokens.size();

    const auto answer = mk::encode_tokens(dialogue.assistant, ids);
    seq.tokens.insert(seq.tokens.end(), answer.begin(), answer.end());
    seq.tokens.push_back(ids.at("<EOS>"));
    return seq;
}

inline std::vector<EncodedDialogue> encode_dialogues(
    const std::vector<mk::Dialogue>& dialogues,
    const std::unordered_map<std::string, TokenId>& ids
) {
    std::vector<EncodedDialogue> result;
    result.reserve(dialogues.size());
    for (const auto& d : dialogues)
        result.push_back(encode_dialogue(d, ids));
    return result;
}

inline std::size_t target_count(
    const EncodedDialogue& seq,
    TokenId unk
) {
    std::size_t n = 0;
    for (std::size_t i = seq.response_start;
         i < seq.tokens.size(); ++i) {
        if (seq.tokens[i] != unk) ++n;
    }
    return n;
}

inline double sequence_loss_sum(
    const Model& model,
    const EncodedDialogue& seq
) {
    const TokenId unk = 1;
    std::vector<double> hidden(HIDDEN_DIM, 0.0);
    std::vector<double> ones(HIDDEN_DIM, 1.0);
    double sum = 0.0;

    for (std::size_t t = 0; t + 1 < seq.tokens.size(); ++t) {
        const auto c = forward_step(model, seq.tokens[t], hidden);
        hidden = c.h;

        const std::size_t target_index = t + 1;
        const TokenId target = seq.tokens[target_index];

        if (target_index < seq.response_start || target == unk)
            continue;

        const auto probs = softmax(output_logits(model, c.h, ones));
        sum -= std::log(std::max(probs[target], 1e-15));
    }
    return sum;
}

inline double evaluate(
    const Model& model,
    const std::vector<EncodedDialogue>& sequences
) {
    double total_loss = 0.0;
    std::size_t total_targets = 0;

    for (const auto& seq : sequences) {
        total_loss += sequence_loss_sum(model, seq);
        total_targets += target_count(seq, 1);
    }
    if (!total_targets) return 0.0;
    return total_loss / static_cast<double>(total_targets);
}

// Accumulates summed gradients for one dialogue. Loss targets are only
// assistant response tokens and EOS; user tokens receive no direct loss.
// Gradients still flow backward through user-token recurrent states.
inline double accumulate_sequence_gradient(
    const Model& model,
    const EncodedDialogue& seq,
    std::mt19937& rng,
    Grad& grad,
    std::size_t& supervised_targets,
    double dropout
) {
    const auto& p = model.params;
    const auto& l = model.layout;
    const TokenId unk = 1;

    std::vector<StepCache> steps;
    steps.reserve(seq.tokens.size() - 1);
    std::vector<double> hidden(HIDDEN_DIM, 0.0);
    std::bernoulli_distribution keep(1.0 - dropout);
    const double kept_scale = dropout > 0.0 ? 1.0 / (1.0 - dropout) : 1.0;
    double loss_sum = 0.0;

    // Forward pass through every input token in sequence order.
    for (std::size_t t = 0; t + 1 < seq.tokens.size(); ++t) {
        StepCache c = forward_step(model, seq.tokens[t], hidden);
        hidden = c.h;

        const std::size_t target_index = t + 1;
        const TokenId target = seq.tokens[target_index];
        const bool supervised =
            target_index >= seq.response_start && target != unk;

        if (supervised) {
            if (dropout > 0.0) {
                for (double& s : c.output_scale)
                    s = keep(rng) ? kept_scale : 0.0;
            }

            const auto probs = softmax(
                output_logits(model, c.h, c.output_scale));
            loss_sum -= std::log(std::max(probs[target], 1e-15));
            ++supervised_targets;

            std::vector<double> dlogits = probs;
            dlogits[target] -= 1.0;

            for (std::size_t v = 0; v < model.vocab_size; ++v) {
                grad.values[l.bo + v] += dlogits[v];
                for (std::size_t j = 0; j < HIDDEN_DIM; ++j) {
                    const std::size_t wi = l.wo + v * HIDDEN_DIM + j;
                    grad.values[wi] +=
                        dlogits[v] * c.h[j] * c.output_scale[j];
                    c.output_dh[j] +=
                        p[wi] * dlogits[v] * c.output_scale[j];
                }
            }
        }
        steps.push_back(std::move(c));
    }

    // Backpropagation through time, from the final token to the first.
    std::vector<double> dh_next(HIDDEN_DIM, 0.0);

    for (std::size_t rev = steps.size(); rev-- > 0;) {
        const auto& c = steps[rev];
        const TokenId input_token =
            c.token < model.vocab_size ? c.token : unk;
        const std::size_t xbase = l.embed + input_token * EMBED_DIM;

        std::vector<double> dh(HIDDEN_DIM), daz(HIDDEN_DIM),
            dar(HIDDEN_DIM), dan(HIDDEN_DIM), dq(HIDDEN_DIM),
            dh_prev(HIDDEN_DIM, 0.0);

        for (std::size_t j = 0; j < HIDDEN_DIM; ++j) {
            dh[j] = c.output_dh[j] + dh_next[j];
            const double dn = dh[j] * (1.0 - c.z[j]);
            const double dz = dh[j] * (c.prev[j] - c.candidate[j]);
            dan[j] = dn * (1.0 - c.candidate[j] * c.candidate[j]);
            daz[j] = dz * c.z[j] * (1.0 - c.z[j]);
        }

        // Candidate gate uses U_n * (reset_gate * previous_hidden).
        for (std::size_t k = 0; k < HIDDEN_DIM; ++k) {
            for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
                dq[k] += p[l.un + j * HIDDEN_DIM + k] * dan[j];
        }

        for (std::size_t j = 0; j < HIDDEN_DIM; ++j) {
            const double dr = dq[j] * c.prev[j];
            dar[j] = dr * c.r[j] * (1.0 - c.r[j]);
            grad.values[l.bz + j] += daz[j];
            grad.values[l.br + j] += dar[j];
            grad.values[l.bn + j] += dan[j];

            for (std::size_t k = 0; k < EMBED_DIM; ++k) {
                const double x = p[xbase + k];
                grad.values[l.wz + j * EMBED_DIM + k] += daz[j] * x;
                grad.values[l.wr + j * EMBED_DIM + k] += dar[j] * x;
                grad.values[l.wn + j * EMBED_DIM + k] += dan[j] * x;
            }

            for (std::size_t k = 0; k < HIDDEN_DIM; ++k) {
                grad.values[l.uz + j * HIDDEN_DIM + k]
                    += daz[j] * c.prev[k];
                grad.values[l.ur + j * HIDDEN_DIM + k]
                    += dar[j] * c.prev[k];
                grad.values[l.un + j * HIDDEN_DIM + k]
                    += dan[j] * c.r[k] * c.prev[k];
            }
        }

        for (std::size_t k = 0; k < HIDDEN_DIM; ++k) {
            dh_prev[k] = dh[k] * c.z[k] + dq[k] * c.r[k];

            for (std::size_t j = 0; j < HIDDEN_DIM; ++j) {
                dh_prev[k] +=
                    p[l.uz + j * HIDDEN_DIM + k] * daz[j]
                  + p[l.ur + j * HIDDEN_DIM + k] * dar[j];
            }
        }

        // Accumulate gradients for the token embedding.
        for (std::size_t k = 0; k < EMBED_DIM; ++k) {
            double dx = 0.0;
            for (std::size_t j = 0; j < HIDDEN_DIM; ++j) {
                dx += p[l.wz + j * EMBED_DIM + k] * daz[j]
                    + p[l.wr + j * EMBED_DIM + k] * dar[j]
                    + p[l.wn + j * EMBED_DIM + k] * dan[j];
            }
            grad.values[xbase + k] += dx;
        }
        dh_next = std::move(dh_prev);
    }
    return loss_sum;
}

inline void apply_adam(
    Model& model,
    Grad& grad,
    Adam& adam,
    std::size_t token_count,
    double learning_rate = LEARNING_RATE
) {
    if (!token_count) return;

    for (double& g : grad.values)
        g /= static_cast<double>(token_count);

    double norm_squared = 0.0;
    for (double g : grad.values) norm_squared += g * g;
    const double norm = std::sqrt(norm_squared);
    const double clip_scale =
        norm > GRAD_CLIP_NORM ? GRAD_CLIP_NORM / norm : 1.0;

    constexpr double BETA1 = 0.9;
    constexpr double BETA2 = 0.999;
    constexpr double EPS = 1e-8;
    ++adam.step;

    const double correction1 = 1.0 - std::pow(BETA1, adam.step);
    const double correction2 = 1.0 - std::pow(BETA2, adam.step);

    for (std::size_t i = 0; i < model.params.size(); ++i) {
        const double g = grad.values[i] * clip_scale;
        adam.first[i] = BETA1 * adam.first[i] + (1.0 - BETA1) * g;
        adam.second[i] = BETA2 * adam.second[i] + (1.0 - BETA2) * g * g;

        const double mhat = adam.first[i] / correction1;
        const double vhat = adam.second[i] / correction2;
        model.params[i] -= learning_rate * mhat
                         / (std::sqrt(vhat) + EPS);
    }
}

inline bool save_model(
    const std::string& path,
    const Model& model,
    std::uint64_t vocab_hash
) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        std::cerr << "Cannot save GRU model: " << path << '\n';
        return false;
    }

    out << "MIKURU_GRU_V1\n"
        << model.vocab_size << ' ' << EMBED_DIM << ' '
        << HIDDEN_DIM << ' ' << vocab_hash << ' '
        << model.params.size() << '\n'
        << std::setprecision(17);

    for (std::size_t i = 0; i < model.params.size(); ++i) {
        if (i) out << ' ';
        out << model.params[i];
    }
    out << '\n';
    return static_cast<bool>(out);
}

inline bool load_model(
    const std::string& path,
    Model& model,
    std::uint64_t expected_hash
) {
    std::ifstream in(path);
    std::string magic;
    std::size_t vocab = 0, embedding = 0, hidden = 0, count = 0;
    std::uint64_t hash = 0;

    if (!(in >> magic >> vocab >> embedding >> hidden >> hash >> count) ||
        magic != "MIKURU_GRU_V1" ||
        vocab != model.vocab_size ||
        embedding != EMBED_DIM ||
        hidden != HIDDEN_DIM ||
        hash != expected_hash ||
        count != model.params.size()) {
        std::cerr << "Invalid GRU model or vocabulary mismatch.\n";
        return false;
    }

    for (double& value : model.params) {
        if (!(in >> value) || !std::isfinite(value)) {
            std::cerr << "Corrupt or incomplete GRU parameter file.\n";
            return false;
        }
    }
    return true;
}

} // namespace mikuru_gru
