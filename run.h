#ifndef RUN_H
#define RUN_H
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cctype>
#include <ctime>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <cstddef>
#include <memory>
#include <algorithm>
#include <chrono>
#include <string>

#define USE_CUDA  // ¿ªÆôCUDA¼ÓËÙ
#include "device_code.h"


class Config {
public:
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
};

template<typename T>
class TransformerWeights {
public:
    // token embedding table
    std::unique_ptr<float[]> token_embedding_table;    // (vocab_size, dim)
    // final rmsnorm
    std::unique_ptr<float[]> rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    // weights for rmsnorms
    std::unique_ptr<float[]> rms_att_weight; // (layer, dim) rmsnorm weights
    std::unique_ptr<float[]> rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    std::unique_ptr<T[]> wq; // (layer, dim, n_heads * head_size)
    std::unique_ptr<T[]> wk; // (layer, dim, n_kv_heads * head_size)
    std::unique_ptr<T[]> wv; // (layer, dim, n_kv_heads * head_size)
    std::unique_ptr<T[]> wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    std::unique_ptr<T[]> w1; // (layer, hidden_dim, dim)
    std::unique_ptr<T[]> w2; // (layer, dim, hidden_dim)
    std::unique_ptr<T[]> w3; // (layer, hidden_dim, dim)
    std::unique_ptr<T[]> wcls;
    // tensor2d freq_cis_real;  // [seq_len, (dim/n_heads)/2]
    // tensor2d freq_cis_imag;  // [seq_len, (dim/n_heads)/2]
    std::unique_ptr<T[]> q_tokens; // (vocab_size, dim)
    
};

template<typename T>
class RunState {
public:
    // current wave of activations
    std::unique_ptr<float[]> x; // activation at current time stamp (dim,)
    std::unique_ptr<float[]> xb; // same, but inside a residual branch (dim,)
    std::unique_ptr<float[]> xb2; // an additional buffer just for convenience (dim,)
    std::unique_ptr<float[]> hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    std::unique_ptr<float[]> hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    std::unique_ptr<float[]> q; // query (dim,)
    std::unique_ptr<float[]> k; // key (dim,)
    std::unique_ptr<float[]> v; // value (dim,)
    std::unique_ptr<float[]> att; // buffer for scores/attention values (n_heads, seq_len)
    std::unique_ptr<float[]> logits; // output logits
    // kv cache
    std::unique_ptr<float[]> key_cache;   // (layer, seq_len, dim)
    std::unique_ptr<float[]> value_cache; // (layer, seq_len, dim)
};

template<>
class RunState<float> {
public:
    // current wave of activations
    std::unique_ptr<float[]> x; // activation at current time stamp (dim,)
    std::unique_ptr<float[]> xb; // same, but inside a residual branch (dim,)
    std::unique_ptr<float[]> xb2; // an additional buffer just for convenience (dim,)
    std::unique_ptr<float[]> hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    std::unique_ptr<float[]> hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    std::unique_ptr<float[]> q; // query (dim,)
    std::unique_ptr<float[]> k; // key (dim,)
    std::unique_ptr<float[]> v; // value (dim,)
    std::unique_ptr<float[]> att; // buffer for scores/attention values (n_heads, seq_len)
    std::unique_ptr<float[]> logits; // output logits
    // kv cache
    std::unique_ptr<float[]> key_cache;   // (layer, seq_len, dim)
    std::unique_ptr<float[]> value_cache; // (layer, seq_len, dim)
};

typedef struct {
    std::string str;
    int id;
} TokenIndex;

bool compare_tokens(const TokenIndex& a, const TokenIndex& b) {
    return a.str < b.str;
}

int str_lookup(const std::string& str, const std::unique_ptr<TokenIndex[]>& sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { str }; // acts as the key to search for

    auto it = std::lower_bound(sorted_vocab.get(), sorted_vocab.get() + vocab_size, tok, compare_tokens);

    // If we didn't reach the end and the string matches
    if (it != (sorted_vocab.get() + vocab_size) && it->str == str) {
        return it->id;
    }

    return -1; // Not found
}

template<typename T>
class Transformer {
private:
    void malloc_weights();
    void malloc_run_state();
public:
    Config config;
    TransformerWeights<T> w;
    RunState<T> s;
    int shared_weights = 1;
    void load_model(const std::string& checkpoint_path); 
    float* forward(int token, int pos);
};

class Tokenizer {
public:
    std::vector<std::unique_ptr<char[]>> vocab;
    std::vector<float> vocab_scores;
    std::unique_ptr<TokenIndex[]> sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
    void build_tokenizer(const std::string& tokenizer_path, int size_for_vacab);
    void encode(const std::string &text, const int8_t &bos, const int8_t &eos, std::unique_ptr<int[]> &tokens, int &n_tokens);
    std::string decode(int prev_token, int token);

};

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling
typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

class Sampler {
private:
    int sample_argmax(float* probabilities, int n);
    int sample_mult(float* probabilities, int n, float coin);
    int sample_topp(float* probabilities, int n, float topp, std::unique_ptr<ProbIndex[]>& probindex, float coin);
    unsigned int random_u32(unsigned long long *state) {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        return (*state * 0x2545F4914F6CDD1Dull) >> 32;
    }
    float random_f32(unsigned long long *state) { // random float32 in [0,1)
        return (random_u32(state) >> 8) / 16777216.0f;
    }
    static bool compare_probindex(const ProbIndex& a, const ProbIndex& b) {
        return a.prob > b.prob;
    }
public:
    int vocab_size;
    std::unique_ptr<ProbIndex[]> probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
    void build_sampler(int vocab_size, float temperature, float topp, unsigned long long rng_seed);
    int sample(float* logits);

};

void safe_print(const std::string& piece) {
    if (piece.empty()) {
        return;
    }
    if (piece.size() == 1) {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    std::cout << piece;
}

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}


void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
#ifdef USE_CUDA
	cuda_matmul(xout, x, w, n, d);
#else


    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
#endif
}

void read_stdin(const std::string& guide, std::string& buffer, size_t max_len) {
    std::cout << guide;
    std::getline(std::cin, buffer);
    if(buffer.length() > max_len) {
        buffer.resize(max_len);
    }
}

#endif