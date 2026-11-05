// llama.cpp referans logit dökümü: aynı token/batch dizisi için echo-core ile karşılaştırılır.
// g++ -O2 -std=c++17 -I$LLAMA/include -I$LLAMA/ggml/include tools/ref_logits.cpp -L$LLAMA/build-vk/bin -lllama -lggml -lggml-base -o ref_logits
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int> ints(const char * s) {
    std::vector<int> v; std::stringstream ss(s); std::string it;
    while (std::getline(ss, it, ',')) v.push_back(std::atoi(it.c_str()));
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 7) { fprintf(stderr, "usage: model tokens steps threads n_ctx out\n"); return 1; }
    auto tokens = ints(argv[2]);
    auto steps  = ints(argv[3]);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = std::atoi(argv[5]);
    cp.n_batch = 2048; cp.n_ubatch = 512;
    cp.n_threads = cp.n_threads_batch = std::atoi(argv[4]);
    llama_context * ctx = llama_init_from_model(model, cp);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    FILE * f = fopen(argv[6], "wb");
    size_t off = 0;
    for (int n : steps) {
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + off, n)) != 0) return 2;
        off += n;
        fwrite(llama_get_logits_ith(ctx, -1), sizeof(float), n_vocab, f);
    }
    fclose(f);
    return 0;
}
