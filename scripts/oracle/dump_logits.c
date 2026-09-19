// dump_logits.c — oracle fixture tool for the Minifield LFM2.5-230M Rust executor.
//
// Loads a GGUF model with llama.cpp, tokenizes a prompt file the same way
// llama-completion does (add_special=true, parse_special=true), runs a single
// llama_decode with logits requested on every input position, and writes:
//
//   u32 n_tokens
//   u32 n_vocab
//   f32 logits[n_tokens][n_vocab]   (row i = logits after position i)
//
// With an optional 4th arg n_gen, it then greedy-decodes up to n_gen tokens
// (argmax == --temp 0 --top-k 1), stopping at EOG. Emitted ids, detokenized
// text, and whether any id >= 64402 appeared go to a sidecar file
// "<out.bin>.gen.txt". This mirrors what llama-completion prints as text,
// but preserves the exact token ids the Rust tokenizer must decode.
//
// Prompt token ids are printed to stderr for the manifest (including whether
// BOS id 1 was prepended by the tokenizer).
//
// Usage: dump_logits <model.gguf> <prompt.txt> <out.bin> [n_gen]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "llama.h"
#include "ggml.h"

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt.txt> <out.bin> [n_gen]\n", argv[0]);
        return 2;
    }
    const int32_t n_gen = argc == 5 ? atoi(argv[4]) : 0;

    FILE * pf = fopen(argv[2], "rb");
    if (!pf) { perror("prompt file"); return 1; }
    fseek(pf, 0, SEEK_END);
    long plen = ftell(pf);
    fseek(pf, 0, SEEK_SET);
    char * prompt = (char *) malloc((size_t) plen + 1);
    if (fread(prompt, 1, (size_t) plen, pf) != (size_t) plen) { perror("read prompt"); return 1; }
    prompt[plen] = '\0';
    fclose(pf);

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only

    struct llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "model load failed: %s\n", argv[1]); return 1; }

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    fprintf(stderr, "vocab_add_bos=%d vocab_add_eos=%d bos=%d eos=%d\n",
            (int) llama_vocab_get_add_bos(vocab), (int) llama_vocab_get_add_eos(vocab),
            (int) llama_vocab_bos(vocab), (int) llama_vocab_eos(vocab));

    // tokenize exactly like llama-completion (completion.cpp: common_tokenize(ctx, prompt, true, true))
    int32_t cap = (int32_t) plen + 64;
    llama_token * tokens = (llama_token *) malloc(sizeof(llama_token) * cap);
    int32_t n = llama_tokenize(vocab, prompt, (int32_t) plen, tokens, cap, true, true);
    if (n < 0) {
        cap = -n;
        tokens = (llama_token *) realloc(tokens, sizeof(llama_token) * cap);
        n = llama_tokenize(vocab, prompt, (int32_t) plen, tokens, cap, true, true);
    }
    if (n <= 0) { fprintf(stderr, "tokenize failed\n"); return 1; }

    fprintf(stderr, "n_prompt_tokens=%d n_vocab=%d\n", n, n_vocab);
    fprintf(stderr, "prompt_token_ids=");
    for (int32_t i = 0; i < n; i++) fprintf(stderr, "%s%d", i ? "," : "", tokens[i]);
    fprintf(stderr, "\n");
    fprintf(stderr, "bos_prepended=%s\n", tokens[0] == llama_vocab_bos(vocab) ? "true" : "false");

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = (uint32_t) n + (uint32_t) n_gen + 16;
    cparams.n_batch   = (uint32_t) (n > 512 ? n : 512);
    cparams.n_threads = 1;
    cparams.type_k    = GGML_TYPE_F32; // f32 KV, mandatory for the oracle
    cparams.type_v    = GGML_TYPE_F32;

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "context init failed\n"); return 1; }

    const int32_t batch_cap = n > n_gen ? n : n_gen;
    struct llama_batch batch = llama_batch_init(batch_cap, 0, 1);
    batch.n_tokens = n;
    for (int32_t i = 0; i < n; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = true; // logits on every position
    }
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    FILE * of = fopen(argv[3], "wb");
    if (!of) { perror("output file"); return 1; }
    uint32_t nt = (uint32_t) n, nv = (uint32_t) n_vocab;
    fwrite(&nt, sizeof(uint32_t), 1, of);
    fwrite(&nv, sizeof(uint32_t), 1, of);
    for (int32_t i = 0; i < n; i++) {
        const float * row = llama_get_logits_ith(ctx, i);
        fwrite(row, sizeof(float), nv, of);
    }
    fclose(of);
    fprintf(stderr, "wrote %s : u32 n_tokens=%d, u32 n_vocab=%d, f32[%d][%d]\n", argv[3], nt, nv, n, n_vocab);

    if (n_gen > 0) {
        // greedy decode: argmax of each step's logits == llama-cli --temp 0 --top-k 1
        char side[4096];
        snprintf(side, sizeof(side), "%s.gen.txt", argv[3]);
        FILE * gf = fopen(side, "wb");
        if (!gf) { perror("gen sidecar"); return 1; }

        llama_token * gen = (llama_token *) malloc(sizeof(llama_token) * n_gen);
        int32_t n_emitted = 0;
        int32_t max_id = 0;
        bool stopped_on_eog = false;
        int32_t pos = n;
        const float * row = llama_get_logits_ith(ctx, n - 1);

        for (int32_t k = 0; k < n_gen; k++) {
            llama_token id = 0;
            for (int32_t i = 1; i < n_vocab; i++) if (row[i] > row[id]) id = i;
            gen[n_emitted++] = id;
            if (id > max_id) max_id = id;
            if (llama_vocab_is_eog(vocab, id)) { stopped_on_eog = true; break; }

            batch.n_tokens    = 1;
            batch.token[0]    = id;
            batch.pos[0]      = pos++;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0]   = true;
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "gen decode failed\n"); return 1; }
            row = llama_get_logits_ith(ctx, 0);
        }

        fprintf(gf, "n_generated=%d\n", n_emitted);
        fprintf(gf, "stopped_on_eog=%s\n", stopped_on_eog ? "true" : "false");
        fprintf(gf, "emitted_ids=");
        for (int32_t i = 0; i < n_emitted; i++) fprintf(gf, "%s%d", i ? "," : "", gen[i]);
        fprintf(gf, "\n");
        fprintf(gf, "max_emitted_id=%d\n", max_id);
        fprintf(gf, "any_emitted_id_ge_64402=%s\n", max_id >= 64402 ? "true" : "false");
        fprintf(gf, "emitted_text=");
        for (int32_t i = 0; i < n_emitted; i++) {
            char piece[512];
            int np = llama_token_to_piece(vocab, gen[i], piece, sizeof(piece), 0, false);
            if (np > 0) fwrite(piece, 1, (size_t) np, gf);
        }
        fprintf(gf, "\n");
        fclose(gf);
        fprintf(stderr, "wrote %s : n_generated=%d max_id=%d\n", side, n_emitted, max_id);
    }
    return 0;
}
