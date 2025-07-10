#include <llama.h>
#include <import>


static void llm_logger(enum ggml_log_level level, const char * text, void * user_data) {
    print("llama.cpp: %s", text);
}


void llm_init(llm m) {
    path llm_path = f(path, "%s/.cache/llama.cpp/%o", getenv("HOME"), m->uri);
    verify(exists(llm_path), "llm not found: %o", llm_path);
    llama_backend_init();

    struct llama_model_params llm_params = llama_model_default_params();
    llm_params.n_gpu_layers = 0;
    llm_params.use_mmap     = true;
    llm_params.use_mlock    = true;

    m->imdl = llama_model_load_from_file(llm_path->chars, llm_params);
    verify(m->imdl, "failed to load llm");
    m->ivocab = (struct llama_vocab *)llama_model_get_vocab(m->imdl);
}


none conversation_init(conversation a) {
    const i32 n_ctx = 1024 * 16;

    struct llama_context_params conversation_params = llama_context_default_params();
    conversation_params.n_ctx   = n_ctx;
    conversation_params.n_batch = n_ctx;

    a->ictx = llama_init_from_model(a->model->imdl, conversation_params);
    verify(a->ictx, "failed to initialize context");

    a->ismpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(a->ismpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(a->ismpl, llama_sampler_init_temp(0.2f));
    llama_sampler_chain_add(a->ismpl, llama_sampler_init_penalties(
        64,    // penalty_last_n — apply to last 64 tokens
        1.2f,  // penalty_repeat — penalize repetition (>1.0)
        0.0f,  // penalty_freq — optional (per-token frequency)
        0.0f   // penalty_present — optional (presence-based)
    ));
    llama_sampler_chain_add(a->ismpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    each(a->content, object, msg) {
        path p = instanceof(msg, path);
        string s = instanceof(msg, string);
        append(a, msg);
    }
}


static string convert_msg(object msg) {
    string     prompt    = null;
    path       file      = instanceof(msg, path);
    if (file) {
        string raw = (string)read(file, typeid(string), null);
        array lines = split(raw, "\n");
        prompt = string(alloc, len(raw) * 4 + 1024);
        append(prompt, "\n\nfilename (source code annotated with line-identifier: code): ");
        append(prompt, file->chars);
        append(prompt, "\n");
        // todo: read from project json
        //append(prompt, "info:\n");
        //append(prompt, "a component runtime for objects using C11 _Generics for constructors w single args, otherwise pairs to make for named constructors.\n");
        append(prompt, "content:\n");
        int line_number = 1;
        for (int i = 0, ln = len(lines); i < ln; i++) {
            string line_content = (string)lines->elements[i];
            string annot = f(string, "%i: %o", line_number, line_content);
            line_number++;
            concat(prompt, annot);
            append(prompt, "\n");
        }
        append(prompt, "end-of-source\n");
    } else
        prompt = instanceof(msg, string);
    return prompt;
}

// query against context (this will not append to it; user case)
string conversation_query(conversation a, object msg) {
    llm        m         = a->model;
    string     prompt    = convert_msg(msg);
    
    verify    (prompt, "type not handled: %s", isa(msg)->name);
    string     response  = string(alloc, 1024);
    const bool is_first  = llama_kv_self_seq_pos_max(a->ictx, 0) == 0;
    const int  n_tokens  = -llama_tokenize(m->ivocab, cstring(prompt), len(prompt), NULL, 0, is_first, true);
    vector     tokens    = vector(alloc, n_tokens, type, typeid(i32));
    i32*       t_data    = (i32*)data(tokens);

    verify(llama_tokenize(m->ivocab, cstring(prompt), len(prompt),
        t_data, n_tokens, is_first, true) >= 0, "tokenization failed");

    llama_token    new_token_id;
    llama_context* ctx = a->ictx;

    struct llama_context_params tmp_params = llama_context_default_params();
    tmp_params.n_ctx   = llama_n_ctx(ctx);
    tmp_params.n_batch = tmp_params.n_ctx;

    ctx = llama_init_from_model(m->imdl, tmp_params);
    verify(ctx, "failed to init temporary context");

    llama_batch batch = llama_batch_get_one(t_data, n_tokens);
    llama_token tok[8];
    int32_t n = llama_tokenize(m->ivocab, "$", 1, tok, 8, false, false);
    verify(n > 0, "failed to tokenize '$'");
    llama_token end_token = tok[0];
        
    print("prompt: %o", prompt);
    for (;;) {
        int n_ctx_used = llama_kv_self_seq_pos_max(ctx, 0);
        verify(n_ctx_used + batch.n_tokens <= llama_n_ctx(ctx), "context overflow");
        verify(llama_decode(ctx, batch) == 0, "decode failure");

        new_token_id = llama_sampler_sample(a->ismpl, ctx, -1);

        if (new_token_id == end_token || llama_vocab_is_eog(m->ivocab, new_token_id)) {
            print("\n");
            break;
        }

        char buf[256];
        int  n = llama_token_to_piece(m->ivocab, new_token_id, buf, sizeof(buf), 0, true);
        verify(n >= 0, "token conversion failure");
        string words = string(alloc, n, ref_length, n, chars, buf);
        put("%o", words);
        fflush(stdout);

        concat(response, words);
        batch = llama_batch_get_one(&new_token_id, 1);
    }

    llama_free(ctx); // free forked context
    return response;
}

// append to context; no processing response (system case)
none conversation_append(conversation a, object msg) {
    llm    m      = a->model;
    string prompt = convert_msg(msg);
    verify(prompt, "append: invalid type: %s", isa(msg)->name);

    bool   is_first   = a->n_past == 0;
    int    n_tokens   = -llama_tokenize(
        m->ivocab, cstring(prompt), len(prompt), NULL, 0, is_first, true);
    vector tokens     = vector(alloc, n_tokens, type, typeid(i32));
    i32*   t_data     = (i32*)data(tokens);

    verify(llama_tokenize(
        m->ivocab, cstring(prompt), len(prompt), 
        t_data, n_tokens, is_first, true) >= 0, "tokenization failed");

    int pos        = 0;
    int total      = n_tokens;
    int chunk_size = 32;
 
    while (pos < total) {
        int chunk = (pos + chunk_size > total) ? (total - pos) : chunk_size;
        llama_batch batch = llama_batch_init(chunk, 0, 1);  // 1 sequence

        for (int i = 0; i < chunk; ++i) {
            batch.token[i]     = t_data[pos + i];
            batch.pos[i]       = a->n_past + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = false;
        }

        batch.n_tokens = chunk;
        batch.logits[chunk - 1] = true;
        verify(llama_decode(a->ictx, batch) == 0, "eval failure (decode)");
        llama_batch_free(batch);
        pos   += chunk;
        a->n_past += chunk;
    }
}
 

void conversation_dealloc(conversation a) {
    llama_sampler_free(a->ismpl);
    llama_free(a->ictx);
}


void llm_dealloc(llm m) {
    llama_model_free(m->imdl);
    llama_log_set(llm_logger, null);
}

define_class(llm, A)
define_class(conversation, A)