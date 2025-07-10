#include <import>
#include <math.h>
#include <immintrin.h>

static sz seek_length(FILE* f, AType of_type) {
    u64 start = ftell(f);
    fseek(f, 0, SEEK_END);
    u64 end = ftell(f);
    sz flen = (end - start) / of_type->size;
    fseek(f, start, SEEK_SET);
    return flen;
}

f32 tensor_sum(tensor a) {
    int t = total(a->shape);
    float sum = 0;
    f32 *data = a->realized;
    for (int i = 0; i < t; i++) {
        sum += data[i];
    }
    return sum;
}

f32 tensor_grad_sum(tensor a) {
    int t = total(a->shape);
    float sum = 0;
    f32 *data = a->grad;
    for (int i = 0; i < t; i++) {
        sum += data[i];
    }
    return sum;
}

none realize(tensor a) {
    if (!a->realized)
         a->realized = vector(type, typeid(f32), vshape, a->shape);

    f32* dst = a->realized;
    u8*  src = (u8*)a->data;  // Now correctly treating as unsigned
    int  t   = total(a->shape);
    int  i  = 0;

    // SIMD conversion from u8 to f32 (with normalization)
    if (false)
    for (; i + 32 <= t; i += 32) { 
        // Load 32 bytes (u8)
        __m256i u8_vals = _mm256_loadu_si256((__m256i*)&src[i]);

        // Zero-extend unsigned 8-bit values to 16-bit
        __m256i lo_i16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(u8_vals, 0));
        __m256i hi_i16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(u8_vals, 1));

        // Extend 16-bit values to 32-bit
        __m256i lo_i32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(lo_i16, 0));
        __m256i hi_i32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(lo_i16, 1));
        __m256i lo2_i32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(hi_i16, 0));
        __m256i hi2_i32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(hi_i16, 1));

        // Convert int32 -> float32
        __m256 lo_f32  = _mm256_cvtepi32_ps(lo_i32);
        __m256 hi_f32  = _mm256_cvtepi32_ps(hi_i32);
        __m256 lo2_f32 = _mm256_cvtepi32_ps(lo2_i32);
        __m256 hi2_f32 = _mm256_cvtepi32_ps(hi2_i32);

        // Normalize by dividing by 255.0f
        __m256 scale = _mm256_set1_ps(1.0f / 255.0f);
        lo_f32  = _mm256_mul_ps(lo_f32, scale);
        hi_f32  = _mm256_mul_ps(hi_f32, scale);
        lo2_f32 = _mm256_mul_ps(lo2_f32, scale);
        hi2_f32 = _mm256_mul_ps(hi2_f32, scale);

        // Store results
        _mm256_storeu_ps(&dst[i], lo_f32);
        _mm256_storeu_ps(&dst[i+8], hi_f32);
        _mm256_storeu_ps(&dst[i+16], lo2_f32);
        _mm256_storeu_ps(&dst[i+24], hi2_f32);
    }

    // Remainder loop for leftover elements
    for (; i < t; i++)
        dst[i] = ((float)(u8)src[i]) / 255.0f;

    a->offset = 0.0;
    a->scale  = 1 / 127.0; // we will want to fix this to -1.0 to 1.0 in training
}

tensor tensor_with_image(tensor a, image img) {
    a->shape = shape_new(img->width, img->height, 1, 0);
    a->data  = vector(type, typeid(i8), vshape, a->shape);
    memcpy(a->data, data(img), img->height * img->width * 1);
    realize(a);
    return a;
}

// called from the generic path read (json parse)
// needs to also load offset and scale
tensor tensor_with_string(tensor a, string loc) {
    path uri = form(path, "%o", loc);
    if (is_ext(uri, "png")) {
        image img = image(uri, uri);
        i8*   dat = data(img);
        verify(img->channels = 1, "not grayscale");
        i64 total = img->width * img->height * img->channels;
        a->shape    = shape_new(img->width, img->height, 1, 0);
        a->data     = vector(type, typeid(i8),  vshape, a->shape);
        a->realized = vector(type, typeid(f32), vshape, a->shape);
        memcpy(a->data, data(img), img->height * img->width * 1);
        a->offset = 0.0;
        a->scale  = 1 / 127.0; // we will want to fix this to -1.0 to 1.0 in training
    } else {
        path uri_f32 = form(path, "models/%o.f32", loc);
        path uri_i8  = form(path, "models/%o.i8", loc);
        FILE* f;
        bool is_f32 = exists(uri_f32);
        f = fopen(is_f32 ? uri_f32->chars : uri_i8->chars, "rb");
        a->shape = shape_read(f);
        sz flen  = seek_length(f, typeid(f32));
        if (is_f32) {
            vector res = vector(type, typeid(f32), vshape, a->shape);
            i64    total = shape_total(a->shape);
            verify(flen == total, "f32 mismatch in size");
            verify(fread(res, sizeof(f32), flen, f) == flen, "could not read path: %o", uri_f32);
            a->realized = res;
        } else {
            /// must really contain two floats for this to make sense.  i do not want this misbound; and its required model-wise
            vector res = vector(type, typeid(i8), vshape, a->shape);
            verify(fread(&a->scale,  sizeof(float), 1, f) == 1, "scale");
            verify(fread(&a->offset, sizeof(float), 1, f) == 1, "offset");
            i64 total = shape_total(a->shape);
            verify(flen == total, "i8 mismatch in size");
            verify(fread(res, 1, flen, f) == flen, "could not read path: %o", uri_i8);
            a->data = res;
        }
        fclose(f);
    }
    return a;
}

/// construct with dimension shape (not the data)
tensor tensor_with_array(tensor a, array dims) {
    num count = len(dims);
    i64 shape[32];
    i64 index = 0;
    each (dims, object, e) {
        i64* i = (i64*)instanceof(e, i64);
        shape[index++] = *i;
    }
    a->shape  = shape_from(index, shape);
    a->total  = shape_total(a->shape); // improve vectors in time
    a->data   = vector(type, typeid(i8), vshape, shape_new(a->total, 0));
    return a;
}

none tensor_init(tensor a) {
    a->total = shape_total(a->shape);
    if (!a->realized) a->realized = vector(type, typeid(f32), vshape, a->shape);
    if (!a->grad)     a->grad     = vector(type, typeid(f32), vshape, a->shape);
    if (!a->data)     a->data     = vector(type, typeid(i8),  vshape, a->shape);
}

static __m256 tanh_approx(__m256 x) {
    // Approximate tanh(x) using a polynomial
    const __m256 alpha = _mm256_set1_ps(1.0f);
    const __m256 beta  = _mm256_set1_ps(0.142857f); // 1/7 approximation
    const __m256 gamma = _mm256_set1_ps(0.333333f); // 1/3 approximation
    const __m256 x2    = _mm256_mul_ps(x, x);  // x^2
    const __m256 num   = _mm256_fmadd_ps(gamma, x2, alpha); // gamma*x^2 + 1
    const __m256 denom = _mm256_fmadd_ps(beta, x2, alpha); // beta*x^2 + 1
    return _mm256_div_ps(num, denom); // Approx tanh(x)
}

none tensor_gemm(tensor a, tensor b, tensor bias, Activation activation, tensor c) {
    i32 m = a->shape->data[0];  // Rows in a (output height)
    i32 k = a->shape->data[1];  // Shared dimension (cols of a, rows of b)
    i32 n = c->shape->data[1];  // Columns in b and c (output width)

    f32* b_data = b->realized;  // Weights (k × n)
    f32* c_data = c->realized;  // Output (m × n)
    f32* bias_data = bias ? data(bias->realized) : NULL;
    f32* a_data = a->realized;  // Input (m × k)

    const i32 vec_size = 8;  // AVX2 processes 8 floats per iteration
    i32 n_aligned = n - (n % vec_size);  // Ensure full AVX2 blocks

    // Perform GEMM: c = a * b + bias
    for (i32 i = 0; i < m; i++) {  // Loop over rows of c
        i32 j = 0;

        // Process 8 output channels at a time
        if (false)
        for (; j < n_aligned; j += vec_size) { 
            __m256 sum = bias_data ? _mm256_loadu_ps(&bias_data[j]) : _mm256_setzero_ps();

            for (i32 k_idx = 0; k_idx < k; k_idx++) {
                __m256 a_vec = _mm256_set1_ps(a_data[i * k + k_idx]); // Broadcast a value
                __m256 b_vec = _mm256_loadu_ps(&b_data[k_idx * n + j]); // Load 8 b values
                sum = _mm256_fmadd_ps(a_vec, b_vec, sum); // Multiply and accumulate
            }

            // Apply activation
            if (activation == Activation_relu) {
                __m256 zero_vec = _mm256_setzero_ps();
                sum = _mm256_max_ps(sum, zero_vec);  // ReLU: max(sum, 0)
            } else if (activation == Activation_tanh) {
                sum = tanh_approx(sum);  // Apply tanh approximation
            }

            _mm256_storeu_ps(&c_data[i * n + j], sum);
        }

        // Process remaining output channels with scalar ops
        for (; j < n; j++) { 
            f32 sum = bias_data ? bias_data[j] : 0.0f;
            for (i32 k_idx = 0; k_idx < k; k_idx++) {
                sum += a_data[i * k + k_idx] * b_data[k_idx * n + j];
            }
            if (activation == Activation_relu)
                c_data[i * n + j] = sum > 0 ? sum : 0;
            else if (activation == Activation_tanh)
                c_data[i * n + j] = tanh(sum);
            else
                c_data[i * n + j] = sum;
        }
    }
}

define_class(tensor, A)