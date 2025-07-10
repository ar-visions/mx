#include <import>
#include <math.h>
#include <immintrin.h>  // AVX2

none pool_finalize(pool a) {
    tensor i = a->tensor;
    verify(i->shape->count == 3, "unexpected input shape: %o", i->shape); 
    i64 h = i->shape->data[0];
    i64 w = i->shape->data[1];
    i64 c = i->shape->data[2];
    i32 pool_h   = a->pool_size->data[0];
    i32 pool_w   = a->pool_size->data[1];
    i32 stride_h = a->strides->data[0];
    i32 stride_w = a->strides->data[1];
    i32 out_h    = (h - pool_h) / stride_h + 1;
    i32 out_w    = (w - pool_w) / stride_w + 1;

    verify(a->output->shape->data[0] == out_h && 
           a->output->shape->data[1] == out_w &&
           a->output->shape->data[2] == c, "pool output shape mismatch");
}

none pool_forward(pool a) {
    verify(a->tensor, "Pool layer has no input tensor!");

    tensor i           = a->tensor;
    tensor o           = a->output;
    i32    in_h        = i->shape->data[0];
    i32    in_w        = i->shape->data[1];
    i32    in_c        = i->shape->data[2];
    i32    pool_h      = a->pool_size->data[0];
    i32    pool_w      = a->pool_size->data[1];
    i32    stride_h    = a->strides->data[0];
    i32    stride_w    = a->strides->data[1];
    i32    out_h       = o->shape->data[0];
    i32    out_w       = o->shape->data[1];
    f32*   input_data  = i->realized;
    f32*   output_data = o->realized;

    // Number of elements to process in parallel with AVX2
    const int vec_size = 8; // AVX2 processes 8 floats at once
    
    // Process channel data in chunks of 8 at a time
    for (i32 oh = 0; oh < out_h; oh++) {
        for (i32 ow = 0; ow < out_w; ow++) {
            // Process channels in chunks of 8
            for (i32 c = 0; c < in_c; c += vec_size) {
                // Load vector of negative infinity values as initial max
                __m256 max_vals = _mm256_set1_ps(-INFINITY);
                
                // Iterate through pool window
                for (i32 kh = 0; kh < pool_h; kh++) {
                    for (i32 kw = 0; kw < pool_w; kw++) {
                        i32 ih = oh * stride_h + kh;
                        i32 iw = ow * stride_w + kw;
                        
                        if (ih < in_h && iw < in_w) {  // Ensure within bounds
                            i32 base_idx = (ih * in_w + iw) * in_c + c;
                            
                            // Load 8 channels at once from this spatial position
                            __m256 input_vec = _mm256_loadu_ps(&input_data[base_idx]);
                            
                            // Update maximum values
                            max_vals = _mm256_max_ps(max_vals, input_vec);
                        }
                    }
                }
                
                // Store results back to memory
                i32 out_base_idx = (oh * out_w + ow) * in_c + c;
                _mm256_storeu_ps(&output_data[out_base_idx], max_vals);
                f32 values[8];
                memcpy(values, &output_data[out_base_idx], vec_size * sizeof(f32));
            }
        }
    }
}

tensor pool_back(pool a, tensor d_output) {
    // Allocate tensor for gradient w.r.t. input
    tensor d_input = tensor(shape, a->tensor->shape);

    // Extract shapes and properties
    i32 in_h    = a->tensor->shape->data[0];
    i32 in_w    = a->tensor->shape->data[1];
    i32 in_c    = a->tensor->shape->data[2];
    i32 pool_h  = a->pool_size->data[0];
    i32 pool_w  = a->pool_size->data[1];
    i32 stride_h = a->strides->data[0];
    i32 stride_w = a->strides->data[1];
    i32 out_h   = d_output->shape->data[0];
    i32 out_w   = d_output->shape->data[1];

    f32* input_data  = a->tensor->realized;
    f32* d_out_data  = d_output->realized;
    f32* d_in_data   = d_input->realized;

    // Initialize d_input to zeros
    memset(d_in_data, 0, total(a->tensor->shape) * sizeof(f32));

    // Backpropagate the max-pooling operation
    for (i32 oh = 0; oh < out_h; oh++) {
        for (i32 ow = 0; ow < out_w; ow++) {
            for (i32 c = 0; c < in_c; c++) {
                // Reset per output location
                i32 max_idx = -1;
                f32 max_val = -INFINITY;

                // Find the max-pooling index
                for (i32 kh = 0; kh < pool_h; kh++) {
                    for (i32 kw = 0; kw < pool_w; kw++) {
                        i32 ih = oh * stride_h + kh;
                        i32 iw = ow * stride_w + kw;
                        if (ih < in_h && iw < in_w) {
                            i32 idx = (ih * in_w + iw) * in_c + c;
                            if (input_data[idx] > max_val) {
                                max_val = input_data[idx];
                                max_idx = idx;
                            }
                        }
                    }
                }

                if (max_idx != -1) {
                    f32 grad_value = d_out_data[(oh * out_w + ow) * in_c + c];

                    // Prevent instability by clamping gradients
                    if (grad_value > 1e6) grad_value = 1e6;
                    if (grad_value < -1e6) grad_value = -1e6;

                    d_in_data[max_idx] += grad_value / (pool_h * pool_w);  // Normalize!
                }

            }
        }
    }

    return d_input;
}

define_class (pool, op)
