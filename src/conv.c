#include <import>
#include <immintrin.h>  // AVX2

// Fixed conv_finalize function
none conv_finalize(conv c) {
    verify(c->tensor, "Conv layer has no input tensor!");

    shape     input_shape = c->tensor->shape;
    shape     out_shape   = c->output->shape;
    i64       out_total   = shape_total(out_shape);
    tensor    f           = c->tensor;
    i32       offset      = f->offset;
    f32       scale       = f->scale;
    i32       in_h        = input_shape     ->data[0];
    i32       in_w        = input_shape     ->data[1];
    i32       in_c        = input_shape     ->data[2];
    i32       filter_h    = c->kernel_size  ->data[0];
    i32       filter_w    = c->kernel_size  ->data[1];
    i32       stride_h    = c->strides      ->data[0];
    i32       stride_w    = c->strides      ->data[1];
    i32       pad_h       = (c->padding == Padding_same) ? (filter_h - 1) / 2 : 0;
    i32       pad_w       = (c->padding == Padding_same) ? (filter_w - 1) / 2 : 0;
    i32       out_h       = (in_h - filter_h + 2 * pad_h) / stride_h + 1;
    i32       out_w       = (in_w - filter_w + 2 * pad_w) / stride_w + 1;
    i32       out_c       = c->out_channels;

    verify(
        out_h == out_shape->data[0] &&
        out_w == out_shape->data[1] &&
        out_c == out_shape->data[2], "shape mismatch");
    
    i32       im2col_h     = out_h * out_w;
    i32       im2col_w     = filter_h * filter_w * in_c;
    shape     im2col_shape = shape_new(im2col_h, im2col_w, 0);

    /// our new input
    c->im2col_matrix = tensor(
        shape, im2col_shape,  
        offset, offset, scale, scale);

    tensor    w = c->weights;
    
    // For GEMM, weights need to be transposed to [im2col_w, out_c]
    // since im2col produces [out_h*out_w, filter_h*filter_w*in_c]
    shape     weights_shape = shape_new(im2col_w, out_c, 0);
    c->weights_matrix = tensor(
        shape, weights_shape, 
        offset, w->offset, scale, w->scale);
    
    f32* w_new  = c->weights_matrix->realized;
    f32* w_orig = c->weights->realized;
    
    // Correct weight reorganization for im2col
    // This transforms weights from [filter_h, filter_w, in_c, out_c] format
    // to [filter_h*filter_w*in_c, out_c] format for matrix multiplication
    for (i32 oc = 0; oc < out_c; oc++) {
        for (i32 kh = 0; kh < filter_h; kh++) {
            for (i32 kw = 0; kw < filter_w; kw++) {
                for (i32 ic = 0; ic < in_c; ic++) {

                    if (in_c == 1) {
                        // Destination index in weights_matrix (im2col compatible format)
                        i32 dest_idx = (kh * filter_w * in_c + kw * in_c + ic) * out_c + oc;
                        
                        // Source index in original weights tensor
                        // Must align with your model's weight order (assuming [h,w,in,out])
                        i32 src_idx = ((kh * filter_w + kw) * in_c + ic) * out_c + oc;
                        
                        w_new[dest_idx] = w_orig[src_idx];
                    } else {
                        // Source index in the original weight tensor
                        // (TensorFlow NHWC format: [kh, kw, ic, oc])
                        i32 src_idx = ((kh * filter_w + kw) * in_c + ic) * out_c + oc;

                        // Destination index in im2col weight matrix
                        // (Rearrange to: [oc, kh, kw, ic] for optimal GEMM access)
                        i32 dest_idx = (((oc * filter_h + kh) * filter_w + kw) * in_c) + ic;

                        w_new[dest_idx] = w_orig[src_idx];
                    }

                }
            }
        }
    }
}

// Fixed im2col implementation optimized for multi-channel inputs
none im2col(conv a, tensor input, tensor result) {
    verify(a->tensor, "Conv layer has no input tensor!");

    shape input_shape  = input->shape;
    shape output_shape = a->output->shape;
    shape im2col_shape = a->im2col_matrix->shape;
    i32   in_h         = input_shape->data[0];
    i32   in_w         = input_shape->data[1];
    i32   in_c         = input_shape->data[2];
    i32   out_h        = output_shape->data[0];
    i32   out_w        = output_shape->data[1];
    i32   out_c        = output_shape->data[2];
    i32   filter_h     = a->kernel_size->data[0];
    i32   filter_w     = a->kernel_size->data[1];
    i32   stride_h     = a->strides->data[0];
    i32   stride_w     = a->strides->data[1];
    i32   pad_h        = (a->padding == Padding_same) ? (filter_h - 1) / 2 : 0;
    i32   pad_w        = (a->padding == Padding_same) ? (filter_w - 1) / 2 : 0;
    i32   im2col_h     = im2col_shape->data[0];  // out_h * out_w
    i32   im2col_w     = im2col_shape->data[1];  // filter_h * filter_w * in_c
    f32*  input_data   = input->realized;
    f32*  im2col_data  = result->realized;

    // Zero the im2col matrix first to handle padding efficiently
    memset(im2col_data, 0, sizeof(f32) * im2col_h * im2col_w);
    
    // Iterate over each output position (oh, ow)
    for (i32 oh = 0; oh < out_h; oh++) {
        for (i32 ow = 0; ow < out_w; ow++) {
            // Current output point index
            i32 out_idx = oh * out_w + ow;
            
            // For each output position, gather all input elements in the receptive field
            for (i32 kh = 0; kh < filter_h; kh++) {
                // Corresponding input row
                i32 ih = oh * stride_h - pad_h + kh;
                
                // Skip if outside input height
                if (ih < 0 || ih >= in_h) continue;
                
                for (i32 kw = 0; kw < filter_w; kw++) {
                    // Corresponding input column
                    i32 iw = ow * stride_w - pad_w + kw;
                    
                    // Skip if outside input width
                    if (iw < 0 || iw >= in_w) continue;
                    
                    // For valid positions, copy all channels at once
                    i32 input_offset = (ih * in_w + iw) * in_c;
                    i32 im2col_offset = out_idx * (filter_h * filter_w * in_c) + 
                                       (kh * filter_w + kw) * in_c;
                    
                    // Copy the entire channel block (in_c values)
                    memcpy(&im2col_data[im2col_offset], 
                           &input_data[input_offset], 
                           sizeof(f32) * in_c);
                }
            }
        }
    }
}


void conv2d_forward(
    float* input, int in_h, int in_w, int in_c, 
    float* weights, int filter_h, int filter_w, int out_c, 
    float* bias, int stride, int pad, 
    float* output, int out_h, int out_w) 
{
    // Loop over output channels
    for (int oc = 0; oc < out_c; oc++) {
        // Loop over output height and width
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                // Start with bias (if available)
                float sum = bias ? bias[oc] : 0;
                
                for (int ic = 0; ic < in_c; ic++) {
                    for (int kh = 0; kh < filter_h; kh++) {
                        for (int kw = 0; kw < filter_w; kw++) {
                            int ih = oh * stride - pad + kh;
                            int iw = ow * stride - pad + kw;
                            
                            if (ih >= 0 && iw >= 0 && ih < in_h && iw < in_w) {
                                // NHWC input indexing
                                int input_idx  = ((ih * in_w + iw) * in_c) + ic;
                                // Keras-compatible weight indexing
                                int weight_idx = ((kh * filter_w + kw) * in_c + ic) * out_c + oc;
                                sum += input[input_idx] * weights[weight_idx];
                            }
                        }
                    }
                }
                
                // Store output with NHWC indexing
                output[((oh * out_w + ow) * out_c) + oc] = (sum > 0) ? sum : 0;
            }
        }
    }
}


none conv_forward(conv a) {
    // Extract tensor shapes
    shape input_shape  = a->tensor->shape;
    shape output_shape = a->output->shape;
    shape kernel_shape = a->kernel_size;
    shape stride_shape = a->strides;

    // Extract dimensions
    int in_h = input_shape->data[0];
    int in_w = input_shape->data[1];
    int in_c = input_shape->data[2];
    int filter_h = kernel_shape->data[0];
    int filter_w = kernel_shape->data[1];
    int out_c = a->out_channels;
    int stride = stride_shape->data[0]; // Assuming square strides
    int pad = (a->padding == Padding_same) ? (filter_h - 1) / 2 : 0;
    int out_h = output_shape->data[0];
    int out_w = output_shape->data[1];

    // Extract tensor data pointers
    float* input_data = a->tensor->realized;
    float* weight_data = a->weights->realized;
    float* bias_data = a->bias ? a->bias->realized : NULL;
    float* output_data = a->output->realized;

    float bias_sum = sum(a->bias);
    float w_sum = sum(a->weights);


    // vanilla conv2d forward function (unoptimized)
    if (true || in_c > 1) {
        // we do NOT want to run this; its inefficient.  we want to use im2col
        conv2d_forward(
            input_data, in_h, in_w, in_c, 
            weight_data, filter_h, filter_w, out_c, 
            bias_data, stride, pad, 
            output_data, out_h, out_w
        );
    } else {
        /// disabled for now
        im2col(a, a->tensor, a->im2col_matrix);
        gemm(a->im2col_matrix, a->weights_matrix, a->bias, a->activation, a->output);
    }
}

tensor conv_back(conv a, tensor d_output) {
    // Extract tensor shapes
    shape input_shape  = a->tensor->shape;
    shape output_shape = a->output->shape;
    shape kernel_shape = a->kernel_size;
    shape stride_shape = a->strides;

    // Extract dimensions
    int in_h        = input_shape->data[0];
    int in_w        = input_shape->data[1];
    int in_c        = input_shape->data[2];
    int filter_h    = kernel_shape->data[0];
    int filter_w    = kernel_shape->data[1];
    int out_c       = a->out_channels;
    int stride      = stride_shape->data[0]; 
    int pad         = (a->padding == Padding_same) ? (filter_h - 1) / 2 : 0;
    int out_h       = output_shape->data[0];
    int out_w       = output_shape->data[1];

    // Create gradient tensors
    tensor d_input   = tensor(shape, a->tensor->shape);  // dL/dInput
    
    // Extract tensor data pointers
    f32* d_out     = d_output->realized;
    f32* d_in      = d_input->realized;
    f32* weights   = a->weights->realized;
    f32* d_w       = a->weights->grad;
    f32* d_b       = a->bias->grad;
    f32* input     = a->tensor->realized;
    f32* output    = a->output->realized;

    // Initialize gradients to zero
    memset(d_in, 0, sizeof(float) * total(input_shape));
    memset(d_w,  0, sizeof(float) * total(a->weights->shape));
    memset(d_b,  0, sizeof(float) * total(a->bias->shape));
    
    // Apply activation derivative before backpropagation
    if (a->activation == Activation_relu) {
        for (int i = 0; i < total(output_shape); i++) {
            d_out[i] *= (output[i] > 0) ? 1.0f : 0.0f; // ReLU derivative
        }
    } else if (a->activation == Activation_tanh) {
        for (int i = 0; i < total(output_shape); i++) {
            float tanh_out = output[i];
            d_out[i] *= (1.0f - tanh_out * tanh_out); // Tanh derivative
        }
    }

    // Compute bias gradient (sum over all output gradients)
    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int out_idx = ((oh * out_w) + ow) * out_c + oc;
                d_b[oc] += d_out[out_idx];
            }
        }
    }

    // Compute weight and input gradient
    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int out_idx = ((oh * out_w) + ow) * out_c + oc;
                float grad_out = d_out[out_idx];

                for (int ic = 0; ic < in_c; ic++) {
                    for (int kh = 0; kh < filter_h; kh++) {
                        int ih = oh * stride - pad + kh;
                        if (ih < 0 || ih >= in_h) continue;

                        for (int kw = 0; kw < filter_w; kw++) {
                            int iw = ow * stride - pad + kw;
                            if (iw < 0 || iw >= in_w) continue;

                            int input_idx = ((ih * in_w) + iw) * in_c + ic;
                            int weight_idx = ((kh * filter_w + kw) * in_c + ic) * out_c + oc;
                            
                            // Compute gradient for weights
                            d_w[weight_idx] += input[input_idx] * grad_out;
                            
                            // Compute gradient for input
                            d_in[input_idx] += weights[weight_idx] * grad_out;
                        }
                    }
                }
            }
        }
    }

    return d_input;
}

define_class (conv, op)
