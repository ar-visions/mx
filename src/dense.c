#include <import>

none dense_finalize(dense a) {
    verify(a->weights,        "no weights");
    verify(a->bias,           "no bias");
    verify(a->output_dim > 0, "no units");
    verify(a->input_dim  > 0, "no input_dim");
    verify(a->output->shape->data[0] == 1 && 
           a->output->shape->data[1] == a->output_dim, "invalid output shape");
}

none dense_forward(dense a) {
    gemm(a->tensor, a->weights, a->bias, a->activation, a->output);
}

tensor dense_back(dense a, tensor d_output) {
    // Backpropagation for dense layer
    tensor  d_input     = tensor(shape, a->tensor->shape);
    tensor  d_weights   = tensor(shape, a->weights->shape);
    tensor  d_bias      = tensor(shape, a->bias->shape);

    i32     input_dim   = a->input_dim;
    i32     output_dim  = a->output_dim;
    f32*    d_out       = d_output->realized;
    f32*    d_in        = d_input->realized;
    f32*    weights     = a->weights->realized;
    f32*    d_w         = a->weights->grad;
    f32*    d_b         = a->bias->grad;
    f32*    output      = a->output->realized;
    f32*    input       = a->tensor->realized;

    // Apply activation gradient
    for (i32 i = 0; i < output_dim; i++) {
        if (a->activation == Activation_relu)
            d_out[i] *= (output[i] > 0) ? 1.0f : 0.0f; // ReLU derivative
        else if (a->activation == Activation_tanh)
            d_out[i] *= (1.0f - output[i] * output[i]); // Tanh derivative
    }

    // dL/dW = d_output * input^T
    for (i32 i = 0; i < output_dim; i++) {
        d_b[i] = d_out[i]; // dL/db = d_output (sum over batch if >1)
        for (i32 j = 0; j < input_dim; j++)
            d_w[i * input_dim + j] += d_out[i] * input[j];
    }

    // dL/dInput = W^T * d_output
    for (i32 j = 0; j < input_dim; j++) {
        d_in[j] = 0;
        for (i32 i = 0; i < output_dim; i++)
            d_in[j] += weights[i * input_dim + j] * d_out[i];
    }
    return d_input;
}

define_class (dense, op)