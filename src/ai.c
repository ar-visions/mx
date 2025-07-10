#include <import>
#include <math.h>

none topo_visit(keras k, op current, map visited, map in_progress) {
    bool* prog = get(in_progress, current->name);
    if (prog && *prog) {
        error("Cycle detected in network graph at operation: %s", current->name);
        return;
    }
    
    // Skip if already visited
    if (get(visited, current->name)) {
        return;
    }
    
    // Mark as in progress (for cycle detection)
    set(in_progress, current->name, A_bool(true));
    
    // Visit all dependencies (inputs) first
    each (current->op_inputs, op, input_op) {
        topo_visit(k, input_op, visited, in_progress);
    }
    
    // Mark as visited and add to execution order
    set(visited, current->name, A_bool(true));
    set(in_progress, current->name, A_bool(false));
    push(k->order, current);
}

// Helper function to build execution order
none build_exec_order(keras k) {
    k->order = array();

    // create a visited flag for each op to track progress
    map visited     = map(hsize, 32);
    map in_progress = map(hsize, 32); // To detect cycles
    
    // for each operation, ensure it's visited
    each (k->ops, op, operation) {
        if (!get(visited, operation->name)) {
            topo_visit(k, operation, visited, in_progress);
        }
    }
}

tensor find_output(keras k, op a) {
    each (k->ops, op, j)
        each (j->op_inputs, op, input_op)
            if (input_op == a) return j->tensor;

    return null;
}

/// load the ops and initialize them after
none keras_init(keras k) {
    /// construct operation map
    k->op_map = map();
    each (k->ops, op, op)
        set (k->op_map, op->name, op);

    /// create op_inputs in op, so we know our input-tensors
    each (k->ops, op, a) {
        /// create resolved op list, corresponding 
        /// with index-of the string-name from inputs
        a->op_inputs = array();
        each (a->inputs, string, input) {
            op res = get(k->op_map, input);
            verify (res, "could not resolve operation: %o", input);
            push (a->op_inputs, hold(res));
        }
    }

    /// create order
    build_exec_order(k);
    op f = get(k->order, 0);
    verify(isa(f) == typeid(input), "input expected");
    k->input = hold(f->tensor); // an inputs output is our input.. it does have a lonely little input tensor sitting there unused though

    
    string k_output2 = string("output");

    /// finalize layers
    each (k->ops, op, a) {
        string k_output7 = string("output");
        a->output = find_output(k, a);
        each(a->op_inputs, op, i) {
            string k_output6 = string("output");
            if (isa(i) == typeid(input)) {
                string k_output5 = string("output");
                verify(compare(a->tensor->shape, i->tensor->shape) == 0, "incompatible shapes");
                drop(a->tensor);
                string k_output4 = string("output");
                a->tensor = hold(i->tensor);
            }
        }
        string k_output3 = string("output");
        finalize(a);
    }

    string k_output = string("output");
    op output = get(k->op_map, k_output);
    k->output = hold(output->tensor);
}

none keras_train(keras k, i32 epochs, map train, f32 learning_rate) {
    for (i32 epoch = 0; epoch < epochs; epoch++) {
        print("epoch %d/%d", epoch + 1, epochs);
        pairs (train, kv) {
            tensor input      = (tensor)kv->key;
            tensor target     = (tensor)kv->value;
            tensor output     = forward(k, input);
            f32    sum0       = sum(input);
            f32    loss       = 0;
            tensor d_output   = tensor(shape, target->shape);
            f32*   d_out_ptr  = data(d_output->realized);
            f32*   output_ptr = data(output->realized);
            f32*   target_ptr = data(target->realized);

            for (i32 i = 0, t = total(target->shape); i < t; i++) {
                f32 diff         = output_ptr[i] - target_ptr[i];
                    loss        += diff * diff;
                    d_out_ptr[i] = 2    * diff; // Gradient of MSE: dL/dy = 2(y - target)
            }

            loss /= total(target->shape);  print("loss: %.5f", loss);
            tensor d_input = back(k, d_output);
            each (k->ops, op, layer) {
                if (layer->weights) {
                    f32* w   = data(layer->weights->realized);
                    f32* d_w = data(layer->weights->grad);  // Assume we store computed gradients here in backprop
                    for (i32 i = 0; i < total(layer->weights->shape); i++)
                        w[i] -= learning_rate * d_w[i];
                }
                if (layer->bias) {
                    f32* b   = data(layer->bias->realized);
                    f32* d_b = data(layer->bias->grad); // Bias gradients should be stored here
                    for (i32 i = 0; i < total(layer->bias->shape); i++)
                        b[i] -= learning_rate * d_b[i];
                }
            }
            drop(d_output);
            drop(d_input);
        }
    }
    print("training complete.");
}

tensor keras_forward(keras k, tensor input) {
    /// copy input tensor, and pass forward
    memcpy(data(k->input->realized), data(input->realized), total(input->shape) * sizeof(f32));
    f32 f2 = sum(input);
    //print("sum of keras input: %.2f", f2);

    each (k->order, op, current) {
        //f32 fi = sum(current->tensor);
        //print("sum of input prior to op: %o: %.2f", current->name, fi);
        //print("%s: %x %x", isa(current)->name, current->tensor->realized, current->output ? current->output->realized : null);
        forward(current);
        //if (current->output) {
        //    f32 f = sum(current->output);
        //    print("sum of output after op: %o: %.2f", current->name, f);
        //}
    }

    tensor res = tensor(shape, k->output->shape);
    //print("keras output ident = %x", k->output->realized);  
    memcpy(data(res->realized), data(k->output->realized), sizeof(f32) * total(k->output->shape));
    return res;
}

tensor keras_back(keras k, tensor d_output) {
    backwards (k->order, op, a) {
        if (a->weights) memset(data(a->weights->grad), 0, total(a->weights->shape) * sizeof(f32));
        if (a->bias)    memset(data(a->bias->grad),    0, total(a->bias->shape)    * sizeof(f32));
        d_output = back(a, d_output);
    }
    return d_output;
}


// post-init is mostly established in keras_init 
// (at this point, we have all model data in props)
none op_finalize(op a) {
    return;
}

// all other layers that perform relu should fuse the operation
none op_forward(op a) {
    if (a->activation) {
        i64 u8_count    = shape_total(a->tensor); // this must return the tensor shape; test this
        tensor i0       = a->tensor;
        tensor o0       = a->output;
        i64 u8_actual   = A_len(i0); /// it can respond to len
        i8* input_data  = vdata(i0);
        i8* output_data = vdata(o0);
        verify(u8_count == u8_actual, "total size mismatch");
        /// we should probably assert the offsets are the same for this tensor
        verify(i0->offset == o0->offset, "expected output tensor to be quantized the same (offset)");
        verify(i0->scale  == o0->scale,  "expected output tensor to be quantized the same (scale)");

        if (a->activation == Activation_relu) {
            for (i64 i = 0; i < u8_count; i++)
                output_data[i] = max(o0->offset, input_data[i]);
        } else if (a->activation == Activation_tanh) {
            for (i64 i = 0; i < u8_count; i++) {
                // Dequantize: Convert i8 -> f32
                float x = (input_data[i] - i0->offset) * i0->scale;
                // Apply tanh function
                float y = tanhf(x);
                // Requantize: Convert f32 -> i8
                i8 quantized_y = (i8)(roundf(y / o0->scale) + o0->offset);
                // Store the result
                output_data[i] = quantized_y;
            }
        }
    }
}

tensor op_back(op a, tensor d_output) {
    return d_output;
}

// Concatenate implementation
none concatenate_finalize(concatenate a) {
    a->axis = -1;  // Default to last dimension
    return;
}

none concatenate_forward(concatenate a) {
    // Implement concatenation along the specified axis
    // Will need to iterate through all inputs and combine them
    return;
}

tensor concatenate_back(concatenate a, tensor d_output) {
    return null;
}


none relu_init(relu a) {
    a->activation = Activation_relu;
    return;
}

none flatten_forward(flatten a) {
    int t = total(a->tensor->shape);
    if (a->output->realized != a->tensor->realized) {
        drop(a->output->realized);
        a->output->realized = hold(a->tensor->realized);
    }
}

tensor flatten_back(flatten a, tensor d_output) {
    int t = total(d_output->shape);
    tensor flat = tensor(shape, shape_new(1, t, 0));
    memcpy(data(flat->realized), data(d_output->realized), t * sizeof(f32));
    return flat;
}

define_enum (Initializer)
define_enum (Activation)
define_enum (Padding)
define_enum (Pooling)
define_class(op,            A)
define_class(input,         op)
define_class(flatten,       op)
define_class(concatenate,   op)
define_class(relu,          op)
define_class(output,        op)
define_class(keras,         A)
define_class(ops,           array, op)

