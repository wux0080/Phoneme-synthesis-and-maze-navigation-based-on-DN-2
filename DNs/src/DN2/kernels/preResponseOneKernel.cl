kernel void preResponseOneKernel(global const float* input,
                                 global const float* weight,
                                 global float* response,
                                 int num_elements,
                                 int input_length)
{
    // Only 1 kernel working
    int thread_id = get_global_id(0);
    if (thread_id == 0){
        for (int neuron_id = 0; neuron_id < num_elements; neuron_id++){
            response[neuron_id] = 0;
            // normalize weight first (input is already normalized).
            float weight_sum = 0;
            int current_weight_id = 0;
            for (int i = 0; i < input_length; i++) {
                current_weight_id = neuron_id + i * num_elements;
                weight_sum += weight[current_weight_id] * weight[current_weight_id];
            }
            weight_sum = sqrt(weight_sum);

            for (int i = 0; i < input_length; i++) {
                current_weight_id = neuron_id + i * num_elements;
                response[neuron_id] += weight[current_weight_id] * input[i] / weight_sum;
            }
        }
    }
}
