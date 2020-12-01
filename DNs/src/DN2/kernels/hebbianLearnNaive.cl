kernel void hebbianLearnNaive(global const float *response, 
                              global const float *input,
                              global float *weights, 
                              global int *age,
                              global float *bottom_up_weight_difference, 
                              const int neuron_num, 
                              const int input_length
                              ) 
{ 
    size_t id  = get_global_id(0); 
    float lr; 
    int current_weight_id; 
    if (response[id] > 0) 
    { 
        age[id] ++ ; 
        lr  = 1.0f / (float)age[id]; 
        for(int i = 0; i< input_length; i++) 
        { 
            current_weight_id = id * input_length + i; 
            weights[current_weight_id] = (1.0f - lr) * weights[current_weight_id] 
                                       + lr * response[id] * input[i]; 
           
            bottom_up_weight_difference[current_weight_id] = (1.0f - lr) * bottom_up_weight_difference[current_weight_id]
            												+ lr * response[id] * fabs(input[i] - weights[current_weight_id]);
        } 
    } 
}