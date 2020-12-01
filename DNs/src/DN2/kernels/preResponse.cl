kernel void preResponse(global const float* input,
                        global const float* weight,
                        global const float* mask, // weights mask
                        global float* response,
                        int num_elements,
                        int input_length)
{
    // get index into global data array
    int neuron_id = get_global_id(0);
    response[neuron_id] = 0;
    
    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (neuron_id >= num_elements)  {
        return;
    }
    
    // normalize weight first (input is already normalized).
    float weight_sum = 0;
    float input_sum  = 0;
    
    
    int current_weight_id = 0;
    for (int i = 0; i < input_length; i++) {
        current_weight_id = neuron_id * input_length + i;//current_weight_id = current_input_id
        if(mask[current_weight_id] > 0){        
		  weight_sum += weight[current_weight_id] * weight[current_weight_id];
		  input_sum  += input[i]  * input[i];
		}
    }
    
    weight_sum = sqrt(weight_sum);
    input_sum = sqrt(input_sum);
    
    //suppose weight and input are all positive
    if (weight_sum >0 && input_sum > 0){
	    for (int i = 0; i < input_length; i++) {
	        current_weight_id = neuron_id * input_length + i;
	        response[neuron_id] += mask[current_weight_id] * weight[current_weight_id] * input[i] /(weight_sum * input_sum);
	    }
    }
}
