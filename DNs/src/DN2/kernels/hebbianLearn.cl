kernel void hebbianLearn(global const float *response,
                         global const int *neuron_type,
                         global const int *k_id,
                         global const float *k_value,
                         global const float *bottom_up_input,
                         global const float *top_down_input,
                         global const float *lateral_input,
                         global float *bottom_up_weights,
                         global int   *bottom_up_weight_age,
                         global float *bottom_up_weight_mask,
                         global float *top_down_weights,
                         global int   *top_down_weight_age,
                         global float *top_down_weight_mask,
                         global float *lateral_weights,
                         global float *lateral_weight_age,
                         global float *lateral_weight_mask,
                         global int *age,
                         global float *bottom_up_weight_difference,
                         global float *top_down_weight_difference,
                         const int neuron_num,
                         const int bottom_up_input_length,
                         const int top_down_input_length,
                         const int lateral_input_length,
                         const int k, 
                         const int current_type,
                         const int smFlag
                        )
{
  size_t thread_id  = get_global_id(0);

  if (thread_id < k)
  {
    float lr;
    int current_weight_id;
    int neuron_id;
    float min_value = k_value[k];
    float max_value = k_value[0];
    float rescaled_response;
    float epsilon = 0.0000001f;
    float sum_bottom_up_input = epsilon;
    float sum_bottom_up_weights = epsilon;
    float sum_top_down_input = epsilon;
    float sum_top_down_weights = epsilon;
    float bottom_up_upper_bound;
    float bottom_up_lower_bound;
    float top_down_upper_bound;
    float top_down_lower_bound;
    int bottom_up_flag = 0;
    int top_down_flag = 0;
    int lateral_flag = 0;

    if(current_type == 5){
      bottom_up_flag = 1;
      top_down_flag = 1;
      lateral_flag = 0;
    } else if (current_type == 3){
      bottom_up_flag = 0;
      top_down_flag = 1;
      lateral_flag = 1;
    }
    neuron_id = k_id[thread_id];
    if (neuron_type[neuron_id] != current_type){
      return;
    }

    //re-scale response: topK & top(K+1)
    age[neuron_id] ++ ;
    rescaled_response = (response[neuron_id] - min_value)/(max_value - min_value + epsilon);

    if (rescaled_response <= epsilon) {
      return;
    }

    //re-normalize bottom up inputs for hebbianLearning
    for(int i = 0; i < bottom_up_input_length; i++)
    {
      int mask_index = neuron_id* bottom_up_input_length + i;
      if(bottom_up_weight_mask[mask_index] > 0)
      {
        sum_bottom_up_input += bottom_up_input[i] * bottom_up_input[i];
      }
    }
    sum_bottom_up_input = sqrt(sum_bottom_up_input);

    //re-normalize top down inputs for hebbianLearning
    for(int i = 0; i < top_down_input_length; i++)
    {
      int mask_index = neuron_id* top_down_input_length + i;
      if(top_down_weight_mask[mask_index] > 0)
      {
        sum_top_down_input += top_down_input[i] * top_down_input[i];
      }
    }
    sum_top_down_input = sqrt(sum_top_down_input);
    
    if (bottom_up_flag){
      // update bottom up weights
      for(int i = 0; i< bottom_up_input_length; i++)
      {
        current_weight_id = neuron_id * bottom_up_input_length + i;
        if (bottom_up_weight_mask[current_weight_id] > 0) {
          bottom_up_weight_age[current_weight_id] ++;
          lr  = 1.0f / (float)bottom_up_weight_age[current_weight_id];
        } else {
          lr = 0.0f;
        }
        bottom_up_weights[current_weight_id] = (1.0f - lr) * bottom_up_weights[current_weight_id]
                                             + lr * rescaled_response * bottom_up_input[i]/sum_bottom_up_input;
      }

      // re-normalize weight after hebbianLearning.
      for(int i = 0; i < bottom_up_input_length; i++)
      {
        int mask_index = neuron_id* bottom_up_input_length + i;
        if(bottom_up_weight_mask[mask_index] > 0)
        {
          sum_bottom_up_weights += bottom_up_weights[mask_index] * bottom_up_weights[mask_index];
        }
      }
      sum_bottom_up_weights = sqrt(sum_bottom_up_weights);

      // update bottom up weight difference.
      for (int i = 0; i < bottom_up_input_length; i++)
      {
        current_weight_id = neuron_id * bottom_up_input_length + i;
        if (bottom_up_weight_mask[current_weight_id] > 0) {
          lr = 1.0f / (float)bottom_up_weight_age[current_weight_id];
        } else {
          lr = 0.0f;
        }
        bottom_up_weight_difference[current_weight_id] = (1.0f -lr) * bottom_up_weight_difference[current_weight_id]
                         + lr * rescaled_response * fabs(bottom_up_input[i]/sum_bottom_up_input 
                        - bottom_up_weights[current_weight_id]/sum_bottom_up_weights);
      }
    }

    if (top_down_flag){
      // update top down weights
      for (int i = 0; i< top_down_input_length; i++)
      {
          current_weight_id = neuron_id * top_down_input_length + i;
          if(top_down_weight_mask[current_weight_id] > 0) {
            top_down_weight_age[current_weight_id]++;
            lr = 1.0f/(float)top_down_weight_age[current_weight_id];
          } else {
            lr = 0;
          }
          top_down_weights[current_weight_id] = (1.0f - lr) * top_down_weights[current_weight_id]
                              +  lr * rescaled_response * top_down_input[i]/sum_top_down_input;
      }

      // re-normalize weight after hebbianLearning.
      for(int i = 0; i <top_down_input_length; i++)
      {
        int mask_index = neuron_id* top_down_input_length + i;
        if(top_down_weight_mask[mask_index] > 0)
        {
          sum_top_down_weights += top_down_weights[mask_index] * top_down_weights[mask_index];
        }
      }
      sum_top_down_weights = sqrt(sum_top_down_weights);

      // update top down weight difference. 
      for (int i = 0; i< top_down_input_length; i++)
      {
          current_weight_id = neuron_id * top_down_input_length + i;
          if(top_down_weight_mask[current_weight_id] > 0) {
            lr = 1.0f/(float)top_down_weight_age[current_weight_id];
          } else {
            lr = 0;
          }
          top_down_weight_difference[current_weight_id] = (1.0f - lr) * top_down_weight_difference[current_weight_id]
                                    + lr * rescaled_response * fabs(top_down_input[i]/sum_top_down_input - top_down_weights[current_weight_id]/sum_top_down_weights);
      }
    }

    if (lateral_flag){
      // update lateral weights
      for (int i = 0; i< lateral_input_length; i++)
      {
          current_weight_id = neuron_id * lateral_input_length + i;
          if(lateral_weight_mask[current_weight_id] > 0) {
            lateral_weight_age[current_weight_id]++;
            lr = 1.0f/(float)lateral_weight_age[current_weight_id];
          } else {
            lr = 0;
          }
          lateral_weights[current_weight_id] = (1.0f - lr) * lateral_weights[current_weight_id]
                                             +  lr * rescaled_response * lateral_input[i];
      }
    }

    //reset mask periodically
    if(age[neuron_id] >= 20 && smFlag == 1){
      age[neuron_id] = 0; //reset neuron age
      float sum = 0; //sum of weight difference for each neuron
      float mean = 0; //mean of weight difference for each neuron

      if (bottom_up_flag){
        //reset  bottom-up weight mask according to weight difference
        sum = 0; //sum of weight difference for each neuron
        mean = 0; //mean of weight difference for each neuron
        for(int i = 0; i < bottom_up_input_length; i++){
          current_weight_id = neuron_id * bottom_up_input_length + i;
          sum += bottom_up_weight_difference[current_weight_id];
        }
        mean = sum/bottom_up_input_length;
        bottom_up_upper_bound = 1.5 * mean + epsilon; // manually set to 1.5
        bottom_up_lower_bound = 0.9 * mean; // manually set to 0.9
        float bottom_up_gap = bottom_up_upper_bound - bottom_up_lower_bound;
        for(int i = 0; i < bottom_up_input_length; i++){
          current_weight_id = neuron_id * bottom_up_input_length + i;
          if(bottom_up_weight_difference[current_weight_id] > bottom_up_upper_bound && 
             bottom_up_weight_mask[current_weight_id] > 0)
            {
              bottom_up_weight_mask[current_weight_id] = 0;
              bottom_up_weights[current_weight_id] = 0;
            }
          else if(bottom_up_weight_difference[current_weight_id] < bottom_up_lower_bound && 
                  bottom_up_weight_mask[current_weight_id] > 0)
            {
              bottom_up_weight_mask[current_weight_id] = 1;
            }
          else if (bottom_up_weight_mask[current_weight_id] > 0)
            {
              bottom_up_weight_mask[current_weight_id] = (bottom_up_upper_bound - bottom_up_weight_difference[current_weight_id])/bottom_up_gap;
            }
        }
        sum = epsilon;
        for (int i = 0; i < bottom_up_input_length; i++) {
          current_weight_id = neuron_id * bottom_up_input_length + i;
          if (bottom_up_weight_mask[current_weight_id] > 0) {
            sum += bottom_up_weights[current_weight_id] * bottom_up_weights[current_weight_id];
          }
        }
        sum = sqrt(sum);
        for (int i = 0; i < bottom_up_input_length; i++) {
          current_weight_id = neuron_id * bottom_up_input_length + i;
          if (bottom_up_weight_mask[current_weight_id] > 0) {
            bottom_up_weights[current_weight_id] = bottom_up_weights[current_weight_id]/sum;
          }
        }
      }

      if (top_down_flag){
        //reset  top-down weight mask according to weight difference
        sum = 0; //sum of weight difference for each neuron
        mean = 0; //mean of weight difference for each neuron
        for(int i = 0; i < top_down_input_length; i++){
          current_weight_id = neuron_id * top_down_input_length + i;
          sum += top_down_weight_difference[current_weight_id];
        }
        mean = sum/top_down_input_length;
        top_down_upper_bound = 1.5 * mean + epsilon; // manually set to 1.5
        top_down_lower_bound = 0.9 * mean; // manually set to 0.9
        float top_down_gap = top_down_upper_bound - top_down_lower_bound;
        for(int i = 0; i < top_down_input_length; i++){
          current_weight_id = neuron_id * top_down_input_length + i;
          if(top_down_weight_difference[current_weight_id] > top_down_upper_bound && 
             top_down_weight_mask[current_weight_id] > 0)
            {
              top_down_weight_mask[current_weight_id] = 0;
              top_down_weights[current_weight_id] = 0;
            }
          else if(top_down_weight_difference[current_weight_id] < top_down_lower_bound && 
                  top_down_weight_mask[current_weight_id] > 0)
            {
              top_down_weight_mask[current_weight_id] = 1;
            }
          else if (top_down_weight_mask[current_weight_id] > 0)
            {
              top_down_weight_mask[current_weight_id] = (top_down_upper_bound - top_down_weight_difference[current_weight_id])/top_down_gap;
            }
        }
        sum = epsilon;
        for (int i = 0; i < top_down_input_length; i++) {
          current_weight_id = neuron_id * top_down_input_length + i;
          if (top_down_weight_mask[current_weight_id] > 0) {
            sum += top_down_weights[current_weight_id] * top_down_weights[current_weight_id];
          }
        }
        sum = sqrt(sum);
        for (int i = 0; i < top_down_input_length; i++) {
          current_weight_id = neuron_id * top_down_input_length + i;
          if (top_down_weight_mask[current_weight_id] > 0) {
            top_down_weights[current_weight_id] = top_down_weights[current_weight_id]/sum;
          }
        }
      }
    }
  }
}
