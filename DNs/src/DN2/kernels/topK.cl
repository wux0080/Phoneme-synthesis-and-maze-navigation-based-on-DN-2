kernel void topK (global float* preresponse,
                  global float* response,
                  global int* neuron_type,
                  global int* intermediate_id,
                  global float* intermediate_response,
                  global int* topK_id,
                  global float* topK_value,
                  int K,
                  int chunk_size,
                  int thread_num,
                  int used_num,
                  float almost_perfect,
                  int max_neuron_num,
                  int current_type,
                  int learning_flag,
                  global int* result) {
    int thread_id = get_global_id(0);
    float max_response;
    int max_id;
    int current_id;
    for (int k = 0; k < (K+1); k++){
        current_id = thread_id;
        max_response = -1;
        max_id = -1;
        for (int i = 0; i < chunk_size; i++) {
            if (preresponse[current_id] > max_response) {
                max_response = preresponse[current_id];
                max_id = current_id;
            }
            current_id += thread_num;
        }
        preresponse[max_id] = -1;
        intermediate_response[thread_id * (K+1) + k] = max_response;
        intermediate_id[thread_id * (K+1) + k] = max_id;
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int max_intermediate_id = -1;
    if (thread_id == 0) {
        for (int k = 0; k < (K+1); k++){
            max_response = -1;
            max_id = -1;
            for (int i = 0; i < thread_num * (K+1); i++) {
                if (intermediate_response[i] > max_response) {
                    max_response = intermediate_response[i];
                    max_id = intermediate_id[i];
                    max_intermediate_id = i;
                }
            }
            intermediate_response[max_intermediate_id] = -1;
            if (k < K){
              if (max_response == 0 && almost_perfect == 0) {
              } else if (max_response > almost_perfect || used_num == max_neuron_num || learning_flag == false){
                  response[max_id] = max_response;
                  topK_value[k] = max_response;
                  topK_id[k] = max_id;
              } else {
                  response[used_num] = almost_perfect;
                  neuron_type[used_num] = current_type;
                  topK_value[k] = almost_perfect;
                  topK_id[k] = used_num;
                  used_num ++ ;
              }
            } else {
              topK_value[k] = max_response;
              topK_id[k] = max_id;
            }
        }

        result[0] = used_num;
    }
}
