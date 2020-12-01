kernel void preResponseJicheng(global float *d_preresponse,
                          global const float *d_x,
                          global const float *d_w,
                          int DATA_SIZE,
                          int neuron_num,
                          int num_of_pe,
                          int chunkSize
                          )
{
    //parameter[4] = {DATA_SIZE, neuron_num, num_of_pe, chunkSize}
    int chunk_id = get_global_id(0);
    int start_neuron_id = (chunk_id) * chunkSize;
    int last_chunk = num_of_pe - 1;;
    int end_neuron_id;
    float sum_weight;
    // dynamic set end_neuron_id
    if(chunk_id == last_chunk)
        end_neuron_id = neuron_num;
    else
        end_neuron_id = (chunk_id+1) * chunkSize;
    
    for (int n_id = start_neuron_id; n_id < end_neuron_id; n_id++)
    {
        d_preresponse[n_id] = 0;
        sum_weight = 0;
        //normalize weights foe each neuron
        for (int j = 0; j < DATA_SIZE; j++ )
        {
            sum_weight += d_w[n_id * DATA_SIZE + j] * d_w[n_id * DATA_SIZE + j];
        }
        sum_weight = sqrt(sum_weight);
        
        for (int j = 0; j < DATA_SIZE ; j++ )
        {
            d_preresponse[n_id] +=  d_w[n_id * DATA_SIZE + j] * d_x[j] / sum_weight;
        }
    }
}
