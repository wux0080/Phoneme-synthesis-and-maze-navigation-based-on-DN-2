// OpenCL Kernel Function for element by element vector addition
kernel void vectorAdd(global float* bottom_up_response, 
                      global float* top_down_response,
                      global float* lateral_response,
                      global int*   neuron_type,
                      global float* pre_response_type3,
                      global float* pre_response_type5,
                      global float* final_response,
                      private int numElements,
                      private float buttom_up_threshhold,
                      private float top_down_threshhold,
                      private float lateral_threshold
                    )
{
  // get index into global data array
  int iGID = get_global_id(0);

  // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
  if (iGID >= numElements)  {
      pre_response_type5[iGID] = 0;
      final_response[iGID] = 0;
      return;
  }

  if(bottom_up_response[iGID] < buttom_up_threshhold)
    bottom_up_response[iGID] = 0;

  if(top_down_response[iGID] < top_down_threshhold)
    top_down_response[iGID] = 0;

  if(lateral_response[iGID] < lateral_threshold)
    lateral_response[iGID] = 0;

  // add the vector elements
  if (neuron_type[iGID] == 5){
    pre_response_type5[iGID] = bottom_up_response[iGID] + top_down_response[iGID];
  } else if (neuron_type[iGID] == 3){
    pre_response_type3[iGID] = lateral_response[iGID] + top_down_response[iGID];
  } else {
    printf("Neuron Type %d at id %d not supported.\n", 
            neuron_type[iGID], iGID);
  }
  
  final_response[iGID] = 0;
}

