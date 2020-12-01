package DN2;

import java.util.Arrays;
import java.util.Random;
import java.io.*;
public class MotorLayer implements Serializable {

	private static final long serialVersionUID = 1L;
	private int mMotorIndex;
	//the height of the motor concept zone
	private int height;
	//the width of the motor concept zone
	private int width;
	//the normalize frequency of bottom-up weight vector
	private int bottomupFrequency;
	//the input vector
	private float[][] input;
    //the number of winner 
	private int topK;	
	private int usedMotorNeurons;
	//the number of z neurons in the motor concept zone
	private int numNeurons;
	//the length of bottom-up weight vector
	private int numBottomUpWeights;
	//the length of lateral weight vector
	private int numLateralWeights;
	//whether do top-k competition
	private boolean isTopk;
	private final float GAMMA = 2000;
	//the machine zero value
	private final float MACHINE_FLOAT_ZERO = 0.00001f;
	//the perfect match value
	private final float TOP_MOTOR_RESPONSE = 1.0f - MACHINE_FLOAT_ZERO;
//  private final float TOP_MOTOR_RESPONSE = 0.5f;
    //the z neuron array	
	public Neuron[] motorNeurons;
	private boolean mode;
	
	// some motor neurons may have lateral connections.
	public MotorLayer(int height, int width, int topK, int hiddenSize, int lateralSize, int frequency){
		mode = false;
		//set the width of the motor concept zone
		this.setWidth(width);
		//set the height of the motor concept zone
		this.setHeight(height);
		//set the number of winner
		this.setTopK(topK);
		//set not use top-k competition
		isTopk = false;
		//set the normalize frequency of bottom-up weight vector 
		bottomupFrequency = frequency;
		//construct the input vector
		input = new float[height][width];
		usedMotorNeurons = topK+39;
		//calculate the number of z neurons in the motor area
		numNeurons = height * width;
		//set the length of bottom-up weight vector
		numBottomUpWeights = hiddenSize;
		//set the length of lateral weight vector
		numLateralWeights = lateralSize;
		//construct the motor neuron array
		motorNeurons = new Neuron[numNeurons];
		//initialize the motor neurons and their locations
		for(int i=0; i<numNeurons; i++){
			float[] temp = new float[3];
			temp[0] = (float)(Math.random()*10);
			temp[1] = (float)(Math.random()*10);
			temp[2] = (float)(Math.random()*10);
			motorNeurons[i] = new Neuron(numBottomUpWeights,0, lateralSize,false,i, bottomupFrequency);
			motorNeurons[i].setlocation(temp);
			
		}
	}

	//construct the motor layer
	public MotorLayer(int width, int height, float[][] input){
		mode = false;
		//set the width of the motor concept zone
		this.setWidth(width);
		//set the height of the motor concept zone
		this.setHeight(height);	
		//set input vector
		this.setInput(input);
	}

	//hebbian learning for the z neurons and update winner neurons' bottom-up weight vector
	public void hebbianLearnMotor(float[] hiddenResponse){
		for (int i = 0; i < numNeurons; i++) {
			//if the neuron's response lagers than or equals to the perfect match value, update its weight  
			if(motorNeurons[i].getnewresponse() >= TOP_MOTOR_RESPONSE){
				motorNeurons[i].hebbianLearnHidden(hiddenResponse);
			}	
		}
	}
	
	//hebbian learning for the z neurons and update winner neurons' lateral weight vector
	public void hebbianLearnLateral(float[] lateralResposne){
		for (int i = 0; i < numNeurons; i++) {
			//if the neuron's response lagers than or equals to the perfect match value, update its weight  
			if(motorNeurons[i].getnewresponse() >= TOP_MOTOR_RESPONSE){
				motorNeurons[i].hebbianLearnLateral(lateralResposne);
		    }
		}
	}

	//hebbian learning for the z neurons and update winner neurons' lateral weight vector
	public void hebbianLearnLateral(float[] lateralResposne, int firing_neuron){
		motorNeurons[firing_neuron].hebbianLearnLateral(lateralResposne);
	}
	
	// convert the response into 1-D Array
	public float[] getNewMotorResponse1D() {
		//construct the new 1-D array
		float[] inputArray = new float[height * width];
		//copy values to the 1-D array
        for(int i = 0; i < numNeurons; i++){			
			inputArray[i] = motorNeurons[i].getnewresponse();
		}		
		return inputArray;
	}

	//get the z neurons' response 
	public float[][] getNewMotorResponse2D() {
		//construct the new array
		float[][] outputArray = new float[height][width];
		//copy the response values to the array
		for (int i = 0; i < numNeurons; i++) {			
			outputArray[i/width][i%width] = motorNeurons[i].getnewresponse();
		}		
		return outputArray;
	}
	
    //get the z neurons' lateral response
	public float[][] getLateralResponse2D() {
		//construct the new array
		float[][] outputArray = new float[height][width];
		//copy the lateral response values to the array
		for (int i = 0; i < numNeurons; i++) {			
			outputArray[i/width][i%width] = motorNeurons[i].getlateralresponse();
		}		
		return outputArray;
	}
	
	//transfer the new response values to old response vector
	public void replaceMotorLayerResponse(){
		for (int i = 0; i < numNeurons; i++) {			
			motorNeurons[i].replaceResponse();
		}
	}
	
	//compute the lateral preResponse
	public void computeLateralResponse(float[] lateral_input) {
		//normalize the lateral input vector
		normalize(lateral_input, lateral_input.length, 2);
		for (int i = 0; i < numNeurons; i++) {
			motorNeurons[i].computeLateralResponse(lateral_input, 1);
		}
	}
	
	//compute the preResponse
	public void computeResponse(float[] hiddenResponse){
		//normalize the bottom-up input vector
		normalize(hiddenResponse, hiddenResponse.length, 2);		
		// do the dot product between the weights
		for (int i = 0; i < numNeurons; i++) {
			motorNeurons[i].computeBottomUpResponse(hiddenResponse);
			motorNeurons[i].computeResponse();
		}	
		

		
		//do top-k competition
		if(isTopk){
			if(mode == true){
				truetopKCompetition();
		     }
			else{
				topKCompetition();
			}
			}
	}
	

	// Sort the topK elements to the beginning of the sort array where the index of the top 
	// elements are still in the pair.
	private static void topKSort(Pair[] sortArray, int topK){
		//find the element with max value and record its index
		for (int i = 0; i < topK; i++) {
			Pair maxPair = sortArray[i]; 
			int maxIndex = i;					
			for (int j = i+1; j < sortArray.length; j++) {
				// select temporary max
				if(sortArray[j].value > maxPair.value){ 
					maxPair = sortArray[j];
					maxIndex = j;					
				}
			}
			
			if(maxPair.index != i){
				// store the value of pivot (top i) element
				Pair temp = sortArray[i];
				// replace with the maxPair object
				sortArray[i] = maxPair; 
				// replace maxPair index elements with the pivot
				sortArray[maxIndex] = temp;  
			}
		}
	}

	//do the top-k competition
	private void topKCompetition(){		
		// Pair is an object that contains the (index,response_value) of each hidden neurons.
		Pair[] sortArray = new Pair[numNeurons]; 
        //copy the z neurons' preResponses to the new pair array and a new array
		for (int i = 0; i < numNeurons; i++) {
			sortArray[i] = new Pair(i, motorNeurons[i].getnewresponse());
//          System.out.println("Motor responses before topK: " + newResponse[i]);
			motorNeurons[i].setnewresponse(0.0f);
		}
		
		// Sort the array of Pair objects by its response_value in non-increasing order.
		// The index is in the Pair, goes with the response ranked.
		topKSort(sortArray, topK);
		

		//System.out.println("Motor top1 value: " + sortArray[0].value);

		// Find the top1 element and set to one.
		/*
		int topIndex = sortArray[0].get_index();							
		newResponse[topIndex] = 1.0f;
		System.out.println("TopIndex: " + topIndex + " , " + "newResponse value: " + newResponse[topIndex]);
		*/
		
		// binary conditioning for the topK neurons.		
		int winnerIndex = 0;		
		while(winnerIndex < topK){			
			// get the index of the top element.
			int topIndex = sortArray[winnerIndex].index;	
			//set new response value for winner neuron
			motorNeurons[topIndex].setnewresponse(1.0f); 			
			winnerIndex++;			
		}				
	}
	
	//do the top-k competition
	private void truetopKCompetition(){		
		int winnerIndex = 0;
		
		float[] copyArray = new float[numNeurons];
		// Pair is an object that contains the (index,response_value) of each hidden neurons.
		Pair[] sortArray = new Pair[numNeurons]; 
        //copy the z neurons' preResponses to the new pair array and a new array
		for (int i = 0; i < numNeurons; i++) {
			sortArray[i] = new Pair(i, motorNeurons[i].getnewresponse());
			copyArray[i] = motorNeurons[i].getnewresponse();
//          System.out.println("Motor responses before topK: " + newResponse[i]);
			motorNeurons[i].setnewresponse(0.0f);
		}
		
		// Sort the array of Pair objects by its response_value in non-increasing order.
		// The index is in the Pair, goes with the response ranked.
		topKSort(sortArray, topK);
		
		if(sortArray[0].value < 0.9f && usedMotorNeurons < numNeurons){ // add one more neuron.
//			System.out.println("This is the new hidden neuron " + usedMotorNeurons);
			motorNeurons[usedMotorNeurons].setnewresponse(1.0f);
			usedMotorNeurons++;
			winnerIndex++;			
		}
		
		// identify the ranks of topk winners
		float value_top1 =  sortArray[0].value;
		float value_topkplus1 = sortArray[topK].value; 
		// binary conditioning for the topK neurons.		
		while(winnerIndex < topK){			
			// get the index of the top element.
			int topIndex = sortArray[winnerIndex].index;	
			//set new response value for winner neuron
			float tempResponse;
			if(value_top1 > value_topkplus1 ){
				tempResponse = ( copyArray[topIndex] - value_topkplus1 ) / (value_top1 - value_topkplus1 );
			}
			else{
				tempResponse = 1;
			}
			motorNeurons[topIndex].setnewresponse(tempResponse); 			
			winnerIndex++;			
		}	
		System.out.println("Used number of motor neuron: "+usedMotorNeurons);
	}

	//set whether do the top-k competition
    public void setisTopk(boolean topk){
    	isTopk = topk;
    }

    //get the height of the motor concept zone
	public int getHeight() {
		return height;
	}

	//set the height of the motor concept zone
	public void setHeight(int height) {
		this.height = height;
	}

	//get the width of the motor concept zone
	public int getWidth() {
		return width;
	}

	//set the width of the motor concept zone
	public void setWidth(int width) {
		this.width = width;
	}

	// convert input into 1-D Array
	public float[] getInput1D() {
		//construct the new 1-D array
		float[] inputArray = new float[height * width];
		//copy the input value to the 1-D array
		for (int i = 0; i < height; i++) {
			System.arraycopy(input[i], 0, inputArray, i * width, width);			
		}		
		return inputArray;
	}

	//set the mode
	public void setMode(boolean Mode) {
		this.mode = Mode;
	}
	
	//get the input array
	public float[][] getInput() {
		return input;
	}

	//set the input array
	public void setInput(float[][] input) {
		this.input = input;
	}

	//get the number of winner
	public int getTopK() {
		return topK;
	}

	//set the number of winner
	public void setTopK(int topK) {
		this.topK = topK;
	}
	
	public void setMotorindex(int index){
		mMotorIndex = index;
	}

	//get the number of z neurons in the motor concept zone
	public int getNumNeurons() {
		return numNeurons;
	}
	
	//get the number of bottom-up connection
	public int getNumBottomUpWeights() {
		return numBottomUpWeights;
	}

	//set the number of bottom-up connection
	public void setNumBottomUpWeights(int numBottomUpWeights) {
		this.numBottomUpWeights = numBottomUpWeights;
	}

	//set the response value
	public void setSupervisedResponse(float[][] supervisedResponse){
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				//calculate the index of neuron
				int index = i*width + j;			
				motorNeurons[index].setnewresponse(supervisedResponse[i][j]);
			}
		}
		
	}

	//the pair class which pairs the value and index
	public class Pair implements Comparable<Pair> {
	    public final int index;
	    public final float value;
        //initialize the value and index pair
	    public Pair(int index, float value) {
	        this.index = index;
	        this.value = value;
	    }

		public int compareTo(Pair other) {
			return -1*Float.valueOf(this.value).compareTo(other.value);
		}
		
		public int get_index(){
			return index;
		}
	}

	//normalize the vector
	public float[] normalize(float[] input, int size, int flag) {		
		  float[] weight = new float[size];
			System.arraycopy(input, 0, weight, 0, size);			
			if (flag ==1){
				float min = weight[0];
				float max = weight[0];			
				for (int i = 0; i < size; i++){
					if(weight[i] < min){min = weight[i];}
					if(weight[i] > max){max = weight[i];}
				}
				
				float diff = max-min + MACHINE_FLOAT_ZERO;			
				for(int i = 0; i < size; i++){
					weight[i] = (weight[i]-min)/diff;
				}		
				
				float mean = 0;
				for (int i = 0; i < size; i++){
					mean += weight[i];
				}
				mean = mean/size;			
				for (int i = 0; i < size; i++){
					weight[i] = weight[i]-mean + MACHINE_FLOAT_ZERO;
				}	
				
				float norm = 0;
				for (int i = 0; i < size; i++){
					norm += weight[i]*weight[i];
				}
				norm = (float) Math.sqrt(norm);
				if (norm > 0){
					for (int i = 0; i < size; i++){
						weight[i] = weight[i]/norm;
						}
				}	
			}
			
			if(flag==2){
				float norm = 0;
				for (int i = 0; i < size; i++){
					norm += weight[i]* weight[i];
				}
				norm = (float) Math.sqrt(norm) + MACHINE_FLOAT_ZERO;
				if (norm > 0){
					for (int i = 0; i < size; i++){
						weight[i] = weight[i]/norm;
						}
				}					
			}
			
			if (flag == 3){
				float norm = 0;
				for (int i = 0; i < size; i++){
					norm += weight[i];
				}
				norm = norm+MACHINE_FLOAT_ZERO;
				if (norm > 0){
					for (int i = 0; i < size; i++){
						weight[i] = weight[i]/norm;
						}
				}		
			}
			return weight;
		}

	//save the weight vectors in the text file
	public void saveWeightToFile(String motor_ind) {
		try {
			PrintWriter wr_weight = new PrintWriter(new File(motor_ind + "bottom_up_weight.txt"));
			for (int i = 0; i < numNeurons; i++){
				for (int j = 0; j < motorNeurons[i].getBottomUpWeights().length; j++){
					wr_weight.print(String.format("% .2f", motorNeurons[i].getBottomUpWeights()[j]) + ',');
				}
				wr_weight.println();
			}
			wr_weight.close();
			
			if(numLateralWeights != 0){
				PrintWriter wr_lw = new PrintWriter(new File(motor_ind + "lateral_w.txt"));
				for (int i = 0; i < numNeurons; i++){
					//System.out.println("Neuron"+i+"'s motor lateral size: "+motorNeurons[i].getLateralWeights().length);

					for (int j = 0; j < motorNeurons[i].getLateralWeights().length; j++){
						wr_lw.print(String.format("% .2f", motorNeurons[i].getLateralWeights()[j]) + ',');
					}
					wr_lw.println();
				}
				wr_lw.close();
			}

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	/* This is the protocol for toy data visualization.
	 * All data are transfered as float to save effort in translation.
	 * There are these things to send over socket:
	 *       1. number of neurons in this motor
	 *       2. length of bottom up weights
	 *       3. bottom up weight
	 */    
	public void sendNetworkOverSocket(PrintWriter string_out, DataOutputStream data_out, int display_num, int display_start_id) throws IOException {
		int start_id = display_start_id - 1 ;
		if (start_id < 0) start_id = 0;
		if (start_id >= numBottomUpWeights) start_id = 0;
		int end_id = start_id + display_num;
		if (end_id > numBottomUpWeights) end_id = numBottomUpWeights;
		if (end_id < 0) end_id = numBottomUpWeights;
		
		data_out.writeInt(numNeurons);
		data_out.writeInt(end_id - start_id);
		data_out.writeInt(numLateralWeights);
		for(int i = 0; i < numNeurons; i++){
			for (int j = start_id; j < end_id; j++){
				data_out.writeFloat((float)motorNeurons[i].getBottomUpWeights()[j]);
			}
		}
		
		for (int i = 0; i < numNeurons; i++){
			for (int j = 0; j < numLateralWeights; j++){
				data_out.writeFloat((float)motorNeurons[i].getLateralWeights()[j]);
			}
		}
	}

	//send the weight and response vectors to the GUI
	public float[][] send_motor(int display_num, int display_start_id){
		float[][] temp_weights;
		int start_id = display_start_id - 1 ;
		if (start_id < 0) start_id = 0;
		if (start_id >= numBottomUpWeights) start_id = 0;
		int end_id = start_id + display_num;
		if (end_id > numBottomUpWeights) end_id = numBottomUpWeights;
		if (end_id < 0) end_id = numBottomUpWeights;
		temp_weights = new float[display_num][numBottomUpWeights];
		for(int i = 0; i < numNeurons; i++){
			for (int j = start_id; j < end_id; j++){
				temp_weights[i][j] = motorNeurons[i].getBottomUpWeights()[j];
			}
		}
		return temp_weights;
	}
}
