package DN1;

import java.util.Random;
import java.io.*;
public class MotorLayer implements Serializable {

	private static final long serialVersionUID = 1L;
	private int height;
	private int width;
	
	private float[][] input;
	
	// new variables
	private int topK;
	
	private boolean isTopk;
	private int numNeurons;
	
	private int numBottomUpWeights;
	private float[][] bottomUpWeights;
	
	
	private float[] newResponse;
	private float[] oldResponse;
	private int[] firingAge;
	
	private final float GAMMA = 2000;
	private final float MACHINE_FLOAT_ZERO = 0.00001f;
	private final float TOP_MOTOR_RESPONSE = MACHINE_FLOAT_ZERO; // Firing threshold.
	//private final float TOP_MOTOR_RESPONSE = 0.5f;
	private final int T1 = 20; 
	private final int T2 = 200;
	private final float C = 2;
	
	public MotorLayer(int height, int width, int topK, int hiddenSize){
		this.setWidth(width);
		this.setHeight(height);
		
		this.setTopK(topK);
		isTopk = false;
		
		input = new float[height][width];
		
		numNeurons = height * width;
		
		firingAge = new int[numNeurons];
		newResponse = new float[numNeurons];
		oldResponse = new float[numNeurons];
		
		numBottomUpWeights = hiddenSize;
		bottomUpWeights = new float[numNeurons][hiddenSize];
		
		
		// initialize the weights
		/*
		for (int i = 0; i < numNeurons; i++) {
			initializeWeight(bottomUpWeights[i], hiddenSize);
		}
		*/
	}
	
	public MotorLayer(int width, int height, float[][] input){
		this.setWidth(width);
		this.setHeight(height);
		
		this.setInput(input);
		isTopk = false;
	}
	
	private long seed = 0;
	private Random rand = new Random(seed);
	
	private void initializeWeight(float[] weights, int size){
		for (int i = 0; i < size; i++) {
			weights[i] = rand.nextFloat();
		}
	}
	
	public void hebbianLearnMotor(float[] hiddenResponse){
		for (int i = 0; i < numNeurons; i++) {
			//System.out.println("New Response: " + newResponse[i] + " > " + TOP_MOTOR_RESPONSE);
			
			
			if(newResponse[i] >= TOP_MOTOR_RESPONSE){
				firingAge[i]++;
				
				normalize(hiddenResponse, hiddenResponse.length, 2);
				normalize(bottomUpWeights[i], bottomUpWeights[i].length, 2);
				
				updateWeights(bottomUpWeights[i], hiddenResponse, getLearningRate(firingAge[i]), i);
			}
		}
	}
	
	public float[] getNewMotorResponse1D() {
		float[] inputArray = new float[height * width];
		System.arraycopy(newResponse, 0, inputArray, 0, numNeurons);
		
		return inputArray;
	}
	
	public float[][] getNewMotorResponse2D() {
		float[][] outputArray = new float[height][width];

		for (int i = 0; i < numNeurons; i++) {
			
			outputArray[i/width][i%width] = newResponse[i];
		}
		
		return outputArray;
	}
	
		
	public void replaceMotorLayerResponse(){
		for (int i = 0; i < numNeurons; i++) {
			oldResponse[i] = newResponse[i];
		}
	}
		
	private void updateWeights(float[] weights, float[] input, float learningRate, int neuron_id){
		
		// make sure both arrays have the same length
		assert weights.length == input.length;
		
		for (int i = 0; i < input.length; i++) {
			weights[i] = (1.0f - learningRate) * weights[i] + learningRate * input[i] * newResponse[neuron_id] ;
		}
	}
	
	private float getLearningRate(int age){
		
		// simple version for learningRate
		return (1.0f / ((float) age));
	}
	
	public void computeResponse(float[] hiddenResponse){

		//normalize the hidden response
		normalize(hiddenResponse, hiddenResponse.length, 2);
		
		// do the dot product between the weights
		for (int i = 0; i < numNeurons; i++) {
			normalize(bottomUpWeights[i], bottomUpWeights[i].length, 2);
			
			newResponse[i] = dotProduct(hiddenResponse, bottomUpWeights[i], bottomUpWeights[i].length);

		}
		
		// We are now using continuous motor responses, so removing top-k competition
		if(isTopk) {
		  topKCompetition();
		}
	}
	
	public float[] normalize(float[] weight, int size, int flag) {
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
			norm = (float) Math.sqrt(norm) + MACHINE_FLOAT_ZERO;
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
		return null;
	}
	
	// Sort the topK elements to the beginning of the sort array where the index of the top 
	// elements are still in the pair.
	private static void topKSort(Pair[] sortArray, int topK){
		
		for (int i = 0; i < topK; i++) {
			Pair maxPair = sortArray[i]; 
			int maxIndex = i;
					
			for (int j = i+1; j < sortArray.length; j++) {
				
				if(sortArray[j].value > maxPair.value){ // select temporary max
					maxPair = sortArray[j];
					maxIndex = j;
					
				}
			}
			
			if(maxPair.index != i){
				Pair temp = sortArray[i]; // store the value of pivot (top i) element
				sortArray[i] = maxPair; // replace with the maxPair object.
				sortArray[maxIndex] = temp; // replace maxPair index elements with the pivot. 
			}
		}
	}

	
	private void topKCompetition(){
		
		// Pair is an object that contains the (index,response_value) of each hidden neurons.
		Pair[] sortArray = new Pair[numNeurons]; 

		for (int i = 0; i < numNeurons; i++) {
			sortArray[i] = new Pair(i, newResponse[i]);
			//System.out.println("Motor responses before topK: " + newResponse[i]);
			newResponse[i] = 0.0f;
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
			newResponse[topIndex] = 1.0f;
			
			
			winnerIndex++;
			
		}
		
		
	}
	
	private float dotProduct(float[] a, float[] b, int size){
		float r = 0.0f;
		
		for (int i = 0; i < size; i++) {
			r += a[i] * b[i];
		}
		
		return r;
	}

	public int getHeight() {
		return height;
	}

	public void setHeight(int height) {
		this.height = height;
	}

	public int getWidth() {
		return width;
	}

	public void setWidth(int width) {
		this.width = width;
	}

	// convert into 1d Array
	public float[] getInput1D() {
		float[] inputArray = new float[height * width];
		
		int beginIndex = 0;
		
		for (int i = 0; i < height; i++) {
			System.arraycopy(input[i], 0, inputArray, i * width, width);			
		}
		
		return inputArray;
	}
	
	public float[][] getInput() {
		return input;
	}

	public void setInput(float[][] input) {
		this.input = input;
	}

	public int getTopK() {
		return topK;
	}

	public void setTopK(int topK) {
		this.topK = topK;
	}

	public int getNumBottomUpWeights() {
		return numBottomUpWeights;
	}

	public void setNumBottomUpWeights(int numBottomUpWeights) {
		this.numBottomUpWeights = numBottomUpWeights;
	}

	public float[][] getBottomUpWeights() {
		return bottomUpWeights;
	}

	public void setBottomUpWeights(float[][] bottomUpWeights) {
		this.bottomUpWeights = bottomUpWeights;
	}
	
	public void setIsTopk(boolean c) {
		isTopk = c;
	}

	public void setSupervisedResponse(float[][] supervisedResponse){
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int index = i*width + j;
			
				newResponse[index] = supervisedResponse[i][j];
			}
		}
		
	}
	
	public class Pair implements Comparable<Pair> {
	    public final int index;
	    public final float value;

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

	public void saveWeightToFile(String motor_ind) {
		try {
			PrintWriter wr_weight = new PrintWriter(new File(motor_ind + "bottom_up_weight.txt"));
			PrintWriter wr_age = new PrintWriter(new File(motor_ind + "neuron_age.txt"));
			for (int i = 0; i < numNeurons; i++){
				for (int j = 0; j < bottomUpWeights[0].length; j++){
					wr_weight.print(String.format("% 2.2f", bottomUpWeights[i][j]) + ',');
				}
				wr_weight.println();
				wr_age.print(Integer.toString(firingAge[i]) + ',');
			}
			wr_weight.close();
			wr_age.close();
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
		if (start_id >= bottomUpWeights[0].length) start_id = 0;
		int end_id = start_id + display_num;
		if (end_id > bottomUpWeights[0].length) end_id = bottomUpWeights[0].length;
		if (end_id < 0) end_id = bottomUpWeights[0].length;
		
		data_out.writeInt(bottomUpWeights.length);
		data_out.writeInt(end_id - start_id);
		for(int i = 0; i < bottomUpWeights.length; i++){
			for (int j = start_id; j < end_id; j++){
				data_out.writeFloat((float)bottomUpWeights[i][j]);
			}
		}
	}
}
