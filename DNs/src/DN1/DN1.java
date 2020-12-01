package DN1;

import java.io.*;
import java.util.concurrent.TimeUnit;

public class DN1 implements Serializable{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1;
	private int numSensor;
	private SensorLayer[] sensor;
	
	private int numMotor;
	private MotorLayer[] motor;
	
	private int numHidden;
	private HiddenLayer[] hidden;
	
	
	public DN1(int numInput, int[][] inputSize, int numMotor, int[][] motorSize,int[] topKMotor, 
			  int numHidden, int rfSize, int rfStride, int[][] rf_id_loc, int[] numHiddenNeurons, 
			  int[] topKHidden, float prescreenPercent){
		
		// Initialize the layers

		// total sizes of the sensor 
		int totalSensor = totalSize(inputSize);
		int totalMotor = totalSize(motorSize);
		
		// Initialize the hidden layers
		this.numHidden = numHidden;
		hidden = new HiddenLayer[numHidden];
		
		for (int i = 0; i < numHidden; i++) {
			hidden[i] = new HiddenLayer(numHiddenNeurons[i], topKHidden[i], totalSensor, totalMotor, 
					                    rfSize, rfStride, rf_id_loc, inputSize, prescreenPercent);
		}
		
		int totalHidden = totalSize(hidden);
		
		// Initialize the sensor layers
		this.numSensor = numInput;
		sensor = new SensorLayer[numSensor];
		
		for (int i = 0; i < numSensor; i++) {
			sensor[i] = new SensorLayer(inputSize[i][0], inputSize[i][1]);
		}
		
		// Initialize the motor layers
		this.numMotor = numMotor;
		motor = new MotorLayer[numMotor];
		
		for (int i = 0; i < numMotor; i++) {
			motor[i] = new MotorLayer(motorSize[i][0], motorSize[i][1], topKMotor[i], totalHidden);
		}
		
	}
	
	public void setSensorInput(int index, float[][] input){
		sensor[index].setInput(input);
	}
	
	public void setMotorInput(int index, float[][] input){
		motor[index].setInput(input);
	}
	
	public HiddenLayer getHiddenLayer(int i){
		assert(i < hidden.length);
		return hidden[i];
	}
	
	public void computeHiddenResponse(float[][][] sensorInput, float[][][] motorInput, boolean learn_flag ){

		// set the input and motors
		float[][] allSensorInput = new float[numSensor][];
		int[] allSensorSize = new int[numSensor];
		for (int i = 0; i < numSensor; i++) {
			sensor[i].setInput(sensorInput[i]);
			
			allSensorInput[i] = sensor[i].getInput1D();
			allSensorSize[i] =  allSensorInput[i].length;
		}
		

		// update the motor input
		float[][] allMotorInput = new float[numMotor][];
		int[] allMotorSize = new int[numMotor];
		for (int i = 0; i < numMotor; i++) {
			motor[i].setInput(motorInput[i]);
			allMotorInput[i] = motor[i].getInput1D();
			allMotorSize[i] =  allMotorInput[i].length;
		}
		
		// computes the new response for Y
		for (int i = 0; i < numHidden; i++) {
			
			hidden[i].computeBottomUpResponse(allSensorInput, allSensorSize);
			hidden[i].computeTopDownResponse(allMotorInput, allMotorSize);
			int where_id = -1;
			// local input
			if (allMotorInput.length >= 5){
			if (allMotorInput[4][1] == 1){
			for (int j = 0; j < allMotorInput[2].length; j++){
				if (allMotorInput[2][j] == 1){
					where_id = j;
					break;
				}				
			}
			}
			}
			hidden[i].computeResponse(where_id, learn_flag);
			
			System.out.println("HiddenResponse");
			displayResponse(hidden[i].getResponse1D());
			
			if (learn_flag)
				hidden[i].hebbianLearnHidden(inputToWeights(allSensorInput, allSensorSize), 
						                     inputToWeights(allMotorInput, allMotorSize));
		}
	}
	
	private void displayResponse(float[] r) {
			System.out.print(r[0]);
			for (int j = 1; j < r.length; j++) {
				System.out.print("," + r[j]);
			}
			System.out.println();
	}

	// compute motor response only, no update for the weights
	public float[][] computeMotorResponse(int motorIndex){
		
		float[][] response;
		
		// get all hiddenLayer responses
		// update the hidden inputs
		float[][] allHiddenInput = new float[numHidden][];
		int[] allHiddenSize = new int[numHidden];
		
		for (int i = 0; i < numHidden; i++) {
			allHiddenInput[i] = hidden[i].getResponse1D();
			allHiddenSize[i] =  allHiddenInput[i].length;
			
		}
		
		// once the final responses are set, compute the motor responses
		motor[motorIndex].computeResponse(inputToWeights(allHiddenInput, allHiddenSize));
		response = motor[motorIndex].getNewMotorResponse2D();
		
		return response;
	}
	
	// compute motor response only, no update for the weights
	public float[][][] computeMotorResponse(){
		float[][][] response = new float[numMotor][][];
		
		// get all hiddenLayer responses
		// update the hidden inputs
		float[][] allHiddenInput = new float[numHidden][];
		int[] allHiddenSize = new int[numHidden];
		
		for (int i = 0; i < numHidden; i++) {
			allHiddenInput[i] = hidden[i].getResponse1D();
			allHiddenSize[i] =  allHiddenInput[i].length;
			
		}
		
		// once the final responses are set, compute the motor responses
		for (int i = 0; i < numMotor; i++) {
			motor[i].computeResponse(inputToWeights(allHiddenInput, allHiddenSize));
			response[i] = motor[i].getNewMotorResponse2D();
		}
		
		
		return response;
		
		
	}
	
	public void setMotorisTopk(boolean c) {
		for (int i = 0; i < numMotor; i++) {
		    motor[i].setIsTopk(c) ;
		}
	}
	
	// Set the new motor response then call motorLearn
	public void updateSupervisedMotorWeights(float[][][] supervisedMotor){
		
		for (int i = 0; i < supervisedMotor.length; i++) {
			
			motor[i].setSupervisedResponse(supervisedMotor[i]);
		}
		
		updateMotorWeights();
	}
	
	// Set the new motor response then call motorLearn
	public void updateSupervisedMotorWeights(int motorIndex, float[][] supervisedMotor){
		
		motor[motorIndex].setSupervisedResponse(supervisedMotor);
		
		
		updateMotorWeights(motorIndex);
	}
	
	public void replaceHiddenResponse(){
		for (int i = 0; i < hidden.length; i++) {
			hidden[i].replaceHiddenLayerResponse();
		}
	}
	
	public void replaceMotorResponse(){
		for (int i = 0; i < motor.length; i++) {
			motor[i].replaceMotorLayerResponse();
		}
	}
	
	// Does the hebbian learning based on the computed response.
	public void updateMotorWeights(){
		
		// get all hiddenLayer responses
		// update the hidden inputs
		float[][] allHiddenInput = new float[numHidden][];
		int[] allHiddenSize = new int[numHidden];
		
		for (int i = 0; i < numHidden; i++) {
			allHiddenInput[i] = hidden[i].getResponse1D();
			allHiddenSize[i] =  allHiddenInput[i].length;	
		}
		
		for (int i = 0; i < numMotor; i++) {
			motor[i].hebbianLearnMotor(inputToWeights(allHiddenInput, allHiddenSize));
			
		}
	}
	

	// Does the hebbian learning based on the computed response.
	public void updateMotorWeights(int motorIndex){
		
		// get all hiddenLayer responses
		// update the hidden inputs
		float[][] allHiddenInput = new float[numHidden][];
		int[] allHiddenSize = new int[numHidden];
		
		for (int i = 0; i < numHidden; i++) {
			allHiddenInput[i] = hidden[i].getResponse1D();
			allHiddenSize[i] =  allHiddenInput[i].length;	
		}
		
		
		motor[motorIndex].hebbianLearnMotor(inputToWeights(allHiddenInput, allHiddenSize));
			
		
	}
	
	private int maxIndex(float[] values){
		int index = (values[0] != 0.0f) ? 0:-1;
		
		float max = values[0];
		
		for (int i = 0; i < values.length; i++) {
			if(values[i] > max){
				max = values[i];
				index = i;
			}
		}
		
		return index;
	}
	
	private float[] inputToWeights(float[][] input, int[] size){
		
		int total = 0;
		// compute the total size
		for (int i = 0; i < size.length; i++) {
			total += size[i];
		}
		
		float[] weights = new float[total];
			
		int beginIndex = 0;
		for (int j = 0; j < input.length; j++) {
			System.arraycopy(input[j], 0, weights, beginIndex, size[j]);
			beginIndex += size[j];
		}
			
		return weights;
		
	}
	
	private int totalSize(HiddenLayer[] hidden2){
		int total = 0;
		
		for (int i = 0; i < hidden2.length; i++) {
			total += (hidden2[i].getNumNeurons());
		}
		
		return total;
	}

	private int totalSize(int[][] size){
		int total = 0;
		
		for (int i = 0; i < size.length; i++) {
			total += (size[i][0] * size[i][1]);
		}
		
		return total;
	}
	
	// Write the network to text files so that we can visualize the weights.
	public void saveToText(){
		  for (int i = 0; i < hidden.length; i++){
			  hidden[i].saveWeightToFile("network/hidden" + Integer.toString(i));
		  }
		  for (int i = 0; i < motor.length; i++){
			  motor[i].saveWeightToFile("network/motor" + Integer.toString(i));
		  }
	}
	
	public void serializeSave(String name) {
		try{
			File f = new File(name);
		    FileOutputStream fileOut = new FileOutputStream(name);
		    ObjectOutputStream out = new ObjectOutputStream(fileOut);
		    out.writeObject(this);
		    out.close();
		    fileOut.close();
		} catch (IOException i){
			i.printStackTrace();
		}
	}
	
	public DN1 deserializeLoad(String name) throws IOException, ClassNotFoundException{
		DN1 new_net = null;
		try{
			FileInputStream fileIn = new FileInputStream("Image_Data/" + name + ".ser");
			ObjectInputStream in = new ObjectInputStream(fileIn);
			new_net = (DN1)in.readObject();
			in.close();
			fileIn.close();
		}catch(IOException i) {
	         i.printStackTrace();
	         return new_net;
	      }
		return new_net;
	}

	public int getNumSensor() {
		return numSensor;
	}


	public void setNumSensor(int numSensor) {
		this.numSensor = numSensor;
	}


	public int getNumMotor() {
		return numMotor;
	}


	public void setNumMotor(int numMotor) {
		this.numMotor = numMotor;
	}


	public int getNumHidden() {
		return numHidden;
	}


	public void setNumHidden(int numHidden) {
		this.numHidden = numHidden;
	}


	public SensorLayer[] getSensor() {
		return sensor;
	}


	public void setSensor(SensorLayer[] sensor) {
		this.sensor = sensor;
	}


	public MotorLayer[] getMotor() {
		return motor;
	}


	public void setMotor(MotorLayer[] motor) {
		this.motor = motor;
	}


	public HiddenLayer[] getHidden() {
		return hidden;
	}


	public void setHidden(HiddenLayer[] hidden) {
		this.hidden = hidden;
	}
    
	// Currently we are sending hiddenLayer[0] over socket.
	public void sendNetworkOverSocket(PrintWriter string_out, DataOutputStream data_out, 
			int display_y_zone, int display_num, int display_start_id, 
			int display_z_zone_1, int display_z_zone_2) throws IOException, InterruptedException {
		// send network version number: 1
		data_out.writeInt(1);
		
		// send hidden layer information
		hidden[0].sendNetworkOverSocket(string_out, data_out, display_y_zone, display_num, display_start_id);
		
		// send number of motor
		data_out.writeInt(motor.length);
		for (int i = 0; i < motor.length; i++){
			motor[i].sendNetworkOverSocket(string_out, data_out, display_num, display_start_id);
		}
		data_out.flush();
		System.out.println("Sending: " + Integer.toString(data_out.size()) + " bytes");
	}
	
	public int[] getUsedNeurons(){
		int[] used_count = new int[numHidden];
		for (int i = 0; i < numHidden; i++){
			used_count[i] = hidden[i].getUsedHiddenNeurons();
		}
		return used_count;
	}
}
