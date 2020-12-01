package DN2;

import java.io.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;
import MazeInterface.Commons;
import MazeInterface.Commons.env;
import MazeInterface.DNCaller;

public class DN2 implements Serializable{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1;
	private int numSensor;
	private SensorLayer[] sensor;
	
	private int numMotor;
	private MotorLayer[] motor;
	private int ZbottomupFrequency;
	private int motorLateralnum;
	
	private int numPrimaryHidden;
	private int numHidden;
	private PrimaryHiddenLayer[] prihidden;
	private HiddenLayer[] hidden;
	private int[] numHiddenNeurons;
	private int[] mHiddentopk;
	
	private int[] lateralZone;
	private float lateralPercent;
	public FileWriter fw;
	
	public static enum MODE {GROUP, MAZE, Speech};
	private MODE mode;
	//construct DN
	public DN2(int numInput, int[][] inputSize, int numMotor, int[][] motorSize,int[] topKMotor, 
			int numPrimaryHidden, int numHidden, int rfSize, int rfStride, int[][] rf_id_loc, int[] numHiddenNeurons, 
			  int[] topKHidden, float prescreenPercent, int[] typeNum, float[][] growthrate, float[][] meanvalue,
			  float lateralpercent, boolean dynamicInhibition, int[] lateral_zone, int lateral_length, MODE network_mode,int Zfrequency){
		
		// Initialize the layers
		this.mode = network_mode;

		//set z neuron's bottom-up weight normalization frequency
		ZbottomupFrequency = Zfrequency;
		// calculate total sizes of the sensor 
		int totalSensor = totalSize(inputSize);
		// calculate total sizes of the motor
		int totalMotor = totalSize(motorSize);
		//get the number of y neurons
		this.numHiddenNeurons = numHiddenNeurons;
		
		// Initialize the hidden layers
		//get the top-k's k value
		this.mHiddentopk = topKHidden;
		this.numPrimaryHidden = numPrimaryHidden;
		//get the number of y layers
		this.numHidden = numHidden;
		//initialize the hidden-layer array
		hidden = new HiddenLayer[numHidden];
		//initialize each component of hidden-layer array
		this.lateralPercent = lateralpercent;
		int totalPriHidden = 0;
		if(numPrimaryHidden != 0) {
			prihidden = new PrimaryHiddenLayer[numPrimaryHidden];
			for (int i = 0; i < numPrimaryHidden; i++) {
				prihidden[i] = new PrimaryHiddenLayer(4, topKHidden[i], totalSensor, totalMotor, inputSize, 2, meanvalue);     //5 for Audition2
				totalPriHidden = totalSize(prihidden);
			}
		}
		for (int i = 0; i < numHidden; i++) {
			int numPriLateral = 0;
			for(int j = 0; j < numPrimaryHidden; j++){
				numPriLateral += prihidden[j].numNeurons*prihidden[j].mDepth;
			}
			hidden[i] = new HiddenLayer(numHiddenNeurons[i], topKHidden[i], totalSensor, numPriLateral, totalMotor, 
					                    rfSize, rfStride, rf_id_loc, inputSize, prescreenPercent,
					                    typeNum, growthrate, meanvalue, dynamicInhibition);
		}
		

		int totalHidden = totalSize(hidden);
		int totalLateral = totalPriHidden+totalHidden;
		
		// Initialize the sensor layers
		this.numSensor = numInput;
		sensor = new SensorLayer[numSensor];
		
		for (int i = 0; i < numSensor; i++) {
			sensor[i] = new SensorLayer(inputSize[i][0], inputSize[i][1]);
		}
		
		// Initialize the motor layers
		this.numMotor = numMotor;
		motor = new MotorLayer[numMotor];
		lateralZone = lateral_zone;
		for (int i = 0; i < numMotor; i++) {
			boolean contains = arrayContains(lateral_zone, i);
			if (!contains){
			    motor[i] = new MotorLayer(motorSize[i][0], motorSize[i][1], topKMotor[i], totalLateral, 0, ZbottomupFrequency);

			} else {
				motor[i] = new MotorLayer(motorSize[i][0], motorSize[i][1], topKMotor[i], totalLateral, lateral_length, ZbottomupFrequency);
			}
		}
		
		motorLateralnum = 0;
		
	}
	
	public void setGrowthRate(float[][] growth_rate){
		for (int i = 0; i < numHidden; i++){
			hidden[i].setGrowthRate(growth_rate);
		}
	}
	
	public boolean arrayContains(int[] array, int target){
		boolean result = false;
		for (int i = 0; i < array.length; i++) {
			if (array[i] == target) {
				result = true;
				break;
			}
		}
		return result;
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
	
    public void setHiddenBottomupMask(int i, int j, float[] mask){
    	hidden[i].setBottomupMask(mask, j);
    }
    
    public void setHiddenTopdownMask(int i, int j, float[] mask){
    	hidden[i].setTopdownMask(mask, j);
    }
    
    public void computePrimaryHiddenResponse(float[][][] sensorInput, boolean learn_flag){

		// set the input and motors
		float[][] allSensorInput = new float[numSensor][];
		int[] allSensorSize = new int[numSensor];
		for (int i = 0; i < numSensor; i++) {
			sensor[i].setInput(sensorInput[i]);
			
			allSensorInput[i] = sensor[i].getInput1D();
			allSensorSize[i] =  allSensorInput[i].length;
		}
		
		// computes the new response for Y
		for (int i = 0; i < numPrimaryHidden; i++) {			
			prihidden[i].computeBottomUpResponse(allSensorInput, allSensorSize);			

			// local input
			prihidden[i].computeResponse(learn_flag, mode);
			
/*			System.out.println("HiddenResponse");
			displayResponse(hidden[i].getResponse1D());
*/
			
			if (learn_flag)
				prihidden[i].hebbianLearnHidden(inputToWeights(allSensorInput, allSensorSize));
		}

	}
	
	public void savePriBottomWeights(){
		for (int i = 0; i < numPrimaryHidden; i++) {
			prihidden[i].saveWeightToFile("PriHidden"+Integer.toString(i));
		}
	}
	
	public boolean getPrilearningflage(int i, int j) {
		return prihidden[0].hiddenNeurons[i][j].getState();
	}
	
	public float sumPrinewresponses() {
		float[][] allHiddenInput = new float[numPrimaryHidden][];
		float summ = 0;
		for (int i = 0; i < numPrimaryHidden; i++) {
			allHiddenInput[i] = prihidden[i].getNewResponse1D();
			for(int j = 0; j < allHiddenInput[i].length; j++) {
				if(allHiddenInput[i][j] != 0) {
					summ += allHiddenInput[i][j];
				}
			}
		}
		return summ;
	}
	
	public float[][] arraySum(float[][] a, float[][] b, float percent) {
		float[][] sum = new float[a.length][a[0].length];
		for (int i = 0; i < a.length; i++) {			
			for(int j = 0; j < a[i].length; j++) {
				sum[i][j] = percent*a[i][j]+(1-percent)*b[i][j];				
			}
		}
		return sum;
	}
	
	public float sumPrioldresponses() {
		float[][] allHiddenInput = new float[numPrimaryHidden][];
		float summ = 0;
		for (int i = 0; i < numPrimaryHidden; i++) {
			allHiddenInput[i] = prihidden[i].getResponse1D(1.0f);
			for(int j = 0; j < allHiddenInput[i].length; j++) {
				summ += allHiddenInput[i][j];
			}
		}
		return summ;
	}
    
	public void computeHiddenResponse(float[][][] sensorInput, float[][][] motorInput, boolean learn_flag,
			env current_type, int current_loc, int current_scale) {
		DNCaller.curr_loc = current_loc;
		DNCaller.curr_scale = current_scale;
		DNCaller.curr_type = current_type.ordinal();
		computeHiddenResponse(sensorInput, motorInput, learn_flag, current_loc, current_scale);
	}	
	
	public void computeHiddenResponse(float[][][] sensorInput, float[][][] motorInput, boolean learn_flag){
		if (mode == DN2.MODE.MAZE){
			DNCaller.curr_loc = -1;
			DNCaller.curr_scale = -1;
			DNCaller.curr_scale = -1;
		}
		computeHiddenResponse(sensorInput, motorInput, learn_flag, -1, -1);
	}
	
	public void computeHiddenResponseParallel(float[][][] sensorInput, float[][][] motorInput, boolean learn_flag){

		computeHiddenResponseParallel(sensorInput, motorInput, learn_flag, -1, -1);
	}
	
	public void computeHiddenResponse(float[][][] sensorInput, float[][][] motorInput, boolean learn_flag, int rf_loc, int rf_size){

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
			
			if(hidden[i].isPriLateral){
				float[][] allHiddenInput = new float[numPrimaryHidden][];
				int[] allHiddenSize = new int[numPrimaryHidden];
				
				for(int j=0; j<numPrimaryHidden;j++){
					allHiddenInput[j] = prihidden[i].getResponse1D(0.3f);       //0.3 for action
					allHiddenSize[j] = allHiddenInput[i].length;
				}
				hidden[i].setPriLateralvector(inputToWeights(allHiddenInput, allHiddenSize));
			} 
			hidden[i].computeLateralResponse();
			
			int where_id = -1;
			if (mode == DN2.MODE.MAZE){
				where_id = 0; // information already stored in DNCaller.
			}
			if(mode == DN2.MODE.GROUP){
				for (int j = 0; j < allMotorInput[1].length-1; j++){
				    if (allMotorInput[1][j] == 1){
					    where_id = j;
					    break;
				    }				
			    }
			}
			// local input
			hidden[i].computeResponse(where_id, learn_flag, mode);
			
			System.out.println("HiddenResponse");
			displayResponse(hidden[i].getResponse1D());
			
			if (learn_flag) {
				hidden[i].hebbianLearnHidden(inputToWeights(allSensorInput, allSensorSize), 
						                     inputToWeights(allMotorInput, allMotorSize));
			}
			else {
				hidden[i].addFiringAges();
			}
		}

	}
	
	public void computeHiddenResponseParallel(float[][][] sensorInput, float[][][] motorInput, boolean learn_flag, int rf_loc, int rf_size){

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
			
			hidden[i].computeBottomUpResponseInParallel(allSensorInput, allSensorSize);
			hidden[i].computeTopDownResponse(allMotorInput, allMotorSize);
			hidden[i].computeLateralResponse();
			
			int where_id = -1;
			// local input
			hidden[i].computeResponse(where_id, learn_flag, mode);
			
			System.out.println("HiddenResponse");
			displayResponse(hidden[i].getResponse1D());
			
			if (learn_flag) {
				hidden[i].hebbianLearnHiddenParallel(allSensorInput, inputToWeights(allSensorInput, allSensorSize),inputToWeights(allMotorInput, allMotorSize));
			}
			else {
				hidden[i].addFiringAges();
			}
		}

	}

	public void computeHiddenResponse(float[][][] sensorInput,  float[][][] motorInput, boolean learn_flag, int type){

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
			int where_id = -1;

			// local input
			hidden[i].computeResponse(where_id, learn_flag, mode, type);
			
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
	
	// compute motor lateral response only, for planning in DN2
	public float[][] computeMotorLateralResponse (float[] lateral_input, int motorIndex){
		float[][] response;
		motor[motorIndex].computeLateralResponse(lateral_input);
		response = motor[motorIndex].getLateralResponse2D();
		return response;
	}

	// compute motor response only, no update for the weights
	public float[][] computeMotorResponse(int motorIndex){
		
		float[][] response;
		
		// get all hiddenLayer responses
		// update the hidden inputs
		float[][] allHiddenInput = new float[numPrimaryHidden+numHidden][];
		int[] allHiddenSize = new int[numPrimaryHidden+numHidden];
		
		for (int i = 0; i < numPrimaryHidden; i++) {
			allHiddenInput[i] = prihidden[i].getResponse1D(0.3f);    //0.3 for action
			allHiddenSize[i] =  allHiddenInput[i].length;			
		}
		
		for (int i = numPrimaryHidden; i < numPrimaryHidden+numHidden; i++) {
			allHiddenInput[i] = hidden[i].getResponse1D();
			allHiddenSize[i] =  allHiddenInput[i].length;			
		}
		
		// once the final responses are set, compute the motor responses
		motor[motorIndex].computeResponse(inputToWeights(allHiddenInput, allHiddenSize));
		response = motor[motorIndex].getNewMotorResponse2D();
		
		return response;
	}
	
	// compute motor response only, no update for the weights
	public float[][] computeMotorResponse(int motorIndex, boolean learning_flag){
		
		float[][] response;
		
		// get all hiddenLayer responses
		// update the hidden inputs
		float[][] allHiddenInput = new float[numPrimaryHidden+numHidden][];
		int[] allHiddenSize = new int[numPrimaryHidden+numHidden];
		if(numPrimaryHidden > 0){
			for (int i = 0; i < numPrimaryHidden; i++) {
				allHiddenInput[i] = prihidden[i].getResponse1D(0.1f);		 //0.3 for action	
				allHiddenSize[i] =  allHiddenInput[i].length;			
			}
		}
		for (int i = numPrimaryHidden; i < numPrimaryHidden+numHidden; i++) {
			allHiddenInput[i] = hidden[i-numPrimaryHidden].getResponse1D();
			allHiddenSize[i] =  allHiddenInput[i].length;			
		}
		
		
		// once the final responses are set, compute the motor responses
		motor[motorIndex].computeResponse(inputToWeights(allHiddenInput, allHiddenSize));
		response = motor[motorIndex].getNewMotorResponse2D();
		
		if(learning_flag){
			motor[motorIndex].hebbianLearnMotor(inputToWeights(allHiddenInput, allHiddenSize));
		}
		
		return response;
	}
	
	// compute motor response only, no update for the weights
	public float[][] computeMotorResponse(int motorIndex, boolean learning_flag, float percent){
		float[][] old_response = new float[motorLateralnum][];
		float[][] lateral_response;
		float[][] temresponse;
		float[][] response;
		
		// get all hiddenLayer responses
		// update the hidden inputs
		float[][] allHiddenInput = new float[numPrimaryHidden+numHidden][];
		int[] allHiddenSize = new int[numPrimaryHidden+numHidden];
		if(numPrimaryHidden > 0){
			for (int i = 0; i < numPrimaryHidden; i++) {
				allHiddenInput[i] = prihidden[i].getResponse1D(percent);		 //0.3 for action	
				allHiddenSize[i] =  allHiddenInput[i].length;			
			}
		}
		for (int i = numPrimaryHidden; i < numPrimaryHidden+numHidden; i++) {
			allHiddenInput[i] = hidden[i-numPrimaryHidden].getResponse1D();
			allHiddenSize[i] =  allHiddenInput[i].length;			
		}

		boolean contains = arrayContains(lateralZone, motorIndex);
		
		for(int i = 0; i < motorLateralnum; i++){
			old_response[i] = motor[i].getNewMotorResponse1D();
		}
		
		int[] allLateralSize = new int[old_response.length];
		for (int k = 0; k < old_response.length; k++) {
			allLateralSize[k] =  old_response[k].length;			
		}		
		if (contains){			

			lateral_response = computeMotorLateralResponse(inputToWeights(old_response, allLateralSize), motorIndex);
			
			// once the final responses are set, compute the motor responses
			motor[motorIndex].computeResponse(inputToWeights(allHiddenInput, allHiddenSize));
			
			temresponse = motor[motorIndex].getNewMotorResponse2D();
			response = arraySum(lateral_response, temresponse, 0.1f);
		}
		else{
			// once the final responses are set, compute the motor responses
			motor[motorIndex].computeResponse(inputToWeights(allHiddenInput, allHiddenSize));
		
			response = motor[motorIndex].getNewMotorResponse2D();
		}

		
		if(learning_flag){
			motor[motorIndex].hebbianLearnMotor(inputToWeights(allHiddenInput, allHiddenSize));

			updateLateralMotorWeights(motorIndex, inputToWeights(old_response, allLateralSize));
						
		}
		
		return response;
	}
	
	// compute motor response only, no update for the weights
	public float[][][] computeMotorResponse(){
		float[][][] response = new float[numMotor][][];
		
		// get all hiddenLayer responses
		// update the hidden inputs
		float[][] allHiddenInput = new float[numPrimaryHidden+numHidden][];
		int[] allHiddenSize = new int[numPrimaryHidden+numHidden];
		
		for (int i = 0; i < numPrimaryHidden; i++) {
			allHiddenInput[i] = prihidden[i].getResponse1D(0.3f);      //0.3 for action
			allHiddenSize[i] =  allHiddenInput[i].length;			
		}
		
		for (int i = numPrimaryHidden; i < numPrimaryHidden+numHidden; i++) {
			allHiddenInput[i] = hidden[i-numPrimaryHidden].getResponse1D();
			allHiddenSize[i] =  allHiddenInput[i].length;			
		}
		
		// once the final responses are set, compute the motor responses
		for (int i = 0; i < numMotor; i++) {
			motor[i].computeResponse(inputToWeights(allHiddenInput, allHiddenSize));
			response[i] = motor[i].getNewMotorResponse2D();
		}
		
		
		return response;
		
		
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
	
	// Set the new motor response then call motorLearn
		public void updateSupervisedMotorWeights(int motorIndex, float[][] supervisedMotor, float percent){
			
			motor[motorIndex].setSupervisedResponse(supervisedMotor);
			
			
			updateMotorWeights(motorIndex, percent);
		}
	
		// Set the new motor response then call motorLearn
		public void updateSupervisedMotorWeights(int motorIndex, float[][] supervisedMotor, float percent, float[][][] oldMotor){
			
			motor[motorIndex].setSupervisedResponse(supervisedMotor);			
			
			updateMotorWeights(motorIndex, percent);
			
			
			float[][] temPattern = new float[motorLateralnum][];
			int[] allMotorSize = new int[motorLateralnum];
			for(int i = 0; i < motorLateralnum; i++){
				for(int j = 0; j < oldMotor[i][0].length; j++){
					temPattern[i] = oldMotor[i][0];
					allMotorSize[i] = oldMotor[i][0].length;
				}
			}
			
			updateLateralMotorWeights(motorIndex, inputToWeights(temPattern, allMotorSize));
		}

		
	public void replacePriHiddenResponse(){
		for (int i = 0; i < prihidden.length; i++) {
			prihidden[i].replaceHiddenLayerResponse();
		}
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
		float[][] allHiddenInput = new float[numPrimaryHidden+numHidden][];
		int[] allHiddenSize = new int[numPrimaryHidden+numHidden];
		
		for (int i = 0; i < numPrimaryHidden; i++) {
			allHiddenInput[i] = prihidden[i].getResponse1D(0.1f);    //0.3 for action
			allHiddenSize[i] =  allHiddenInput[i].length;			
		}
		
		for (int i = numPrimaryHidden; i < numPrimaryHidden+numHidden; i++) {
			allHiddenInput[i] = hidden[i-numPrimaryHidden].getResponse1D();
			allHiddenSize[i] =  allHiddenInput[i].length;			
		}
		for (int i = 0; i < numMotor; i++) {
			motor[i].hebbianLearnMotor(inputToWeights(allHiddenInput, allHiddenSize));
			
		}
	}
	
	// Does the hebbian learning based on the computed response.
	public void updateMotorWeights(int motorIndex, float percent){
		// get all hiddenLayer responses
		// update the hidden inputs
		float[][] allHiddenInput = new float[numPrimaryHidden+numHidden][];
		int[] allHiddenSize = new int[numPrimaryHidden+numHidden];
		
		for (int i = 0; i < numPrimaryHidden; i++) {
			allHiddenInput[i] = prihidden[i].getResponse1D(percent);       //0.3 for action
			allHiddenSize[i] =  allHiddenInput[i].length;			
		}
		
		for (int i = numPrimaryHidden; i < numPrimaryHidden+numHidden; i++) {
			allHiddenInput[i] = hidden[i-numPrimaryHidden].getResponse1D();
			allHiddenSize[i] =  allHiddenInput[i].length;			
		}
		
		motor[motorIndex].hebbianLearnMotor(inputToWeights(allHiddenInput, allHiddenSize));
	}
	
	
	// Does the hebbian learning based on the computed response.
	public void updateMotorWeights(int motorIndex){
		// get all hiddenLayer responses
		// update the hidden inputs
		float[][] allHiddenInput = new float[numPrimaryHidden+numHidden][];
		int[] allHiddenSize = new int[numPrimaryHidden+numHidden];
		
		for (int i = 0; i < numPrimaryHidden; i++) {
			allHiddenInput[i] = prihidden[i].getResponse1D(0.1f);       //0.3 for action
			allHiddenSize[i] =  allHiddenInput[i].length;			
		}
		
		for (int i = numPrimaryHidden; i < numPrimaryHidden+numHidden; i++) {
			allHiddenInput[i] = hidden[i-numPrimaryHidden].getResponse1D();
			allHiddenSize[i] =  allHiddenInput[i].length;			
		}
		
		motor[motorIndex].hebbianLearnMotor(inputToWeights(allHiddenInput, allHiddenSize));
	}
	
	public void updateLateralMotorWeights(int motorIndex, float[] response, int firing_neuron){
		if (arrayContains(lateralZone, motorIndex) && firing_neuron != Commons.NULLVALUE){
			motor[motorIndex].hebbianLearnLateral(response, firing_neuron);
		}
	} 
	
	public void updateLateralMotorWeights(int motorIndex, float[] response){
		if (arrayContains(lateralZone, motorIndex)){
			motor[motorIndex].hebbianLearnLateral(response);
		}
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
		for (int j = 0; j < size.length; j++) {
			System.arraycopy(input[j], 0, weights, beginIndex, size[j]);
			beginIndex += size[j];
		}
			
		return weights;
		
	}
	
	private int totalSize(PrimaryHiddenLayer[] hidden2){
		int total = 0;
		
		for (int i = 0; i < hidden2.length; i++) {
			total += (hidden2[i].getNumNeurons());
		}
		
		return total;
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
			  hidden[i].saveWeightToFile("maps/hidden" + Integer.toString(i));
		  }
		  for (int i = 0; i < motor.length; i++){
			  System.out.println("Save the "+i+"-th motor layer");
			  motor[i].saveWeightToFile("maps/motor" + Integer.toString(i));
		  }
	}
	
	// Write the network to text files so that we can visualize the weights.
	public void saveMotorToText(){
		  for (int i = 0; i < motor.length; i++){
			  motor[i].saveWeightToFile("Audition_Data/DN2/motor" + Integer.toString(i));
		  }
	}
	
	public void saveAgeToText(int ind){
		  for (int i = 0; i < hidden.length; i++){
			  hidden[i].saveAgeToFile("Audition_Data/DN2/hidden" + Integer.toString(i), Integer.toString(ind));
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
	
	public DN2 deserializeLoad(String name) throws IOException, ClassNotFoundException{
		DN2 new_net = null;
		try{
			FileInputStream fileIn = new FileInputStream("Image_Data/" + name + ".ser");
			ObjectInputStream in = new ObjectInputStream(fileIn);
			new_net = (DN2)in.readObject();
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
			int display_y_zone, int display_y2_zone, int display_num, int display_start_id, 
			int display_z_zone_1, int display_z_zone_2) throws IOException, InterruptedException {
		
		// send out version number: DN-1 = 1, DN-2_CPU_1D = 2, DN-2_GPU_1D = 3, DN-2_CPU_2D = 4
		if (Commons.vision_2D_flag == false){
		    data_out.writeInt(2);
		} else {
			data_out.writeInt(4);
		}
		hidden[0].sendNetworkOverSocket(string_out, data_out, display_y_zone, display_y2_zone, display_num, display_start_id);
		
		// send number of motor
		data_out.writeInt(motor.length);
		for (int i = 0; i < motor.length; i++){
			motor[i].sendNetworkOverSocket(string_out, data_out, display_num, display_start_id);
		}
		data_out.flush();
		System.out.println("Sending: " + Integer.toString(data_out.size()) + " bytes");
	}
	
	public void updateHiddenLocation(){
		float pullrate = 0.1f;
		for(int i = 0; i < numHidden; i++){
			hidden[i].pullneurons(pullrate);
			hidden[i].calcuateNeiDistance();
		}
	}
	
	public void updatePriHiddenLocation(float a){
		float pullrate = a;
		for(int i = 0; i < numPrimaryHidden; i++){
			prihidden[i].pullneurons(pullrate);
		}
	}
	
	public void outputPriHiddenLocation(String a){
		String filename = "Audition_Data/prihlocation"+a+".txt";
		try {
			   PrintWriter wr = new PrintWriter(new File(filename));

//				wr.println("PriHidden Neourons Location");
				for (int i = 0; i < numPrimaryHidden; i++) {
//					wr.println("Pri ßHidden area: " + i);
					for (int j = 0; j < prihidden[i].numNeurons; j++) {
						for (int k = 0; k < prihidden[i].mDepth; k++) {
							float[] temp2 = prihidden[i].hiddenNeurons[j][k].getlocation();
							//wr.println("Neuron "+index+"'s location: ("+temp2[0]+", "+temp2[1]+", "+temp2[2]+")");
							wr.println(temp2[0]+" "+temp2[1]+" "+temp2[2]);
						}
					}
					
					wr.println();
				}
				wr.close();
			}catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}			
	}
	
	public void outputHiddenLocation(String a){
		String filename = "Audition_Data/hlocation"+a+".txt";
		try {
			   PrintWriter wr = new PrintWriter(new File(filename));
			   
/*			    wr.println("Glial cells Location");
			    
					for (int i = 0; i < numHidden; i++) {
						for (int j = 0; j < hidden[i].numGlial; j++) {
							int index = hidden[i].glialcells[j].getindex();
							float[] temp1 = hidden[i].glialcells[j].getlocation();
							//wr.println("Glialcell "+index+"'s location: ("+temp1[0]+", "+temp1[1]+", "+temp1[2]+")");
							wr.println(temp1[0]+", "+temp1[1]+", "+temp1[2]);
						}
						wr.println();
			    }
				wr.println("Hidden Neourons Location");   */
				for (int i = 0; i < numHidden; i++) {
//					wr.println("Hidden area: " + i);
					for (int j = 0; j < hidden[i].usedNeurons; j++) {
						int index = hidden[i].hiddenNeurons[j].getindex();
						float[] temp2 = hidden[i].hiddenNeurons[j].getlocation();
						//wr.println("Neuron "+index+"'s location: ("+temp2[0]+", "+temp2[1]+", "+temp2[2]+")");
						wr.println(temp2[0]+" "+temp2[1]+" "+temp2[2]);
					}
					
					wr.println();
				}
				wr.close();
			}catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}			
	}

	public void setMotorisTopk(boolean isTopk){
		for(int i=0; i<motor.length; i++){
			motor[i].setisTopk(isTopk);
		}
	}
	
	public void setMotorCompitition(int index){
	    		motor[index].setMode(true);
	}
	
	public void setNeuronGrow(int type, boolean d){
		for(int i=0; i<hidden.length; i++){
			   hidden[i].setNeuronGrowthrate(type, d);;
			   }
	}
	public void setConcept(boolean c3){
		for(int i=0; i<hidden.length; i++){
		   hidden[i].setConcept(c3);
		   }
	}
	
	public int getHiddenWinner(){
		return hidden[0].getWinner();
	}
	
	public void setLearning(int index, boolean l){
		for(int i=0; i<hidden.length; i++){
			
			hidden[i].setNeuronLearning(index, l);
		  
		}
	}

	
	public float[][][] send_y_location(){
		float[][][] temp = new float[numHidden][hidden[0].getUsedHiddenNeurons()][3];
		for (int i = 0; i < numHidden; i++) {
			for (int j = 0; j < hidden[i].getUsedHiddenNeurons(); j++) {
				for(int k = 0; k < 3; k++){
				 temp[i][j][k] = hidden[i].hiddenNeurons[j].getlocation()[k]/9.0f;}
				}
			}
		return temp;
	}
	
	public float[][][] send_z_location(){
		float[][][] temp = new float[numMotor][][];
		for (int i = 0; i < numMotor; i++){
			temp[i] = new float[motor[i].getNumNeurons()][3];
		}
		for (int i = 0; i < numMotor; i++) {
			for (int j = 0; j < motor[i].getNumNeurons(); j++) {
				for(int k = 0; k < 3; k++){
				 temp[i][j][k] = Math.min((motor[i].motorNeurons[j].getlocation()[k]+5.0f),12.0f)/15.0f;}
				}
			}
		return temp;
	}
	
	public float[][] send_y_bottomup_weights(int num, int start_id){
		
		float[][]temp = hidden[0].send_y(num, start_id, 1); 
		normalize(temp);
		return temp;
	}	
	
	public float[][] send_y_topdown_weights(int num, int start_id){
		
		float[][] temp = hidden[0].send_y(num, start_id, 5); 
		normalize(temp);
		return temp;
	}	
	
	public float[][] send_y_lateral_weights(int num, int start_id){
		
		float[][] temp = hidden[0].send_y(num, start_id, 9); 
		normalize(temp);
		return temp;
	}	
	
	public float[][] send_y_lateral_masks(int num, int start_id){
		
		float[][] temp = hidden[0].send_y(num, start_id, 11); 
		return temp;
	}	
	
	public float[][] send_y_inhibition_weights(int num, int start_id){
		
		float[][] temp = hidden[0].send_y(num, start_id, 13); 
		normalize(temp);
		return temp;
	}	
	
	public float[][] send_y_inhibition_masks(int num, int start_id){
		
		float[][] temp = hidden[0].send_y(num, start_id, 14); 
//		normalize(temp);
		return temp;
	}
	
	//send y neurons' responses to GUI
	public float[] send_y_response(int num, int start_id){
        //get the responses vector
		float[] temp = hidden[0].getResponse1D(); 
		//normalize to [0,1]
		normalize2(temp);
		return temp;
	}
	
	//send z neurons' bottom-up weights to GUI
	public float[][] send_z_bottomup_weights(int start_id){
		int num = 0;
		//calculate the number of all z neurons
		for(int i=0; i<numMotor; i++){
			num += motor[i].getNumNeurons();
		}
		//initialize the weights matrix
		float[][] temp = new float[num][];
		
		int index=0;
		//copy weights from z neurons to weights matrix
		for(int i=0; i<numMotor; i++){
			 System.arraycopy(motor[i].send_motor(motor[i].getNumNeurons(), 1), 0, temp, index , motor[i].getNumNeurons());
			 index += motor[i].getNumNeurons();
		}
        for(int i = 0; i < num; i++){
        	normalize2(temp[i]);
        }
		return temp;
	}
	
	public float[][] send_z_bottomup_weihts(int num, int start_id){
		
		float[][]temp = motor[0].send_motor(num, start_id); 
		normalize(temp);
		return temp;
	}
	
	public int[] getUsedNeurons(){
		int[] used_count = new int[numHidden];
		for (int i = 0; i < numHidden; i++){
			used_count[i] = hidden[i].getUsedHiddenNeurons();
		}
		return used_count;
	}
	
	public void normalize(float[][] weight){
		float min = weight[0][0];
		float max = weight[0][0];			
		for (int i = 0; i < weight.length; i++){
			for (int j = 0; j < weight[0].length; j++){
			    if(weight[i][j] < min){min = weight[i][j];}
			    if(weight[i][j] > max){max = weight[i][j];}
			}
		}
		
		float diff = max-min + 0.0001f;			
		for(int i = 0; i < weight.length; i++){
			for(int j = 0; j < weight[0].length; j++){
			    weight[i][j] = (weight[i][j]-min)/diff;
			}
		}
	}
	
	//normalize vector to [0,1]
	public void normalize2(float[] weight){
		float min = weight[0];
		float max = weight[0];
		//find max and min value
		for (int i = 0; i < weight.length; i++){
			    if(weight[i] < min){min = weight[i];}
			    if(weight[i] > max){max = weight[i];}			
		}
		//calculate the difference between max and min value
		float diff = max-min + 0.0001f;			
		for(int i = 0; i < weight.length; i++){
			    weight[i] = (weight[i]-min)/diff;
		}
	}
	
	
	public int[] getConnections() {
		int[] result = new int[numHidden];
		for(int i=0; i<numHidden; i++) {
			result[i] = hidden[i].sumConnections();
		}
		return result;
	}
}
