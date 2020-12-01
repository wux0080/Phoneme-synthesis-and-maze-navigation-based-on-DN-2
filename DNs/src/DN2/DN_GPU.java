package DN2;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;

import DN2.DN2.MODE;
import MazeInterface.Commons;

import java.io.DataOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Random;

import javax.management.RuntimeErrorException;

import static java.lang.System.*;
import static com.jogamp.opencl.CLMemory.Mem.*;
import static java.lang.Math.*;

// We are going to make the DN_GPU structure really flat, since OPENCL does not support
// classes and such. Thus all weights and all the ages would be visible inside DN_GPU.
// There will be no hidden layer objects,  but only the weights.
public class DN_GPU {
	private boolean DEBUG_FLAG = false;
	private final float MACHINE_FLOAT_ZERO = 0.000001f;
	private int inputNum;      // number of input zones.
	private int[][] inputSize; // size of each input zone.
	private int motorNum;      // number of motors.
	private int[][] motorSize; // size of each motor.
	private int[] topKMotor;   // K value for each motor.
	private int numHiddenNeurons; // determines the number of hidden neurons
	                              // used for this network.
	                              // currently should only be an array of
	                              // length 1.
	private int topKHidden; // top k value for each hidden layer.
	// currently should only be an array of length 1.
	private int usedNeuronNum;

	// hidden layer
	private CLBuffer<IntBuffer> hiddenAge;
	private CLBuffer<IntBuffer> hiddenNeuronTypeIndex; 
	
	// bottom up connection
	private int hiddenBottomUpInputLength;
	private CLBuffer<FloatBuffer> hiddenBottomUpWeight;
	private CLBuffer<FloatBuffer> hiddenBottomUpWeightMask; //mask for hidden layer bottom up weights
	private CLBuffer<FloatBuffer> hiddenBottomUpResponse;
	private CLBuffer<FloatBuffer> hiddenBottomUpInput;
	private CLBuffer<FloatBuffer> hiddenBottomUpWeightDifference;
	private CLBuffer<IntBuffer>   hiddenBottomUpWeightAge;
	
	// top down connection
	private int hiddenTopDownInputLength;
	private CLBuffer<FloatBuffer> hiddenTopDownWeight;
	private CLBuffer<FloatBuffer> hiddenTopDownWeightMask;
	private CLBuffer<FloatBuffer> hiddenTopDownWeightDifference;
	private CLBuffer<IntBuffer>   hiddenTopDownWeightAge;
	private CLBuffer<FloatBuffer> hiddenTopDownResponse;
	private CLBuffer<FloatBuffer> hiddenTopDownInput;
	
	// lateral connection
	private int hiddenLateralInputLength;
	private CLBuffer<FloatBuffer> hiddenLateralWeight;
	private CLBuffer<FloatBuffer> hiddenLateralResponse;
	private CLBuffer<IntBuffer>   hiddenLateralWeightAge;
	private CLBuffer<FloatBuffer> hiddenLateralWeightMask;
	
	// top-K related variables
	private CLBuffer<IntBuffer>   hiddenIntermediateTopkId; // stores top K+1 value and Id.
	private CLBuffer<FloatBuffer> hiddenIntermediateTopkValue;
	private CLBuffer<IntBuffer>   hiddenUsedNeuronNum;
	private CLBuffer<FloatBuffer> hiddenTopKValueType5;
	private CLBuffer<IntBuffer>   hiddenTopKIdType5;
	private CLBuffer<FloatBuffer> hiddenTopKValueType3;
	private CLBuffer<IntBuffer>   hiddenTopKIdType3;

	// final response
	private CLBuffer<FloatBuffer> hiddenPreResponseType5;
	private CLBuffer<FloatBuffer> hiddenPreResponseType3;
	private CLBuffer<FloatBuffer> hiddenFinalResponseNew;
	private CLBuffer<FloatBuffer> hiddenFinalResponseOld;
	private float[][] growthRate;
	private boolean[] typeFlag;
	private int smFlag = 0;
	private float almostPerfectResponseType5 = 1.99999f;
	private float almostPerfectResponseType3 = 1.99999f;
	
	// motor layer
	private int motorBottomUpInputLength;
	private CLBuffer<FloatBuffer> motorBottomUpWeight;
	private CLBuffer<FloatBuffer> motorBottomUpWeightMask;
	private CLBuffer<IntBuffer> motorAge;
	private CLBuffer<FloatBuffer> motorResponseNew;
	private CLBuffer<FloatBuffer> motorResponseOld;
	private float[][][] finalMotorResponse;
	private CLBuffer<FloatBuffer> motorBottomUpWeightDifference;
	private int           motorLateralLength;
	private float[][][][] motorLateralWeights;
	private int[][][]     motorLateralAge;
	private float[][][]   motorLateralResponse;

	// opencl related stuff.
	private int localWorkSize_hidden_preresponse = 4;
	private int globalWorkSize_hidden_preresponse; // decided by neuron number;
	private int localWorkSize_hidden_addition = 256;
	private int globalWorkSize_hidden_addition;
	private int localWorkSize_hidden_topK = 32;
	private int globalWorkSize_hidden_topK = 32;
	private int localWorkSize_hidden_hebbian = 32;
	private int globalWorkSize_hidden_hebbian;    // decided by the K value;
	private int localWorkSize_motor_preresponse = 32; // generally we have fewer neurons in motor.
	private int globalWorkSize_motor_preresponse; // decided by neuron number;
	private int localWorkSize_motor_hebbian = 32;
	private int globalWorkSize_motor_hebbian;    // decided by the firing value in motor;
	private float buttom_up_threshhold = 0f;
	private float top_down_threshhold = 0f;
	private float lateral_threshold = 0f;
	private CLContext context;
	private CLProgram program_preresponse;
	private CLProgram program_add;
	private CLProgram program_topK;
	private CLProgram program_hebbian_hidden;
	private CLProgram program_hebbian_motor;
	private CLKernel kernel_preresponse;
	private CLKernel kernel_add;
	private CLKernel kernel_topK;
	private CLKernel kernel_hebbian_hidden;
	private CLKernel kernel_hebbian_motor;
	private CLCommandQueue queue;

	public DN_GPU(int numInput, int[][] inputSize, int numMotor, int[][] motorSize, int[] topKMotor, int numHidden,
			int rfSize, int rfStrde, int[][] rf_id_loc, int[] numHiddenNeurons, int[] topKHidden, float preprescreenPercent,
			int[] typeNum, float[][] growthrate, float[][] meanvalue, float lateralpercent, boolean dynamicInhibition, 
			int[] lateral_zone, int lateral_length, MODE network_mode, int Zfrequency) {
		inputNum = numInput;
		this.inputSize = new int[inputNum][];
		hiddenBottomUpInputLength = 0;
		for (int i = 0; i < inputNum; i++) {
			this.inputSize[i] = new int[2];
			this.inputSize[i][0] = inputSize[i][0];
			this.inputSize[i][1] = inputSize[i][1];
			hiddenBottomUpInputLength += inputSize[i][0] * inputSize[i][1];
		}

		motorNum = numMotor;
		this.motorSize = new int[motorNum][];
		hiddenTopDownInputLength = 0;
		for (int i = 0; i < motorNum; i++) {
			this.motorSize[i] = new int[2];
			this.motorSize[i][0] = motorSize[i][0];
			this.motorSize[i][1] = motorSize[i][1];
			hiddenTopDownInputLength += motorSize[i][0] * motorSize[i][1];
		}
		
		motorLateralLength = lateral_length;
		motorLateralWeights = new float[motorNum][][][];
		motorLateralAge     = new int[motorNum][][];
		motorLateralResponse = new float[motorNum][][];
		for (int i = 0; i < motorNum; i++){
			motorLateralWeights[i] = new float[motorSize[i][0]][motorSize[i][1]][motorLateralLength];
			motorLateralAge[i] = new int[motorSize[i][0]][motorSize[i][1]];
			motorLateralResponse[i] = new float[motorSize[i][0]][motorSize[i][1]];
		}
		
		this.topKMotor = new int[topKMotor.length];
		for (int i = 0; i < topKMotor.length; i++) {
			this.topKMotor[i] = topKMotor[i];
		}
		
		if (dynamicInhibition == true){
			throw new java.lang.Error("dynamic inhibition not implemented");
		}

		if (numHidden != 1 || numHiddenNeurons.length != 1 || topKHidden.length != 1) {
			throw new java.lang.Error("numHidden > 1 not supported");
		}
		this.numHiddenNeurons = numHiddenNeurons[0];
		this.hiddenLateralInputLength = this.numHiddenNeurons;
		// we are selecting the top k plus one response, neuron with the least response would be scaled to response 0.
		this.topKHidden = topKHidden[0];  
		this.motorBottomUpInputLength = this.numHiddenNeurons;
		usedNeuronNum = 0;
		setGrowthRate(growthrate);
		this.typeFlag = new boolean[typeNum.length + 1];
		for (int i = 0; i < typeNum.length; i++) {
			if (typeNum[i] > 0){
			    this.typeFlag[i + 1] = true;
			} else {
				this.typeFlag[i + 1] = false;
			}
		}

		finalMotorResponse = new float[motorNum][][];
		for(int i = 0; i < motorNum; i++){
			finalMotorResponse[i] = new float[motorSize[i][0]][motorSize[i][1]];
		}

		// initialize things for opencl.
		globalWorkSize_hidden_preresponse = roundUp(localWorkSize_hidden_preresponse, this.numHiddenNeurons);
		globalWorkSize_hidden_addition    = roundUp(localWorkSize_hidden_addition, this.numHiddenNeurons);
		globalWorkSize_hidden_hebbian     = roundUp(localWorkSize_hidden_hebbian, this.topKHidden);
		globalWorkSize_motor_preresponse  = roundUp(localWorkSize_motor_preresponse, this.hiddenTopDownInputLength);
		globalWorkSize_motor_hebbian      = roundUp(localWorkSize_motor_hebbian, this.hiddenTopDownInputLength);

		context = CLContext.create();
		try {
			// Create Kernels
			program_preresponse = context.createProgram(DN_GPU.class.getResourceAsStream("kernels/preResponse.cl")).build();
			program_add = context.createProgram(DN_GPU.class.getResourceAsStream("kernels/vectorAdd.cl")).build();
			program_topK = context.createProgram(DN_GPU.class.getResourceAsStream("kernels/topK.cl")).build();
			program_hebbian_hidden = context.createProgram(DN_GPU.class.getResourceAsStream("kernels/hebbianLearn.cl")).build();
			program_hebbian_motor = context.createProgram(DN_GPU.class.getResourceAsStream("kernels/hebbianLearnNaive.cl")).build();
			kernel_preresponse = program_preresponse.createCLKernel("preResponse");
			kernel_add = program_add.createCLKernel("vectorAdd");
			kernel_topK = program_topK.createCLKernel("topK");
			kernel_hebbian_hidden = program_hebbian_hidden.createCLKernel("hebbianLearn");
			kernel_hebbian_motor = program_hebbian_motor.createCLKernel("hebbianLearnNaive");

			// Create weights, ages, and responses for hidden layer.
			hiddenAge = context.createIntBuffer(this.numHiddenNeurons);
			hiddenNeuronTypeIndex =context.createIntBuffer(this.numHiddenNeurons);
			hiddenBottomUpInput = context.createFloatBuffer(hiddenBottomUpInputLength);
			hiddenBottomUpWeight = context.createFloatBuffer(this.numHiddenNeurons * hiddenBottomUpInputLength);
			hiddenBottomUpWeightMask = context.createFloatBuffer(this.numHiddenNeurons * hiddenBottomUpInputLength);
			hiddenBottomUpWeightAge = context.createIntBuffer(this.numHiddenNeurons * hiddenBottomUpInputLength);
			hiddenBottomUpResponse = context.createFloatBuffer(this.numHiddenNeurons);
			hiddenBottomUpWeightDifference = context.createFloatBuffer(this.numHiddenNeurons * hiddenBottomUpInputLength);
			
			hiddenTopDownInput = context.createFloatBuffer(hiddenTopDownInputLength);
			hiddenTopDownWeight  = context.createFloatBuffer(this.numHiddenNeurons * hiddenTopDownInputLength);
			hiddenTopDownWeightMask = context.createFloatBuffer(this.numHiddenNeurons * hiddenTopDownInputLength);
			hiddenTopDownWeightAge = context.createIntBuffer(this.numHiddenNeurons * hiddenTopDownInputLength);
			hiddenTopDownResponse = context.createFloatBuffer(this.numHiddenNeurons);
			hiddenTopDownWeightDifference = context.createFloatBuffer(this.numHiddenNeurons * hiddenTopDownInputLength);
			
			hiddenLateralWeight = context.createFloatBuffer(this.numHiddenNeurons * hiddenLateralInputLength);
			hiddenLateralResponse = context.createFloatBuffer(this.numHiddenNeurons);
			hiddenLateralWeightAge = context.createIntBuffer(this.numHiddenNeurons * hiddenLateralInputLength);
			hiddenLateralWeightMask = context.createFloatBuffer(this.numHiddenNeurons * hiddenLateralInputLength);
			
			hiddenPreResponseType5 = context.createFloatBuffer(this.numHiddenNeurons);
			hiddenPreResponseType3 = context.createFloatBuffer(this.numHiddenNeurons);
			hiddenFinalResponseNew = context.createFloatBuffer(this.numHiddenNeurons);
			hiddenFinalResponseOld = context.createFloatBuffer(this.numHiddenNeurons);
			hiddenIntermediateTopkId = context.createIntBuffer(globalWorkSize_hidden_topK * (this.topKHidden+1));
			hiddenIntermediateTopkValue = context.createFloatBuffer(globalWorkSize_hidden_topK * (this.topKHidden+1));
			hiddenTopKValueType5 = context.createFloatBuffer(this.topKHidden + 1);
			hiddenTopKIdType5 = context.createIntBuffer(this.topKHidden + 1);
			hiddenTopKValueType3 = context.createFloatBuffer(this.topKHidden + 1);
			hiddenTopKIdType3 = context.createIntBuffer(this.topKHidden + 1);
			hiddenUsedNeuronNum = context.createIntBuffer(1);

			// Create weights, ages, and responses for motor layer.
			motorAge = context.createIntBuffer(this.hiddenTopDownInputLength);
			motorBottomUpWeight = context.createFloatBuffer(this.hiddenTopDownInputLength * motorBottomUpInputLength);
			motorBottomUpWeightMask = context.createFloatBuffer(this.hiddenTopDownInputLength * motorBottomUpInputLength);
			motorResponseNew = context.createFloatBuffer(this.hiddenTopDownInputLength);
			motorResponseOld = context.createFloatBuffer(this.hiddenTopDownInputLength);
			motorBottomUpWeightDifference = context.createFloatBuffer(this.hiddenTopDownInputLength * motorBottomUpInputLength);
			
			// fill all weights, responses, and ages to 0.
			fillBuffer(hiddenAge.getBuffer(),0);
			fillBuffer(hiddenNeuronTypeIndex.getBuffer(),-1); //set neuron type to -1 by default
			
			fillBuffer(hiddenBottomUpWeight.getBuffer(),0);
			fillBuffer(hiddenBottomUpWeightAge.getBuffer(), 0);
			fillBuffer(hiddenBottomUpWeightMask.getBuffer(),1); // set all mask elements to 1
			fillBuffer(hiddenBottomUpWeightDifference.getBuffer(),0);
			fillBuffer(hiddenBottomUpResponse.getBuffer(),0);
			fillBuffer(hiddenBottomUpInput.getBuffer(),0);
			
			fillBuffer(hiddenTopDownWeight.getBuffer(),0);
			fillBuffer(hiddenTopDownWeightAge.getBuffer(),0);
			fillBuffer(hiddenTopDownWeightMask.getBuffer(),1);
			fillBuffer(hiddenTopDownWeightDifference.getBuffer(), 0);
			fillBuffer(hiddenTopDownResponse.getBuffer(),0);
			fillBuffer(hiddenTopDownInput.getBuffer(),0);
			
			fillBuffer(hiddenLateralWeight.getBuffer(),0);
			fillBuffer(hiddenLateralResponse.getBuffer(),0);
			fillBuffer(hiddenLateralWeightAge.getBuffer(),0);
			fillBuffer(hiddenLateralWeightMask.getBuffer(), 1);
			
			fillBuffer(hiddenPreResponseType5.getBuffer(),0);
			fillBuffer(hiddenPreResponseType3.getBuffer(),0);
			fillBuffer(hiddenFinalResponseNew.getBuffer(),0);
			fillBuffer(hiddenFinalResponseOld.getBuffer(),0);
			fillBuffer(hiddenUsedNeuronNum.getBuffer(),0);
			
			fillBuffer(motorAge.getBuffer(),0);
			fillBuffer(motorBottomUpWeight.getBuffer(),0);
			fillBuffer(motorBottomUpWeightMask.getBuffer(),1);
			fillBuffer(motorResponseNew.getBuffer(),0);
			fillBuffer(motorResponseOld.getBuffer(),0);
			fillBuffer(motorBottomUpWeightDifference.getBuffer(),0);
			queue = context.getMaxFlopsDevice().createCommandQueue();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	public void computeHiddenResponse(float[][][] sensorInput, float[][][] motorInput, boolean learning_flag) {
		// Bottom up response calculation.
		float[] bottomUpInput = convertInputTo1D(sensorInput);
		copyInputToBuffer(bottomUpInput, hiddenBottomUpInput.getBuffer());
		kernel_preresponse.setArg(0, hiddenBottomUpInput);
		kernel_preresponse.setArg(1, hiddenBottomUpWeight);
		kernel_preresponse.setArg(2, hiddenBottomUpWeightMask); 
		kernel_preresponse.setArg(3, hiddenBottomUpResponse);
		kernel_preresponse.setArg(4, usedNeuronNum);
		kernel_preresponse.setArg(5, hiddenBottomUpInputLength);
		queue.putWriteBuffer(hiddenBottomUpInput, false)
		.putWriteBuffer(hiddenBottomUpWeight, false)
		.putWriteBuffer(hiddenBottomUpWeightMask, false)
		.putWriteBuffer(hiddenBottomUpResponse, true)
		.put1DRangeKernel(kernel_preresponse, 0, globalWorkSize_hidden_preresponse, 0)
		.putReadBuffer(hiddenBottomUpResponse, true)
		.putReadBuffer(hiddenBottomUpWeight, true)
		.putReadBuffer(hiddenBottomUpWeightMask, true)
		.putReadBuffer(hiddenBottomUpInput, true);
		debug("after bottom up response");

		// Top down response calculation.
		float[] topDownInput = convertInputTo1D(motorInput);
		copyInputToBuffer(topDownInput, hiddenTopDownInput.getBuffer());
		kernel_preresponse.setArg(0, hiddenTopDownInput);
		kernel_preresponse.setArg(1, hiddenTopDownWeight);
		kernel_preresponse.setArg(2, hiddenTopDownWeightMask);
		kernel_preresponse.setArg(3, hiddenTopDownResponse);
		kernel_preresponse.setArg(4, usedNeuronNum);
		kernel_preresponse.setArg(5, hiddenTopDownInputLength);
		queue.putWriteBuffer(hiddenTopDownInput, false)
		.putWriteBuffer(hiddenTopDownWeight, false)
		.putWriteBuffer(hiddenTopDownResponse, true)
		.putWriteBuffer(hiddenTopDownWeightMask, false)
		.put1DRangeKernel(kernel_preresponse, 0, globalWorkSize_hidden_preresponse, 0)
		.putReadBuffer(hiddenTopDownResponse, true)
		.putReadBuffer(hiddenTopDownWeight, true)
		.putReadBuffer(hiddenTopDownWeightMask, true)
		.putReadBuffer(hiddenTopDownInput, true);
		queue.finish();
		debug("after top down response");
		
		// Lateral response calculation.
		kernel_preresponse.setArg(0, hiddenFinalResponseOld);
		kernel_preresponse.setArg(1, hiddenLateralWeight);
		kernel_preresponse.setArg(2, hiddenLateralWeightMask);
		kernel_preresponse.setArg(3, hiddenLateralResponse);
		kernel_preresponse.setArg(4, usedNeuronNum);
		kernel_preresponse.setArg(5, hiddenLateralInputLength);
		queue.putWriteBuffer(hiddenFinalResponseOld, false)
		.putWriteBuffer(hiddenLateralWeight, false)
		.putWriteBuffer(hiddenLateralWeightMask, false)
		.putWriteBuffer(hiddenLateralResponse, true)
		.put1DRangeKernel(kernel_preresponse, 0, globalWorkSize_hidden_preresponse, 0)
		.putReadBuffer(hiddenFinalResponseOld, true)
		.putReadBuffer(hiddenLateralWeight, true)
		.putReadBuffer(hiddenLateralWeightMask, true)
		.putReadBuffer(hiddenLateralResponse, true);
		queue.finish();
		debug("after lateral response");

		// Pre response = bottom up response + top down response + lateral response;
		kernel_add.setArg(0, hiddenBottomUpResponse);
		kernel_add.setArg(1, hiddenTopDownResponse);
		kernel_add.setArg(2, hiddenLateralResponse);
		kernel_add.setArg(3, hiddenNeuronTypeIndex);
		kernel_add.setArg(4, hiddenPreResponseType3);
		kernel_add.setArg(5, hiddenPreResponseType5);
		kernel_add.setArg(6, hiddenFinalResponseNew);
		kernel_add.setArg(7, usedNeuronNum);
		kernel_add.setArg(8, buttom_up_threshhold);
		kernel_add.setArg(9, top_down_threshhold);
		kernel_add.setArg(10, lateral_threshold);
		queue.putWriteBuffer(hiddenBottomUpResponse, false)
		.putWriteBuffer(hiddenTopDownResponse, false)
		.putWriteBuffer(hiddenLateralResponse, false)
		.putWriteBuffer(hiddenNeuronTypeIndex, false)
		.putWriteBuffer(hiddenPreResponseType3, true)
		.putWriteBuffer(hiddenPreResponseType5, true)
		.putWriteBuffer(hiddenFinalResponseNew, true)
		.put1DRangeKernel(kernel_add, 0, globalWorkSize_hidden_addition, 0)
		.putReadBuffer(hiddenPreResponseType5, true)
		.putReadBuffer(hiddenPreResponseType3, true)
		.putReadBuffer(hiddenBottomUpResponse, true)
		.putReadBuffer(hiddenTopDownResponse, true)
		.putReadBuffer(hiddenLateralResponse, true)
		.putReadBuffer(hiddenNeuronTypeIndex, true)
		.putReadBuffer(hiddenFinalResponseNew, true);
		queue.finish();
		debug("after adding");

		// Top-K and initialize new neurons if not perfect.
		int chunk_size = roundUp(globalWorkSize_hidden_topK, usedNeuronNum)/globalWorkSize_hidden_topK;
		int currentType = 5;
		float currentAlmostPerfect = 0;
		if (growthRate[0][currentType]>= 0.1f) {
			currentAlmostPerfect = growthRate[0][currentType] * almostPerfectResponseType5;
		}
		if (typeFlag[currentType] == true && currentAlmostPerfect > 0){
			if (chunk_size < 1) {chunk_size = 1;}
			kernel_topK.setArg(0, hiddenPreResponseType5);
			kernel_topK.setArg(1, hiddenFinalResponseNew);
			kernel_topK.setArg(2, hiddenNeuronTypeIndex);
			kernel_topK.setArg(3, hiddenIntermediateTopkId);
			kernel_topK.setArg(4, hiddenIntermediateTopkValue);
			kernel_topK.setArg(5, hiddenTopKIdType5);
			kernel_topK.setArg(6, hiddenTopKValueType5);
			kernel_topK.setArg(7, topKHidden);
			kernel_topK.setArg(8, chunk_size);
			kernel_topK.setArg(9, globalWorkSize_hidden_topK);
			kernel_topK.setArg(10, usedNeuronNum);
			kernel_topK.setArg(11, currentAlmostPerfect);
			kernel_topK.setArg(12, numHiddenNeurons);
			kernel_topK.setArg(13, currentType);
			kernel_topK.setArg(14, (learning_flag)?1:0);
			kernel_topK.setArg(15, hiddenUsedNeuronNum);
			queue.putWriteBuffer(hiddenPreResponseType5, true)
			.putWriteBuffer(hiddenFinalResponseNew, true)
			.putWriteBuffer(hiddenIntermediateTopkId, true)
			.putWriteBuffer(hiddenIntermediateTopkValue, true)
			.putWriteBuffer(hiddenTopKIdType5, true)
			.putWriteBuffer(hiddenTopKValueType5, true)
			.putWriteBuffer(hiddenUsedNeuronNum, true)
			.putWriteBuffer(hiddenNeuronTypeIndex, true)
			.put1DRangeKernel(kernel_topK, 0, globalWorkSize_hidden_topK, 0)
			.putReadBuffer(hiddenPreResponseType5, true)
			.putReadBuffer(hiddenFinalResponseNew, true)
			.putReadBuffer(hiddenIntermediateTopkValue, false)
			.putReadBuffer(hiddenIntermediateTopkId, false)
			.putReadBuffer(hiddenTopKValueType5, false)
			.putReadBuffer(hiddenTopKIdType5, false)
			.putReadBuffer(hiddenUsedNeuronNum, false)
			.putReadBuffer(hiddenNeuronTypeIndex, true);
			queue.finish();
			usedNeuronNum = hiddenUsedNeuronNum.getBuffer().get();
			hiddenUsedNeuronNum.getBuffer().rewind();
			debug("after type 5 top k");
		}
		
		currentType = 3;
		currentAlmostPerfect = 0;
		if (growthRate[0][currentType]>= 0.1f) {
			currentAlmostPerfect = growthRate[0][currentType] * almostPerfectResponseType3;
		}
		if (typeFlag[currentType] == true && currentAlmostPerfect > 0){
			kernel_topK.setArg(0, hiddenPreResponseType3);
			kernel_topK.setArg(1, hiddenFinalResponseNew);
			kernel_topK.setArg(2, hiddenNeuronTypeIndex);
			kernel_topK.setArg(3, hiddenIntermediateTopkId);
			kernel_topK.setArg(4, hiddenIntermediateTopkValue);
			kernel_topK.setArg(5, hiddenTopKIdType3);
			kernel_topK.setArg(6, hiddenTopKValueType3);
			kernel_topK.setArg(7, topKHidden);
			kernel_topK.setArg(8, chunk_size);
			kernel_topK.setArg(9, globalWorkSize_hidden_topK);
			kernel_topK.setArg(10, usedNeuronNum);
			kernel_topK.setArg(11, currentAlmostPerfect);
			kernel_topK.setArg(12, numHiddenNeurons);
			kernel_topK.setArg(13, currentType);
			kernel_topK.setArg(14, (learning_flag)?1:0);
			kernel_topK.setArg(15, hiddenUsedNeuronNum);
			queue.putWriteBuffer(hiddenPreResponseType3, true)
			.putWriteBuffer(hiddenFinalResponseNew, true)
			.putWriteBuffer(hiddenIntermediateTopkId, true)
			.putWriteBuffer(hiddenIntermediateTopkValue, true)
			.putWriteBuffer(hiddenTopKIdType3, true)
			.putWriteBuffer(hiddenTopKValueType3, true)
			.putWriteBuffer(hiddenUsedNeuronNum, true)
			.putWriteBuffer(hiddenNeuronTypeIndex, true)
			.put1DRangeKernel(kernel_topK, 0, globalWorkSize_hidden_topK, 0)
			.putReadBuffer(hiddenPreResponseType3, true)
			.putReadBuffer(hiddenFinalResponseNew, true)
			.putReadBuffer(hiddenIntermediateTopkValue, false)
			.putReadBuffer(hiddenIntermediateTopkId, false)
			.putReadBuffer(hiddenTopKValueType3, false)
			.putReadBuffer(hiddenTopKIdType3, false)
			.putReadBuffer(hiddenUsedNeuronNum, false)
			.putReadBuffer(hiddenNeuronTypeIndex, true);
			queue.finish();
		}
		
		usedNeuronNum = hiddenUsedNeuronNum.getBuffer().get();
		hiddenUsedNeuronNum.getBuffer().rewind();
		System.out.println("used neuron number: " + usedNeuronNum);
		int firing_neuron_id;
		firing_neuron_id = hiddenTopKIdType5.getBuffer().get();
		hiddenTopKIdType5.getBuffer().rewind();
		System.out.println("firing neuron id: " + firing_neuron_id);
		debug("after type 3 topK");
		
		// Hebbian learning after top-K competition.
		if (learning_flag){
			currentType = 5;
			currentAlmostPerfect = 0;
			if (growthRate[0][currentType]>= 0.1f) {
				currentAlmostPerfect = growthRate[0][currentType] * almostPerfectResponseType5;
			}
			if (typeFlag[currentType] == true && currentAlmostPerfect > 0){
				kernel_hebbian_hidden.setArg(0, hiddenFinalResponseNew);
				kernel_hebbian_hidden.setArg(1, hiddenNeuronTypeIndex);
				kernel_hebbian_hidden.setArg(2, hiddenTopKIdType5);
				kernel_hebbian_hidden.setArg(3, hiddenTopKValueType5);
				kernel_hebbian_hidden.setArg(4, hiddenBottomUpInput);
				kernel_hebbian_hidden.setArg(5, hiddenTopDownInput);
				kernel_hebbian_hidden.setArg(6, hiddenFinalResponseOld);
				kernel_hebbian_hidden.setArg(7, hiddenBottomUpWeight);
				kernel_hebbian_hidden.setArg(8, hiddenBottomUpWeightAge);
				kernel_hebbian_hidden.setArg(9, hiddenBottomUpWeightMask);
				kernel_hebbian_hidden.setArg(10, hiddenTopDownWeight);
				kernel_hebbian_hidden.setArg(11, hiddenTopDownWeightAge);
				kernel_hebbian_hidden.setArg(12, hiddenTopDownWeightMask);
				kernel_hebbian_hidden.setArg(13, hiddenLateralWeight);
				kernel_hebbian_hidden.setArg(14, hiddenLateralWeightAge);
				kernel_hebbian_hidden.setArg(15, hiddenLateralWeightMask);
				kernel_hebbian_hidden.setArg(16, hiddenAge);
				kernel_hebbian_hidden.setArg(17, hiddenBottomUpWeightDifference);
				kernel_hebbian_hidden.setArg(18, hiddenTopDownWeightDifference);
				kernel_hebbian_hidden.setArg(19, numHiddenNeurons);
				kernel_hebbian_hidden.setArg(20, hiddenBottomUpInputLength);
				kernel_hebbian_hidden.setArg(21, hiddenTopDownInputLength);
				kernel_hebbian_hidden.setArg(22, hiddenLateralInputLength);
				kernel_hebbian_hidden.setArg(23, topKHidden);
				kernel_hebbian_hidden.setArg(24, currentType);
				kernel_hebbian_hidden.setArg(25, smFlag);
				queue.putWriteBuffer(hiddenFinalResponseNew, false)
				.putWriteBuffer(hiddenBottomUpInput, false)
				.putWriteBuffer(hiddenTopDownInput, false)
				.putWriteBuffer(hiddenBottomUpWeight, true)
				.putWriteBuffer(hiddenBottomUpWeightMask, false)
				.putWriteBuffer(hiddenTopDownWeight, true)
				.putWriteBuffer(hiddenTopDownWeightMask, false)
				.putWriteBuffer(hiddenBottomUpWeightDifference, true)
				.putWriteBuffer(hiddenTopDownWeightDifference, true)
				.putWriteBuffer(hiddenLateralWeight, true)
				.putWriteBuffer(hiddenLateralWeightAge, true)
				.putWriteBuffer(hiddenLateralWeightMask, true)
				.putWriteBuffer(hiddenAge, true)
				.putWriteBuffer(hiddenTopKIdType5, false)
				.putWriteBuffer(hiddenTopKValueType5, false)
				.putWriteBuffer(hiddenBottomUpWeightAge, true)
				.putWriteBuffer(hiddenTopDownWeightAge, true)
				.put1DRangeKernel(kernel_hebbian_hidden, 0, globalWorkSize_hidden_hebbian, 0)
				.putReadBuffer(hiddenAge, true)
				.putReadBuffer(hiddenBottomUpWeight, true)
				.putReadBuffer(hiddenBottomUpWeightMask, true)
				.putReadBuffer(hiddenTopDownWeight, true)
				.putReadBuffer(hiddenTopDownWeightMask, true)
				.putReadBuffer(hiddenLateralWeight, true)
				.putReadBuffer(hiddenLateralWeightMask, true)
				.putReadBuffer(hiddenBottomUpWeightDifference, true)
				.putReadBuffer(hiddenTopDownWeightDifference, true)
				.putReadBuffer(hiddenBottomUpWeightAge, true)
				.putReadBuffer(hiddenTopDownWeightAge, true)
				.putReadBuffer(hiddenLateralWeightAge, true);
				
				queue.finish();
				debug("after hebbian learn type 5");
			}
			
			currentType = 3;
			currentAlmostPerfect = 0;
			if (growthRate[0][currentType]>= 0.1f) {
				currentAlmostPerfect = growthRate[0][currentType] * almostPerfectResponseType3;
			}
			if (typeFlag[currentType] == true && currentAlmostPerfect > 0) {
				kernel_hebbian_hidden.setArg(0, hiddenFinalResponseNew);
				kernel_hebbian_hidden.setArg(1, hiddenNeuronTypeIndex);
				kernel_hebbian_hidden.setArg(2, hiddenTopKIdType3);
				kernel_hebbian_hidden.setArg(3, hiddenTopKValueType3);
				kernel_hebbian_hidden.setArg(4, hiddenBottomUpInput);
				kernel_hebbian_hidden.setArg(5, hiddenTopDownInput);
				kernel_hebbian_hidden.setArg(6, hiddenFinalResponseOld);
				kernel_hebbian_hidden.setArg(7, hiddenBottomUpWeight);
				kernel_hebbian_hidden.setArg(8, hiddenBottomUpWeightAge);
				kernel_hebbian_hidden.setArg(9, hiddenBottomUpWeightMask);
				kernel_hebbian_hidden.setArg(10, hiddenTopDownWeight);
				kernel_hebbian_hidden.setArg(11, hiddenTopDownWeightAge);
				kernel_hebbian_hidden.setArg(12, hiddenTopDownWeightMask);
				kernel_hebbian_hidden.setArg(13, hiddenLateralWeight);
				kernel_hebbian_hidden.setArg(14, hiddenLateralWeightAge);
				kernel_hebbian_hidden.setArg(15, hiddenLateralWeightMask);
				kernel_hebbian_hidden.setArg(16, hiddenAge);
				kernel_hebbian_hidden.setArg(17, hiddenBottomUpWeightDifference);
				kernel_hebbian_hidden.setArg(18, hiddenTopDownWeightDifference);
				kernel_hebbian_hidden.setArg(19, numHiddenNeurons);
				kernel_hebbian_hidden.setArg(20, hiddenBottomUpInputLength);
				kernel_hebbian_hidden.setArg(21, hiddenTopDownInputLength);
				kernel_hebbian_hidden.setArg(22, hiddenLateralInputLength);
				kernel_hebbian_hidden.setArg(23, topKHidden);
				kernel_hebbian_hidden.setArg(24, currentType);
				kernel_hebbian_hidden.setArg(25, smFlag);
				queue.putWriteBuffer(hiddenFinalResponseNew, false)
				.putWriteBuffer(hiddenBottomUpInput, false)
				.putWriteBuffer(hiddenTopDownInput, false)
				.putWriteBuffer(hiddenBottomUpWeight, true)
				.putWriteBuffer(hiddenBottomUpWeightMask, false)
				.putWriteBuffer(hiddenTopDownWeight, true)
				.putWriteBuffer(hiddenTopDownWeightMask, false)
				.putWriteBuffer(hiddenBottomUpWeightDifference, true)
				.putWriteBuffer(hiddenTopDownWeightDifference, true)
				.putWriteBuffer(hiddenLateralWeight, true)
				.putWriteBuffer(hiddenLateralWeightAge, true)
				.putWriteBuffer(hiddenLateralWeightMask, true)
				.putWriteBuffer(hiddenAge, true)
				.putWriteBuffer(hiddenTopKIdType3, false)
				.putWriteBuffer(hiddenTopKValueType3, false)
				.putWriteBuffer(hiddenBottomUpWeightAge, true)
				.putWriteBuffer(hiddenTopDownWeightAge, true)
				.put1DRangeKernel(kernel_hebbian_hidden, 0, globalWorkSize_hidden_hebbian, 0)
				.putReadBuffer(hiddenAge, true)
				.putReadBuffer(hiddenBottomUpWeight, true)
				.putReadBuffer(hiddenBottomUpWeightMask, true)
				.putReadBuffer(hiddenTopDownWeight, true)
				.putReadBuffer(hiddenTopDownWeightMask, true)
				.putReadBuffer(hiddenLateralWeight, true)
				.putReadBuffer(hiddenLateralWeightMask, true)
				.putReadBuffer(hiddenBottomUpWeightDifference, true)
				.putReadBuffer(hiddenTopDownWeightDifference, true)
				.putReadBuffer(hiddenBottomUpWeightAge, true)
				.putReadBuffer(hiddenTopDownWeightAge, true)
				.putReadBuffer(hiddenLateralWeightAge, true);
				
				queue.finish();
				debug("after hebbian learn type 3");
			}
		}
	}

	public float[][][] computeMotorResponse() {
		kernel_preresponse.setArg(0, hiddenFinalResponseOld);
		kernel_preresponse.setArg(1, motorBottomUpWeight);
		kernel_preresponse.setArg(2, motorBottomUpWeightMask);
		kernel_preresponse.setArg(3, motorResponseNew);
		kernel_preresponse.setArg(4, hiddenTopDownInputLength);
		kernel_preresponse.setArg(5, numHiddenNeurons);
		queue.putWriteBuffer(hiddenFinalResponseOld, false)
		.putWriteBuffer(motorBottomUpWeight, false)
		.putWriteBuffer(motorBottomUpWeightMask, false)
		.put1DRangeKernel(kernel_preresponse, 0, globalWorkSize_motor_preresponse, 0)
		.putReadBuffer(motorResponseNew, true)
		.putReadBuffer(hiddenFinalResponseOld, true)
		.putReadBuffer(motorBottomUpWeightMask, true);
		
		queue.finish();

		for(int i = 0; i < finalMotorResponse.length; i++) {
			for (int j = 0; j < finalMotorResponse[i].length; j++){
				for (int k = 0; k < finalMotorResponse[i][j].length; k++){
					finalMotorResponse[i][j][k] = motorResponseNew.getBuffer().get();
				}
			}
		}
		motorResponseNew.getBuffer().rewind();
		debug("computing motor response");
		System.out.println(" motor compute response");
		
		for(int i = 0; i < finalMotorResponse.length; i++){
			finalMotorResponse[i][0] = topKCompetition(finalMotorResponse[i][0], 1);
		}

		return finalMotorResponse;
	}

	public void replaceHiddenResponse() {
		queue.putWriteBuffer(hiddenFinalResponseNew, false)
		.putWriteBuffer(hiddenFinalResponseOld, true)
		.putCopyBuffer(hiddenFinalResponseNew, hiddenFinalResponseOld)
		.putReadBuffer(hiddenFinalResponseNew, true)
		.putReadBuffer(hiddenFinalResponseOld, false);
	}

	public void replaceMotorResponse() {
		queue.putWriteBuffer(motorResponseNew, false)
		.putWriteBuffer(motorResponseOld, true)
		.putCopyBuffer(motorResponseNew, motorResponseOld)
		.putReadBuffer(motorResponseNew, true)
		.putReadBuffer(motorResponseOld, false);
	}

	public void updateSupervisedMotorWeights(float[][][] supervisedMotor) {
		float[] motorResponse = convertInputTo1D(supervisedMotor);
		if (motorResponse.length > hiddenTopDownInputLength) {
			return;
		}
		copyInputToBuffer(motorResponse, motorResponseNew.getBuffer());
		kernel_hebbian_motor.setArg(0, motorResponseNew);
		kernel_hebbian_motor.setArg(1, hiddenFinalResponseOld);
		kernel_hebbian_motor.setArg(2, motorBottomUpWeight);
		kernel_hebbian_motor.setArg(3, motorAge); 
		kernel_hebbian_motor.setArg(4, motorBottomUpWeightDifference);
		kernel_hebbian_motor.setArg(5, hiddenTopDownInputLength); 
		kernel_hebbian_motor.setArg(6, numHiddenNeurons); 

		// Hebbian learning in motor area.
		queue.putWriteBuffer(motorResponseNew, false) 
		.putWriteBuffer(hiddenFinalResponseOld, false) 
		.putWriteBuffer(motorBottomUpWeight, true) 
		.putWriteBuffer(motorAge, true)
		.putWriteBuffer(motorBottomUpWeightDifference, true)
		.put1DRangeKernel(kernel_hebbian_motor, 0, globalWorkSize_motor_hebbian, localWorkSize_motor_hebbian) 
		.putReadBuffer(motorAge, true) 
		.putReadBuffer(motorBottomUpWeight, true)
		.putReadBuffer(motorBottomUpWeightDifference, true);
		queue.finish();
		debug("after motor hebbian learn");
	}
	
	private void debug(String string){
		if (DEBUG_FLAG){
			int[] hiddenAge = get1DbufferToArray(this.hiddenAge.getBuffer());
			int[] hiddenNeuronTypeIndex = get1DbufferToArray(this.hiddenNeuronTypeIndex.getBuffer());
			
			float[][] hiddenBottomUpWeight = get2DbufferToArray(this.hiddenBottomUpWeight.getBuffer(), numHiddenNeurons, hiddenBottomUpInputLength);
			float[][] hiddenBottomUpMask = get2DbufferToArray(this.hiddenBottomUpWeightMask.getBuffer(), numHiddenNeurons, hiddenBottomUpInputLength);
			float[][] hiddenBottomUpDiff = get2DbufferToArray(this.hiddenBottomUpWeightDifference.getBuffer(), numHiddenNeurons, hiddenBottomUpInputLength);
			int[][]   hiddenBottomUpAge  = get2DbufferToArray(this.hiddenBottomUpWeightAge.getBuffer(), numHiddenNeurons, hiddenBottomUpInputLength);
			float[]   hiddenBottomUpResponse = get1DbufferToArray(this.hiddenBottomUpResponse.getBuffer());
			float[]   hiddenBottomUpInput = get1DbufferToArray(this.hiddenBottomUpInput.getBuffer());
			

			float[][] hiddenTopDownWeight = get2DbufferToArray(this.hiddenTopDownWeight.getBuffer(), numHiddenNeurons, hiddenTopDownInputLength);
			float[][] hiddenTopDownMask   = get2DbufferToArray(this.hiddenTopDownWeightMask.getBuffer(), numHiddenNeurons, hiddenTopDownInputLength);
			float[][] hiddenTopDownDiff   = get2DbufferToArray(this.hiddenTopDownWeightDifference.getBuffer(), numHiddenNeurons, hiddenTopDownInputLength);
			int[][]   hiddenTopDownAge    = get2DbufferToArray(this.hiddenTopDownWeightAge.getBuffer(), numHiddenNeurons, hiddenTopDownInputLength);
			float[]   hiddenTopDownResponse = get1DbufferToArray(this.hiddenTopDownResponse.getBuffer());
			float[]   hiddenTopDownInput  = get1DbufferToArray(this.hiddenTopDownInput.getBuffer());
			
			float[][] hiddenLateralWeight = get2DbufferToArray(this.hiddenLateralWeight.getBuffer(), numHiddenNeurons, hiddenLateralInputLength);
			float[][] hiddenLateralWeightMask = get2DbufferToArray(this.hiddenLateralWeightMask.getBuffer(), numHiddenNeurons, hiddenLateralInputLength);
			int[][]   hiddenLateralWeightAge = get2DbufferToArray(this.hiddenLateralWeightAge.getBuffer(), numHiddenNeurons, hiddenLateralInputLength);
			float[]   hiddenLateralResponse = get1DbufferToArray(this.hiddenLateralResponse.getBuffer());
			
			int[]   hiddenIntermediateTopkId = get1DbufferToArray(this.hiddenIntermediateTopkId.getBuffer()); 
			float[] hiddenIntermediateTopkValue = get1DbufferToArray(this.hiddenIntermediateTopkValue.getBuffer());
			int[]   hiddenUsedNeuronNum = get1DbufferToArray(this.hiddenUsedNeuronNum.getBuffer());
			float[] hiddenTopKValueType5 = get1DbufferToArray(this.hiddenTopKValueType5.getBuffer());
			int[]   hiddenTopKIdType5 = get1DbufferToArray(this.hiddenTopKIdType5.getBuffer());
			float[] hiddenTopKValueType3 = get1DbufferToArray(this.hiddenTopKValueType3.getBuffer());
			int[]   hiddenTopKIdType3 = get1DbufferToArray(this.hiddenTopKIdType3.getBuffer());

			// final response
			float[] hiddenPreResponseType5 = get1DbufferToArray(this.hiddenPreResponseType5.getBuffer());
			float[] hiddenPreResponseType3 = get1DbufferToArray(this.hiddenPreResponseType3.getBuffer());
			float[] hiddenFinalResponseNew = get1DbufferToArray(this.hiddenFinalResponseNew.getBuffer());
			float[] hiddenFinalResponseOld = get1DbufferToArray(this.hiddenFinalResponseOld.getBuffer());

			// motor layer
			float[][] motorBottomUpWeight = get2DbufferToArray(this.motorBottomUpWeight.getBuffer(), hiddenTopDownInputLength, numHiddenNeurons);
			float[] motorBottomUpMask = get1DbufferToArray(this.motorBottomUpWeightMask.getBuffer());
			int[] motorAge = get1DbufferToArray(this.motorAge.getBuffer());
			float[] motorResponseNew = get1DbufferToArray(this.motorResponseNew.getBuffer());
			float[] motorResponseOld = get1DbufferToArray(this.motorResponseOld.getBuffer());
		
		    System.out.println("debugging: " + string);
		    System.out.println("forward");
		}
	}
	
	// Currently we are sending hiddenLayer[0] over socket.
	public void sendNetworkOverSocket(PrintWriter string_out, DataOutputStream data_out, int display_y_zone,
			int display_y2_zone, int display_num, int display_start_id, int display_z_zone_1, int display_z_zone_2)
			throws IOException, InterruptedException {
		// display_y2_zone is not used in DN_GPU
		// TODO: handle display_y2_zone.
		
		int[] hiddenAge = get1DbufferToArray(this.hiddenAge.getBuffer());
		
		float[][] hiddenBottomUpWeight = get2DbufferToArray(this.hiddenBottomUpWeight.getBuffer(), numHiddenNeurons, hiddenBottomUpInputLength);
		float[][] hiddenBottomUpMask = get2DbufferToArray(this.hiddenBottomUpWeightMask.getBuffer(), numHiddenNeurons, hiddenBottomUpInputLength);
		float[][] hiddenBottomUpDiff = get2DbufferToArray(this.hiddenBottomUpWeightDifference.getBuffer(), numHiddenNeurons, hiddenBottomUpInputLength);
		int[][]   hiddenBottomUpAge  = get2DbufferToArray(this.hiddenBottomUpWeightAge.getBuffer(), numHiddenNeurons, hiddenBottomUpInputLength);
		float[]   hiddenBottomUpResponse = get1DbufferToArray(this.hiddenBottomUpResponse.getBuffer());
		float[]   hiddenBottomUpInput = get1DbufferToArray(this.hiddenBottomUpInput.getBuffer());
		

		float[][] hiddenTopDownWeight = get2DbufferToArray(this.hiddenTopDownWeight.getBuffer(), numHiddenNeurons, hiddenTopDownInputLength);
		float[][] hiddenTopDownMask   = get2DbufferToArray(this.hiddenTopDownWeightMask.getBuffer(), numHiddenNeurons, hiddenTopDownInputLength);
		float[][] hiddenTopDownDiff   = get2DbufferToArray(this.hiddenTopDownWeightDifference.getBuffer(), numHiddenNeurons, hiddenTopDownInputLength);
		int[][]   hiddenTopDownAge    = get2DbufferToArray(this.hiddenTopDownWeightAge.getBuffer(), numHiddenNeurons, hiddenTopDownInputLength);
		float[]   hiddenTopDownResponse = get1DbufferToArray(this.hiddenTopDownResponse.getBuffer());
		float[]   hiddenTopDownInput  = get1DbufferToArray(this.hiddenTopDownInput.getBuffer());
		
		int[]   hiddenIntermediateTopkId = get1DbufferToArray(this.hiddenIntermediateTopkId.getBuffer()); 
		float[]   hiddenIntermediateTopkValue = get1DbufferToArray(this.hiddenIntermediateTopkValue.getBuffer());
		int[]   hiddenUsedNeuronNum = get1DbufferToArray(this.hiddenUsedNeuronNum.getBuffer());
		float[] hiddenTopKValue = get1DbufferToArray(this.hiddenTopKValueType5.getBuffer());
		int[]   hiddenTopKId = get1DbufferToArray(this.hiddenTopKIdType5.getBuffer());

		// final response
		float[] hiddenPreResponse = get1DbufferToArray(this.hiddenPreResponseType5.getBuffer());
		float[] hiddenFinalResponseNew = get1DbufferToArray(this.hiddenFinalResponseNew.getBuffer());
		float[] hiddenFinalResponseOld = get1DbufferToArray(this.hiddenFinalResponseOld.getBuffer());

		// motor layer
		float[][] motorBottomUpWeight = get2DbufferToArray(this.motorBottomUpWeight.getBuffer(), hiddenTopDownInputLength, numHiddenNeurons);
		float[] motorBottomUpMask = get1DbufferToArray(this.motorBottomUpWeightMask.getBuffer());
		int[] motorAge = get1DbufferToArray(this.motorAge.getBuffer());
		float[] motorResponseNew = get1DbufferToArray(this.motorResponseNew.getBuffer());
		float[] motorResponseOld = get1DbufferToArray(this.motorResponseOld.getBuffer());
		
		int start_id = display_start_id - 1;
		if (start_id < 0)
			start_id = 0;
		if (start_id >= hiddenBottomUpWeight.length)
			start_id = 0;
		int end_id = start_id + display_num;
		if (end_id > hiddenBottomUpWeight.length)
			end_id = hiddenBottomUpWeight.length;
		if (end_id < 0)
			end_id = hiddenBottomUpWeight.length;
		
		// send out version number
		data_out.writeInt(3);

		// number of hidden neurons
		data_out.writeInt(end_id - start_id);

		// length of bottom up input
		data_out.writeInt(hiddenBottomUpInputLength);

		// length of topDown input
		data_out.writeInt(hiddenTopDownInputLength);

		// bottom up weight
		if (display_y_zone == 1) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < hiddenBottomUpWeight[0].length; j++) {
					data_out.writeFloat((float) hiddenBottomUpWeight[i][j] * hiddenBottomUpMask[i][j]);
				}
			}
		}

		// bottom up age
		else if (display_y_zone == 2) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < hiddenBottomUpWeight[0].length; j++) {
					data_out.writeFloat((float) hiddenBottomUpAge[i][j]);
				}
			}
		}

		// bottom up mask
		else if (display_y_zone == 3) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < hiddenBottomUpWeight[0].length; j++) {
					data_out.writeFloat((float) hiddenBottomUpMask[i][j]);
				}
			}
		}

		// bottom up diff
		else if (display_y_zone == 4) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < hiddenBottomUpWeight[0].length; j++) {
					data_out.writeFloat((float) hiddenBottomUpDiff[i][j]);
				}
			}
		}

		// topDown weight
		else if (display_y_zone == 5) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < hiddenTopDownWeight[0].length; j++) {
					data_out.writeFloat((float) hiddenTopDownWeight[i][j]);
				}
			}
		}

		// topDown age
		else if (display_y_zone == 6) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < hiddenTopDownWeight[0].length; j++) {
					data_out.writeInt(hiddenTopDownAge[i][j]);
				}
			}
		}

		// topDown mask
		else if (display_y_zone == 7) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < hiddenTopDownWeight[0].length; j++) {
					data_out.writeFloat((float) hiddenTopDownMask[i][j]);
				}
			}
		}

		// topDown variance
		else if (display_y_zone == 8) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < hiddenTopDownWeight[0].length; j++) {
					data_out.writeFloat((float) hiddenTopDownDiff[i][j]);
				}
			}
		}

		// bottom up input
		for (int i = 0; i < hiddenBottomUpInputLength; i++) {
			data_out.writeFloat((float) hiddenBottomUpInput[i]);
		}

		// top down input
		for (int i = 0; i < hiddenTopDownInputLength; i++) {
			data_out.writeFloat((float) hiddenTopDownInput[i]);
		}

		// bottom up response
		for (int i = start_id; i < end_id; i++) {
			data_out.writeFloat((float) hiddenBottomUpResponse[i]);
		}

		// top down response
		for (int i = start_id; i < end_id; i++) {
			data_out.writeFloat((float) hiddenTopDownResponse[i]);
		}

		// final response
		for (int i = start_id; i < end_id; i++) {
			data_out.writeFloat((float) hiddenFinalResponseNew[i]);
		}

		// send number of motor
		data_out.writeInt(motorNum);
		int current_count = 0;
		for (int i_motor = 0; i_motor < motorNum; i_motor++) {
			start_id = display_start_id - 1 ;
			if (start_id < 0) start_id = 0;
			if (start_id >= numHiddenNeurons) start_id = 0;
			end_id = start_id + display_num;
			if (end_id > numHiddenNeurons) end_id = numHiddenNeurons;
			if (end_id < 0) end_id = numHiddenNeurons;
			
			data_out.writeInt(motorSize[i_motor][0] * motorSize[i_motor][1]);
			data_out.writeInt(end_id - start_id);
			for(int i = 0; i < motorSize[i_motor][0] * motorSize[i_motor][1]; i++){
				for (int j = start_id; j < end_id; j++){
					data_out.writeFloat((float)motorBottomUpWeight[current_count][j]);
				}
				current_count ++;
			}
		}
		data_out.flush();
		System.out.println("Sending: " + Integer.toString(data_out.size()) + " bytes");
	}
	
	private float[][] get2DbufferToArray(FloatBuffer buffer, int height, int width){
		float[][] result = new float[height][width];
		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				result[i][j] = buffer.get();
			}
		}
		buffer.rewind();
		return result; 
	}
	
	private int[][] get2DbufferToArray(IntBuffer buffer, int height, int width){
		int[][] result = new int[height][width];
		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				result[i][j] = buffer.get();
			}
		}
		buffer.rewind();
		return result; 
	}
	
	private float[] get1DbufferToArray(FloatBuffer buffer){
		float[] result = new float[buffer.limit()];
		for (int i = 0; i < result.length; i++){
			result[i] = buffer.get();
		}
		buffer.rewind();
		return result;
	}
	
	private int[] get1DbufferToArray(IntBuffer buffer){
		int[] result = new int[buffer.limit()];
		for (int i = 0; i < result.length; i++){
			result[i] = buffer.get();
		}
		buffer.rewind();
		return result;
	}


	private int roundUp(int groupSize, int globalSize) {
		int r = globalSize % groupSize;
		if (r == 0) {
			return globalSize;
		} else {
			return globalSize + groupSize - r;
		}
	}

	public void close(){
		context.release();
	}

	private void copyInputToBuffer(float[] input, FloatBuffer buffer){
		// we first normalize the input.
		float input_sum = 0.000001f;
		for (int i = 0; i < input.length; i++){
			input_sum += input[i] * input[i];
		}
		input_sum = (float) Math.sqrt(input_sum);
		for (int i = 0; i < input.length; i++){
			buffer.put(input[i]/input_sum);
		}
		buffer.rewind();
	}

	private void fillBuffer(FloatBuffer buffer, float value) {
		while (buffer.remaining() != 0)
			buffer.put(value);
		buffer.rewind();
	}

	private void fillBuffer(IntBuffer buffer, int value){
		while (buffer.remaining() != 0)
			buffer.put(value);
		buffer.rewind();
	}

	public static float[] convertInputTo1D(float[][][] input){
		// first we figure out how many elements there are in the input.
		int count = 0;
		for (int i = 0; i < input.length; i++){
			for (int j = 0; j < input[i].length; j++){
				for (int k = 0; k < input[i][j].length; k++){
					count++;
				}
			}
		}
		float[] result = new float[count];
		// now we copy the elements into the result;
		count = 0;
		for (int i = 0; i < input.length; i++){
			for (int j = 0; j < input[i].length; j++){
				for (int k = 0; k < input[i][j].length; k++){
					result[count] = input[i][j][k];
					count ++;
				}
			}
		}
		return result;
	}

	private float[] topKCompetition(float[] response, int topK){
		float max_value;
		int max_id;
		float[] result = new float[response.length];
		for (int k = 0; k < topK; k++) {
			max_value = -1;
			max_id = -1;
			for (int i = 0; i < response.length; i++) {
				if (response[i] > max_value){
					max_value = response[i];
					max_id = i;
				}
			}
			response[max_id] = 0;
			result[max_id] = 1;
		}
		return result;
	}

	public int[] getUsedNeurons() {
		return new int[] {usedNeuronNum};
	}

	public float[][] computeMotorLateralResponse(float[] lateral_input, int current_zone) {
		float[][] newLateralResponse = new float[motorSize[current_zone][0]][motorSize[current_zone][1]];
		for (int i = 0; i < motorSize[current_zone][0]; i++) {
			for (int j = 0; j < motorSize[current_zone][1]; j++) {
				newLateralResponse[i][j] = computeResponse(lateral_input, motorLateralWeights[current_zone][i][j]);
			}
		}
		return newLateralResponse;
	}

	public void updateLateralMotorWeights(int motor_zone, float[] lateral_input, int firing_neuron) {
		if (firing_neuron != Commons.NULLVALUE){
			if (lateral_input.length != motorLateralLength) {
				throw new java.lang.Error("lateral length does not match");
			}
			motorLateralAge[motor_zone][0][firing_neuron] ++ ;
			float learning_rate = getLearningRate(motorLateralAge[motor_zone][0][firing_neuron]);
			for (int i = 0; i < lateral_input.length; i++) {
				motorLateralWeights[motor_zone][0][firing_neuron][i] = (1 - learning_rate) 
						* motorLateralWeights[motor_zone][0][firing_neuron][i] + learning_rate * lateral_input[i];
			}
		}
	}

	public void updateSupervisedMotorWeights(int current_zone, float[][] current_zone_response) {
		float[][][] motorResponse = new float[motorNum][][];
		for (int i = 0; i < motorNum; i++){
			motorResponse[i] = new float[motorSize[i][0]][motorSize[i][1]];
		}
		if (motorResponse[current_zone].length != current_zone_response.length) {
			throw new java.lang.Error("zone length does not match");
		}
		for (int i = 0; i < motorResponse[current_zone].length; i++){
			if (motorResponse[current_zone][i].length != current_zone_response[i].length){
				throw new java.lang.Error("zone length does not match");
			}
			for (int j = 0; j < motorResponse[current_zone][i].length; j++){
				motorResponse[current_zone][i][j] = current_zone_response[i][j];
			}
		}
		updateSupervisedMotorWeights(motorResponse);
	}

	public void setGrowthRate(float[][] growth_table){
        //construct the growth rate array
		growthRate = new float[growth_table.length][];
		for(int i = 0; i < growth_table.length; i++){
			growthRate[i] = new float[growth_table[i].length];
			System.arraycopy(growth_table[i], 0, growthRate[i], 0, growth_table[i].length);
		}
	}
	
	private float getLearningRate(int age){		
		// simple version for learningRate
		return (1.0f / ((float) age));
	}
	
	public float computeResponse(float[] hiddenResponse, float[] bottomUpWeights){		
		normalize(hiddenResponse, hiddenResponse.length, 2);
		normalize(bottomUpWeights, bottomUpWeights.length, 2);
		return dotProduct(hiddenResponse, bottomUpWeights, bottomUpWeights.length);
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
	
	private float dotProduct(float[] a, float[] b, int size){
		float r = 0.0f;
		
		for (int i = 0; i < size; i++) {
			r += a[i] * b[i];
		}
		
		return r;
	}

}
