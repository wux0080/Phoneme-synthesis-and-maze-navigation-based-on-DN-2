package GUI;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

/*
 * PerformanceMeasure
 * 
 * This class evaluates the error rate of the network. It evaluates the error rate for:
 * a) Timestep (subsequence)  error rate: Represents whether the network's motor response matches with the expected motor response.
 * 						   For this contest the error rate for each timestep will be either zero (computer motor response is different than expected motor response) or one otherwise. 
 * 
 * b) Concept (Motor) error rate: Represents the error from each timestep measured in (a) and averages it out by the number of unsupervised subsequences.
 * 
 * c) Segment error rate:  Adds the error rates of each concept or motor response in (b) and average it out by the number of motors.
 * 
 * d) Total error rate: Adds all error rates measured per segment and averages them out by the number of testing segments.
 * 
 *     
 */
public class PerformanceMeasure {
	
	private int numMotor; // Number of motors the network has. 
	
	private int numSegments; // The number of segments where each segment can be training, resubstitution, or testing.
	
	private int startTestSegment; // Indicate where the training segments ends and the testing or resubstitution set.
	
	private int[] lenSequence; // The number of timesteps or samples each segment has. All segments have the same length.
	
	private float errorRate; // The total error rate of the network's performance.
	
	private float[][] trueAction;
	private float[][] generatedAction;
	private int[][] gAction;
	 
	private float[] errorSegment; // The error rate per segment.
	
	private float[][] errorConcept; // The error rate of each motor per segment.
	
	private float[][][] errorTime; // The error rate of each timestep per concept and per segment.
	
	private int[][] numUnsupervisedInstances; // Count the number of samples that are not supervised.
	
	private int numHidden; // Number of hidden layers the network has.
	
	private int[] numHiddenNeurons; // Number of neurons each hidden layer has.
	private int[] numConnections;
	
	private char modality; // Indicates which modality the network is running.
	
	private int cnum;
	/**
	 * Argument Constructor
	 * <p>
	 * This method initializes all the parameters needed for measuring the performance. 
	 * 
	 * @param Segments
	 * @param motors
	 * @param len
	 * @param start
	 * @param numHidden
	 * @param numHiddenNeurons
	 * @param modality
	 */
	public PerformanceMeasure(int Segments, int motors, int[] len, int start, int numHidden, int[] numHiddenNeurons, char modality){
		
		numSegments = Segments;
		
		startTestSegment = start;
		
		numMotor = motors;
		lenSequence = len;
		
		errorRate = 0.0f;
		errorSegment = new float[numSegments];
		errorConcept = new float[numSegments][numMotor];
		errorTime = new float[numSegments][numMotor][];
		int maxlength = 0;
		for(int i=0; i<lenSequence.length; i++){
			if(lenSequence[i]>maxlength){
				maxlength = lenSequence[i];
			}
		}
		trueAction = new float[numSegments][maxlength];
		generatedAction = new float[numSegments][maxlength];
		gAction = new int[maxlength][numMotor];
		
		for(int i=0; i < numSegments; i++){
			for (int j = 0; j < numMotor; j++) {
				errorTime[i][j] = new float[lenSequence[i]];
			}
		}
		
		numUnsupervisedInstances = new int[numSegments][numMotor];
		
		this.numHidden = numHidden;
		this.numHiddenNeurons = numHiddenNeurons;
		numConnections = new int[numHidden];
		
		this.modality = modality;
		
		cnum = 0;
		
	}
	
	/**
	 * Determine whether the computed motor response is the same as the expected motor response for each (segment, motor, sequence) combination.
	 * 
	 * @param Segment
	 * @param motorIndex
	 * @param sequenceIndex
	 * @param computedMotorResponse
	 * @param expectedMotorResponse
	 */
	public void updateErrorTimeRate(int Segment, int motorIndex, int sequenceIndex, float[][][] computedMotorResponse, float[][][] expectedMotorResponse){
		float errorValue = 0.0f; // assume that both motor elements are the same (zero error).
		for (int i = 0; i < expectedMotorResponse[motorIndex].length; i++) { // The height dimension of each motor
			for (int j = 0; j < expectedMotorResponse[motorIndex][i].length; j++) { // The width dimension of each motor
				if(expectedMotorResponse[motorIndex][i][j] > 0 ){
					if(motorIndex == 60 && (j != 21 && j != 20)){   //change
						cnum = cnum+1;
					}
				
				if(expectedMotorResponse[motorIndex][i][j] != computedMotorResponse[motorIndex][i][j]){
					if(motorIndex == 60){              //need change
						if(j != 21){
							errorValue = 1.0f;
							break;
						}
					}
					else{
						errorValue = 1.0f;
						break;
					}
				}
				}
			}
		}
		
		errorTime[Segment][motorIndex][sequenceIndex] = errorValue;
		
				
		if(motorIndex == 60){      //change
			int true_Action = 0;
			for (int i = 0; i < expectedMotorResponse[motorIndex][0].length; i++){
				if (expectedMotorResponse[motorIndex][0][i] == 1) {true_Action = i; break;}
			}
			int computedAction = 0;
			for (int i = 0; i < computedMotorResponse[motorIndex][0].length; i++){
				if (computedMotorResponse[motorIndex][0][i] == 1) {computedAction = i; break;}
			}
			trueAction[Segment][sequenceIndex] = true_Action;
			generatedAction[Segment][sequenceIndex] = computedAction;
		}
	}
	
	public void updateComputedAction(int motorIndex, int sequenceIndex, float[][][] computedMotorResponse){
		int computedAction = 0;
		for (int i = 0; i < computedMotorResponse[motorIndex][0].length; i++){
			if (computedMotorResponse[motorIndex][0][i] == 1) {computedAction = i+1; break;}
		}
		gAction[sequenceIndex][motorIndex] = computedAction;
	}
	
	/**
	 * This method sums up the error rate of each time step sequence and averages it by the number of unsupervised instances.
	 * 
	 * @param Segment
	 */
	public void updateErrorMotorRate(int Segment){
		
		for (int i = 0; i < numMotor; i++) {
			for (int m = 0; m < lenSequence[Segment]; m++) {
				errorConcept[Segment][i] += errorTime[Segment][i][m];
			}
			
			if(numUnsupervisedInstances[Segment][i] > 0)
				if(i == 60){       
					System.out.println("error nnumber: "+errorConcept[Segment][i]);    //need change
					errorConcept[Segment][i] /= 80;
					System.out.println("total nnumber: "+cnum);					

				}
				else{
					errorConcept[Segment][i] /= ((float) numUnsupervisedInstances[Segment][i]);

				}
		}
	}
	
	/**
	 * This method sums up the error rate of each motor and averages it by the number of motors.
	 * 
	 * @param Segment
	 */
	public void updateErrorSegmentRate(int Segment){
		
		for (int i = 0; i < numMotor; i++) {
			errorSegment[Segment] += errorConcept[Segment][i];
		}
		
		errorSegment[Segment] /= ((float) numMotor);
		
	}
	
	/**
	 * This method computes the total error rate by adding error rate per segment and averaging it by the number of testing segments.
	 */
	public void computeErrorRate(){
		for (int i = 0; i < numSegments; i++) {
			errorRate += errorSegment[i];
		}
		
		errorRate /= ((float) (numSegments - startTestSegment + 1));
		
	}
	
	/**
	 * Count the number of unsupervised motor sequences.
	 * 
	 * @param Segment
	 * @param motorIndex
	 */
	public void updateInstanceCount(int Segment, int motorIndex){
		numUnsupervisedInstances[Segment][motorIndex]++;
	}
	
	public void getConnectionNumbers(int[] conn) {
		for(int i=0; i<numHidden; i++) {
			numConnections[i] = conn[i];
		}
	}
	
	/**
	 * Write the performance of the network. 
	 * <p>
	 * This report file writes the number of neurons that were used by the network, the total error rate, total error rate per segment and the total error rate for each motor per segment.
	 * 
	 * @param filename
	 */
	
	public void writeAction(int seq, String filename){

		try {
			PrintWriter wr = new PrintWriter(new File(filename));
			for(int i=0; i<lenSequence[seq]; i++){
				for(int j=0; j<numMotor; j++){
					wr.print(Integer.toString(gAction[i][j])+ ',');
				}
				wr.println();
			}
			wr.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void writePerformance(String filename) {
		try {
			PrintWriter wr = new PrintWriter(new File(filename));
			
			switch(modality){
			case 'V':
				wr.println("Vision Modality");
				break;
				
			case 'T':
				wr.println("Language Modality");
				break;
				
			case 'A':
				wr.println("Audition Modality");
				break;
			}
			
			wr.println("Performance of the network");
			
			
			for (int i = 0; i < numHidden; i++) {
				wr.println("Number of neurons for HiddenLayer " + (i+1) + ": " + Integer.toString(numHiddenNeurons[i]));
			}
			
			for (int i = 0; i < numHidden; i++) {
				wr.println("Number of connections for Y Neurons in hidden layer " + (i+1) + ": " + Integer.toString(numConnections[i]));
				wr.println();
			}
			
			wr.println("Total Error Rate: " + Float.toString(errorRate));
			
			
			for (int i = 0; i < errorSegment.length; i++) {

				wr.println("Error for Segment " + Integer.toString(i+1) + ": " + Float.toString(errorSegment[i]));
				for (int j = 0; j < errorConcept[i].length; j++) {
					wr.println("\tError for Concept " + Integer.toString(j+1) + ": " + Float.toString(errorConcept[i][j]));
				}
				
				wr.println();
			}
			
			for (int segmentId = 0; segmentId < numSegments; segmentId++){
				wr.println("Motor Errors: " + Integer.toString(segmentId) + "====================");
			    for (int i = 0; i < errorTime[segmentId][60].length; i++) {    //change
			    	if(errorTime[segmentId][60][i] == 1){
				    wr.println("Time: " + Integer.toString(i+1) + ":" + Float.toString(errorTime[segmentId][60][i]) + ", Expected type: " + 
					    		Float.toString(trueAction[segmentId][i]) + ", Generated type: " + Float.toString(generatedAction[segmentId][i]));
			    	}
			    }
			    wr.println();
			}
			

			wr.close();
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

}