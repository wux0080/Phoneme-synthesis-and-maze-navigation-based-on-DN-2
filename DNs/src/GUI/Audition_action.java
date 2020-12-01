package GUI;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.*;
import java.util.concurrent.TimeUnit;
import DN2.DN2;

public class Audition_action {
	public static void runInterface(String[] args) throws IOException,
	InterruptedException {

// Stores all the intialization information for the network
// Indicate the filename to initialize the settings.

Scanner sc = new Scanner(System.in);
	int modalityOption = 0;
	boolean mContinue = false;
String modalityDirectory = "";
String modalityName = "";
	
do{
	
	try{
		System.out.println("This is listening and action modality:");

		System.out.print("Please ");
		mContinue = getKey();
		if(mContinue) {
			modalityOption = 3;
		}
		System.out.println();
		
	}
	catch(Exception e){
		System.out.println("Option must be an integer number");
	}
	
}while(modalityOption != 3);

Settings st = null;

switch (modalityOption) {
case 1:
	modalityDirectory = "Image_Data";
	modalityName = "vision";
	VisionInterface.removeSamplingRecords();
	VisionInterface.restoreOriginalTrainingFile();
	break;

case 2:
	modalityDirectory = "Language_Data";
	modalityName = "language";
	break;

case 3:
	modalityDirectory = "Audition_Data";
	modalityName = "audition";
	break;
}

st = new Settings(modalityDirectory + "/DN2/action/Input/settings_audition.txt");

/*
 * Get the information from the settings file.
 */
int numSegments = st.getNumSegments();
int startTestSegment = st.getTestSegmentStart(); // this index indicates
													// when training
													// ends and testing
													// begins in terms
													// of Segments.
int[] lenSequence = st.getLenSequence(); // total elements within the
											// sequence.
int numInput = st.getNumInput(); // The number of sensor the network
									// will use. For this contest, we
									// will use one sensor.
int[][] inputSize = st.getInputSize(); // This array will have the
										// (height, width) of each
										// sensor.

int numMotor = st.getNumMotor(); // The number of motors or effectors
									// the DN will have.
int[][] motorSize = st.getMotorSize(); // (height, width) of each motor.
int[] topKMotor = st.getTopKMotor(); // Topk winning neurons of each
										// motor layer.

int numHidden = st.getNumHidden(); // Number of hidden layers.
int[] numHiddenNeurons = st.getHiddenSize(); // The number of neurons
												// for each hidden
												// layer. This value can
												// be changed, only to a
												// smaller value.
int[] topKHidden = st.getTopKHidden(); // Topk winning neurons for each
										// hidden layer.

int rfSize = st.getRfSize();
int rfStride = st.getRfStride();
int[][] rf_id_loc = VisionInterface.configure_where_count(rfSize,
		rfStride, inputSize);

float prescreenPercent = st.getPrescreeningPercent();

/*
 * The modality of the input. V -> image data A -> audition data T ->
 * language (text) data
 */
char modality = st.getModality();

// The network is initialized the same way is was discussed during the
// workshop.
int[] typenum={0,0,0,2,0,0,2};
//int[] lateral_zone = {};
int[] lateral_zone = {};
int lateral_length = 0;   //128*60
float lateral_percent = 0;
int  zFrequency = 1;
float[][] growthtable = new float[50][];
float[][] meantable = new float[20][];
try {
	growthtable = getTablevector("Audition_Data/DN2/action/Input/Audition_growthrate_table.txt");
	meantable = getTablevector("Audition_Data/DN2/action/Audition_inhibition_table.csv");
} catch (IOException e1) {
	// TODO Auto-generated catch block
	e1.printStackTrace();
}
int numPrehidden = 1;
DN2 network = new DN2(numInput, inputSize, numMotor, motorSize,
		topKMotor, numPrehidden, numHidden, rfSize, rfStride, rf_id_loc,
		numHiddenNeurons, topKHidden, prescreenPercent,typenum,growthtable,meantable,lateral_percent, true, lateral_zone,
		lateral_length, DN2.MODE.Speech, zFrequency);

//network.setNeuronGrow(5, false);
//network.setLearning(5, false);
network.setNeuronGrow(7, false);
network.setLearning(7, false);

// Initialize the input and motor streams.
InputDataReader inputReader = null;
MotorDataReader motorReader = null;
MotorDataReader correctLabelReader = null;

boolean[] supervisedMotors = new boolean[numMotor];
float[][][] oldInputPattern = new float[numInput][][];
float[][][] oldMotorPattern = new float[numMotor][][];
float[][][] currentMotorPattern = new float[numMotor][][];
float[][][] newMotorPattern = new float[numMotor][][];
float[][][] performanceMotorPattern = new float[numMotor][][];

int[] winners = new int[lenSequence[2]] ;
/*
 * Start the Segment iteration This case we run two Segments. One for
 * training and other for testing.
 */
int temp1 = st.getNumPractices();
int temp2 = st.getNumPracticesPerTest();
int total_iteration = temp1/temp2;


/*
 * This part deals with GUI specific parameters. If the GUI flag is true
 * then would wait for the GUI signal. Default value for the continue_count
 * is -1. 
 */

for (int iter = 0; iter < total_iteration; iter++) {
	/*
	 * Measures the network performance.
	 * 
	 * Computing the error rate is hidden from the participant. This object
	 * will display the computing error for debugging purposes.
	 */
	PerformanceMeasure performance = new PerformanceMeasure(numSegments,
			numMotor, lenSequence, startTestSegment, numHidden,
			numHiddenNeurons, modality);
	//System.out.println("run into before loop");
	for (int k = 0; k < numSegments; k++) {
	
		int numPractices = 1;
	  
		// Keep track of the current input/motor sequence.
		for (int p = 0; p < numPractices; p++) {

		int seqCount = 0;
		if(modality=='A' && k==0){


				InputDataReader preinput = new InputDataReader("Audition_Data/DN2/action/Input/audition_pretraining_input.mat", 2000);
				float[][][] oldMotor = new float[numMotor][][];
				float[][] tempmotor = new float[1][128];
				for(int ii=0; ii<128; ii++){
					tempmotor[0][ii] = 0.0f;
				}
				for(int jj=0; jj<numMotor; jj++){
					oldMotor[jj] = tempmotor;
				}
				int ts = 1;
				do {
					System.out.println("type 4 computation: "+ts);
					float[][][] oldInput = new float[numInput][][];
					oldInput = preinput.getStreamInput();
					network.computeHiddenResponse(oldInput, oldMotor, true);

					ts++;
				} while (preinput.hasInput());

			
//pre-hidden					
					InputDataReader input = new InputDataReader("Audition_Data/DN2/action/Input/audition_training_input.mat", lenSequence[k]);
					int tt = 1;
					do {
							System.out.println("primary hidden computation: "+tt);
							float[][][] oldInput = new float[numInput][][];
							oldInput = input.getStreamInput();
							network.computePrimaryHiddenResponse(oldInput, true);

							tt++;
					} while (input.hasInput());
					network.savePriBottomWeights();

/*
				
				boolean g = true;

				while(g) {
					System.out.println("finish train primary hidden");
						if(getKey()) {
						g = false;
					}
				}
*/	
				
				//freeze type 4 neuron		
				network.setLearning(4, false);
//				network.setLearning(5, true);
				network.setLearning(7, true);
				network.setMotorisTopk(true);
//				network.setNeuronGrow(5, true);
				network.setNeuronGrow(7, true);
		}
			
		inputReader = getInputDataReader(k, lenSequence[k],
						modality);
		motorReader = getMotorDataReader(k, lenSequence[k],
						modality);

		// In the future, maybe training including testing.
		correctLabelReader = getPerformanceMotorDataReader(k,
						lenSequence[k], modality);					
																										

			System.out.println("Segment " + k);

			// t = 1 (initialization of inputs and motors).
			// read the old input
			System.out.println("t = 1");
//mode

			oldInputPattern = inputReader.getStreamInput();
			oldMotorPattern = motorReader.getStreamInput();
			correctLabelReader.getStreamInput();

			seqCount++;

			// t = 2 (compute the first hidden response).
			System.out.println("t = 2");
//pre-hidden
			network.computePrimaryHiddenResponse(oldInputPattern, false);
			// compute the hidden response with the old input
			network.computeHiddenResponse(oldInputPattern,
					oldMotorPattern, true);
			
//			if(k==2){
//				winners[seqCount] = network.getHiddenWinner();
//			}
			
			network.replaceHiddenResponse();
//pre-hidden			
			network.replacePriHiddenResponse();


//mode
			oldInputPattern = inputReader.getStreamInput();
			oldMotorPattern = motorReader.getStreamInput();
			correctLabelReader.getStreamInput();

			seqCount++;

			// start at t = 3
			do {
//				 mContinue = getKey();
//			   if(modality=='A' && k == 1) {
//				   network.saveAgeToText(1);
//			   }				
/*                if(k==1&&seqCount%2000==0){
                	g = true;
                }
				while(g) {
				System.out.println("stop");
					if(getKey()) {
						g = false;
					}
				}
*/
				System.out.println("t = " + (seqCount + 1));

				// compute the hidden response with the old input
				// The order doesn't matter, we choose to compute the Y
				// area
				// first.
				
//pre-hidden				
				network.computePrimaryHiddenResponse(oldInputPattern, false);
				
				if (k < 2){
				    network.computeHiddenResponse(oldInputPattern,
						oldMotorPattern, true);
				} else {
					network.computeHiddenResponse(oldInputPattern,
						oldMotorPattern, false);
					winners[seqCount] = network.getHiddenWinner();
				}
//mode						network.replaceHiddenResponse(); //mode
                /*
				if(k < 2){
				      network.trackHiddenwinners(seqCount);
				}*/
				
				// Read the current motor pattern to determine if
				// supervision is required.
				currentMotorPattern = motorReader.getStreamInput();
				performanceMotorPattern = correctLabelReader
						.getStreamInput();

				// using the old response from Y compute the new motor
				// all zeros will represent that the pattern is to be
				// emergent.
				supervisedMotors = allZeros(currentMotorPattern);

				// compute the response for each motor individually.
				for (int i = 0; i < numMotor; i++) {

					// If the current motor pattern has at least one
					// non-zero element.
					// we compute the motor response and update its
					// weights without supervision.
					if (supervisedMotors[i]) {
	             	if(i == 60){                                                   //need change
							newMotorPattern[i] = network
									.computeMotorResponse(i,false, 0.3f);
						}
						else{
							newMotorPattern[i] = network
									.computeMotorResponse(i,false, 0);
						}
						
						// network.updateMotorWeights(i);

						// Indicate that this instance is to count on
						// final
						// errorComputation
						performance.updateInstanceCount(k, i);

						// compute the error per sequence.
						
						performance.updateErrorTimeRate(k, i, seqCount,
								newMotorPattern,
								performanceMotorPattern);
//count generated actions						
						if(k == 2){
							performance.updateComputedAction(i, seqCount, newMotorPattern);
						}
					}

					else { // new state is supervised to be the current
						newMotorPattern[i] = currentMotorPattern[i];
						if (k<2){
							if(i == 60){                                     //need change								
								network.updateSupervisedMotorWeights(i,
										newMotorPattern[i], 0.3f, oldMotorPattern);
							}
							else{
								network.updateSupervisedMotorWeights(i,
										newMotorPattern[i], 0);
							}
						}								
					}

				}

//				System.out.println("Old Motor Pattern");
//				displayResponse(oldMotorPattern);

//				System.out.println("Current Motor Pattern");
//				displayResponse(currentMotorPattern);

//				System.out.println("New Motor Pattern");
//				displayResponse(newMotorPattern);

				// set the current input as the old input for next
				// computation
				oldInputPattern = inputReader.getStreamInput();
				oldMotorPattern = newMotorPattern;
				
				// Updage continue_count for GUI

				// Increment the sequence index counter.
				if (motorReader.hasMotor())
					seqCount++;

				// replace old by new.
				// see DN book algorithm 6.1, Step 2(b)
//mode	
				network.replaceHiddenResponse(); // All computation was
													// using
													// oldResponses
													// for individual
													// neurons, now new
													// response replace
													// the
//pre-hidden													// old.
				network.replacePriHiddenResponse();

				network.replaceMotorResponse(); // All computation was
												// using
												// oldResponses for
												// individual neurons,
												// now
												// new response replace
												// the
												// old.
//				if((seqCount%150 == 0)&&(k == 1)){
//					network.updateHiddenLocation();	
//				}
				
//				if((seqCount%150 == 0)&&(k == 1)){
//					network.updatePriHiddenLocation(0.1f);
//				}
				
//				if(k == 1 && seqCount == 2000){
//					network.outputPriHiddenLocation("1");	
//					network.outputHiddenLocation("1");
//				}
				
//				}
//				if(k == 2 && seqCount == 10){
//					network.outputPriHiddenLocation("4");
//				    network.outputHiddenLocation("4");	
//				}
		 
			} while (inputReader.hasInput() && motorReader.hasMotor());

			// measure the error per concept
			performance.updateErrorMotorRate(k);

			// measure the error per Segment
			performance.updateErrorSegmentRate(k);
			
		}
	}

	// We divide the total error per Segment by the number of Segments.
	performance.computeErrorRate();		
    int[] connections = network.getConnections();
	performance.getConnectionNumbers(connections);
//	network.saveAgeToText(2);
	performance.writePerformance(modalityDirectory + "/DN2/action/Input/Preformance_"
			+ modalityName + Integer.toString(iter) + ".txt");
	performance.writeAction(2, modalityDirectory + "/DN2/action/actions.txt");
	// network.serializeSave(modalityDirectory + "/network_" + Integer.toString(iter) + ".ser");
	//network.saveToText();
	//writeIndexes(modalityDirectory + "/DN2/action/winner_indexes.txt", winners);
}
}

/*
* Initialize the InputDataReader with the corresponding dataset Segment + 1
* = 1 -> Training dataset. Segment + 1 = 2 -> Resubstitution Test dataset.
* Same sequence as training dataset. Segment + 1 = 3 -> Disjoint Test
* dataset. New sequence.
*/
public static InputDataReader getInputDataReader(int Segment,
	int lenSequence, char modality) {

InputDataReader inRead = null;

switch (Segment) {
case 0:
	switch (modality) {

	case 'V':
		inRead = new InputDataReader(
				"Image_Data/Input/vision_training_input.mat",
				lenSequence);
		break;

	case 'A':
		inRead = new InputDataReader(
				"Audition_Data/DN2/action/Input/audition_training_input.mat",
				lenSequence);
		break;

	case 'T':
		inRead = new InputDataReader(
				"Language_Data/Input/language_training_input.mat",
				lenSequence);
		break;

	default:
		inRead = null;
		break;
	}
	break;

case 1:
	switch (modality) {

	case 'V':
		inRead = new InputDataReader(
				"Image_Data/Input/vision_resubstitution_input.mat",
				lenSequence);
		break;

	case 'A':
		inRead = new InputDataReader(
				"Audition_Data/DN2/action/Input/audition_training_input.mat",
				lenSequence);
		break;

	case 'T':
		inRead = new InputDataReader(
				"Language_Data/Input/language_resubstitution_input.mat",
				lenSequence);
		break;

	default:
		inRead = null;
		break;
	}
	break;

case 2:
	switch (modality) {

	case 'V':
		inRead = new InputDataReader(
				"Image_Data/Input/vision_disjoint_1_input.mat",
				lenSequence);
		break;

	case 'A':
		inRead = new InputDataReader(
				"Audition_Data/DN2/action/Input/audition_training_input.mat",
				lenSequence);
		break;

	case 'T':
		inRead = new InputDataReader(
				"Language_Data/Input/language_disjoint_1_input.mat",
				lenSequence);
		break;

	default:
		inRead = null;
		break;
	}
	break;

case 3:
	switch (modality) {

	case 'V':
		inRead = new InputDataReader(
				"Image_Data/Input/vision_disjoint_2_input.mat",
				lenSequence);
		break;

	case 'A':
		inRead = new InputDataReader(
				"Audition_Data/DN2/action/Input/audition_disjoint1_input.mat",
				lenSequence);
		break;

	case 'T':
		inRead = new InputDataReader(
				"Language_Data/Input/language_disjoint_2_input.mat",
				lenSequence);
		break;

	default:
		inRead = null;
		break;
	}
	break;

case 4:
	switch (modality) {

	case 'V':
		inRead = new InputDataReader(
				"Image_Data/Input/vision_disjoint_3_input.mat",
				lenSequence);
		break;

	case 'A':
		inRead = new InputDataReader(
				"Audition_Data/DN2/action/Input/audition_disjoint2_input.mat",
				lenSequence);
		break;

	case 'T':
		inRead = new InputDataReader(
				"Language_Data/Input/language_disjoint_3_input.mat",
				lenSequence);
		break;

	default:
		inRead = null;
		break;
	}
	break;
	
case 5:
	switch (modality) {

	case 'V':
		inRead = new InputDataReader(
				"Image_Data/Input/vision_disjoint_3_input.mat",
				lenSequence);
		break;

	case 'A':
		inRead = new InputDataReader(
				"Audition_Data/DN2/action/Input/audition_disjoint3_input.mat",
				lenSequence);
		break;

	case 'T':
		inRead = new InputDataReader(
				"Language_Data/Input/language_disjoint_3_input.mat",
				lenSequence);
		break;

	default:
		inRead = null;
		break;
	}
	break;
}

return inRead;
}

/*
* Initialize the MotorDataReader with the corresponding dataset for the
* corresponding modality. Segment + 1 = 1 -> Training dataset. All motors
* are supervised so the network can learn all sequences. Segment + 1 = 2 ->
* Resubstitution Test dataset. Same sequence as the training dataset. The
* first two motors are supervised, the other motors are free. Segment + 1 =
* 3 -> Disjoint Test dataset. New sequence, not previously trained. The
* first two motors are supervised, the other motors are free.
*/
public static MotorDataReader getMotorDataReader(int Segment,
	int lenSequence, char modality) {

MotorDataReader moRead = null;

switch (Segment) {
case 0:
	switch (modality) {

	case 'V':
		moRead = new MotorDataReader(
				"Image_Data/Input/vision_training_motor.mat",
				lenSequence);
		break;

	case 'A':
		moRead = new MotorDataReader(
				"Audition_Data/DN2/action/Input/audition_training_motor.mat",
				lenSequence);
		break;

	case 'T':
		moRead = new MotorDataReader(
				"Language_Data/Input/language_training_motor.mat",
				lenSequence);
		break;

	default:
		moRead = null;
		break;
	}
	break;

case 1:
	switch (modality) {

	case 'V':
		moRead = new MotorDataReader(
				"Image_Data/Input/vision_resubstitution_motor.mat",
				lenSequence);
		break;

	case 'A':
		moRead = new MotorDataReader(
				"Audition_Data/DN2/action/Input/audition_training_motor.mat",
				lenSequence);
		break;

	case 'T':
		moRead = new MotorDataReader(
				"Language_Data/Input/language_resubstitution_motor.mat",
				lenSequence);
		break;

	default:
		moRead = null;
		break;
	}
	break;

case 2:
	switch (modality) {

	case 'V':
		moRead = new MotorDataReader(
				"Image_Data/Input/vision_disjoint_1_motor.mat",
				lenSequence);
		break;

	case 'A':
		moRead = new MotorDataReader(
				"Audition_Data/DN2/action/Input/audition_resub_motor.mat",     
				lenSequence);
		break;

	case 'T':
		moRead = new MotorDataReader(
				"Language_Data/Input/language_disjoint_1_motor.mat",
				lenSequence);
		break;

	default:
		moRead = null;
		break;
	}
	break;

case 3:
	switch (modality) {

	case 'V':
		moRead = new MotorDataReader(
				"Image_Data/Input/vision_disjoint_2_motor.mat",
				lenSequence);
		break;

	case 'A':
		moRead = new MotorDataReader(
				"Audition_Data/DN2/action/Input/audition_resub_motor.mat",
				lenSequence);
		break;

	case 'T':
		moRead = new MotorDataReader(
				"Language_Data/Input/language_disjoint_2_motor.mat",
				lenSequence);
		break;

	default:
		moRead = null;
		break;
	}
	break;

case 4:
	switch (modality) {

	case 'V':
		moRead = new MotorDataReader(
				"Image_Data/Input/vision_disjoint_3_motor.mat",
				lenSequence);
		break;

	case 'A':
		moRead = new MotorDataReader(
				"Audition_Data/DN2/action/Input/audition_resub_motor.mat",
				lenSequence);
		break;

	case 'T':
		moRead = new MotorDataReader(
				"Language_Data/Input/language_disjoint_3_motor.mat",
				lenSequence);
		break;

	default:
		moRead = null;
		break;
	}
	break;
	
case 5:
	switch (modality) {

	case 'V':
		moRead = new MotorDataReader(
				"Image_Data/Input/vision_disjoint_3_motor.mat",
				lenSequence);
		break;

	case 'A':
		moRead = new MotorDataReader(
				"Audition_Data/DN2/action/Input/audition_resub_motor.mat",
				lenSequence);
		break;

	case 'T':
		moRead = new MotorDataReader(
				"Language_Data/Input/language_disjoint_3_motor.mat",
				lenSequence);
		break;

	default:
		moRead = null;
		break;
	}
	break;
}

return moRead;
}

/*
* Initialize the MotorDataReader with the corresponding dataset for the
* corresponding modality. Segment + 1 = 1 -> Training dataset. All motors
* are supervised so the network can learn all sequences. Segment + 1 = 2 ->
* Resubstitution Test dataset. Same sequence as the training dataset. The
* first two motors are supervised, the other motors are free. Segment + 1 =
* 3 -> Disjoint Test dataset. New sequence, not previously trained. The
* first two motors are supervised, the other motors are free.
*/
public static MotorDataReader getPerformanceMotorDataReader(int Segment,
	int lenSequence, char modality) {

MotorDataReader moRead = null;

switch (Segment) {
case 0:
	switch (modality) {

	case 'V':
		moRead = new MotorDataReader(
				"Image_Data/Input/vision_training_performance_motor.mat",
				lenSequence);
		break;

	case 'A':
		moRead = new MotorDataReader(
				"Audition_Data/DN2/action/Input/audition_training_performance_motor.mat",
				lenSequence);
		break;

	case 'T':
		moRead = new MotorDataReader(
				"Language_Data/Input/language_training_performance_motor.mat",
				lenSequence);
		break;

	default:
		moRead = null;
		break;
	}
	break;

case 1:
	switch (modality) {

	case 'V':
		moRead = new MotorDataReader(
				"Image_Data/Input/vision_resubstitution_performance_motor.mat",
				lenSequence);
		break;

	case 'A':
		moRead = new MotorDataReader(
				"Audition_Data/DN2/action/Input/audition_training_performance_motor.mat",
				lenSequence);
		break;

	case 'T':
		moRead = new MotorDataReader(
				"Language_Data/Input/language_resubstitution_performance_motor.mat",
				lenSequence);
		break;

	default:
		moRead = null;
		break;
	}
	break;

case 2:
	switch (modality) {

	case 'V':
		moRead = new MotorDataReader(
				"Image_Data/Input/vision_disjoint_1_performance_motor.mat",
				lenSequence);
		break;

	case 'A':
		moRead = new MotorDataReader(
				"Audition_Data/DN2/action/Input/audition_training_performance_motor.mat",
				lenSequence);
		break;

	case 'T':
		moRead = new MotorDataReader(
				"Language_Data/Input/language_disjoint_1_performance_motor.mat",
				lenSequence);
		break;

	default:
		moRead = null;
		break;
	}
	break;

case 3:
	switch (modality) {

	case 'V':
		moRead = new MotorDataReader(
				"Image_Data/Input/vision_disjoint_2_performance_motor.mat",
				lenSequence);
		break;

	case 'A':
		moRead = new MotorDataReader(
				"Audition_Data/DN2/action/Input/audition_training_performance_motor.mat",
				lenSequence);
		break;

	case 'T':
		moRead = new MotorDataReader(
				"Language_Data/Input/language_disjoint_2_performance_motor.mat",
				lenSequence);
		break;

	default:
		moRead = null;
		break;
	}
	break;

case 4:
	switch (modality) {

	case 'V':
		moRead = new MotorDataReader(
				"Image_Data/Input/vision_disjoint_3_performance_motor.mat",
				lenSequence);
		break;

	case 'A':
		moRead = new MotorDataReader(
				"Audition_Data/DN2/action/Input/audition_training_performance_motor.mat",
				lenSequence);
		break;

	case 'T':
		moRead = new MotorDataReader(
				"Language_Data/Input/language_disjoint_3_performance_motor.mat",
				lenSequence);
		break;

	default:
		moRead = null;
		break;
	}
	break;
	
case 5:
	switch (modality) {

	case 'V':
		moRead = new MotorDataReader(
				"Image_Data/Input/vision_disjoint_3_performance_motor.mat",
				lenSequence);
		break;

	case 'A':
		moRead = new MotorDataReader(
				"Audition_Data/DN2/action/Input/audition_training_performance_motor.mat",
				lenSequence);
		break;

	case 'T':
		moRead = new MotorDataReader(
				"Language_Data/Input/language_disjoint_3_performance_motor.mat",
				lenSequence);
		break;

	default:
		moRead = null;
		break;
	}
	break;
}

return moRead;
}

/*
* This methods displays a 3d array of response on console.
* 
* You can use it to see the motor responses for debugging purposes.
*/
private static void displayResponse(float[][][] r) {

for (int k = 0; k < r.length; k++) {
	if (k!=2){
	for (int i = 0; i < r[k].length; i++) {
		System.out.print("Motor " + (k + 1) + " " + r[k][i][0]);
		for (int j = 1; j < r[k][i].length; j++) {
			System.out.print("," + r[k][i][j]);
		}
		System.out.println();
	}
	}
}
}

/*
* This procedure checks if all the elements of a motor pattern are zero. If
* all the elements are zero, then supervision is not required.
*/
public static boolean[] allZeros(float[][][] r) {

boolean[] sup = new boolean[r.length]; // indicates whether the motor
										// needs to be supervised.

for (int i = 0; i < r.length; i++) { // motor loop

	int count = 0;

	for (int j = 0; j < r[i].length; j++) { // height of each motor loop

		for (int k = 0; k < r[i][j].length; k++) { // width of each
													// motor loop
			// if there is at least one motor neuron active, increase
			// the count.
			if (((int) r[i][j][k]) == 1) {
				count++;
			}
		}

	}

	// If the number of active motor neurons is greater than zero,
	// supervise the motor response.
	if (count == 0)
		sup[i] = true;

}

return sup;
}

public static void writeIndexes(String filename, int[] data){
	try {
		PrintWriter wr = new PrintWriter(new File(filename));
		for(int i=0; i<data.length; i++){
				wr.print(Integer.toString(data[i]));
			
			wr.println();
		}
		wr.close();
	} catch (FileNotFoundException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}
}

public static float[][] getTablevector(String filename) throws IOException{
return new TableReader(filename).getTable();
}

public static boolean getKey() throws IOException{
boolean a = false;
System.out.println("Please continue....");
int b = System.in.read();
if(b == 10){
	a = true;
}
return a;
}

public static void main(String args[]) throws IOException, ClassNotFoundException, InterruptedException {
Audition_action.runInterface(args);
}


}