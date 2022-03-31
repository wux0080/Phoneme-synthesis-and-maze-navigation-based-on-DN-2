/* 
 *  DN caller is the interface between the game environment and the DN.
 *  DN caller is implemented as a singleton. Meaning that in the entire game there can only be one DN.
 *  The network is accessed by calling DNCaller.getInstance().
 *  The underlying network program can be replaced by future versions of DN, as long as the training and
 *  testing procedure remains unchanged. 
 */

package MazeInterface;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.PrintWriter;

import DN1.DN1;
import DN2.DN2;
import DN2.DN_GPU;
import GUI.*;
import MazeInterface.Agent.Action;
import MazeInterface.Commons.env;
import MazeInterface.Commons.gps;

public class DNCaller implements Commons {
	private static DNCaller instance = null; // This is the instance of
												// DNCaller, implemented as
												// singleton.
	private DN1 network1; // This is the underlying network to train and test.
	private DN2 network_cpu;
	private DN_GPU network_gpu;
	
	private static int numInput = 3; // Vision and GPS as two inputs.
	private static int[][] inputSize; 
	//3; delete last 3
	private static int numMotor = 13; // We have visions, concept (skills),
										// destinations, and
										// destinations_pain_low,
										// destination_pain_high.
	// we are using type 011 and 101. 
	private static int[] typeNum = { 2, 2, 2, 2, 2, 2, 2 };
	private static float[][] growthrate = { 
			{ 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 1.0f, 0.05f, 0.05f },
			{ 0.10f, 0.05f, 0.05f, 0.05f, 0.05f, 1.0f, 0.05f, 0.05f }, 
			{ 0.15f, 0.05f, 0.05f, 0.05f, 0.05f, 1.0f, 0.05f, 0.05f },
			{ 0.20f, 0.05f, 0.05f, 0.05f, 0.05f, 1.0f, 0.05f, 0.05f }, 
			{ 0.25f, 0.05f, 0.05f, 0.05f, 0.05f, 1.0f, 0.05f, 0.05f },
			{ 0.30f, 0.05f, 0.05f, 0.05f, 0.05f, 1.0f, 0.05f, 0.05f }, 
			{ 0.35f, 0.05f, 0.05f, 0.05f, 0.05f, 1.0f, 0.05f, 0.05f },
			{ 0.40f, 0.05f, 0.05f, 0.05f, 0.05f, 1.0f, 0.05f, 0.05f }, 
			{ 0.45f, 0.05f, 0.05f, 0.05f, 0.05f, 1.0f, 0.05f, 0.05f },
			{ 0.50f, 0.05f, 0.05f, 0.05f, 0.05f, 1.0f, 0.05f, 0.05f }, 
			{ 0.55f, 0.05f, 0.05f, 0.05f, 0.05f, 1.0f, 0.05f, 0.05f },
			{ 0.60f, 0.05f, 0.05f, 0.05f, 0.05f, 1.0f, 0.05f, 0.05f }, 
			{ 0.65f, 0.05f, 0.05f, 0.05f, 0.05f, 1.0f, 0.05f, 0.05f },
			{ 0.70f, 0.05f, 0.05f, 0.05f, 0.05f, 1.0f, 0.05f, 0.05f }, 
			{ 0.75f, 0.05f, 0.05f, 0.05f, 0.05f, 1.0f, 0.05f, 0.05f },
			{ 0.80f, 0.05f, 0.05f, 0.05f, 0.05f, 1.0f, 0.05f, 0.05f }, 
			{ 0.85f, 0.05f, 0.05f, 0.05f, 0.05f, 1.0f, 0.05f, 0.05f },
			{ 0.90f, 0.05f, 0.05f, 0.05f, 0.05f, 1.0f, 0.05f, 0.05f }, 
			{ 0.95f, 0.05f, 0.05f, 0.05f, 0.05f, 1.0f, 0.05f, 0.05f },
			{ 1.00f, 0.05f, 0.05f, 0.05f, 0.05f, 1.0f, 0.05f, 0.05f } };
	private static float[][] meanValue = { { 0.05f, 0.01f }, { 0.1f, 0.01f }, { 0.15f, 0.01f }, { 0.2f, 0.01f },
			{ 0.25f, 0.01f }, { 0.3f, 0.01f }, { 0.35f, 0.01f }, { 0.4f, 0.01f }, { 0.45f, 0.01f }, { 0.5f, 0.01f },
			{ 0.55f, 0.01f }, { 0.6f, 0.01f }, { 0.65f, 0.01f }, { 0.7f, 0.01f }, { 0.75f, 0.01f }, { 0.8f, 0.1f },
			{ 0.85f, 0.4f }, { 0.9f, 0.6f }, { 0.95f, 0.7f }, { 1, 1 } };
	//3; delete 3
	private static int[] topKMotor = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	// Action motor, concept motor, and destination motor. Concept motor is
	// dynamic, starting with size 0.
	// The motors are : action, skill, means/destination, means_pain_low,
	// means_pain_high, cost_1, cost_2, comparison, covert.
	// means and means pain low, means pain high are actually same neuron.
	public static final int scale_num = 6;  //21
	public static final int loc_num = 6;   //43
	//3; add 7; 8th add 1, 9th add 1
	private static int[][] motorSize = { { 1, Agent.action_num }, { 1, 0 }, { 1, 0 }, { 1, 0 }, { 1, 0 }, { 1, 18 },{ 1, 18 },
			{ 1, 18 }, { 1, 4 }, { 1, 2},{1, loc_num}, {1, env.values().length}, {1, scale_num} };
	int[] lateral_zone = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int lateral_length;
	private static float pain_inhibit_rate_low = 2f;
	private static float pain_inhibit_rate_high = 4f;
	private static int numHidden = 1;
	private static int[][] rf_id_loc;
	private static int[] numHiddenNeurons = { 620 };
	private static int[] topKHidden = { 1 };
	private static float prescreenPercent = 0.4f;
	private static float lateral_percent = 0.001f;
	private boolean continuous_motor_flag = false; // If continuous flag is
													// false, then we do top-K
													// competition
													// during testing. If
													// continuous, then the
													// motor output is then
													// converted to a turning
													// degree.

	private static boolean initialized; // At the beginning of we need
										// to reset the network.

	// The network uses oldInputPattern and oldMotorPattern for the current
	// computation and generates
	// a newMotorPattern as the output corresponding to the oldInputPattern.
	private static float[][][] oldInputPattern;
	private static float[][][] oldMotorPattern;
	private static float[][][] newMotorPattern;
	private static float[][][] lateralMotorPattern;
	private static float[][][] tempMotorPattern;

	// These params are used to learn lateral weights for planning.
	// Lateral weights are learned during destination training phase.
	private int current_skill;
	private int old_skill;
	private int current_means;
	private int currentCost;
	private int old_means;
	private int overt_skill;
	private int[] cost_firing_id = { NULLVALUE, NULLVALUE };
	private int get_back_skill_id;
	private int get_back_dest_id;
	private boolean overt;
	private int lessMore_id;
	private int old_landmark_loc;
	private env old_landmark_type;
	private int old_landmark_size;
	private int zFrequency;
	
	// Training where what. Referenced inside DN to initialize RFs.
	public static int curr_type = NULLVALUE;
	public static int curr_loc  = NULLVALUE;
	public static int curr_scale = NULLVALUE;
	
	// This is how the game environment (agent) get access to the singleton.
	// Once accessed, if the instance is not initialized, then initialization
	// will occur.
	// Otherwise, just return the instance. This aurantees that only one
	// instance will be created during the
	// entire process.
	public static DNCaller getInstance() {
		if (instance == null) {
			instance = new DNCaller();
		}
		return instance;
	}

	// Each map reset will set the initialized flag to false. Thus the network
	// would know to reset its internal
	// response with the default input and default motor response.
	public void setInitialized(boolean initialized) {
		DNCaller.initialized = initialized;
		currentCost = 0;
	}

	// Send the network over socket for GUI debugging and weights visualizaiton.
	// The GUI is in matlab. To use the GUI:
	// 1) set Commons.use_gui_flag to true.
	// 2) start Java program.
	// 3) start matlab one_by_one_gui
	public void sendOverSocket(PrintWriter string_out, DataOutputStream data_out, int display_y_zone, int display_num,
			int display_start_id, int display_z_zone_1, int display_z_zone_2) throws IOException, InterruptedException {
		switch (DNVERSION) {
		case 1:
			network1.sendNetworkOverSocket(string_out, data_out, display_y_zone, display_num, display_start_id,
					display_z_zone_1, display_z_zone_2);
			break;
		case 2:
			if (computing_mode == ComputingMode.CPU){
			    network_cpu.sendNetworkOverSocket(string_out, data_out, display_y_zone, 0, display_num, display_start_id,
					display_z_zone_1, display_z_zone_2);
			} else {
				network_gpu.sendNetworkOverSocket(string_out, data_out, display_y_zone, 0, display_num, display_start_id,
						display_z_zone_1, display_z_zone_2);
			}
			break;
		default:
			throw new java.lang.Error("DN version not supported");
		}
	};

	// DN creating method. The newly created DN is not initialized (no learning
	// has taken place yet).
	public void createDN(int num_skills, int num_destinations) {
		zFrequency = 1;
		if (vision_2D_flag == false) {
		    inputSize = new int[][]{ { 1, Agent.vision_num * 3 }, { 1, Agent.vision_num * 3 }, { 1, Agent.vision_num * 1 } }; 
		} else {
			int vision_width = Agent.vision_num;
			int vision_height = Agent.vision_num * 3/4;
			int vision_length = vision_width * vision_height;
			inputSize = new int[][]{{1, vision_length * 3}, {1, vision_length * 3}, {1, vision_length}};
		}
		motorSize[1][1] = num_skills + 1;
		motorSize[2][1] = num_destinations + 1;
		motorSize[3][1] = num_destinations + 1;
		motorSize[4][1] = num_destinations + 1;
		get_back_skill_id = num_skills;
		get_back_dest_id = num_destinations;
		//3; 5,6,7
		lateral_length = motorSize[0][1] + motorSize[1][1] + motorSize[2][1] + motorSize[5][1] + motorSize[6][1]
				+ motorSize[7][1] + motorSize[8][1] + motorSize[9][1];
		switch (DNVERSION) {
		case 1:
			network1 = new DN1(numInput, inputSize, numMotor, motorSize, topKMotor, numHidden, 0, 0, rf_id_loc,
					numHiddenNeurons, topKHidden, prescreenPercent);
			break;
		case 2:
			// DN 2 initialization method.
			// Skills concept zone has connection from skills and destinations.
			cost_firing_id = new int[motorSize[2][1]];
			if (computing_mode == ComputingMode.CPU){
			    network_cpu = new DN2(numInput, inputSize, numMotor, motorSize, topKMotor, 0, numHidden, 0, 0, rf_id_loc,
					numHiddenNeurons, topKHidden, prescreenPercent, typeNum, growthrate, meanValue, 0, false,
					lateral_zone, lateral_length, DN2.MODE.MAZE, zFrequency);
			} else {
				network_gpu = new DN_GPU(numInput, inputSize, numMotor, motorSize, topKMotor, numHidden, 0, 0, rf_id_loc,
						numHiddenNeurons, topKHidden, prescreenPercent, typeNum, growthrate, meanValue, 0, false,
						lateral_zone, lateral_length, DN2.MODE.MAZE, zFrequency);
			}
			break;
		default:
			throw new java.lang.Error("DN version not supported");
		}
		initialized = false;
	}

	// Save the DN to a specific path.
	public void saveDN(String path) {
		switch (DNVERSION) {
		case 1:
			network1.serializeSave(path);
			break;
		case 2:
			if (computing_mode == ComputingMode.CPU){
			    //network_cpu.serializeSave(path);
			    network_cpu.saveToText();
			} else {
				// TODO: save the network.
			}
			break;
		}
	}

	// Load a pre-trained DN.
	public void loadDN(String path) throws IOException, ClassNotFoundException {
		switch (DNVERSION) {
		case 1:
			network1.deserializeLoad(path);
			break;
		case 2:
			if (computing_mode == ComputingMode.CPU){
			    network_cpu.deserializeLoad(path);
			} else {
				throw new java.lang.Error("gpu load not available yet");
			}
			break;
		}
	}

	// Training is only training individual skills. At this stage there is no
	// cost, no values, no destinations.
	// During training, we have the vision input, gps input, and the supervised
	// motors.
	// visions: are the vision lines with types of recognized objects and
	// distance of that object.
	// gps is represented as gps_diff, becuase the AI that generates supervision
	// needs this diff info.
	// For DN training, gps is either left, right or forward.
	public int train(VisionLine[] visions, BufferedImage vision_image, int gps_diff, Action supervised_action, int supervised_skill,
			int supervised_destination, int previous_skill, float current_value, boolean block_change_flag,
			int landmark_loc, env landmark_type, int landmark_size) {
		// If not initialized, the network needs to update once to reset its
		// internal responses.
		int land_type;
		if(landmark_type == null){
			land_type = NULLVALUE;
		}
		else{
			land_type = landmark_type.ordinal();
		}
		if (!initialized) {
			oldInputPattern = convert_mat(visions, vision_image, gps_diff, block_change_flag);
			oldMotorPattern = getDefaultMotor(previous_skill);
			newMotorPattern = getDiscreteMotorPattern(supervised_action, supervised_skill, supervised_destination,
					current_value, true, landmark_loc, land_type, landmark_size);

			switch (DNVERSION) {
			case 1:
				network1.computeHiddenResponse(oldInputPattern, oldMotorPattern, true);
				network1.replaceHiddenResponse();
				for (int i = 0; i < numMotor; i++) {
					network1.updateSupervisedMotorWeights(i, newMotorPattern[i]);
				}
				network1.replaceMotorResponse();
				break;
			case 2:
				if (computing_mode == ComputingMode.CPU){
				    network_cpu.computeHiddenResponse(oldInputPattern, oldMotorPattern, true);
				    network_cpu.replaceHiddenResponse();
				    for (int i = 0; i < numMotor; i++) {
					    network_cpu.updateSupervisedMotorWeights(i, newMotorPattern[i]); 
				    }
				    network_cpu.replaceMotorResponse();
				} else {
					network_gpu.computeHiddenResponse(oldInputPattern, oldMotorPattern, true);
				    network_gpu.replaceHiddenResponse();
					network_gpu.updateSupervisedMotorWeights(newMotorPattern); 
				    network_gpu.replaceMotorResponse();
				}
				break;
			}

			for (int i = 0; i < oldMotorPattern.length; i++) {
				for (int j = 0; j < oldMotorPattern[i].length; j++) {
					if (newMotorPattern[i][j].length != oldMotorPattern[i][j].length)
						throw new AssertionError(
								"Length not equal: " + Integer.toString(i) + ", " + Integer.toString(j));
					System.arraycopy(newMotorPattern[i][j], 0, oldMotorPattern[i][j], 0, oldMotorPattern[i][j].length);
				}
			}
			initialized = true;
		}
		// This is to count the number of supervision during the entire process.
		// If emergent motor is wrong, then supervise motor response, set
		// supervised to be true.
		boolean supervised = false;
		oldInputPattern = convert_mat(visions, vision_image, gps_diff, block_change_flag);
		Action old_action = getActionFromMotorPattern(oldMotorPattern);
		// Supervised learning makes sure the pattern is correct, even if the
		// emergent pattern is wrong.
		newMotorPattern = getDiscreteMotorPattern(supervised_action, supervised_skill, supervised_destination,
				current_value, true, landmark_loc, land_type, landmark_size);
		switch (DNVERSION) {
		case 1:
			network1.computeHiddenResponse(oldInputPattern, oldMotorPattern, true);
			network1.replaceHiddenResponse();
			for (int i = 0; i < numMotor; i++) {
				network1.updateSupervisedMotorWeights(i, newMotorPattern[i]);
			}
			// All computation was using oldResponses for individual neurons,
			// now new response replace the old.
			network1.replaceMotorResponse();
			break;
		case 2:
			if (computing_mode == ComputingMode.CPU){
				// first update
				network_cpu.computeHiddenResponse(oldInputPattern, oldMotorPattern, true);
				for (int i = 0; i < numMotor; i++) {
					network_cpu.updateSupervisedMotorWeights(i, oldMotorPattern[i]);
					float[] current_lateral_input = getLateralInput(old_action.ordinal(), NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE,
							NULLVALUE);
					updateLateralWeights(current_lateral_input, old_action.ordinal(), NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE,
							NULLVALUE, NULLVALUE, NULLVALUE);
				}
				// All computation was using oldResponses for individual neurons,
				// now new response replace the old.
				network_cpu.replaceHiddenResponse();
				network_cpu.replaceMotorResponse();
	
				// second update
				network_cpu.computeHiddenResponse(oldInputPattern, oldMotorPattern, true);
				for (int i = 0; i < numMotor; i++) {
					network_cpu.updateSupervisedMotorWeights(i, newMotorPattern[i]);
					float[] current_lateral_input = getLateralInput(old_action.ordinal(), NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE,
							NULLVALUE);
					updateLateralWeights(current_lateral_input, supervised_action.ordinal(), NULLVALUE, NULLVALUE, NULLVALUE,
							NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE);
				}
				// All computation was using oldResponses for individual neurons,
				// now new response replace the old.
				network_cpu.replaceHiddenResponse();
				network_cpu.replaceMotorResponse();
				break;
			} else {
				// first update
				network_gpu.computeHiddenResponse(oldInputPattern, oldMotorPattern, true);
				network_gpu.updateSupervisedMotorWeights(oldMotorPattern);
				for (int i = 0; i < numMotor; i++) {
					float[] current_lateral_input = getLateralInput(old_action.ordinal(), NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE,
							NULLVALUE);
					updateLateralWeights(current_lateral_input, old_action.ordinal(), NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE,
							NULLVALUE, NULLVALUE, NULLVALUE);
				}
				// All computation was using oldResponses for individual neurons,
				// now new response replace the old.
				network_gpu.replaceHiddenResponse();
				network_gpu.replaceMotorResponse();
	
				// second update
				network_gpu.computeHiddenResponse(oldInputPattern, oldMotorPattern, true);
				network_gpu.updateSupervisedMotorWeights(newMotorPattern);
				for (int i = 0; i < numMotor; i++) {
					float[] current_lateral_input = getLateralInput(old_action.ordinal(), NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE,
							NULLVALUE);
					updateLateralWeights(current_lateral_input, supervised_action.ordinal(), NULLVALUE, NULLVALUE, NULLVALUE,
							NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE);
				}
				// All computation was using oldResponses for individual neurons,
				// now new response replace the old.
				network_gpu.replaceHiddenResponse();
				network_gpu.replaceMotorResponse();
				break;
			}
		}

		// For the next computation, the new pattern becomes the old pattern.
		for (int i = 0; i < oldMotorPattern.length; i++) {
			for (int j = 0; j < oldMotorPattern[i].length; j++) {
				if (newMotorPattern[i][j].length != oldMotorPattern[i][j].length)
					throw new AssertionError("Length not equal: " + Integer.toString(i) + ", " + Integer.toString(j));
				System.arraycopy(newMotorPattern[i][j], 0, oldMotorPattern[i][j], 0, oldMotorPattern[i][j].length);
			}
		}
		
		old_landmark_loc = landmark_loc;
		old_landmark_type = landmark_type;
		old_landmark_size = landmark_size;

		if (supervised) {
			return 1;
		} else {
			return 0;
		}
	} 
	
	// Destination can be supervised. E.g. We can train "turning left when
	// seeing
	// obstacle" concept with "turning left" concept and "seeing obstacle"
	// concept pre-trained.
	// Thus during training there is this emergent concept and supervised
	// concept. We supervised one coarse concept to be firing all the time but
	// the other finer concepts may also fire.
	// The newly supervised action would learn according to hebbian learning
	// rule.
	// During testing we only have the coarse concept (or the chained concept).
	// The chained concept would still be updating.
	// One or more of the supervised_skill and supervised_destination will be
	// null.
	public ActionConceptPair test(VisionLine[] visions, BufferedImage vision_image, int gps_diff, int supervised_skill, int supervised_destination,
			int previous_skill, boolean block_change_flag) {
		if (!initialized) {
			oldInputPattern = convert_mat(visions, vision_image, gps_diff, block_change_flag);
			oldMotorPattern = getDefaultMotor(previous_skill);
			switch (DNVERSION) {
			case 1:
				network1.computeHiddenResponse(oldInputPattern, oldMotorPattern, false);
				network1.replaceHiddenResponse();
				initialized = true;
				for (int i = 0; i < numMotor; i++) {
					newMotorPattern[i] = network1.computeMotorResponse(i);
				}
				network1.replaceMotorResponse();
				break;
			case 2:
				if (computing_mode == ComputingMode.CPU){
					network_cpu.computeHiddenResponse(oldInputPattern, oldMotorPattern, false);
					network_cpu.replaceHiddenResponse();
					initialized = true;
					for (int i = 0; i < numMotor; i++) {
						newMotorPattern[i] = network_cpu.computeMotorResponse(i);
					}
					network_cpu.replaceMotorResponse();
					break;
				} else {
					network_gpu.computeHiddenResponse(oldInputPattern, oldMotorPattern, false);
					network_gpu.replaceHiddenResponse();
					initialized = true;
					newMotorPattern = network_gpu.computeMotorResponse();
					network_gpu.replaceMotorResponse();
					break;
				}
			}

			// This part acts as top-k in the motor. 
			Action currentAction = getActionFromMotorPattern(newMotorPattern);
			old_skill = current_skill;
			current_skill = getSkillFromMotorPattern(newMotorPattern);
			int currentDestination = getDestinationFromMotorPattern(newMotorPattern);
			int current_landmark_loc = getLandmarkLocFromMotorPattern(newMotorPattern);
			int current_landmark_type = getLandmarkTypeFromMotorPattern(newMotorPattern);
			int current_landmark_size = getLandmarkSizeFromMotorPattern(newMotorPattern);
			
			if (supervised_skill == NULLVALUE) {
				supervised_skill = current_skill;
			}

			if (supervised_destination == NULLVALUE) {
				supervised_destination = currentDestination;
			}

			oldMotorPattern = getDiscreteMotorPattern(currentAction, supervised_skill, supervised_destination,
					currentCost, true);

			if (supervised_destination != NULLVALUE) {
				switch (DNVERSION) {
				case 1:
					network1.updateSupervisedMotorWeights(2, oldMotorPattern[2]);
					break;
				case 2:
					if (computing_mode == ComputingMode.CPU){
					    network_cpu.updateSupervisedMotorWeights(2, oldMotorPattern[2]);
					} else {
						// TODO: make sure this copy is correct.
						float[][][] tempPattern = new float[numMotor][][];
						for (int i = 0; i < numMotor; i++) {
							tempPattern[i] = new float[motorSize[i][0]][motorSize[i][1]];
						}
						tempPattern[2] = oldMotorPattern[2];
					    network_gpu.updateSupervisedMotorWeights(tempPattern);
					}
					break;
				}
			}
		}
		oldInputPattern = convert_mat(visions, vision_image, gps_diff, block_change_flag);
		switch (DNVERSION) {
		case 1:
			network1.computeHiddenResponse(oldInputPattern, oldMotorPattern, false);
			network1.replaceHiddenResponse();
			for (int i = 0; i < numMotor; i++) {
				newMotorPattern[i] = network1.computeMotorResponse(i);
			}
			network1.replaceMotorResponse();
			break;
		case 2:
			// first update
			if (computing_mode == ComputingMode.CPU){
                network_cpu.computeHiddenResponse(oldInputPattern, oldMotorPattern, false);
			} else {
				network_gpu.computeHiddenResponse(oldInputPattern, oldMotorPattern, false);
			}
            tempMotorPattern = getDefaultMotor(0);
            if (computing_mode == ComputingMode.CPU){
	            for (int i = 0; i < numMotor; i++) {
	                tempMotorPattern[i] = network_cpu.computeMotorResponse(i);
	            }
            } else {
            	tempMotorPattern = network_gpu.computeMotorResponse();
            }
            lateralMotorPattern = getDefaultMotor(0);
            Action old_action = getActionFromMotorPattern(oldMotorPattern);
            float[] lateral_input = getLateralInput(old_action.ordinal(), supervised_skill, supervised_destination, NULLVALUE, NULLVALUE,
                    NULLVALUE, NULLVALUE, true);
            
            if (computing_mode == ComputingMode.CPU){
	            for (int i = 0; i < lateral_zone.length; i++) {
	                int current_zone = lateral_zone[i];
	                lateralMotorPattern[current_zone] = network_cpu.computeMotorLateralResponse(lateral_input,
	                        current_zone);
	            }
            } else {
            	for (int i = 0; i < lateral_zone.length; i++) {
                    int current_zone = lateral_zone[i];
                    lateralMotorPattern[current_zone] = network_gpu.computeMotorLateralResponse(lateral_input,
                            current_zone);
                }
            }
            for (int i = 0; i < numMotor; i++) {
                for (int j = 0; j < newMotorPattern[i].length; j++) {
                    for (int k = 0; k < newMotorPattern[i][j].length; k++) {
                        tempMotorPattern[i][j][k] += lateral_percent * lateralMotorPattern[i][j][k];
                    }
                }
            }
            
            if (computing_mode == ComputingMode.CPU){
	            network_cpu.replaceHiddenResponse();
	            network_cpu.replaceMotorResponse();
            } else {
            	network_gpu.replaceHiddenResponse();
            	network_gpu.replaceMotorResponse();
            }
            Action currentAction = getActionFromMotorPattern(newMotorPattern);
            old_skill = current_skill;
    		current_skill = getSkillFromMotorPattern(newMotorPattern);
    		int currentDestination = getDestinationFromMotorPattern(newMotorPattern);
			
    		if (supervised_skill == NULLVALUE) {
    			supervised_skill = current_skill;
    		}
    		supervised_destination = currentDestination;
            tempMotorPattern = getDiscreteMotorPattern(currentAction, supervised_skill, supervised_destination, currentCost,
    				true);
            
            // second update
            if (computing_mode == ComputingMode.CPU){
	            network_cpu.computeHiddenResponse(oldInputPattern, tempMotorPattern, false);
	            for (int i = 0; i < numMotor; i++) {
	                newMotorPattern[i] = network_cpu.computeMotorResponse(i);
	            }
            } else {
            	network_gpu.computeHiddenResponse(oldInputPattern, tempMotorPattern, false);
            	newMotorPattern = network_gpu.computeMotorResponse();
            }
            lateralMotorPattern = getDefaultMotor(0);
            old_action = getActionFromMotorPattern(tempMotorPattern);
            lateral_input = getLateralInput(currentAction.ordinal(), supervised_skill, supervised_destination, NULLVALUE, NULLVALUE,
                    NULLVALUE, NULLVALUE, true);
            if (computing_mode == ComputingMode.CPU){
	            for (int i = 0; i < lateral_zone.length; i++) {
	                int current_zone = lateral_zone[i];
	                lateralMotorPattern[current_zone] = network_cpu.computeMotorLateralResponse(lateral_input,
	                        current_zone);
	            }
            } else {
            	 for (int i = 0; i < lateral_zone.length; i++) {
                     int current_zone = lateral_zone[i];
                     lateralMotorPattern[current_zone] = network_gpu.computeMotorLateralResponse(lateral_input,
                             current_zone);
                 }
            } 
            for (int i = 0; i < numMotor; i++) {
                for (int j = 0; j < newMotorPattern[i].length; j++) {
                    for (int k = 0; k < newMotorPattern[i][j].length; k++) {
                        newMotorPattern[i][j][k] += lateral_percent * lateralMotorPattern[i][j][k];
                    }
                }
            }
            if (computing_mode == ComputingMode.CPU){
	            network_cpu.replaceHiddenResponse();
	            network_cpu.replaceMotorResponse();
            } else {
            	network_gpu.replaceHiddenResponse();
	            network_gpu.replaceMotorResponse();
            }
            
            break;
		}

		Action currentAction = getActionFromMotorPattern(newMotorPattern);
		old_skill = current_skill;
		current_skill = getSkillFromMotorPattern(newMotorPattern);
		int currentDestination = getDestinationFromMotorPattern(newMotorPattern);
		
		int oldCost;
		if (block_change_flag) {
			currentCost++;
			oldCost = currentCost - 1;
		} else {
			oldCost = currentCost;
		}

		if (supervised_skill == NULLVALUE) {
			supervised_skill = current_skill;
		}

		supervised_destination = currentDestination;
		
        System.out.println("Current dest: " + currentDestination);
		oldMotorPattern = getDiscreteMotorPattern(currentAction, supervised_skill, supervised_destination, currentCost,
				true);

		return new ActionConceptPair(currentAction, supervised_skill, supervised_destination);
	}

	// Only DN-2 has this functionality in current implementation.
	public PlanResult planOneStep(VisionLine[] visions, BufferedImage vision_image, int gps_diff, boolean block_change_flag) {
		Action currentAction = Action.FORWARD;
		if (!initialized) {
			oldInputPattern = convert_mat(visions, vision_image, gps_diff, block_change_flag);
			oldMotorPattern = getDefaultMotor(NULLVALUE);
			
			if (computing_mode == ComputingMode.CPU){
				network_cpu.computeHiddenResponse(oldInputPattern, oldMotorPattern, false);
				network_cpu.replaceHiddenResponse();
				
				initialized = true;
				for (int i = 0; i < numMotor; i++) {
					newMotorPattern[i] = network_cpu.computeMotorResponse(i);
				}
				network_cpu.replaceMotorResponse();
			} else {
				network_gpu.computeHiddenResponse(oldInputPattern, oldMotorPattern, false);
				network_gpu.replaceHiddenResponse();
				
				initialized = true;
				newMotorPattern = network_gpu.computeMotorResponse();
				network_gpu.replaceMotorResponse();
			}

			currentAction = getActionFromMotorPattern(newMotorPattern);
			current_skill = getSkillFromMotorPattern(newMotorPattern);
			current_means = getDestinationFromMotorPattern(newMotorPattern);
			old_skill = current_skill;
			overt_skill = current_skill;
			old_means = current_means;
			overt = false;
			lessMore_id = NULLVALUE;
			cost_firing_id = new int[cost_firing_id.length];
		} else {
			if (overt) {
				ActionConceptPair temp_result = test(visions, vision_image, gps_diff, NULLVALUE, old_means, NULLVALUE,
						block_change_flag);
				currentAction = temp_result.action;
				current_skill = temp_result.skill;
			} else {
				float[][][] newCovertMotorPattern = new float[numMotor][][];
				float[][][] newBottomUpMotorPattern = new float[numMotor][][];
				float[][][] oldPlanningMotorPattern = getDiscreteMotorPattern0(currentAction, old_skill, old_means,
						NULLVALUE, false);
				oldPlanningMotorPattern[2][0][0] = 1;
				oldPlanningMotorPattern[2][0][1] = 1;
				//3;
				oldPlanningMotorPattern[2][0][2] = 1;
				oldPlanningMotorPattern[5][0][cost_firing_id[0]] = 1;
				oldPlanningMotorPattern[6][0][cost_firing_id[1]] = 1;
				//3;
				oldPlanningMotorPattern[7][0][cost_firing_id[2]] = 1;
				for (int i = 0; i < numMotor; i++) {
					newCovertMotorPattern[i] = new float[motorSize[i][0]][motorSize[i][1]];
					newBottomUpMotorPattern[i] = new float[motorSize[i][0]][motorSize[i][1]];
				}
				float[][][] planning_bottom_up = getDefaultInput();
				if (computing_mode == ComputingMode.CPU){
					network_cpu.computeHiddenResponse(planning_bottom_up, oldPlanningMotorPattern, false);
					network_cpu.replaceHiddenResponse();
					for (int i = 0; i < numMotor; i++) {
						newBottomUpMotorPattern[i] = network_cpu.computeMotorResponse(i);
					}
				} else {
					network_gpu.computeHiddenResponse(planning_bottom_up, oldPlanningMotorPattern, false);
					network_gpu.replaceHiddenResponse();
					newBottomUpMotorPattern = network_gpu.computeMotorResponse();
				}

				float[] lateral_input = getLateralInput(currentAction.ordinal(), old_skill, old_means,
						(int) cost_firing_id[0], (int) cost_firing_id[1], (int) cost_firing_id[2], lessMore_id, false);
				// We are currently choosing between these two destinations.
				lateral_input[motorSize[0][1] + motorSize[1][1] + 0] = 1;
				lateral_input[motorSize[0][1] + motorSize[1][1] + 1] = 1;
				//3;
				lateral_input[motorSize[0][1] + motorSize[1][1] + 2] = 1;
				for (int i = 0; i < lateral_zone.length; i++) {
					int current_zone = lateral_zone[i];
					if (computing_mode == ComputingMode.CPU){
					    newCovertMotorPattern[current_zone] = network_cpu.computeMotorLateralResponse(lateral_input,
							    current_zone);
					} else {
						newCovertMotorPattern[current_zone] = network_gpu.computeMotorLateralResponse(lateral_input,
								current_zone);
					}
				}
				for (int i = 0; i < numMotor; i++) {
					for (int j = 0; j < newCovertMotorPattern[i].length; j++) {
						for (int k = 0; k < newCovertMotorPattern[i][j].length; k++) {
							newCovertMotorPattern[i][j][k] = newCovertMotorPattern[i][j][k] * lateral_percent
									+ newBottomUpMotorPattern[i][j][k];
						}
					}
				}

				if (computing_mode == ComputingMode.CPU){
				    network_cpu.replaceMotorResponse();
				} else {
					network_gpu.replaceMotorResponse();
				}
				currentAction = getActionFromMotorPattern(newCovertMotorPattern);
				current_skill = getSkillFromMotorPattern(newCovertMotorPattern);
				old_skill = current_skill;
				current_means = getDestinationFromMotorPattern(newCovertMotorPattern);

				for (int i = 0; i < motorSize[2][1]; i++) {
					int current_id = getCostFromMotorPattern(newCovertMotorPattern, i);
					cost_firing_id[i] = current_id;
				}

				lessMore_id = getComparisonResultFromMotorPattern(newCovertMotorPattern);
				if (lessMore_id == 0) {
					current_means = 0;
				} else if (lessMore_id == 1) {
					current_means = 1;
				}else if (lessMore_id == 2) {
					current_means = 2;
				}
				overt = getCovertOvertFromMotorPattern(newCovertMotorPattern);

				// Planning thread change
				if (current_means != old_means) {
					old_skill = overt_skill;
				}
				old_means = current_means;
			}
		}

		ActionConceptPair result = new ActionConceptPair(currentAction, current_skill, current_means);
		PlanResult plan = new PlanResult(cost_firing_id.length, cost_firing_id, result);
		return plan;
	}

	boolean getCovertOvertFromMotorPattern(float[][][] motorPattern) {
		if (motorPattern[9][0][1] < motorPattern[9][0][0]) {
			return true;
		} else {
			return false;
		}
	}

	//3;
	private int getComparisonResultFromMotorPattern(float[][][] motorPattern) {
		int max_id = NULLVALUE;
		float max_value = 0;
		for (int i = 0; i < motorSize[8][1]; i++) {
			if (max_value < motorPattern[8][0][i]) {
				max_id = i;
				max_value = motorPattern[8][0][i];
			}
		}
		return max_id;
	}

	private int getCostFromMotorPattern(float[][][] motorPattern, int current_destination) {
		int max_id = 0;
		if (current_destination == 0) {
			float max_value = motorPattern[5][0][0];
			for (int i = 0; i < motorPattern[5][0].length; i++) {
				if (motorPattern[5][0][i] > max_value) {
					max_id = i;
					max_value = motorPattern[5][0][i];
				}
			}
		} else if (current_destination == 1) {
			float max_value = motorPattern[6][0][0];
			for (int i = 0; i < motorPattern[6][0].length; i++) {
				if (motorPattern[6][0][i] > max_value) {
					max_id = i;
					max_value = motorPattern[6][0][i];
				}
			}
		}else if (current_destination == 2) {    //3;
			float max_value = motorPattern[7][0][0];
			for (int i = 0; i < motorPattern[7][0].length; i++) {
				if (motorPattern[7][0][i] > max_value) {
					max_id = i;
					max_value = motorPattern[7][0][i];
				}
			}
		}
		return max_id;
	}

	public void learnPlanning(int[] skill_sequence, int curr_destination, int reward_value, float state_value) {
		for (int i = 0; i < skill_sequence.length - 1; i++) {
			float[][][] current_bottom_up_input = getDefaultInput();
			int old_skill = skill_sequence[i];
			int new_skill = skill_sequence[i + 1];
			float[][][] current_top_down_input = getDiscreteMotorPattern(Action.FORWARD, old_skill, curr_destination,
					state_value, false);
			if (computing_mode == ComputingMode.CPU){
				network_cpu.computeHiddenResponse(current_bottom_up_input, current_top_down_input, true);
				network_cpu.replaceHiddenResponse();
				float[][][] new_top_down_pattern = getDiscreteMotorPattern(Action.FORWARD, new_skill, curr_destination,
						state_value, true);
				for (int j = 0; j < numMotor; j++) {
					network_cpu.updateSupervisedMotorWeights(j, new_top_down_pattern[j]);
				}
				//3;
				network_cpu.updateLateralMotorWeights(0,
						getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), NULLVALUE);
				network_cpu.updateLateralMotorWeights(1,
						getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), new_skill);
				network_cpu.updateLateralMotorWeights(2,
						getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), curr_destination);
				if (curr_destination == 0) {
					network_cpu.updateLateralMotorWeights(5,
							getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE),
							(int) Math.floor(state_value));
					network_cpu.updateLateralMotorWeights(6,
							getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), NULLVALUE);
					network_cpu.updateLateralMotorWeights(7,
							getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), NULLVALUE);
				} else if (curr_destination == 1) {
					network_cpu.updateLateralMotorWeights(5,
							getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), NULLVALUE);
					network_cpu.updateLateralMotorWeights(6,
							getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE),
							(int) Math.floor(state_value));
					network_cpu.updateLateralMotorWeights(7,
							getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), NULLVALUE);
				}else if (curr_destination == 2) {
					network_cpu.updateLateralMotorWeights(5,
							getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), NULLVALUE);
					network_cpu.updateLateralMotorWeights(6,
							getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), NULLVALUE);
					network_cpu.updateLateralMotorWeights(7,
							getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE),
							(int) Math.floor(state_value));
				}
			} else {
				network_gpu.computeHiddenResponse(current_bottom_up_input, current_top_down_input, true);
				network_gpu.replaceHiddenResponse();
				float[][][] new_top_down_pattern = getDiscreteMotorPattern(Action.FORWARD, new_skill, curr_destination,
						state_value, true);
				network_gpu.updateSupervisedMotorWeights(new_top_down_pattern);
				network_gpu.updateLateralMotorWeights(0,
						getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), NULLVALUE);
				network_gpu.updateLateralMotorWeights(1,
						getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), new_skill);
				network_gpu.updateLateralMotorWeights(2,
						getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), curr_destination);
				if (curr_destination == 0) {
					network_gpu.updateLateralMotorWeights(5,
							getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE),
							(int) Math.floor(state_value));
					network_gpu.updateLateralMotorWeights(6,
							getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), NULLVALUE);
					network_gpu.updateLateralMotorWeights(7,
							getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), NULLVALUE);
				} else if (curr_destination == 1) {
					network_gpu.updateLateralMotorWeights(5,
							getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), NULLVALUE);
					network_gpu.updateLateralMotorWeights(6,
							getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE),
							(int) Math.floor(state_value));
					network_gpu.updateLateralMotorWeights(7,
							getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), NULLVALUE);
				}else if (curr_destination == 2) {
					network_gpu.updateLateralMotorWeights(5,
							getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), NULLVALUE);
					network_gpu.updateLateralMotorWeights(6,
							getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), NULLVALUE);
					network_gpu.updateLateralMotorWeights(7,
							getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE),
							(int) Math.floor(state_value));
				}
			}
		}
		
		// Learn final reward.
		int old_skill = skill_sequence[skill_sequence.length - 1];
		float[][][] current_top_down_input = getDiscreteMotorPattern(Action.FORWARD, old_skill, curr_destination,
				state_value, true);
		if (computing_mode == ComputingMode.CPU){
			if (reward_value == LOW_PAIN) {
				network_cpu.updateSupervisedMotorWeights(3, current_top_down_input[2]);
				network_cpu.updateLateralMotorWeights(3,
						getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), curr_destination);
			} else if (reward_value == HIGH_PAIN) {
				network_cpu.updateSupervisedMotorWeights(4, current_top_down_input[2]);
				network_cpu.updateLateralMotorWeights(4,
						getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), curr_destination);
			}
		} else {
			if (reward_value == LOW_PAIN) {
				network_gpu.updateSupervisedMotorWeights(3, current_top_down_input[2]);
				network_gpu.updateLateralMotorWeights(3,
						getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), curr_destination);
			} else if (reward_value == HIGH_PAIN) {
				network_gpu.updateSupervisedMotorWeights(4, current_top_down_input[2]);
				network_gpu.updateLateralMotorWeights(4,
						getLateralInput(NULLVALUE, old_skill, curr_destination, NULLVALUE, NULLVALUE, NULLVALUE), curr_destination);
			}
		}
	}

	public void learnReward(int value) {
		int currentDestination = getDestinationFromMotorPattern(newMotorPattern);
		float[][] current_z_p_pattern = oldMotorPattern[2];
		if (computing_mode == ComputingMode.CPU){
			if (value == LOW_PAIN) {
				network_cpu.updateSupervisedMotorWeights(3, current_z_p_pattern);
				network_cpu.updateLateralMotorWeights(3,
						getLateralInput(NULLVALUE, old_skill, currentDestination, NULLVALUE, NULLVALUE, NULLVALUE),
						currentDestination);
			} else if (value == HIGH_PAIN) {
				network_cpu.updateSupervisedMotorWeights(4, current_z_p_pattern);
				network_cpu.updateLateralMotorWeights(4,
						getLateralInput(NULLVALUE, old_skill, currentDestination, NULLVALUE, NULLVALUE, NULLVALUE),
						currentDestination);
			}
		} else {
			if (value == LOW_PAIN) {
				network_gpu.updateSupervisedMotorWeights(3, current_z_p_pattern);
				network_gpu.updateLateralMotorWeights(3,
						getLateralInput(NULLVALUE, old_skill, currentDestination, NULLVALUE, NULLVALUE, NULLVALUE),
						currentDestination);
			} else if (value == HIGH_PAIN) {
				network_gpu.updateSupervisedMotorWeights(4, current_z_p_pattern);
				network_gpu.updateLateralMotorWeights(4,
						getLateralInput(NULLVALUE, old_skill, currentDestination, NULLVALUE, NULLVALUE, NULLVALUE),
						currentDestination);
			}
		}
	}

	public void learnCovertToCovert() {
		float[][][] current_bottom_up_input = getDefaultInput();
		float[][][] covert_top_down_pattern = getMoreLessMotorPattern(NULLVALUE, NULLVALUE, NULLVALUE, 1); // all
																								// zero
		covert_top_down_pattern[5][0][17] = 1;
		covert_top_down_pattern[6][0][17] = 1;
		//3;
		covert_top_down_pattern[7][0][17] = 1;
		covert_top_down_pattern[8][0][3] = 1;
		covert_top_down_pattern[9][0][COVERT] = 1;
		if (computing_mode == ComputingMode.CPU){
			network_cpu.computeHiddenResponse(current_bottom_up_input, covert_top_down_pattern, true);
			network_cpu.replaceHiddenResponse();
			for (int j = 0; j < numMotor; j++) {
				network_cpu.updateSupervisedMotorWeights(j, covert_top_down_pattern[j]);
			}
		} else {
			network_gpu.computeHiddenResponse(current_bottom_up_input, covert_top_down_pattern, true);
			network_gpu.replaceHiddenResponse();
			network_gpu.updateSupervisedMotorWeights(covert_top_down_pattern);
		}
		//3;
		float[] current_lateral_input = getLateralInput(NULLVALUE, NULLVALUE, NULLVALUE, 17, 17, 17, 3, false);
		updateLateralWeights(current_lateral_input, NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE,
				COVERT);
	}
	

	public void learnMeansLoop(int means_num) {
		// we are now releasing 011 neurons to form loop of means. 
		for (int i = 0; i < growthrate.length; i++){
			//growthrate[i][2] = 1.0f;
			growthrate[i][3] = 0.63f;
		}
		if (computing_mode == ComputingMode.CPU){
		    network_cpu.setGrowthRate(growthrate);
		} else {
			network_gpu.setGrowthRate(growthrate);
		}
		// we train the means Y-Y connection with background image.
		float[][][] current_bottom_up_input = getDefaultInput();
		float[][][] top_down_pattern = getMoreLessMotorPattern(NULLVALUE, NULLVALUE, NULLVALUE, 1); // all 0
		top_down_pattern[2][0][means_num] = 1;
		int train_num = 5;
		for (int i = 0; i < train_num; i++){
			if (computing_mode == ComputingMode.CPU){
				network_cpu.computeHiddenResponse(current_bottom_up_input, top_down_pattern, true);
				network_cpu.replaceHiddenResponse();
				for (int j = 0; j < numMotor; j++) {
					network_cpu.updateSupervisedMotorWeights(j, top_down_pattern[j]);
				}
			} else {
				network_gpu.computeHiddenResponse(current_bottom_up_input, top_down_pattern, true);
				network_gpu.replaceHiddenResponse();
				network_gpu.updateSupervisedMotorWeights(top_down_pattern);
			}
		}
	}

	public void learnMoreLess() {
		for (int x = 8; x < 17; x++) {
			for (int y = 8; y < 17; y++) {
				float[][][] current_bottom_up_input = getDefaultInput();
				float[][][] current_top_down_input = getMoreLessMotorPattern(x, y, 1);
				float[][][] comparison_top_down_pattern = getMoreLessMotorPattern(x, y, 2);
				if (computing_mode == ComputingMode.CPU){
					network_cpu.computeHiddenResponse(current_bottom_up_input, current_top_down_input, true);
					network_cpu.replaceHiddenResponse();
					for (int j = 0; j < numMotor; j++) {
						network_cpu.updateSupervisedMotorWeights(j, comparison_top_down_pattern[j]);
					}
				} else {
					network_gpu.computeHiddenResponse(current_bottom_up_input, current_top_down_input, true);
					network_gpu.replaceHiddenResponse();
					network_gpu.updateSupervisedMotorWeights(comparison_top_down_pattern);
				}
				float[] current_lateral_input = getLateralInput(NULLVALUE, NULLVALUE, NULLVALUE, x, y);
				int compair_result = NULLVALUE;
				if (x < y) {
					compair_result = 0;
				}
				if (x == y) {
					compair_result = 1;
				}
				if (x > y) {
					compair_result = 2;
				}
				updateLateralWeights(current_lateral_input, NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE,
						compair_result, NULLVALUE);
				// Current motor pattern is with comparison result.
				if (computing_mode == ComputingMode.CPU){
				    network_cpu.computeHiddenResponse(current_bottom_up_input, comparison_top_down_pattern, true);
				    network_cpu.replaceHiddenResponse();
				} else {
					network_gpu.computeHiddenResponse(current_bottom_up_input, comparison_top_down_pattern, true);
				    network_gpu.replaceHiddenResponse();
				}
				float[][][] overt_top_down_pattern = getMoreLessMotorPattern(NULLVALUE, NULLVALUE, 1);
				int current_means;
				if (x <= y) {
					current_means = 0;
				} else {
					current_means = 1;
				}
				overt_top_down_pattern[2][0][current_means] = 1;
				if (computing_mode == ComputingMode.CPU){
					for (int j = 0; j < numMotor; j++) {
						network_cpu.updateSupervisedMotorWeights(j, overt_top_down_pattern[j]);
					}
				} else {
					network_gpu.updateSupervisedMotorWeights(overt_top_down_pattern);

				}
				current_lateral_input = getLateralInput(NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE,
						compair_result, true);
				updateLateralWeights(current_lateral_input, NULLVALUE, NULLVALUE, current_means, NULLVALUE, NULLVALUE,
						NULLVALUE, OVERT);
			}
		}
	}
	
	//3;
	public void learnMoreLess3() {
		for (int x = 7; x < 17; x++) {
			for (int y = 7; y < 17; y++) {
				for (int z = 7; z < 17; z++) {
					float[][][] current_bottom_up_input = getDefaultInput();
					float[][][] current_top_down_input = getMoreLessMotorPattern(x, y, z, 1);
					float[][][] comparison_top_down_pattern = getMoreLessMotorPattern(x, y, z, 2);
					if (computing_mode == ComputingMode.CPU){
						network_cpu.computeHiddenResponse(current_bottom_up_input, current_top_down_input, true);
						network_cpu.replaceHiddenResponse();
						for (int j = 0; j < numMotor; j++) {
							network_cpu.updateSupervisedMotorWeights(j, comparison_top_down_pattern[j]);
						}
					} else {
						network_gpu.computeHiddenResponse(current_bottom_up_input, current_top_down_input, true);
						network_gpu.replaceHiddenResponse();
						network_gpu.updateSupervisedMotorWeights(comparison_top_down_pattern);
					}
					float[] current_lateral_input = getLateralInput(NULLVALUE, NULLVALUE, NULLVALUE, x, y, z);
					int compair_result = NULLVALUE;
					if (x <= y && x <= z) {
						compair_result = 0;
					}
					if (x > y && y <= z) {
						compair_result = 1;
					}
					if (x > z && z < y) {
						compair_result = 2;
					}

					updateLateralWeights(current_lateral_input, NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE,
						compair_result, NULLVALUE);
					// Current motor pattern is with comparison result.
					if (computing_mode == ComputingMode.CPU){
						network_cpu.computeHiddenResponse(current_bottom_up_input, comparison_top_down_pattern, true);
						network_cpu.replaceHiddenResponse();
					} else {
						network_gpu.computeHiddenResponse(current_bottom_up_input, comparison_top_down_pattern, true);
						network_gpu.replaceHiddenResponse();
					}
					float[][][] overt_top_down_pattern = getMoreLessMotorPattern(NULLVALUE, NULLVALUE, NULLVALUE, 1);
					int current_means;
					if (x <= y && x <= z) {
						current_means = 0;
					} else if(y < x && y <= z){
						current_means = 1;
					}else {
						current_means = 2;
					}
					overt_top_down_pattern[2][0][current_means] = 1;
					if (computing_mode == ComputingMode.CPU){
						for (int j = 0; j < numMotor; j++) {
							network_cpu.updateSupervisedMotorWeights(j, overt_top_down_pattern[j]);
						}
					} else {
						network_gpu.updateSupervisedMotorWeights(overt_top_down_pattern);

					}
					current_lateral_input = getLateralInput(NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE, NULLVALUE,
						compair_result, true);
					updateLateralWeights(current_lateral_input, NULLVALUE, NULLVALUE, current_means, NULLVALUE, NULLVALUE, NULLVALUE,
						NULLVALUE, OVERT);
				}
			}
		}
	}

	private void updateLateralWeights(float[] current_lateral_input, int action, int skill, int means, int cost1,
			int cost2, int compair, int overt_flag) {
		if (computing_mode == ComputingMode.CPU){
			network_cpu.updateLateralMotorWeights(0, current_lateral_input, action);
			network_cpu.updateLateralMotorWeights(1, current_lateral_input, skill);
			network_cpu.updateLateralMotorWeights(2, current_lateral_input, means);
			network_cpu.updateLateralMotorWeights(5, current_lateral_input, cost1);
			network_cpu.updateLateralMotorWeights(6, current_lateral_input, cost2);
			network_cpu.updateLateralMotorWeights(7, current_lateral_input, compair);
			if (overt_flag == COVERT) {
				network_cpu.updateLateralMotorWeights(8, current_lateral_input, COVERT);
			} else if (overt_flag == OVERT) {
				network_cpu.updateLateralMotorWeights(8, current_lateral_input, OVERT);
			}
		} else {
			network_gpu.updateLateralMotorWeights(0, current_lateral_input, action);
			network_gpu.updateLateralMotorWeights(1, current_lateral_input, skill);
			network_gpu.updateLateralMotorWeights(2, current_lateral_input, means);
			network_gpu.updateLateralMotorWeights(5, current_lateral_input, cost1);
			network_gpu.updateLateralMotorWeights(6, current_lateral_input, cost2);
			network_gpu.updateLateralMotorWeights(7, current_lateral_input, compair);
			if (overt_flag == COVERT) {
				network_gpu.updateLateralMotorWeights(8, current_lateral_input, COVERT);
			} else if (overt_flag == OVERT) {
				network_gpu.updateLateralMotorWeights(8, current_lateral_input, OVERT);
			}
		}
	}
	
	//3;
	private void updateLateralWeights(float[] current_lateral_input, int action, int skill, int means, int cost1,
			int cost2,int cost3,  int compair, int overt_flag) {
		if (computing_mode == ComputingMode.CPU){
			network_cpu.updateLateralMotorWeights(0, current_lateral_input, action);
			network_cpu.updateLateralMotorWeights(1, current_lateral_input, skill);
			network_cpu.updateLateralMotorWeights(2, current_lateral_input, means);
			network_cpu.updateLateralMotorWeights(5, current_lateral_input, cost1);
			network_cpu.updateLateralMotorWeights(6, current_lateral_input, cost2);
			network_cpu.updateLateralMotorWeights(7, current_lateral_input, cost3);
			network_cpu.updateLateralMotorWeights(8, current_lateral_input, compair);
			if (overt_flag == COVERT) {
				network_cpu.updateLateralMotorWeights(9, current_lateral_input, COVERT);
			} else if (overt_flag == OVERT) {
				network_cpu.updateLateralMotorWeights(9, current_lateral_input, OVERT);
			}
		} else {
			network_gpu.updateLateralMotorWeights(0, current_lateral_input, action);
			network_gpu.updateLateralMotorWeights(1, current_lateral_input, skill);
			network_gpu.updateLateralMotorWeights(2, current_lateral_input, means);
			network_gpu.updateLateralMotorWeights(5, current_lateral_input, cost1);
			network_gpu.updateLateralMotorWeights(6, current_lateral_input, cost2);
			network_gpu.updateLateralMotorWeights(7, current_lateral_input, cost3);
			network_gpu.updateLateralMotorWeights(8, current_lateral_input, compair);
			if (overt_flag == COVERT) {
				network_gpu.updateLateralMotorWeights(9, current_lateral_input, COVERT);
			} else if (overt_flag == OVERT) {
				network_gpu.updateLateralMotorWeights(9, current_lateral_input, OVERT);
			}
		}
	}

	private float[][][] getMoreLessMotorPattern(int num1, int num2, int mode) {
		float[][][] result = new float[numMotor][][];
		for (int i = 0; i < numMotor; i++) {
			result[i] = new float[motorSize[i][0]][motorSize[i][1]];
		}
		if (mode == 1) {
			if (num1 != NULLVALUE) {
				result[5][0][num1] = 1;
			}
			if (num2 != NULLVALUE) {
				result[6][0][num2] = 1;
			}
		} else if (mode == 2) {
			if (num2 > num1) {
				result[7][0][0] = 1;
			} else if (num2 == num1) {
				result[7][0][1] = 1;
			} else if (num2 < num1) {
				result[7][0][2] = 1;
			}
		} else if (mode == 3) {
			for (int i = 0; i < numMotor; i++) {
				if ((i != 8) && (i != 7)) {
					for (int j = 0; j < motorSize[i][0]; j++) {
						for (int k = 0; k < motorSize[i][1]; k++) {
							result[i][j][k] = 1;
						}
					}
				}
			}
		}
		return result;
	}

	private float[][][] getMoreLessMotorPattern(int num1, int num2, int num3, int mode) {
		float[][][] result = new float[numMotor][][];
		for (int i = 0; i < numMotor; i++) {
			result[i] = new float[motorSize[i][0]][motorSize[i][1]];
		}
		if (mode == 1) {
			if (num1 != NULLVALUE) {
				result[5][0][num1] = 1;
			}
			if (num2 != NULLVALUE) {
				result[6][0][num2] = 1;
			}
			if (num3 != NULLVALUE) {
				result[7][0][num2] = 1;
			}
			if (num1 == NULLVALUE && num2 == NULLVALUE && num3 == NULLVALUE){
			//add comparison none
				result[8][0][motorSize[8][1]-1] = 1;
			}
		} else if (mode == 2) {
			if (num2 >= num1 && num3 >= num1 ) {
				result[8][0][0] = 1;
			}  else if (num2 < num1 && num2 <= num3) {
				result[8][0][1] = 1;
			} else if (num3 < num1 && num3 < num1) {
				result[8][0][2] = 1;
			}
		} else if (mode == 3) {
			for (int i = 0; i < numMotor; i++) {
				if ((i != 8) && (i != 9)) {
					for (int j = 0; j < motorSize[i][0]; j++) {
						for (int k = 0; k < motorSize[i][1]; k++) {
							result[i][j][k] = 1;
						}
					}
				}
			}
		}
		return result;
	}
	
	// Get the lateral input for Z area. In the current setting, Z_skills only
	// accept input from Z_skills
	// and Z_destination.
	private float[] getLateralInput(int old_action, int old_skill, int supervised_destination, int cost1, int cost2) {
		float[] result = new float[lateral_length];
		if (old_action != NULLVALUE) {
			result[old_action] = 1;
		}
		if (old_skill != NULLVALUE) {
			result[motorSize[0][1] + old_skill] = 1;
		}
		if (supervised_destination != NULLVALUE) {
			result[motorSize[0][1] + motorSize[1][1] + supervised_destination] = 1;
		}
		if (cost1 != NULLVALUE) {
			result[motorSize[0][1] + motorSize[1][1] + motorSize[2][1] + cost1] = 1;
		}
		if (cost2 != NULLVALUE) {
			result[motorSize[0][1] + motorSize[1][1] + motorSize[2][1] + motorSize[5][1] + cost2] = 1;
		}
		return result;
	}

	//3;
	private float[] getLateralInput(int old_action, int old_skill, int supervised_destination, int cost1, int cost2, int cost3) {
		float[] result = new float[lateral_length];
		if (old_action != NULLVALUE) {
			result[old_action] = 1;
		}
		if (old_skill != NULLVALUE) {
			result[motorSize[0][1] + old_skill] = 1;
		}
		if (supervised_destination != NULLVALUE) {
			result[motorSize[0][1] + motorSize[1][1] + supervised_destination] = 1;
		}
		else{ // add none
			result[motorSize[0][1] + motorSize[1][1] + motorSize[2][1]] = 1;
		}
		if (cost1 != NULLVALUE) {
			result[motorSize[0][1] + motorSize[1][1] + motorSize[2][1] + cost1] = 1;
		}
		else{ // add none
			result[motorSize[0][1] + motorSize[1][1] + motorSize[2][1] + motorSize[5][1]] = 1;
		}
		if (cost2 != NULLVALUE) {
			result[motorSize[0][1] + motorSize[1][1] + motorSize[2][1] + motorSize[5][1] + cost2] = 1;
		}
		else{ // add none
			result[motorSize[0][1] + motorSize[1][1] + motorSize[2][1] + motorSize[5][1] + motorSize[6][1]] = 1;
		}
		if (cost3 != NULLVALUE) {
			result[motorSize[0][1] + motorSize[1][1] + motorSize[2][1] + motorSize[5][1] + motorSize[6][1] + cost3] = 1;
		}
		else{ // add none
			result[motorSize[0][1] + motorSize[1][1] + motorSize[2][1] + motorSize[5][1] + motorSize[6][1] + motorSize[7][1]] = 1;
		}
		return result;
	}
	private float[] getLateralInput(int old_action, int old_skill, int supervised_destination, int cost1, int cost2,
			int compair_result, boolean overt) {
		float[] result = new float[lateral_length];
		if (old_action != NULLVALUE) {
			result[old_action] = 1;
		}
		if (old_skill != NULLVALUE) {
			result[motorSize[0][1] + old_skill] = 1;
		}
		if (supervised_destination != NULLVALUE) {
			result[motorSize[0][1] + motorSize[1][1] + supervised_destination] = 1;
		}
		if (cost1 != NULLVALUE) {
			result[motorSize[0][1] + motorSize[1][1] + motorSize[2][1] + cost1] = 1;
		}
		if (cost2 != NULLVALUE) {
			result[motorSize[0][1] + motorSize[1][1] + motorSize[2][1] + motorSize[5][1] + cost2] = 1;
		}
		if (compair_result != NULLVALUE) {
			result[motorSize[0][1] + motorSize[1][1] + motorSize[2][1] + motorSize[5][1] + motorSize[6][1]
					+ compair_result] = 1;
		}
		if (overt == false) {
			result[motorSize[0][1] + motorSize[1][1] + motorSize[2][1] + motorSize[5][1] + motorSize[6][1]
					+ motorSize[7][1] + 1] = 1;
		}
		return result;
	}
	
	//3
	private float[] getLateralInput(int old_action, int old_skill, int supervised_destination, int cost1, int cost2, int cost3,
			int compair_result, boolean overt) {
		float[] result = new float[lateral_length];
		if (old_action != NULLVALUE) {
			result[old_action] = 1;
		}
		if (old_skill != NULLVALUE) {
			result[motorSize[0][1] + old_skill] = 1;
		}
		if (supervised_destination != NULLVALUE) {
			result[motorSize[0][1] + motorSize[1][1] + supervised_destination] = 1;
		}
		if (cost1 != NULLVALUE) {
			result[motorSize[0][1] + motorSize[1][1] + motorSize[2][1] + cost1] = 1;
		}
		if (cost2 != NULLVALUE) {
			result[motorSize[0][1] + motorSize[1][1] + motorSize[2][1] + motorSize[5][1] + cost2] = 1;
		}
		if (cost3 != NULLVALUE) {
			result[motorSize[0][1] + motorSize[1][1] + motorSize[2][1] + motorSize[5][1] + motorSize[6][1] + cost3] = 1;
		}
		if (compair_result != NULLVALUE) {
			result[motorSize[0][1] + motorSize[1][1] + motorSize[2][1] + motorSize[5][1] + motorSize[6][1] + motorSize[7][1]
					+ compair_result] = 1;
		}
		if (overt == false) {
			result[motorSize[0][1] + motorSize[1][1] + motorSize[2][1] + motorSize[5][1] + motorSize[6][1]
					+ motorSize[7][1] + motorSize[8][1] + 1] = 1;
		}
		return result;
	}

	// The default motor response.
	private float[][][] getDefaultMotor(int previous_skill) {
		float[][][] result = new float[numMotor][][];
		for (int i = 0; i < numMotor; i++) {
			result[i] = new float[motorSize[i][0]][motorSize[i][1]];
			//add none concept
			
			if (i == 2 || (i >=5  && i <= 8)){
				result[i][0][motorSize[i][1]-1] = 1.0f;
			}
			
		}
		result[0][0][Action.FORWARD.ordinal()] = 1; // Action Motor;
		if (previous_skill != NULLVALUE) {
			result[1][0][previous_skill] = 1;
		}
//		else{  //add none
//			result[1][0][motorSize[1][1]-1] = 1;
//		}

		return result;
	}

	private float[][][] getDefaultInput() {
		float[][][] input_pattern = new float[numInput][][];
		for (int i = 0; i < numInput; i++) {
			input_pattern[i] = new float[inputSize[i][0]][inputSize[i][1]];
			for (int j = 0; j < inputSize[i][0]; j++) {
				for (int k = 0; k < inputSize[i][1]; k++) {
					input_pattern[i][j][k] = 1.0f;
				}
			}
		}
		return input_pattern;
	}

	// Convert the supervised action and concept to our desired format (i.e. 3D
	// array of motor response).
	private float[][][] getDiscreteMotorPattern(Action currentAction, int current_skill, int current_destination,
			float current_value, boolean overt) {
		return getDiscreteMotorPattern(currentAction, current_skill, current_destination, current_value, overt,
				                       NULLVALUE, NULLVALUE, NULLVALUE);
	}
	
	private float[][][] getDiscreteMotorPattern0(Action currentAction, int current_skill, int current_destination,
			float current_value, boolean overt) {
		return getDiscreteMotorPattern0(currentAction, current_skill, current_destination, current_value, overt,
				                       NULLVALUE, NULLVALUE, NULLVALUE);
	}
	
	// Convert the supervised action and concept to our desired format (i.e. 3D
	// array of motor response).
	//3; from 7 add 1
	private float[][][] getDiscreteMotorPattern(Action currentAction, int current_skill, int current_destination,
			float current_value, boolean overt, int landmark_loc, int landmark_type, int landmark_size) {
		float[][][] result = new float[numMotor][][];
		for (int i = 0; i < numMotor; i++) {
			result[i] = new float[motorSize[i][0]][motorSize[i][1]];
		}
		result[0][0][currentAction.ordinal()] = 1; // Action Motor;
		if (current_skill != NULLVALUE) {
			result[1][0][current_skill] = 1; // Skill Motor;
		}
		if (current_destination != NULLVALUE) {
			result[2][0][current_destination] = 1; // Destination Motor;
		}
		else{ // add none
			result[2][0][motorSize[2][1]-1] = 1;
		}
		if (current_value != NULLVALUE) {
			if (current_destination == 0) {
				result[5][0][(int) Math.floor(current_value)] = 1;
			} else if (current_destination == 1) {
				result[6][0][(int) Math.floor(current_value)] = 1;
			}else if (current_destination == 2) {
				result[7][0][(int) Math.floor(current_value)] = 1;
			}
		}
		else{ // add none
			result[5][0][motorSize[5][1]-1] = 1;
			result[6][0][motorSize[6][1]-1] = 1;
			result[7][0][motorSize[7][1]-1] = 1;
			result[8][0][motorSize[8][1]-1] = 1;
		}  

		if (overt == false) {
			result[9][0][1] = 1;
		}
		
		if((landmark_loc==NULLVALUE)&&(landmark_type==NULLVALUE)&&(landmark_size==NULLVALUE)) {
			;
		}
		else {
			if((landmark_loc==NULLVALUE)||(landmark_type==NULLVALUE)||(landmark_size==NULLVALUE)) {
				System.out.println("type:"+landmark_type+" loc:"+landmark_loc+" size:"+landmark_size);
				throw new java.lang.Error("landmark recognition failure");
			}
			result[10][0][landmark_loc] = 1;
			result[11][0][landmark_type] = 1;
			result[12][0][landmark_size] = 1;
			
		}
		
		return result;
	}
		
		private float[][][] getDiscreteMotorPattern0(Action currentAction, int current_skill, int current_destination,
				float current_value, boolean overt, int landmark_loc, int landmark_type, int landmark_size) {
			float[][][] result = new float[numMotor][][];
			for (int i = 0; i < numMotor; i++) {
				result[i] = new float[motorSize[i][0]][motorSize[i][1]];
			}
			result[0][0][currentAction.ordinal()] = 1; // Action Motor;
			if (current_skill != NULLVALUE) {
				result[1][0][current_skill] = 1; // Skill Motor;
			}
			if (current_destination != NULLVALUE) {
				result[2][0][current_destination] = 1; // Destination Motor;
			}
			if (current_value != NULLVALUE) {
				if (current_destination == 0) {
					result[5][0][(int) Math.floor(current_value)] = 1;
				} else if (current_destination == 1) {
					result[6][0][(int) Math.floor(current_value)] = 1;
				}else if (current_destination == 2) {
					result[7][0][(int) Math.floor(current_value)] = 1;
				}
			}  

			if (overt == false) {
				result[9][0][1] = 1;
			}
		
		// first check the where what are valid.
		if ((landmark_loc == NULLVALUE) && (landmark_type == NULLVALUE) && (landmark_size == NULLVALUE)) {
			; // this is fine. move on.
		} else {
			if ((landmark_loc == NULLVALUE) || (landmark_type == NULLVALUE) || (landmark_size == NULLVALUE)) {
				throw new java.lang.Error("land mark recognition failure");
			}
			result[10][0][landmark_loc] = 1;
			result[11][0][landmark_type] = 1;
			result[12][0][landmark_size] = 1;
		}
		
		return result;
	}

	// Get the concept id from the motor pattern.
	private int getSkillFromMotorPattern(float[][][] motorPattern) {
		int max_id = NULLVALUE;
		float max_val = 0;
		for (int i = 0; i < motorPattern[1][0].length; i++) {
			if (motorPattern[1][0][i] > max_val) {
				max_val = motorPattern[1][0][i];
				max_id = i;
			}
		}
		return max_id;
	}
	
	//3;
	private int getLandmarkLocFromMotorPattern(float[][][] motorPattern){
		int max_id = NULLVALUE;
		float max_val = 0;
		for (int i = 0; i < motorPattern[10][0].length; i++) {
			if (motorPattern[10][0][i] > max_val) {
				max_val = motorPattern[10][0][i];
				max_id = i;
			}
		}
		return max_id;
	}
	//3;
	private int getLandmarkTypeFromMotorPattern(float[][][] motorPattern){
		int max_id = NULLVALUE;
		float max_val = 0;
		for (int i = 0; i < motorPattern[11][0].length; i++) {
			if (motorPattern[11][0][i] > max_val) {
				max_val = motorPattern[11][0][i];
				max_id = i;
			}
		}
		return max_id;
	}
	//3;
	private int getLandmarkSizeFromMotorPattern(float[][][] motorPattern){
		int max_id = NULLVALUE;
		float max_val = 0;
		for (int i = 0; i < motorPattern[12][0].length; i++) {
			if (motorPattern[12][0][i] > max_val) {
				max_val = motorPattern[12][0][i];
				max_id = i;
			}
		}
		return max_id;
	}

	// Get the destination id from the motor pattern.
	private int getDestinationFromMotorPattern(float[][][] motorPattern) {
		int max_id = NULLVALUE;
		float max_val = 0;
		float[] destination_response = new float[motorPattern[2][0].length];
		for (int i = 0; i < motorPattern[2][0].length; i++) {
			destination_response[i] = motorPattern[2][0][i];
		}

		for (int i = 0; i < destination_response.length; i++) {
			if (destination_response[i] > max_val) {
				max_val = destination_response[i];
				max_id = i;
			}
		}
		return max_id;
	}

	// Get the value of the current destination
	public float getEmergentValueFromMotorPattern(float[][][] pattern, int current_destination) {
		float result;
		result = pattern[2][0][current_destination] - pain_inhibit_rate_low * pattern[3][0][current_destination]
				- pain_inhibit_rate_high * pattern[4][0][current_destination];
		return result;
	}

	public void trainWhereWhat(VisionLine[] visions, BufferedImage vision_image, env current_type,
			int current_loc, int current_scale) {
		if (DNVERSION == 1) {
			throw new java.lang.Error("DN1 does not support this.");
		}
		oldInputPattern = convert_mat(visions, vision_image, NULLVALUE, false);
		oldMotorPattern = whereWhatMotor(current_type, current_loc, current_scale);
		curr_loc = NULLVALUE;
		curr_type = NULLVALUE;
		curr_scale = NULLVALUE;
		if (computing_mode == ComputingMode.CPU){
			//network_cpu.computeHiddenResponse(oldInputPattern, oldMotorPattern, true, current_type, current_loc, current_scale);
			network_cpu.computeHiddenResponse(oldInputPattern, oldMotorPattern, true);
			curr_loc = current_loc;
			curr_scale = current_scale;
			if(current_type == null){
				curr_type = NULLVALUE;
			}
			else{
				curr_type = current_type.ordinal();
			}
			network_cpu.replaceHiddenResponse();
		}
		newMotorPattern = whereWhatMotor(current_type, current_loc, current_scale);
		network_cpu.updateSupervisedMotorWeights(newMotorPattern);
	}
	
	public int[] testWhereWhat(VisionLine[] visions, BufferedImage vision_image, env current_type,
			int current_loc, int current_scale){
		if (DNVERSION == 1) {
			throw new java.lang.Error("DN1 does not support this.");
		}
		oldInputPattern = convert_mat(visions, vision_image, NULLVALUE, false);
		oldMotorPattern = whereWhatMotor(null, NULLVALUE, NULLVALUE);
		if (computing_mode == ComputingMode.CPU){
			network_cpu.computeHiddenResponse(oldInputPattern, oldMotorPattern, false);
			network_cpu.replaceHiddenResponse();
		}
		newMotorPattern = whereWhatMotor(null, NULLVALUE, NULLVALUE);
		for (int i = 0; i < numMotor; i++) {
			newMotorPattern[i] = network_cpu.computeMotorResponse(i);
        }
		
		// return where, what and scale
		int[] result = new int[3];
		result[0] = getLandmarkLocFromMotorPattern(newMotorPattern);
		result[1] = getLandmarkTypeFromMotorPattern(newMotorPattern);
		result[2] = getLandmarkSizeFromMotorPattern(newMotorPattern);
		
		return result;
	}
	
	//3; every index add 1
	private float[][][] whereWhatMotor(env current_type, int current_loc, int current_scale) {
		float[][][] result = new float[numMotor][][];
		for (int i = 0; i < numMotor; i++) {
			result[i] = new float[motorSize[i][0]][motorSize[i][1]];
		}
		if (current_loc != NULLVALUE){
		    result[10][0][current_loc] = 1;
		}
		if (current_type != null){
		    result[11][0][current_type.ordinal()] = 1;
		}
		if (current_scale != NULLVALUE){
		    result[12][0][current_scale] = 1;
		}
		return result;
	}

	// This is used when getting back from the destination or moving towards the
	// learning start point.
	public void computeIdle(VisionLine[] visions, BufferedImage vision_image, int gps_diff, float current_value, 
			                int curr_skill, int curr_means, boolean block_change_flag) {
//		oldInputPattern = convert_mat(visions, gps_diff, block_change_flag);
//		oldMotorPattern = getGoBackMotor(curr_skill);
//		if (curr_skill != NULLVALUE && curr_means == NULLVALUE) {
//			switch (DNVERSION) {
//			case 1:
//				network1.computeHiddenResponse(oldInputPattern, oldMotorPattern, true);
//				network1.replaceHiddenResponse();
//				break;
//			case 2:
//				network2.computeHiddenResponse(oldInputPattern, oldMotorPattern, true);
//				network2.replaceHiddenResponse();
//				break;
//			}
//			newMotorPattern = getGoBackMotor(curr_skill);
//			switch (DNVERSION) {
//			case 1:
//				for (int i = 0; i < numMotor; i++) {
//					network1.updateSupervisedMotorWeights(i, newMotorPattern[i]);
//				}
//				network1.replaceMotorResponse();
//				break;
//			case 2:
//				for (int i = 0; i < numMotor; i++) {
//					network2.updateSupervisedMotorWeights(i, newMotorPattern[i]);
//				}
//				network2.replaceMotorResponse();
//				break;
//			}
//		} else {
//			switch (DNVERSION) {
//			case 1:
//				network1.computeHiddenResponse(oldInputPattern, oldMotorPattern, false);
//				network1.replaceHiddenResponse();
//				break;
//			case 2:
//				network2.computeHiddenResponse(oldInputPattern, oldMotorPattern, false);
//				network2.replaceHiddenResponse();
//				break;
//			}
//		}
	}

	private float[][][] getGoBackMotor(int curr_skill) {
		float[][][] result = new float[numMotor][][];
		for (int i = 0; i < numMotor; i++) {
			result[i] = new float[motorSize[i][0]][motorSize[i][1]];
		}
		if (curr_skill != NULLVALUE){
		    result[1][0][curr_skill] = 1;
		}
		result[2][0][get_back_dest_id] = 1;

		return result;
	}

	// Read the action instruction from the input motor pattern.
	// If continuous motor, then we need to convert the generated heading angle
	// to an action.
	// The emergent motor pattern is treated as left, right, and forward forces.
	// And the direction of
	// that force is the desired heading direction.
	// If not continuous motor, then we just do top-1 competition among all
	// motor neurons.
	private Action getActionFromMotorPattern(float[][][] motorPattern) {
		if (!continuous_motor_flag) {
			int max_id = -1;
			float max_val = 0;
			for (int i = 0; i < newMotorPattern[0][0].length; i++) {
				if (newMotorPattern[0][0][i] > max_val) {
					max_val = newMotorPattern[0][0][i];
					max_id = i;
				}
			}
			if (max_id == -1) {
				max_id = Action.STOP.ordinal();
			}
			return Action.values()[max_id];
		} else {
			float right_force = motorPattern[0][0][Action.RIGHT.ordinal()];
			float left_force = motorPattern[0][0][Action.LEFT.ordinal()];
			float forward_force = motorPattern[0][0][Action.FORWARD.ordinal()];

			if (right_force > left_force) {
				float theta;
				if (forward_force > 0) {
					theta = (float) Math.toDegrees(Math.atan((right_force - left_force) / (forward_force)));
				} else {
					theta = 90;
				}
				if (theta > 45) {
					return Action.RIGHT;
				} else {
					return Action.FORWARD;
				}
			} else if (left_force > right_force) {
				float theta;
				if (forward_force > 0) {
					theta = (float) Math.toDegrees(Math.atan((left_force - right_force) / (forward_force)));
				} else {
					theta = 90;
				}
				if (theta > 45) {
					return Action.LEFT;
				} else {
					return Action.FORWARD;
				}
			} else {
				return Action.FORWARD;
			}
		}
	}

	// Convert the vision and gps input to our desired format.
	// Returns a 3D array of size oldInputPattern.
	// Each VisionLine is represented as continuous r,g,b bits in a 1-d array.
	// Each type of vision (meaning that the recognized result of that object)
	// has a specific color.
	// The intensity of that color is the distance of that recognized object.
	// GPS is represented as 3 bits, with [1, 0, 0] as left, [0, 1, 0] as
	// forward, and [0, 0, 1] as right.
	// In current environment there is no need for an extra bit of stop, as the
	// map automatically resets once
	// it reaches the destination.
	private float[][][] convert_mat(VisionLine[] visions, BufferedImage vision_image, int gps_diff, boolean block_change_flag) {
		float[][][] input_pattern = new float[numInput][][];
		for (int i = 0; i < numInput; i++) {
			input_pattern[i] = new float[inputSize[i][0]][inputSize[i][1]];
		}
		if (vision_2D_flag == false){
			for (int i_vision = 0; i_vision < visions.length; i_vision++) {
				if (visions[i_vision].type == env.WALL) {
					input_pattern[0][0][i_vision * 3 + 0] = visions[i_vision].getLength();
				}
				if (visions[i_vision].type == env.OPEN || visions[i_vision].type == env.DEST
						|| visions[i_vision].type == env.RWRD) {
					input_pattern[0][0][i_vision * 3 + 1] = visions[i_vision].getLength();
				}
				if (visions[i_vision].type == env.OBST) {
					input_pattern[0][0][i_vision * 3 + 2] = visions[i_vision].getLength();
				}
				if (visions[i_vision].type == env.LIGHT_PASS) {
					input_pattern[0][0][i_vision * 3 + 1] = visions[i_vision].getLength();
					input_pattern[0][0][i_vision * 3 + 2] = visions[i_vision].getLength();
				}
				if (visions[i_vision].type == env.LIGHT_STOP) {
					input_pattern[0][0][i_vision * 3 + 0] = visions[i_vision].getLength();
					input_pattern[0][0][i_vision * 3 + 1] = visions[i_vision].getLength();
				}
			}
		} else {
			int count = 0;
			for (int i = 0; i < vision_image.getHeight(); i++){
				for (int j = 0; j < vision_image.getWidth(); j++){
					Color current_color = new Color(vision_image.getRGB(j, i));
					input_pattern[0][0][count * 3 + 0] = ((float)current_color.getRed()) 
							                           * ((float)Agent.vision_range) / 255.0f;
					input_pattern[0][0][count * 3 + 1] = ((float)current_color.getGreen()) 
							                           * ((float)Agent.vision_range) / 255.0f;
					input_pattern[0][0][count * 3 + 2] = ((float)current_color.getBlue()) 
							                           * ((float)Agent.vision_range) / 255.0f;
					count ++ ;
				}
			}
			
		}
		if (gps_diff != NULLVALUE) {
			for (int i = 0; i < inputSize[1][1]/3; i++){
				if (gps_diff < 0) {
					input_pattern[1][0][0 + 3 * i] = 1 * (float) Agent.vision_range;
					input_pattern[1][0][1 + 3 * i] = 0;
					input_pattern[1][0][2 + 3 * i] = 0;
				} else if (gps_diff > 0) {
					input_pattern[1][0][0 + 3 * i] = 0;
					input_pattern[1][0][1 + 3 * i] = 0;
					input_pattern[1][0][2 + 3 * i] = 1 * (float) Agent.vision_range;
					;
				} else {
					input_pattern[1][0][0 + 3 * i] = 0;
					input_pattern[1][0][1 + 3 * i] = 1 * (float) Agent.vision_range;
					input_pattern[1][0][2 + 3 * i] = 0;
				}
			}
		}
		if (block_change_flag) {
			for (int i = 0; i < inputSize[2][1]; i++){
			    input_pattern[2][0][0 + i] = 1;
			}
		} else {
			for (int i = 0; i < Agent.vision_num; i++){
			    input_pattern[2][0][0 + i] = 0;
			}
		}
		return input_pattern;
	}

	// Return oldMotorPattern for debugging purposes.
	public float[][][] getMotorPattern() {
		return oldMotorPattern;
	}

	// Return the number of used neurons for each hidden layer.
	public int[] getUsedNeurons() {
		switch (DNVERSION) {
		case 1:
			return network1.getUsedNeurons();
		case 2:
			if (computing_mode == ComputingMode.CPU){
			    return network_cpu.getUsedNeurons();
			} else {
				return network_gpu.getUsedNeurons();
			}
		default:
			return null;
		}
	}

	public void sendInfoToGui() {
		if (computing_mode == ComputingMode.GPU){
			throw new java.lang.Error("GPU GUI not supported");
		}
		int numTypes = 0;
		I_GUI.numSensor = numInput;
		I_GUI.numMotor = numMotor;
		I_GUI.motorSize = motorSize;
		I_GUI.inputSize = inputSize;		
		for (int i = 0; i < typeNum.length; i++){
			if (typeNum[i] != 0) {
				numTypes ++;
			}
		}
		int totalNum = numTypes * 2 + numHiddenNeurons[0];
		I_GUI.mUsedneurons = network_cpu.getUsedNeurons()[0] - numTypes * 2;
		I_GUI.numTypes = numTypes;
		I_GUI.numNeurons = numHiddenNeurons;
		I_GUI.numNeurons[0] = network_cpu.getUsedNeurons()[0] - numTypes * 2;
		
		I_GUI.mZLocation = network_cpu.send_z_location();
		I_GUI.mXLocation = set_x_location();
		I_GUI.oldmotor = oldMotorPattern;
		I_GUI.oldinput = getDefaultInput();
		for (int i = 0; i < oldInputPattern.length; i++){
			for (int j = 0; j < oldInputPattern[i].length; j++){
				for (int k = 0; k < oldInputPattern[i][j].length; k++){
					I_GUI.oldinput[i][j][k] = oldInputPattern[i][j][k]/(float)Agent.vision_range;
				}
			}
		}
		
		float[][] temp5 = network_cpu.send_y_bottomup_weights(totalNum, 1);
		I_GUI.mBottomupweights = new float[temp5.length][temp5[0].length];
		for (int j = 0; j < temp5.length; j++) {
			System.arraycopy(temp5[j], 0, I_GUI.mBottomupweights[j], 0, temp5[j].length);
		}

		float[][] temp6 = network_cpu.send_y_topdown_weights(totalNum, 1);
		I_GUI.mTopdownweights = new float[temp6.length][temp6[0].length];
		for (int j = 0; j < temp6.length; j++) {
			System.arraycopy(temp6[j], 0, I_GUI.mTopdownweights[j], 0, temp6[j].length);
		}

		float[][] temp1 = network_cpu.send_y_inhibition_weights(totalNum, 1);
		I_GUI.mInhibitionweights = new float[temp1.length][temp1[0].length];
		for (int j = 0; j < temp1.length; j++) {
			System.arraycopy(temp1[j], 0, I_GUI.mInhibitionweights[j], 0, temp1[j].length);
		}

		float[][] temp2 = network_cpu.send_y_inhibition_masks(totalNum, 1);
		I_GUI.mInhibitionMask = new float[temp2.length][temp2[0].length];
		for (int j = 0; j < temp2.length; j++) {
			System.arraycopy(temp2[j], 0, I_GUI.mInhibitionMask[j], 0, temp2[j].length);
		}
		float[][] temp4 = network_cpu.send_y_lateral_weights(totalNum, 1);
		I_GUI.mLateralweights = new float[temp4.length][temp4[0].length];
		for (int j = 0; j < temp4.length; j++) {
			System.arraycopy(temp4[j], 0, I_GUI.mLateralweights[j], 0, temp4[j].length);
		}
		float[][][] temp3 = network_cpu.send_y_location();
		I_GUI.mLocation = new float[temp3[0].length][temp3[0][0].length];
		I_GUI.mUsedneurons = temp3[0].length;
		for (int j = 0; j < temp3[0].length; j++) {
			System.arraycopy(temp3[0][j], 0, I_GUI.mLocation[j], 0, temp3[0][j].length);
		}
		float[] temp7 = network_cpu.send_y_response(totalNum, 1);
		I_GUI.yresponse = new float[temp7.length];
		System.arraycopy(temp7, 0, I_GUI.yresponse, 0, temp7.length);

		float[][] temp8 = network_cpu.send_z_bottomup_weights(1);
		I_GUI.mZBottomupweights = new float[temp8.length][temp8[0].length];
		for (int j = 0; j < temp8.length; j++) {
			System.arraycopy(temp8[j], 0, I_GUI.mZBottomupweights[j], 0, temp8[j].length);
		}
	}

	public int getDisplayNum() {
		int numTypes = 0;
		for (int i = 0; i < typeNum.length; i++){
			if (typeNum[i] != 0) {
				numTypes ++;
			}
		}
		int totalNum = numTypes * 2 + numHiddenNeurons[0];
		return totalNum;
	}

	public int getTopDownNum() {
		int numTopDown = 0;
        for(int i = 0; i < numMotor; i++){
            numTopDown += motorSize[i][0]*motorSize[i][1];
        }
        return numTopDown;
	}
	
	public float[][][] set_x_location(){
		float[][][] temp = new float[numInput][][];
		for(int i=0; i<numInput; i++){
        	temp[i] = new float[inputSize[i][0]*inputSize[i][1]][3];
        }
		for (int i = 0; i <numInput; i++) {
			for(int j=0; j < inputSize[i][0]*inputSize[i][1]; j++){
				for(int k = 0; k < 3; k++){
					temp[i][j][k] = Math.min((float)(Math.random()*10+4.0f),10.0f)/12.0f;}
				}
			}
		return temp;
	}
}
