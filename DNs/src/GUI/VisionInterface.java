package GUI;
import java.io.*;
import java.util.*;

public class VisionInterface {
	private int[][] input_size; // num_of_inputs * 2, should be [[38, 38]]
	private int scale_count = 2; // global and local scale.
	private int what_count = 5; // type = {'ob', 'cr', 'of', 'n', 're'};
	private static int where_count; // where_count is dependent on the receptive
									// field
	// size and stride.
	private int rf_size;
	private int rf_stride;
	private int[][] rf_id_loc; // rf_id_loc[i] is the [height, width] of the ith
								// rf.

	// Initialization function reads settings file and creates a DN network.
	public VisionInterface() {
		// The first stage we need to train where what
		Settings st = new Settings("Image_Data/settings_vision.txt");

		int numInput = st.getNumInput(); // The number of sensor the network
											// will use. For this contest, we
											// will use one sensor.
		int[][] inputSize = st.getInputSize(); // This array will have the
												// (height, width) of each
												// sensor.
		input_size = inputSize;
		int rfSize = st.getRfSize();
		int rfStride = st.getRfStride();
		rf_size = rfSize;
		rf_stride = rfStride;
		rf_id_loc = configure_where_count(rf_size, rf_stride, input_size);

		int numMotor = 3; // We only have where what scale motor at first stage.
		int[][] motorSize = { { 1, where_count }, { 1, what_count },
				{ 1, scale_count } }; // (height, width) of each motor.
		int[] topKMotor = { 1, 1, 1 }; // Topk winning neurons of each
										// motor layer.

		int numHidden = 1; // Number of hidden layers.
		int[] numHiddenNeurons = st.getHiddenSize();
		int[] topKHidden = { 1 }; // Topk winning neurons for each hidden layer.
	};

	public static int[][] configure_where_count(int rfSize, int rfStride,
			int[][] inputSize) {
		// in matlab we use the rf_id in col major order.
		// but rf_id_loc is [id][height][width]
		where_count = 0;
		int[][] rfIdLoc;
		if(rfSize!=0){
		  int half_rf_size = (rfSize - 1) / 2;
	      for (int width = half_rf_size; width < inputSize[0][1] - half_rf_size; width += rfStride) {
			for (int height = half_rf_size; height < inputSize[0][0] - half_rf_size; height += rfStride) {
				where_count++;
			}
	      }
		

		  rfIdLoc = new int[where_count][2];
		  int id = 0;
		  for (int width = half_rf_size; width < inputSize[0][1] - half_rf_size; width += rfStride) {
			for (int height = half_rf_size; height < inputSize[0][0] - half_rf_size; height += rfStride) {
				rfIdLoc[id][0] = height;
				rfIdLoc[id][1] = width;
				id++;
			}
		  }
		}
		else{ 
		  rfIdLoc = new int[1][2];
		  rfIdLoc[0][0] = 0;
		  rfIdLoc[0][1] = 0;
		  
		}
		return rfIdLoc;
	}

	public float[][][] generateSupervisedMotor(int current_where_id,
			int current_what_id, int current_scale_id) {
		float[][][] current_motor_pattern = new float[3][1][];
		current_motor_pattern[0][0] = new float[where_count];

		if (current_where_id > 0) {
			current_motor_pattern[0][0][current_where_id] = 1;
		}

		current_motor_pattern[1][0] = new float[what_count];
		current_motor_pattern[1][0][current_what_id] = 1;

		current_motor_pattern[2][0] = new float[scale_count];
		current_motor_pattern[2][0][current_scale_id] = 1;

		return current_motor_pattern;
	}

	public float[][][] generatePartiallySupervisedMotor(int current_where_id) {
		float[][][] current_motor_pattern = new float[3][1][];
		current_motor_pattern[0][0] = new float[where_count];

		if (current_where_id > 0) {
			current_motor_pattern[0][0][current_where_id] = 1;
		}

		current_motor_pattern[1][0] = new float[what_count];
		current_motor_pattern[2][0] = new float[scale_count];
		return current_motor_pattern;
	}

	static void shuffleArray(int[] ar) {
		// If running on Java 6 or older, use `new Random()` on RHS here
		Random rnd = new Random();
		rnd.setSeed(5);
		for (int i = ar.length - 1; i > 0; i--) {
			int index = rnd.nextInt(i + 1);
			// Simple swap
			int a = ar[index];
			ar[index] = ar[i];
			ar[i] = a;
		}
	}

	public static void removeSamplingRecords() throws IOException,
			InterruptedException {
		Process p = Runtime.getRuntime().exec(
				"rm Image_Data/Input/randomly_sample/previous_shifts.mat");
		p.waitFor();
		p.destroy();
	}

	public static void restoreOriginalTrainingFile() throws IOException,
			InterruptedException {
		Process p = Runtime
				.getRuntime()
				.exec("cp Image_Data/Input/original_data/vision_training_input.mat Image_Data/Input/");
		p.waitFor();
		p.destroy();

		Process p1 = Runtime
				.getRuntime()
				.exec("cp Image_Data/Input/original_data/vision_training_motor.mat Image_Data/Input/");
		p1.waitFor();
		p1.destroy();

		Process p2 = Runtime
				.getRuntime()
				.exec("cp Image_Data/Input/original_data/vision_training_performance_motor.mat Image_Data/Input/");
		p2.waitFor();
		p2.destroy();
	}
	
	static String generateOriginalTrainingSegment() { 
	    String output = "", error = ""; 
	    try { 
	      String commandToRun = "/Applications/MATLAB_R2016a.app/bin/matlab -nodisplay < " 
	          + System.getProperty("user.dir") + "/test_mat_cmd_original.m"; 
	      System.out.println(commandToRun); 
	      Process p = Runtime.getRuntime().exec(commandToRun); 
	 
	      String s; 
	 
	      BufferedReader stdInput = new BufferedReader(new InputStreamReader( 
	          p.getInputStream())); 
	 
	      BufferedReader stdError = new BufferedReader(new InputStreamReader( 
	          p.getErrorStream())); 
	 
	      // read the output from the command 
	      System.out 
	          .println("\nHere is the standard output of the command:\n"); 
	      while ((s = stdInput.readLine()) != null) { 
	        output += s + "\n"; 
	        System.out.println(s); 
	      } 
	 
	      // read any errors from the attempted command 
	      System.out 
	          .println("\nHere is the standard error of the command (if any):\n"); 
	      while ((s = stdError.readLine()) != null) { 
	        error += s + "\n"; 
	        System.out.println(s); 
	      } 
	 
	      Process p0 = Runtime 
	          .getRuntime() 
	          .exec("mv Image_Data/Input/randomly_sample/vision_training_input.mat Image_Data/Input/"); 
	      p0.waitFor(); 
	      p0.destroy(); 
	 
	      Process p1 = Runtime 
	          .getRuntime() 
	          .exec("mv Image_Data/Input/randomly_sample/vision_training_motor.mat Image_Data/Input/"); 
	      p1.waitFor(); 
	      p1.destroy(); 
	 
	      Process p2 = Runtime 
	          .getRuntime() 
	          .exec("mv Image_Data/Input/randomly_sample/vision_training_performance_motor.mat Image_Data/Input/"); 
	      p2.waitFor(); 
	      p2.destroy(); 
	    } catch (Exception e) { 
	      System.out.println("exception happened 鈥� here鈥檚 what I know: "); 
	      e.printStackTrace(); 
	      System.exit(-1); 
	    } 
	    return output + error; 
	  } 

	public static String generateNewTrainingSegment() {
		String output = "", error = "";
		try {
			String commandToRun = "/Applications/MATLAB_R2016a.app/bin/matlab -nodisplay < "
					+ System.getProperty("user.dir") + "/test_mat_cmd.m";
			System.out.println(commandToRun);
			Process p = Runtime.getRuntime().exec(commandToRun);

			String s;

			BufferedReader stdInput = new BufferedReader(new InputStreamReader(
					p.getInputStream()));

			BufferedReader stdError = new BufferedReader(new InputStreamReader(
					p.getErrorStream()));

			// read the output from the command
			System.out
					.println("\nHere is the standard output of the command:\n");
			while ((s = stdInput.readLine()) != null) {
				output += s + "\n";
				System.out.println(s);
			}

			// read any errors from the attempted command
			System.out
					.println("\nHere is the standard error of the command (if any):\n");
			while ((s = stdError.readLine()) != null) {
				error += s + "\n";
				System.out.println(s);
			}

			Process p0 = Runtime
					.getRuntime()
					.exec("mv Image_Data/Input/randomly_sample/vision_training_input.mat Image_Data/Input/");
			p0.waitFor();
			p0.destroy();

			Process p1 = Runtime
					.getRuntime()
					.exec("mv Image_Data/Input/randomly_sample/vision_training_motor.mat Image_Data/Input/");
			p1.waitFor();
			p1.destroy();

			Process p2 = Runtime
					.getRuntime()
					.exec("mv Image_Data/Input/randomly_sample/vision_training_performance_motor.mat Image_Data/Input/");
			p2.waitFor();
			p2.destroy();
		} catch (Exception e) {
			System.out.println("exception happened 鈥� here鈥檚 what I know: ");
			e.printStackTrace();
			System.exit(-1);
		}
		return output + error;
	}
}
