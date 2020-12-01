package DN1;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import javax.swing.Spring;

import java.io.*;

public class HiddenLayer implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	// new variables
	private int topK;

	private int numNeurons; // This is a static number of Y (hidden) neurons.

	private float[] currentBottomUpInput;
	private float[] currentTopDownInput;

	private int[] inputSize;

	private int numBottomUpWeights;
	private float[][] bottomUpWeights;
	private float[][] bottomUpVariance;
	private float[][] bottomUpMask; // receptive field of hidden neurons. Binary
									// value not changeable once it is
									// initialized.
	private int[][] bottomUpAge;
	private float[] bottomUpResponse;

	private int numTopDownWeights;
	private float[][] topDownWeights;
	private float[][] topDownMask;
	private float[][] topDownVariance;
	private int[][] topDownAge;
	private float[] topDownResponse;

	private float[] newResponse;
	private float[] oldResponse;
	private int[] firingAge;

	private int rfSize;
	private int rfStride;
	private int[][] rf_id_loc;

	private boolean[] winnerFlag;
	private int usedHiddenNeurons; // The is the number of Y neurons that have
									// been used for topK winning.

	private final float GAMMA = 2000;
	private final float MACHINE_FLOAT_ZERO = 0.000001f;
	private final float ALMOST_PERFECT_MATCH_RESPONSE = 2.0f - 300 * MACHINE_FLOAT_ZERO;
	private final int T1 = 20;
	private final int T2 = 200;
	private final float C = 2;
	private final int SMAGE = 20;
	private final float SMUPPERTHRESH = 1.5f;
	private final float SMLOWERTHRESH = 0.9f;

	private float prescreenPercent;

	public HiddenLayer(int initialNumNeurons, int topK, int sensorSize, int motorSize, int rfSize, int rfStride,
			int[][] rf_id_loc, int[][] inputSize, float prescreenPercent) {
		this.setTopK(topK);
		this.usedHiddenNeurons = topK + 1; // bound of used neurons.

		this.numNeurons = (initialNumNeurons + usedHiddenNeurons);

		this.rfSize = rfSize;
		this.rfStride = rfStride;
		this.rf_id_loc = rf_id_loc;

		assert (inputSize.length == 1);
		this.inputSize = inputSize[0];

		this.bottomUpMask = new float[this.numNeurons][sensorSize];
		// The first two neurons have global receptive fields.
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < sensorSize; j++) {
				this.bottomUpMask[i][j] = 1;
			}
		}

		firingAge = new int[this.numNeurons];
		newResponse = new float[this.numNeurons];
		oldResponse = new float[this.numNeurons];
		bottomUpResponse = new float[this.numNeurons];
		topDownResponse = new float[this.numNeurons];

		numBottomUpWeights = sensorSize;
		bottomUpWeights = new float[this.numNeurons][sensorSize];
		bottomUpVariance = new float[this.numNeurons][sensorSize];
		bottomUpAge = new int[this.numNeurons][sensorSize];
		currentBottomUpInput = new float[numBottomUpWeights];

		numTopDownWeights = motorSize;
		topDownWeights = new float[this.numNeurons][motorSize];
		topDownMask = new float[this.numNeurons][motorSize];
		topDownVariance = new float[this.numNeurons][motorSize];
		topDownAge = new int[this.numNeurons][motorSize];
		currentTopDownInput = new float[motorSize];
		// topDownMask are ones at the beginning
		for (int i = 0; i < numNeurons; i++) {
			for (int j = 0; j < motorSize; j++) {
				topDownMask[i][j] = 1;
			}
		}

		winnerFlag = new boolean[this.numNeurons];

		// initialize the weights
		for (int i = 0; i < numNeurons; i++) {
			initializeWeight(bottomUpWeights[i], sensorSize);
			initializeWeight(topDownWeights[i], motorSize);
		}

		this.prescreenPercent = prescreenPercent;
	}

	public void saveWeightToFile(String hidden_ind) {
		try {
			PrintWriter wr_weight = new PrintWriter(new File(hidden_ind + "bottom_up_weight.txt"));
			PrintWriter wr_mask = new PrintWriter(new File(hidden_ind + "bottom_up_mask.txt"));
			PrintWriter wr_topdown = new PrintWriter(new File(hidden_ind + "top_down_weight.txt"));
			PrintWriter wr_topdownMask = new PrintWriter(new File(hidden_ind + "top_down_mask.txt"));
			PrintWriter wr_topdownAge = new PrintWriter(new File(hidden_ind + "top_down_age.txt"));
			PrintWriter wr_topDownVar = new PrintWriter(new File(hidden_ind + "top_down_var.txt"));
			for (int i = 0; i < numNeurons; i++) {
				for (int j = 0; j < bottomUpWeights[0].length; j++) {
					wr_weight.print(String.format("% .2f", bottomUpWeights[i][j]) + ',');
					wr_mask.print(Float.toString(bottomUpMask[i][j]) + ',');
				}
				wr_weight.println();
				wr_mask.println();

				for (int j = 0; j < topDownWeights[i].length; j++) {
					wr_topdown.print(String.format("% .2f", topDownWeights[i][j]) + ',');
				}
				wr_topdown.println();

				for (int j = 0; j < topDownMask[i].length; j++) {
					wr_topdownMask.print(String.format("% .2f", topDownMask[i][j]) + ',');
				}
				wr_topdownMask.println();

				for (int j = 0; j < topDownAge[i].length; j++) {
					wr_topdownAge.print(String.format("% 4d", topDownAge[i][j]) + ',');
				}
				wr_topdownAge.println();

				for (int j = 0; j < topDownVariance[i].length; j++) {
					wr_topDownVar.print(String.format("% .2f", topDownVariance[i][j]) + ',');
				}
				wr_topDownVar.println();
			}
			wr_weight.close();
			wr_mask.close();
			wr_topdown.close();
			wr_topdownMask.close();
			wr_topdownAge.close();
			wr_topDownVar.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	// seed for the random number generator
	private long seed = 0; // System.currentTimeMillis();
	Random rand = new Random(seed);

	private void initializeWeight(float[] weights, int size) {
		for (int i = 0; i < size; i++) {
			weights[i] = rand.nextFloat();
		}
	}

	public float[] elementWiseProduct(float[] vec1, float[] vec2) {
		assert vec1.length == vec2.length;
		int size = vec1.length;
		float[] result = new float[size];
		for (int i = 0; i < size; i++) {
			result[i] = vec1[i] * vec2[i];
		}
		return result;
	}

	// Initialize receptive field for the ith neuron, according to the whereID.
	// The center of the receptive field is located at rf_id_loc[i].
	// size of the receptive field is rf_size.
	public void initializeRfMask(int i, int whereID) {
		int im_height = (int) Math.sqrt(bottomUpMask[i].length);
		int im_width = im_height;

		if (whereID >= 0) {
			int half_rf_size = (rfSize - 1) / 2;
			int rf_begin_row = rf_id_loc[whereID][0] - half_rf_size;
			int rf_end_row = rf_id_loc[whereID][0] + half_rf_size;
			int rf_begin_col = rf_id_loc[whereID][1] - half_rf_size;
			int rf_end_col = rf_id_loc[whereID][1] + half_rf_size;
			assert im_height * im_width == bottomUpMask[i].length;
			for (int row = rf_begin_row; row <= rf_end_row; row++) {
				for (int col = rf_begin_col; col <= rf_end_col; col++) {
					int current_idx = row * im_width + col;
					bottomUpMask[i][current_idx] = 1.0f;
				}
			}
		} else {
			for (int pixel_ind = 0; pixel_ind < bottomUpMask[i].length; pixel_ind++) {
				bottomUpMask[i][pixel_ind] = 1.0f;
			}
		}
		System.out.println("Rf initialized.");
	}

	public void hebbianLearnHidden(float[] sensorInput, float[] motorInput) {

		int oldUsed = usedHiddenNeurons;
		for (int i = 0; i < oldUsed; i++) {
			if (winnerFlag[i] == true) {
				firingAge[i]++;

				// Bottom up weights, age, and variance.
				float[] currentSensorInput = elementWiseProduct(sensorInput, sign(bottomUpMask[i]));
				currentSensorInput = normalize(currentSensorInput, sensorInput.length, 2);
				incrementBottomUpAge(i);
				updateWeights(bottomUpWeights[i], currentSensorInput,
						getAmnesicLearningRate(bottomUpAge[i], bottomUpMask[i]), true);

				float[] currBottomUpVar = calculateDiff(bottomUpWeights[i], currentSensorInput, bottomUpMask[i]);
				updateWeights(bottomUpVariance[i], bottomUpMask[i], currBottomUpVar,
						getAmnesicLearningRate(bottomUpAge[i], bottomUpMask[i]), false);

				// Top down weights, age, and variance.
				float[] currentMotorInput = elementWiseProduct(motorInput, sign(topDownMask[i]));
				currentMotorInput = epsilon_normalize(currentMotorInput, currentMotorInput.length, topDownMask[i]);

				incrementTopDownAge(i);
				updateWeights(topDownWeights[i], currentMotorInput,
						getAmnesicLearningRate(topDownAge[i], topDownMask[i]), true);

				float[] currTopDownVar = calculateDiff(topDownWeights[i], currentMotorInput, topDownMask[i]);
				updateWeights(topDownVariance[i], topDownMask[i], currTopDownVar,
						getAmnesicLearningRate(topDownAge[i], topDownMask[i]), false);

				// Synapse Maintenance
				if (firingAge[i] > SMAGE) {
					// Calculate mean
					float topDownMeanVariance = mean(topDownVariance[i], topDownMask[i]);
					float bottomUpMeanVariance = mean(bottomUpVariance[i], bottomUpMask[i]);
					if (bottomUpMeanVariance < 0.01f) {
						bottomUpMeanVariance = 0.01f;
					}

					// Update Mask
					updateBottomUpMask(i, bottomUpMeanVariance, currentSensorInput);
					updateTopDownMask(i, topDownMeanVariance);
				}
			}

		}
	}

	private float[] sign(float[] fs) {
		float[] result = new float[fs.length];
		for (int i = 0; i < fs.length; i++) {
			result[i] = (fs[i] > 0) ? 1 : 0;
		}
		return result;
	}

	private float[] epsilon_normalize(float[] weight, int length, float[] mask) {
		float norm = 0;
		float mean = 0;
		int useful_length = 0;
		for (int i = 0; i < length; i++) {
			if (mask[i] > 0) {
				mean += weight[i];
				useful_length++;
			}
		}
		mean = mean / useful_length - MACHINE_FLOAT_ZERO;
		for (int i = 0; i < length; i++) {
			if (mask[i] > 0) {
				weight[i] -= mean;
			}
		}
		for (int i = 0; i < length; i++) {
			if (mask[i] > 0) {
				norm += weight[i] * weight[i];
			}
		}
		norm = (float) Math.sqrt(norm) + MACHINE_FLOAT_ZERO;
		if (norm > 0) {
			for (int i = 0; i < length; i++) {
				weight[i] = weight[i] / norm;
			}
		}
		return weight;
	}

	private void updateBottomUpMask(int i, float meanVariance, float[] currentSensorInput) {
		if (meanVariance > MACHINE_FLOAT_ZERO) {
			ArrayList<Integer> growlist = new ArrayList<Integer>();
			ArrayList<Integer> cutlist = new ArrayList<Integer>();

			for (int j = 0; j < numBottomUpWeights; j++) {
				float ratio = bottomUpVariance[i][j] / meanVariance;
				if (ratio < SMLOWERTHRESH && bottomUpMask[i][j] > 0) {
					bottomUpMask[i][j] = 1;
					// Grow nearby (neighbor of neuron i to z neuron j).
					if (bottomUpAge[i][j] > SMAGE) {
						growlist.add(j);
					}
				} else if (ratio > SMUPPERTHRESH && bottomUpMask[i][j] > 0) {
					// Cut connection.
					if (bottomUpAge[i][j] > SMAGE) {
						cutlist.add(j);
					}
				} else {
					if (bottomUpAge[i][j] > SMAGE && bottomUpMask[i][j] > 0) {
						// Version 1: Mahalanobis distance.
						bottomUpMask[i][j] = (1 / ratio - 1 / SMUPPERTHRESH) / (1 / SMLOWERTHRESH - 1 / SMUPPERTHRESH);

						// Version 2: Linear distance.
						// bottomUpMask[i][j] = (SMUPPERTHRESH - ratio)/
						// (SMUPPERTHRESH - SMLOWERTHRESH);
					}
				}
			}

			for (int j = 0; j < cutlist.size(); j++) {
				bottomUpMask[i][cutlist.get(j)] = 0;
				bottomUpWeights[i][cutlist.get(j)] = 0;
				bottomUpAge[i][cutlist.get(j)] = 0;
				bottomUpVariance[i][cutlist.get(j)] = 0;
			}

			for (int j = 0; j < growlist.size(); j++) {
				// growBottomUpConnection(i, growlist.get(j),
				// currentSensorInput);
			}

			firingAge[i] = 0;
			// for (int j = 0; j < bottomUpWeights[i].length; j++){
			// bottomUpAge[i][j] = 0;
			// bottomUpVariance[i][j] = 0;
			// }
			bottomUpWeights[i] = elementWiseProduct(bottomUpWeights[i], sign(bottomUpMask[i]));
			bottomUpWeights[i] = normalize(bottomUpWeights[i], bottomUpWeights[i].length, 2);
		}
	}

	private void updateTopDownMask(int i, float meanVariance) {
		if (meanVariance > MACHINE_FLOAT_ZERO) {
			ArrayList<Integer> growlist = new ArrayList<Integer>();
			ArrayList<Integer> cutlist = new ArrayList<Integer>();

			for (int j = 0; j < numTopDownWeights; j++) {
				float ratio = topDownVariance[i][j] / meanVariance;
				if (ratio < SMLOWERTHRESH && topDownMask[i][j] > 0) {
					topDownMask[i][j] = 1;
					// Grow nearby (neighbor of neuron i to z neuron j).
					if (topDownAge[i][j] > SMAGE) {
						growlist.add(j);
					}
				} else if (ratio > SMUPPERTHRESH && topDownMask[i][j] > 0) {
					// Cut connection.
					if (topDownAge[i][j] > SMAGE) {
						cutlist.add(j);
					}
				} else {
					if (topDownAge[i][j] > SMAGE && topDownMask[i][j] > 0) {
						// Version 1: Mahalanobis distance.
						topDownMask[i][j] = (1 / ratio - 1 / SMUPPERTHRESH) / (1 / SMLOWERTHRESH - 1 / SMUPPERTHRESH);

						// Version 2: Linear distance.
						// topDownMask[i][j] = (SMUPPERTHRESH - ratio)/
						// (SMUPPERTHRESH - SMLOWERTHRESH);
					}
				}
			}

			for (int j = 0; j < growlist.size(); j++) {
				growTopDownConnection(i, growlist.get(j));
			}

			for (int j = 0; j < cutlist.size(); j++) {
				topDownMask[i][cutlist.get(j)] = 0;
				topDownWeights[i][cutlist.get(j)] = 0;
				topDownAge[i][cutlist.get(j)] = 0;
				topDownVariance[i][cutlist.get(j)] = 0;
			}

			topDownWeights[i] = epsilon_normalize(topDownWeights[i], topDownWeights[i].length, topDownMask[i]);
		}
	}

	private void incrementTopDownAge(int i) {
		for (int j = 0; j < topDownMask[i].length; j++) {
			if (topDownMask[i][j] > 0) {
				topDownAge[i][j]++;
			}
		}
	}

	private void incrementBottomUpAge(int i) {
		for (int j = 0; j < bottomUpMask[i].length; j++) {
			if (bottomUpMask[i][j] > 0) {
				bottomUpAge[i][j]++;
			}
		}
	}

	// TODO: implement this.
	private void growBottomUpConnection(int i, int j, float[] currentSensorInput) {
		int[] sub = ind2sub(j, inputSize); // ind: [height, width]
		if (sub[0] - 1 >= 0) {
			int[] growSub = { sub[0] - 1, sub[1] };
			int growInd = sub2ind(growSub, inputSize);
			if (bottomUpMask[i][growInd] == 0) {
				bottomUpMask[i][growInd] = 1;
				bottomUpWeights[i][growInd] = currentSensorInput[growInd];
				bottomUpAge[i][growInd] = 0;
				bottomUpVariance[i][growInd] = 0;
			}
		}
		if (sub[0] + 1 < inputSize[0]) {
			int[] growSub = { sub[0] + 1, sub[1] };
			int growInd = sub2ind(growSub, inputSize);
			if (bottomUpMask[i][growInd] == 0) {
				bottomUpMask[i][growInd] = 1;
				bottomUpWeights[i][growInd] = currentSensorInput[growInd];
				bottomUpAge[i][growInd] = 0;
				bottomUpVariance[i][growInd] = 0;
			}
		}
		if (sub[1] - 1 >= 0) {
			int[] growSub = { sub[0], sub[1] - 1 };
			int growInd = sub2ind(growSub, inputSize);
			if (bottomUpMask[i][growInd] == 0) {
				bottomUpMask[i][growInd] = 1;
				bottomUpWeights[i][growInd] = currentSensorInput[growInd];
				bottomUpAge[i][growInd] = 0;
				bottomUpVariance[i][growInd] = 0;
			}
		}
		if (sub[1] + 1 < inputSize[1]) {
			int[] growSub = { sub[0], sub[1] + 1 };
			int growInd = sub2ind(growSub, inputSize);
			if (bottomUpMask[i][growInd] == 0) {
				bottomUpMask[i][growInd] = 1;
				bottomUpWeights[i][growInd] = currentSensorInput[growInd];
				bottomUpAge[i][growInd] = 0;
				bottomUpVariance[i][growInd] = 0;
			}
		}
	}

	private int[] ind2sub(int i, int[] shape) {
		int[] result = new int[2];
		result[0] = (int) i / shape[1];
		result[1] = i - result[0] * shape[1];
		return result;
	}

	private int sub2ind(int[] ind, int[] shape) {
		int result = ind[1] + ind[0] * shape[1];
		return result;
	}

	// Grow the jth top-down connection for the ith neuron.
	private void growTopDownConnection(int i, int j) {
		if (j > 0) {
			if (topDownMask[i][j - 1] == 0) {
				topDownMask[i][j - 1] = 1;
				topDownAge[i][j - 1] = 0;
				topDownVariance[i][j - 1] = 0;
				topDownWeights[i][j - 1] = 0;
			}
		}
		if (j < topDownMask[i].length - 1) {
			if (topDownMask[i][j + 1] == 0) {
				topDownMask[i][j + 1] = 1;
				topDownAge[i][j + 1] = 0;
				topDownVariance[i][j + 1] = 0;
				topDownWeights[i][j + 1] = 0;
			}
		}
	}

	private float[] calculateDiff(float[] input_1, float[] input_2, float[] mask) {
		assert (input_1.length == input_2.length);
		float[] result = new float[input_1.length];
		for (int i = 0; i < input_1.length; i++) {
			if (mask[i] > 0) {
				result[i] = (Math.abs(input_1[i] - input_2[i]));
			} else {
				result[i] = 0;
			}
		}
		return result;
	}

	// convert into 1d Array
	public float[] getResponse1D() {
		float[] inputArray = new float[numNeurons];
		System.arraycopy(oldResponse, 0, inputArray, 0, numNeurons);

		return inputArray;
	}

	private void updateWeights(float[] weights, float[] mask, float[] input, float[] learningRate,
			boolean normalize_flag) {
		// make sure both arrays have the same length
		assert weights.length == input.length;

		for (int i = 0; i < input.length; i++) {
			if (mask[i] > 0) {
				weights[i] = (1.0f - learningRate[i]) * weights[i] + learningRate[i] * input[i];
			} else {
				weights[i] = 0;
			}
		}

		if (normalize_flag) {
			weights = normalize(weights, input.length, 2);
		}
	}

	private void updateWeights(float[] weights, float[] input, float[] learningRate, boolean normalize_flag) {

		// make sure both arrays have the same length
		assert weights.length == input.length;

		for (int i = 0; i < input.length; i++) {
			weights[i] = (1.0f - learningRate[i]) * weights[i] + learningRate[i] * input[i];
		}

		if (normalize_flag) {
			weights = normalize(weights, input.length, 2);
		}
	}

	private void updateWeights(float[] weights, float[] input, float learningRate, boolean normalize_flag) {

		// make sure both arrays have the same length
		assert weights.length == input.length;

		for (int i = 0; i < input.length; i++) {
			weights[i] = (1.0f - learningRate) * weights[i] + learningRate * input[i];
		}

		if (normalize_flag) {
			weights = normalize(weights, input.length, 2);
		}
	}

	private float[] getAmnesicLearningRate(int[] age, float[] mask) {
		float[] result = new float[age.length];
		for (int i = 0; i < age.length; i++) {
			if (mask[i] > 0) {
				assert (age[i] > 0);
				result[i] = getAmnesicLearningRate(age[i]);
			}
		}
		return result;
	}

	/*
	 * This method is to get the amnesic learning rate.
	 */
	private float getAmnesicLearningRate(int age) {

		float mu, learning_rate;

		if (age < T1) {
			mu = 0.0f;
		}

		else if ((age < T2) && (age >= T1)) {
			mu = C * ((float) age - T1) / (T2 - T1);
		}

		else {
			mu = C * ((float) age - T2) / GAMMA;
		}

		learning_rate = (1 + mu) / ((float) age);

		return learning_rate;
	}

	public void computeBottomUpResponse(float[][] sensorInput, int[] sensorSize) {
		// Keep track of the sensorInput.
		int beginIndex = 0;
		for (int j = 0; j < sensorSize.length; j++) {
			System.arraycopy(sensorInput[j], 0, currentBottomUpInput, beginIndex, sensorSize[j]);
			beginIndex += sensorSize[j];
		}

		for (int i = 0; i < usedHiddenNeurons; i++) {

			bottomUpResponse[i] = 0.0f;
			beginIndex = 0;

			float[] currentInput = new float[numBottomUpWeights];
			float[] currentWeight = new float[numBottomUpWeights];

			for (int j = 0; j < sensorSize.length; j++) {
				System.arraycopy(sensorInput[j], 0, currentInput, beginIndex, sensorSize[j]);
				beginIndex += sensorSize[j];
			}

			System.arraycopy(bottomUpWeights[i], 0, currentWeight, 0, numBottomUpWeights);

			currentInput = elementWiseProduct(currentInput, sign(bottomUpMask[i]));
			currentInput = normalize(currentInput, numBottomUpWeights, 2);

			currentWeight = elementWiseProduct(currentWeight, sign(bottomUpMask[i]));
			currentWeight = normalize(currentWeight, numBottomUpWeights, 2);

			bottomUpResponse[i] += dotProduct(currentWeight, currentInput, numBottomUpWeights);
		}
	}

	public void computeTopDownResponse(float[][] motorInput, int[] motorSize) {
		int beginIndex = 0;
		for (int j = 0; j < motorSize.length; j++) {
			System.arraycopy(motorInput[j], 0, currentTopDownInput, beginIndex, motorSize[j]);
			beginIndex += motorSize[j];
		}
		for (int i = 0; i < usedHiddenNeurons; i++) {
			topDownResponse[i] = 0.0f;
			beginIndex = 0;

			float[] currentInput = new float[numTopDownWeights];
			float[] currentWeight = new float[numTopDownWeights];

			for (int j = 0; j < motorSize.length; j++) {
				System.arraycopy(motorInput[j], 0, currentInput, beginIndex, motorSize[j]);
				beginIndex += motorSize[j];
			}

			System.arraycopy(topDownWeights[i], 0, currentWeight, 0, numTopDownWeights);

			// If using SM.
			currentInput = elementWiseProduct(currentInput, sign(topDownMask[i]));
			currentInput = epsilon_normalize(currentInput, numTopDownWeights, topDownMask[i]);
			currentWeight = epsilon_normalize(currentWeight, numTopDownWeights, topDownMask[i]);
			topDownResponse[i] += dotProduct(currentWeight, currentInput, numTopDownWeights);
		}
	}

	public void computeResponse(int whereId, boolean learn_flag) {
		prescreenResponse();
		for (int i = 0; i < usedHiddenNeurons; i++) {
			newResponse[i] = (bottomUpResponse[i] + topDownResponse[i]);
		}
		newResponse[0] = 0;
		newResponse[1] = 0;

		// do the topKcompetition
		topKCompetition(whereId, learn_flag);
	}

	private void prescreenResponse() {
		// Prescreen bottomUpResponse
		float[] tempArray = new float[bottomUpResponse.length];
		System.arraycopy(bottomUpResponse, 0, tempArray, 0, bottomUpResponse.length);
		Arrays.sort(tempArray);
		int cutOffPos = (int) Math.ceil((double) tempArray.length * (double) prescreenPercent);
		if (cutOffPos >= bottomUpResponse.length - 1)
			cutOffPos = bottomUpResponse.length - 1;
		float cutOffValue = tempArray[cutOffPos];
		for (int i = 0; i < bottomUpResponse.length; i++) {
			if (bottomUpResponse[i] < cutOffValue) {
				bottomUpResponse[i] = 0;
			}
		}

		// Prescreen topDownResponse
		tempArray = new float[topDownResponse.length];
		System.arraycopy(topDownResponse, 0, tempArray, 0, topDownResponse.length);
		Arrays.sort(tempArray);
		cutOffPos = (int) Math.ceil((double) tempArray.length * (double) prescreenPercent);
		if (cutOffPos >= topDownResponse.length - 1)
			cutOffPos = topDownResponse.length - 1;
		cutOffValue = tempArray[cutOffPos];
		for (int i = 0; i < bottomUpResponse.length; i++) {
			if (topDownResponse[i] < cutOffValue) {
				topDownResponse[i] = 0;
			}
		}
	}

	// Sort the topK elements to the beginning of the sort array where the index
	// of the top
	// elements are still in the pair.
	private static void topKSort(Pair[] sortArray, int topK) {

		for (int i = 0; i < topK; i++) {
			Pair maxPair = sortArray[i];
			int maxIndex = i;

			for (int j = i + 1; j < sortArray.length; j++) {

				if (sortArray[j].value > maxPair.value) { // select temporary
															// max
					maxPair = sortArray[j];
					maxIndex = j;

				}
			}

			if (maxPair.index != i) {
				Pair temp = sortArray[i]; // store the value of pivot (top i)
											// element
				sortArray[i] = maxPair; // replace with the maxPair object.
				sortArray[maxIndex] = temp; // replace maxPair index elements
											// with the pivot.
			}
		}
	}

	private void topKCompetition(int attentionId, boolean learn_flag) {

		// initializing the indexes
		// winnerIndex is only for the winner neurons among the active neurons.
		int winnerIndex = 0;

		float[] copyArray = new float[usedHiddenNeurons];

		// Pair is an object that contains the (index,response_value) of each
		// hidden neurons.
		Pair[] sortArray = new Pair[usedHiddenNeurons];

		for (int i = 0; i < usedHiddenNeurons; i++) {
			sortArray[i] = new Pair(i, newResponse[i]);
			copyArray[i] = newResponse[i];
			newResponse[i] = 0.0f;
			winnerFlag[i] = false;
		}

		// Sort the array of Pair objects by its response_value in
		// non-increasing order.
		// The index is in the Pair, goes with the response ranked.
		topKSort(sortArray, topK);

		System.out.println("High top1 value of hidden: " + sortArray[0].value);

		// check if the top winner has almost perfect match.
		System.out.println(sortArray[0].value < ALMOST_PERFECT_MATCH_RESPONSE && usedHiddenNeurons < numNeurons);
		if (learn_flag) {
			if (sortArray[0].value < ALMOST_PERFECT_MATCH_RESPONSE && usedHiddenNeurons < numNeurons) { // add
																										// one
																										// more
																										// neuron.
				System.out.println("This is the new hidden neuron " + usedHiddenNeurons);
				winnerFlag[usedHiddenNeurons] = true;
				newResponse[usedHiddenNeurons] = 1.0f; // set to perfect match.
				initializeRfMask(usedHiddenNeurons, attentionId); // initialize
				usedHiddenNeurons++;
				winnerIndex++;
			}
		}

		// identify the ranks of topk winners
		float value_top1 = sortArray[0].value;
		float value_topkplus1 = sortArray[topK].value; // this neurons is the
														// largerst response
														// neurons to be set to
														// zero.

		// Find the topK winners and their indexes.
		while (winnerIndex < topK) {
			// get the index of the top element.
			int topIndex = sortArray[winnerIndex].get_index();
			newResponse[topIndex] = (copyArray[topIndex] - value_topkplus1)
					/ (value_top1 - value_topkplus1 + MACHINE_FLOAT_ZERO);
			winnerIndex++;

			winnerFlag[topIndex] = true;
		}

	}

	public float[] normalize(float[] input, int size, int flag) {

		float[] weight = new float[size];
		System.arraycopy(input, 0, weight, 0, size);

		if (flag == 1) {
			float min = weight[0];
			float max = weight[0];
			for (int i = 0; i < size; i++) {
				if (weight[i] < min) {
					min = weight[i];
				}
				if (weight[i] > max) {
					max = weight[i];
				}
			}

			float diff = max - min + MACHINE_FLOAT_ZERO;
			for (int i = 0; i < size; i++) {
				weight[i] = (weight[i] - min) / diff;
			}

			float mean = 0;
			for (int i = 0; i < size; i++) {
				mean += weight[i];
			}
			mean = mean / size;
			for (int i = 0; i < size; i++) {
				weight[i] = weight[i] - mean + MACHINE_FLOAT_ZERO;
			}

			float norm = 0;
			for (int i = 0; i < size; i++) {
				norm += weight[i] * weight[i];
			}
			norm = (float) Math.sqrt(norm);
			if (norm > 0) {
				for (int i = 0; i < size; i++) {
					weight[i] = weight[i] / norm;
				}
			}
		}

		if (flag == 2) {
			float norm = 0;

			for (int i = 0; i < size; i++) {
				norm += weight[i] * weight[i];
			}
			norm = (float) Math.sqrt(norm) + MACHINE_FLOAT_ZERO;
			if (norm > 0) {
				for (int i = 0; i < size; i++) {
					weight[i] = weight[i] / norm;
				}
			}
		}

		if (flag == 3) {
			float norm = 0;
			for (int i = 0; i < size; i++) {
				norm += weight[i];
			}
			norm = norm + MACHINE_FLOAT_ZERO;
			if (norm > 0) {
				for (int i = 0; i < size; i++) {
					weight[i] = weight[i] / norm;
				}
			}
		}
		return weight;
	}

	public void replaceHiddenLayerResponse() {
		for (int i = 0; i < usedHiddenNeurons; i++) {
			oldResponse[i] = newResponse[i];
		}

	}

	private float dotProduct(float[] a, float[] b, int size) {
		float r = 0.0f;

		for (int i = 0; i < size; i++) {
			r += (a[i] * b[i]);
		}

		return r;
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

	public int getNumTopDownWeights() {
		return numTopDownWeights;
	}

	public void setNumTopDownWeights(int numTopDownWeights) {
		this.numTopDownWeights = numTopDownWeights;
	}

	public float[][] getTopDownWeights() {
		return topDownWeights;
	}

	public void setTopDownWeights(float[][] topDownWeights) {
		this.topDownWeights = topDownWeights;
	}

	public int getNumNeurons() {
		return numNeurons;
	}

	public float[][] getRfMask() {
		return bottomUpMask;
	}

	public void setNumNeurons(int numNeurons) {
		this.numNeurons = numNeurons;
	}

	public int getUsedHiddenNeurons() {
		return usedHiddenNeurons;
	}

	public void resetResponses() {

		Arrays.fill(bottomUpResponse, 0.0f);
		Arrays.fill(topDownResponse, 0.0f);

	}

	public class Pair implements Comparable<Pair> {
		public final int index;
		public final float value;

		public Pair(int index, float value) {
			this.index = index;
			this.value = value;
		}

		public int compareTo(Pair other) {
			return -1 * Float.valueOf(this.value).compareTo(other.value);
		}

		public int get_index() {
			return index;
		}
	}

	public static float mean(float[] m, float[] mask) {
		float sum = 0;
		float count = 0;
		for (int i = 0; i < m.length; i++) {
			if (mask[i] > 0) {
				sum += m[i];
				count++;
			}
		}
		return sum / count;
	}

	/*
	 * This is the protocol for toy data visualization. All data are transfered
	 * as float to save effort in translation. There are these things to send
	 * over socket: 1. number of hidden neurons 2. length of bottomUp input 3.
	 * bottom up weight 4. bottom up mask 5. length of topDown input 6. topDown
	 * Age 7. topDown Mask 8. topDown Variance 9. topdown Weight 10.bottom up
	 * input 11.top down input
	 */
	public void sendNetworkOverSocket(PrintWriter string_out, DataOutputStream data_out, int display_y_zone,
			int display_num, int display_start_id) throws IOException, InterruptedException {
		int start_id = display_start_id - 1;
		if (start_id < 0)
			start_id = 0;
		if (start_id >= bottomUpWeights.length)
			start_id = 0;
		int end_id = start_id + display_num;
		if (end_id > bottomUpWeights.length)
			end_id = bottomUpWeights.length;
		if (end_id < 0)
			end_id = bottomUpWeights.length;

		// number of hidden neurons
		data_out.writeInt(end_id - start_id);

		// length of bottom up input
		data_out.writeInt(bottomUpWeights[0].length);

		// length of topDown input
		data_out.writeInt(topDownWeights[0].length);

		// bottom up weight
		if (display_y_zone == 1) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < bottomUpWeights[0].length; j++) {
					data_out.writeFloat((float) bottomUpWeights[i][j] * bottomUpMask[i][j]);
				}
			}
		}

		// bottom up age
		else if (display_y_zone == 2) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < bottomUpWeights[0].length; j++) {
					data_out.writeFloat((float) bottomUpAge[i][j]);
				}
			}
		}

		// bottom up mask
		else if (display_y_zone == 3) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < bottomUpWeights[0].length; j++) {
					data_out.writeFloat((float) bottomUpMask[i][j]);
				}
			}
		}

		// bottom up age
		else if (display_y_zone == 4) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < bottomUpWeights[0].length; j++) {
					data_out.writeFloat((float) bottomUpVariance[i][j]);
				}
			}
		}

		// topDown weight
		else if (display_y_zone == 5) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < topDownWeights[0].length; j++) {
					data_out.writeFloat((float) topDownWeights[i][j]);
				}
			}
		}

		// topDown age
		else if (display_y_zone == 6) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < topDownWeights[0].length; j++) {
					data_out.writeInt(topDownAge[i][j]);
				}
			}
		}

		// topDown mask
		else if (display_y_zone == 7) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < topDownWeights[0].length; j++) {
					data_out.writeFloat((float) topDownMask[i][j]);
				}
			}
		}

		// topDown variance
		else if (display_y_zone == 8) {
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < topDownWeights[0].length; j++) {
					data_out.writeFloat((float) topDownVariance[i][j]);
				}
			}
		}

		// bottom up input
		for (int i = 0; i < currentBottomUpInput.length; i++) {
			data_out.writeFloat((float) currentBottomUpInput[i]);
		}

		// top down input
		for (int i = 0; i < currentTopDownInput.length; i++) {
			data_out.writeFloat((float) currentTopDownInput[i]);
		}

		// bottom up response
		for (int i = start_id; i < end_id; i++) {
			data_out.writeFloat((float) bottomUpResponse[i]);
		}

		// top down response
		for (int i = start_id; i < end_id; i++) {
			data_out.writeFloat((float) topDownResponse[i]);
		}

		// final response
		for (int i = start_id; i < end_id; i++) {
			data_out.writeFloat((float) newResponse[i]);
		}
	}
}
