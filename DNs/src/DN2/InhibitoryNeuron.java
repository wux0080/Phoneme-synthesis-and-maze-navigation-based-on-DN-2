package DN2;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class InhibitoryNeuron implements Serializable{
	//neuron's state: 1 for learning; 0 for initial
    private boolean mState; 
	//The age for all inhibition synapses of the neuron
    private int[] mInAge;
	//each neuron's firing age
	private int mFiringAge;
	//y neurons' type
	private int mType;
	//true for y neuron; false for z neuron
	private boolean mCategory;
	//each neuron's index
	private int mIndex;
	//true for winning in top-k competition
	private boolean winnerFlag;
	//each neuron's 3D location
	private float[] mLocation;
	//each neuron's bottom-up preResponse
	private float bottomUpResponse;
	//each neuron's top-down preResponse
	private float topDownResponse;
	//each neuron's lateral preResponse
	private float lateralResponse;
    //each neuron's current Response
	private float newResponse;
	//each neuron's old Response in last frame
	private float oldResponse;
	//each neuron's bottom-up weights
	private float[] mBottomUpWeights;
	//each neuron's top-down weights
	private float[] mTopDownWeights;
	//each neuron's lateral weights
    private float[] mLateralWeights;
	//record each neuron's variances between bottom-up weights and bottom-up inputs
	private float[] mBottomUpVariance;
	//record each neuron's variances between top-down weights and top-down inputs
	private float[] mTopDownVariance;
	//record each neuron's variances between lateral weights and lateral inputs
	private float[] mLateralVariance;
	//each neuron's bottom-up mask
	private float[] mBottomUpMask;
	//each neuron's top-down mask
	private float[] mTopDownMask;
	//each neuron's lateral mask
	private float[] mLateralMask;
	//each bottom-up synapse's age of the neuron
	private int[] bottomUpAge;
	//each top-down synapse's age of the neuron
	private int[] topDownAge;
	//each lateral synapse's age of the neuron
	private int[] lateralAge;
	//the parameter for bottom-up preResponse's contribution
	private float bottomupRatio;
	//the parameter for top-down preResponse's contribution
	private float topdownRatio;
	//the parameter for lateral preResponse's contribution
    private float lateralRatio;
	//record the size of input
	private int[] inputSize;
	//The parameter of learning rate
	private final float GAMMA = 2000;
	private float MACHINE_FLOAT_ZERO;
	//The parameter of learning rate
	private final int T1 = 20;
	//The parameter of learning rate
	private final int T2 = 200;
	private final float C = 2;
	private final int SMAGE = 20;
	//The parameter for synaptic maintenance
	private final float SMUPPERTHRESH = 1.5f;
	//The parameter for synaptic maintenance
	private final float SMLOWERTHRESH = 0.9f;
	//construct inhibitory neuron
	public InhibitoryNeuron(int bottomupsize,int topdownsize, int lateralsize, boolean category, int type, int index, int[] inputsize){
		//set neuron's index        
		mIndex=index;
		//set the neuron's type: 0 for z neuron; 1 for y neuron
		mType=type;
		//initialize neuron's category; 1 for y neuron; 0 for z neuron
	    mCategory = category;
	    //construct the age vector of inhibition synapses
        mInAge = new int[lateralsize];
        //initialize the age vector of inhibition synapses
		for(int i = 0; i < lateralsize; i++){
		    mInAge[i]=1;
		}
		//initialize neuron's firing age
		mFiringAge=0;
		//initialize neuron's winner flag
		winnerFlag=false;
		/* 		  
		  newResponse=0.0f;
		  oldResponse=0.0f;
		  winnerFlag=false;
	    */ 
		//set the machine zero
	    MACHINE_FLOAT_ZERO = 0.00001f;
	    //set the input size  		  
		inputSize = new int[inputsize.length];
		for(int i = 0; i < inputsize.length; i++){
			  inputSize[i] = inputsize[i];
	    }
		//construct the 3D location vector		  
		mLocation = new float[3];
		//construct each neuron's bottom-up weight vector
	    mBottomUpWeights = new float[bottomupsize];
	    //construct each neuron's top-down weight vector
		mTopDownWeights = new float[topdownsize];
		//construct each neuron's lateral weight vector
		mLateralWeights = new float[lateralsize];
		//initialize weights for y neuron  
		if(mCategory){
			//set the machine zero
		    MACHINE_FLOAT_ZERO = 0.0001f;
		    //randomly initialize y neuron's bottom-up weights for the neurons which have bottom-up connections
		    if(mType == 4 || mType == 5 || mType == 6 || mType == 7){
		        initializeWeight(mBottomUpWeights,bottomupsize);
		    }
		    //set the bottom-up weights to be 0 for the neurons which don't have bottom-up connections
		    else{
		    	for(int i=0;i<bottomupsize;i++){
		    		mBottomUpWeights[i]=0;
		    	}
		    }
		    //initialize y neuron's top-down weights for the neurons which have top-down connections
		    if(mType == 1 || mType == 3 || mType ==5 || mType == 7){
		        initializeWeight(mTopDownWeights,topdownsize);
		    }
		    //set the top-down weights to be 0 for the neurons which don't have top-down connections
		    else{
		    	for(int i=0;i<topdownsize;i++){
		    		mTopDownWeights[i]=0;
		    	}
		    }
		    /*
		    if(mType == 2 || mType == 3 || mType == 6 || mType == 7){
		        initializeWeight(mLateralWeights,lateralsize);
		    }*/
		    //set the lateral weights to be 0
		    for(int i=0;i<lateralsize;i++){
	    		mLateralWeights[i]=0;
	    	}
		    //construct bottom-up mask vector
		    mBottomUpMask = new float[bottomupsize];
		    //construct bottom-up variance vector
		    mBottomUpVariance = new float[bottomupsize];
		    //construct bottom-up age vector
		    bottomUpAge = new int[bottomupsize];
			/* for(int i = 0; i < bottomupsize; i++){
	                bottomUpAge[i] = 0;
	           }			 
	        */
		    //construct the top-down mask vector
		    mTopDownMask = new float[topdownsize];
		    //construct the top-down variance vector
		    mTopDownVariance = new float[topdownsize];
		    //construct the top-down age vector
		    topDownAge = new int[topdownsize];
		    
		    /* for(int j = 0; j < topdownsize; j++){
		    	topDownAge[j] = 0;
		       }
		    */
		    //construct the lateral mask vector
		    mLateralMask = new float[lateralsize];
		    //construct the lateral variance vector
		    mLateralVariance = new float[lateralsize];
		    //construct the lateral age vector
		    lateralAge = new int[lateralsize];
		  }	  		  
	  }
	  
	  // seed for the random number generator
	  private long seed = 0; // System.currentTimeMillis();
	  Random rand = new Random(seed);
		
	  //the method of randomly initialize the weights
	  private void initializeWeight(float[] weights, int size){
			for (int i = 0; i < size; i++) {
				weights[i] = rand.nextFloat();
			}
		}
	  
	  //do element product between two vectors
	  public float[] elementWiseProduct(float[] vec1, float[] vec2) {
		    //check the length of 2 vectors
			assert vec1.length == vec2.length;
			int size = vec1.length;
			//construct the result vector
			float[] result = new float[size];
			//calculate the product between each element of 2 vectors
			for (int i = 0; i < size; i++) {
				result[i] = vec1[i] * vec2[i];
			}
			return result;
	  }
	  
		public void addFiringAge() {
			if(winnerFlag) {
				mFiringAge++;
			}
		}

	  //hebbian learning for each neuron
	  public void hebbianLearnHidden(float[] sensorInput, float[] motorInput, float[] preResponse, float[][] areamask){
		  float bottomUpMeanVariance = 0;
		  float topDownMeanVariance = 0;
		  float lateralMeanVariance = 0;
		  float[] currentpreresponse = new float[preResponse.length];
		  System.arraycopy(preResponse, 0, currentpreresponse, 0, preResponse.length);
		  if(winnerFlag == true){
//				System.out.println("Inhibition update neuron id: "+this.getindex());
			    //increase the firing age
				mFiringAge++;	
				//update the bottom-up connections
				if(mType == 4 || mType == 5 || mType == 6 || mType == 7){
					//filter the bottom-up input vector through bottom-up mask
				    float[] currentSensorInput = elementWiseProduct(sensorInput, sign(mBottomUpMask));
				    //normalize the bottom-up input vector
				    currentSensorInput = normalize(currentSensorInput, sensorInput.length, 2);
				    //increase bottom-up synapses' ages
				    incrementBottomUpAge();
				    //update the bottom-up weights
				    updateWeights(mBottomUpWeights, currentSensorInput,
						getAmnesicLearningRate(bottomUpAge, mBottomUpMask), true);
				    //accumulate the variance between the bottom-up weight and input vector
				    float[] currBottomUpVar = calculateDiff(mBottomUpWeights, currentSensorInput, mBottomUpMask);
				    updateWeights(mBottomUpVariance, mBottomUpMask, currBottomUpVar,
						getAmnesicLearningRate(bottomUpAge, mBottomUpMask), false);
				    //if the neuron reach the stable stage, start bottom-up synaptic maintenance
				    if (mFiringAge > SMAGE) {
					    bottomUpMeanVariance = mean(mBottomUpVariance, mBottomUpMask);
						if (bottomUpMeanVariance < 0.01f) {
						       bottomUpMeanVariance = 0.01f;
						}
						//update the bottom-up mask
						updateBottomUpMask(bottomUpMeanVariance, currentSensorInput);
				    }
				}
				//update the top-down connections
				if(mType == 1 || mType == 3 || mType ==5 || mType == 7){
					//filter the top-down input vector through top-down mask
				    float[] currentMotorInput = elementWiseProduct(motorInput, sign(mTopDownMask));
				    //normalize the top-down input vector
				    currentMotorInput = normalize(currentMotorInput, currentMotorInput.length, mTopDownMask);
				    //increase top-down synapses' ages				
				    incrementTopDownAge();
				    //update the top-down weights
				    updateWeights(mTopDownWeights, currentMotorInput,
						getAmnesicLearningRate(topDownAge, mTopDownMask), true);
				    //accumulate the variance between the top-down weight and input vector
				    float[] currTopDownVar = calculateDiff(mTopDownWeights, currentMotorInput, mTopDownMask);
				    updateWeights(mTopDownVariance, mTopDownMask, currTopDownVar,
						getAmnesicLearningRate(topDownAge, mTopDownMask), false);
			    	//if the neuron reach the stable stage, start top-down synaptic maintenance				    
				    if (mFiringAge > SMAGE) {
				    	topDownMeanVariance = mean(mTopDownVariance, mTopDownMask);
					    if (topDownMeanVariance < 0.01f) {
						       topDownMeanVariance = 0.01f;
						    }
					    //update the top-down mask
						updateTopDownMask(topDownMeanVariance);
				    }
				}
				for(int i=0; i<areamask[this.mType-1].length; i++){
					if(areamask[this.mType-1][i] != 0){
						mInAge[i] += 1; 
					}
					else{
						currentpreresponse[i] = 0;
					}
				}
				//update the lateral weights
				updateWeights(mLateralWeights, currentpreresponse, getAmnesicLearningRate(mInAge), true);

	
								
		}

	  }
      
      private float ratioParameter(float ratio){
    	  float Ratio;      
			if (ratio < SMLOWERTHRESH) {
				Ratio = 1;

			} else if (ratio > SMUPPERTHRESH) {
				Ratio = 0;
			}
			else {

					Ratio = (1 / ratio - 1 / SMUPPERTHRESH) / (1 / SMLOWERTHRESH - 1 / SMUPPERTHRESH);

		    }
		return Ratio;
      }
      
      //binaryzation of each element in the vector: 1 for the ones larger than 0; 0 for the others 
      private float[] sign(float[] fs) {
    	  //construct the result rector
	      float[] result = new float[fs.length];
		  for (int i = 0; i < fs.length; i++) {
		      result[i] = (fs[i] > 0) ? 1 : 0;
		  }
		  return result;
		}

      //normalize the non-zero elements in the vector
	  private float[] normalize(float[] weight, int length, float[] mask) {
			float norm = 0;
			float mean = 0;
			int useful_length = 0;
			for (int i = 0; i < length; i++) {
				if (mask[i] > 0) {
					mean += weight[i];
					useful_length++;
				}
			}
			//calculate the mean value of non-zero elements in the vector
			mean = mean / useful_length - MACHINE_FLOAT_ZERO;
			//each non-zero element minuses the mean value
			for (int i = 0; i < length; i++) {
				if (mask[i] > 0) {
					weight[i] -= mean;
				}
			}
			//calculate the norm-2 for the non-zero elements
			for (int i = 0; i < length; i++) {
				if (mask[i] > 0) {
					norm += weight[i] * weight[i];
				}
			}
			norm = (float) Math.sqrt(norm) + MACHINE_FLOAT_ZERO;
			//normalize the non-zero element
			if (norm > 0) {
				for (int i = 0; i < length; i++) {
					weight[i] = weight[i] / norm;
				}
			}
			return weight;
		}

	    //update the bottom-up mask vector
		private void updateBottomUpMask(float meanVariance, float[] currentSensorInput) {
			if (meanVariance > MACHINE_FLOAT_ZERO) {
				//construct the grow array list
				ArrayList<Integer> growlist = new ArrayList<Integer>();
				//construct the cut array list
				ArrayList<Integer> cutlist = new ArrayList<Integer>();

				for (int j = 0; j < mBottomUpWeights.length; j++) {
					//calculate the ratio for each variance element
					float ratio = mBottomUpVariance[j] / meanVariance;
					if (ratio < SMLOWERTHRESH && mBottomUpMask[j] > 0) {
						mBottomUpMask[j] = 1;
						// Grow nearby (neighbor of neuron i to z neuron j).
						if (bottomUpAge[j] > SMAGE) {
							growlist.add(j);
						}
					} else if (ratio > SMUPPERTHRESH && mBottomUpMask[j] > 0) {
						// Cut connection.
						if (bottomUpAge[j] > SMAGE) {
							cutlist.add(j);
						}
					} else {
						if (bottomUpAge[j] > SMAGE && mBottomUpMask[j] > 0) {
							// Version 1: Mahalanobis distance.
							mBottomUpMask[j] = (1 / ratio - 1 / SMUPPERTHRESH) / (1 / SMLOWERTHRESH - 1 / SMUPPERTHRESH);

							/* Version 2: Linear distance.
							   bottomUpMask[i][j] = (SMUPPERTHRESH - ratio)/
							   (SMUPPERTHRESH - SMLOWERTHRESH);
							*/
						}
					}
				}

				for (int j = 0; j < cutlist.size(); j++) {
					mBottomUpMask[cutlist.get(j)] = 0;
					mBottomUpWeights[cutlist.get(j)] = 0;
					bottomUpAge[cutlist.get(j)] = 0;
					mBottomUpVariance[cutlist.get(j)] = 0;
				}
				
				/*  for (int j = 0; j < growlist.size(); j++) {
		        growBottomUpConnection(i, growlist.get(j), currentSensorInput);
		        } */
				
				mFiringAge = 0;
				/* for (int j = 0; j < bottomUpWeights[i].length; j++){
			       bottomUpAge[i][j] = 0;
			       bottomUpVariance[i][j] = 0;
			    }  */
			   //filter the bottom-up weights through new bottom-up mask
			   mBottomUpWeights = elementWiseProduct(mBottomUpWeights, sign(mBottomUpMask));
			   //normalize the bottom-up weight
			   mBottomUpWeights = normalize(mBottomUpWeights, mBottomUpWeights.length, 2);
			}
		}

		//update the top-down mask vector
		private void updateTopDownMask(float meanVariance) {
			if (meanVariance > MACHINE_FLOAT_ZERO) {
				//construct the grow array list
				ArrayList<Integer> growlist = new ArrayList<Integer>();
				//construct the cut array list
				ArrayList<Integer> cutlist = new ArrayList<Integer>();

				for (int j = 0; j < mTopDownWeights.length; j++) {
					//calculate the ratio for each variance element
					float ratio = mTopDownVariance[j] / meanVariance;
					if (ratio < SMLOWERTHRESH && mTopDownMask[j] > 0) {
						mTopDownMask[j] = 1;
						// Grow nearby (neighbor of neuron i to z neuron j).
						if (topDownAge[j] > SMAGE) {
							growlist.add(j);
						}
					} else if (ratio > SMUPPERTHRESH && mTopDownMask[j] > 0) {
						// Cut connection.
						if (topDownAge[j] > SMAGE) {
							cutlist.add(j);
						}
					} else {
						if (topDownAge[j] > SMAGE && mTopDownMask[j] > 0) {
							// Version 1: Mahalanobis distance.
							mTopDownMask[j] = (1 / ratio - 1 / SMUPPERTHRESH) / (1 / SMLOWERTHRESH - 1 / SMUPPERTHRESH);

							// Version 2: Linear distance.
							/* topDownMask[i][j] = (SMUPPERTHRESH - ratio)
							   (SMUPPERTHRESH - SMLOWERTHRESH);  */
						}
					}
				}

				for (int j = 0; j < growlist.size(); j++) {
					growTopDownConnection(growlist.get(j));
				}

				for (int j = 0; j < cutlist.size(); j++) {
					mTopDownMask[cutlist.get(j)] = 0;
					mTopDownWeights[cutlist.get(j)] = 0;
					topDownAge[cutlist.get(j)] = 0;
					mTopDownVariance[cutlist.get(j)] = 0;
				}
				//normalize the top-down weight vector
				mTopDownWeights = normalize(mTopDownWeights, mTopDownWeights.length, mTopDownMask);
			}
		}
/*
		private void updateLateralMask(float meanVariance, float[] currentSensorInput) {
			if (meanVariance > MACHINE_FLOAT_ZERO) {
				ArrayList<Integer> growlist = new ArrayList<Integer>();
				ArrayList<Integer> cutlist = new ArrayList<Integer>();

				for (int j = 0; j < mLateralExcitationWeights.length; j++) {
					float ratio = mLateralVariance[j] / meanVariance;
					if (ratio < SMLOWERTHRESH && mLateralMask[j] > 0) {
						mLateralMask[j] = 1;
						// Grow nearby (neighbor of neuron i to z neuron j).
						if (lateralAge[j] > SMAGE) {
							growlist.add(j);
						}
					} else if (ratio > SMUPPERTHRESH && mLateralMask[j] > 0) {
						// Cut connection.
						if (lateralAge[j] > SMAGE) {
							cutlist.add(j);
						}
					} else {
						if (lateralAge[j] > SMAGE && mLateralMask[j] > 0) {
							// Version 1: Mahalanobis distance.
							mLateralMask[j] = (1 / ratio - 1 / SMUPPERTHRESH) / (1 / SMLOWERTHRESH - 1 / SMUPPERTHRESH);

							// Version 2: Linear distance.
							// bottomUpMask[i][j] = (SMUPPERTHRESH - ratio)/
							// (SMUPPERTHRESH - SMLOWERTHRESH);
						}
					}
				}

				for (int j = 0; j < cutlist.size(); j++) {
					mLateralMask[cutlist.get(j)] = 0;
					mLateralExcitationWeights[cutlist.get(j)] = 0;
					lateralAge[cutlist.get(j)] = 0;
					mLateralVariance[cutlist.get(j)] = 0;
				}
				
				//for (int j = 0; j < growlist.size(); j++) {
					// growBottomUpConnection(i, growlist.get(j), currentSensorInput);
				//}
				
			//	mFiringAge = 0;
				// for (int j = 0; j < bottomUpWeights[i].length; j++){
				// 	bottomUpAge[i][j] = 0;
				//	bottomUpVariance[i][j] = 0;
				// }
				mLateralExcitationWeights = elementWiseProduct(mLateralExcitationWeights, sign(mLateralMask));
				mLateralExcitationWeights = normalize(mLateralExcitationWeights, mLateralExcitationWeights.length, 2);
			}
		}
*/
		//update the lateral mask vector	
		private void updateLateralMask(float meanVariance) {
			if (meanVariance > MACHINE_FLOAT_ZERO) {
				//construct the grow array list
				ArrayList<Integer> growlist = new ArrayList<Integer>();
				//construct the cut array list
				ArrayList<Integer> cutlist = new ArrayList<Integer>();

				for (int j = 0; j < mLateralWeights.length; j++) {
					//calculate the ratio for each variance element
					float ratio = mLateralVariance[j] / meanVariance;
					if (ratio < SMLOWERTHRESH && mLateralMask[j] > 0) {
						mLateralMask[j] = 1;
						// Grow nearby (neighbor of neuron i to z neuron j).
						if (lateralAge[j] > SMAGE) {
							growlist.add(j);
						}
					} else if (ratio > SMUPPERTHRESH && mLateralMask[j] > 0) {
						// Cut connection.
						if (lateralAge[j] > SMAGE) {
							cutlist.add(j);
						}
					} else {
						if (lateralAge[j] > SMAGE && mLateralMask[j] > 0) {
							// Version 1: Mahalanobis distance.
							mLateralMask[j] = (1 / ratio - 1 / SMUPPERTHRESH) / (1 / SMLOWERTHRESH - 1 / SMUPPERTHRESH);

							// Version 2: Linear distance.
							/* topDownMask[i][j] = (SMUPPERTHRESH - ratio)
							   (SMUPPERTHRESH - SMLOWERTHRESH);   */
						}
					}
				}

				for (int j = 0; j < growlist.size(); j++) {
					growLateralConnection(growlist.get(j));
				}

				for (int j = 0; j < cutlist.size(); j++) {
					mLateralMask[cutlist.get(j)] = 0;
					mLateralWeights[cutlist.get(j)] = 0;
					lateralAge[cutlist.get(j)] = 0;
					mLateralVariance[cutlist.get(j)] = 0;
				}
				//update the lateral weight vector
				mLateralWeights = normalize(mLateralWeights, mLateralWeights.length, mLateralMask);
			}
		}
		
		//increase the top-down connections ages
		private void incrementTopDownAge() {
			for (int j = 0; j < mTopDownMask.length; j++) {
				//increase the top-down weight age which in the mask
				if (mTopDownMask[j] > 0) {
					topDownAge[j]++;
				}
			}
		}

		//increase the lateral connections ages
		private void incrementLateralAge() {
			for (int j = 0; j < mLateralMask.length; j++) {
				//increase the lateral weight age which in the mask
				if (mLateralMask[j] > 0) {
					lateralAge[j]++;
				}
			}
		}

		//increase the bottom-up connections ages
		private void incrementBottomUpAge() {
			//increase the bottom-up weight age which in the mask
			for (int j = 0; j < mBottomUpMask.length; j++) {
				if (mBottomUpMask[j] > 0) {
					bottomUpAge[j]++;
				}
			}
		}

		/*
		// TODO: implement this.
		private void growBottomUpConnection(int j, float[] currentSensorInput) {
			int[] sub = ind2sub(j, inputSize); // ind: [height, width]
			if (sub[0] - 1 >= 0) {
				int[] growSub = { sub[0] - 1, sub[1] };
				int growInd = sub2ind(growSub, inputSize);
				if (mBottomUpMask[growInd] == 0) {
					mBottomUpMask[growInd] = 1;
					mBottomUpWeights[growInd] = currentSensorInput[growInd];
				    bottomUpAge[growInd] = 0;
				    mBottomUpVariance[growInd] = 0;
				}
			}
			if (sub[0] + 1 < inputSize[0]) {
				int[] growSub = { sub[0] + 1, sub[1] };
				int growInd = sub2ind(growSub, inputSize);
				if (mBottomUpMask[growInd] == 0) {
					mBottomUpMask[growInd] = 1;
					mBottomUpWeights[growInd] = currentSensorInput[growInd];
				    bottomUpAge[growInd] = 0;
				    mBottomUpVariance[growInd] = 0;
				}
			}
			if (sub[1] - 1 >= 0) {
				int[] growSub = { sub[0], sub[1] - 1 };
				int growInd = sub2ind(growSub, inputSize);
				if (mBottomUpMask[growInd] == 0) {
					mBottomUpMask[growInd] = 1;
					mBottomUpWeights[growInd] = currentSensorInput[growInd];
				    bottomUpAge[growInd] = 0;
				    mBottomUpVariance[growInd] = 0;
				}
			}
			if (sub[1] + 1 < inputSize[1]) {
				int[] growSub = { sub[0], sub[1] + 1 };
				int growInd = sub2ind(growSub, inputSize);
				if (mBottomUpMask[growInd] == 0) {
					mBottomUpMask[growInd] = 1;
					mBottomUpWeights[growInd] = currentSensorInput[growInd];
				    bottomUpAge[growInd] = 0;
				    mBottomUpVariance[growInd] = 0;
				}
			}
		}  */
		
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

		//add new top-down connections for each neuron
		private void growTopDownConnection(int j) {
			//add the neuron's j-th (if j is larger than 0) top-down connection's neighbor (j-1)-th in the connections
			if (j > 0) {
				if (mTopDownMask[j - 1] == 0) {
					mTopDownMask[j - 1] = 1;
					//set the new connection's age
					topDownAge[j-1] = 0;
					mTopDownVariance[j-1] = 0;
					mTopDownWeights[j-1] = 0;
				}
			}
			//add the neuron's j-th (if j is less than length of the vector) top-down connection's neighbor (j+1)-th in the connections
			if (j < mTopDownMask.length - 1) {
				if (mTopDownMask[j + 1] == 0) {
					mTopDownMask[j + 1] = 1;
					//set the new connection's age
					topDownAge[j+1] = 0;
					mTopDownVariance[j+1] = 0;
					mTopDownWeights[j+1] = 0;
				}
			}
		}

		//add new lateral connections for each neuron
		private void growLateralConnection(int j) {
			//add the neuron's j-th (if j is larger than 0) lateral connection's neighbor (j-1)-th in the connections
			if (j > 0) {
				if (mLateralMask[j - 1] == 0) {
					mLateralMask[j - 1] = 1;
					//set the new connection's age
					lateralAge[j-1] = 0;
					mLateralVariance[j-1] = 0;
					mLateralWeights[j-1] = 0;
				}
			}
			//add the neuron's j-th (if j is less than length of the vector) lateral connection's neighbor (j+1)-th in the connections
			if (j < mLateralMask.length - 1) {
				if (mLateralMask[j + 1] == 0) {
					mLateralMask[j + 1] = 1;
					//set the new connection's age
					lateralAge[j+1] = 0;
					mLateralVariance[j+1] = 0;
					mLateralWeights[j+1] = 0;
				}
			}
		}
		
		//calculate the difference of each element in 2 vectors
		private float[] calculateDiff(float[] input_1, float[] input_2, float[] mask) {
			//check the 2 vectors have same length
			assert (input_1.length == input_2.length);
			//construct the result vector
			float[] result = new float[input_1.length];
			for (int i = 0; i < input_1.length; i++) {
				//calculate the difference if the element is in the mask
				if (mask[i] > 0) {
					result[i] = (Math.abs(input_1[i] - input_2[i]));
				} else {
					result[i] = 0;
				}
			}
			return result;
		}
		
		//update the weight vector
		private void updateWeights(float[] weights, float[] mask, float[] input, float[] learningRate,
				boolean normalize_flag) {
			// make sure both arrays have the same length
			assert weights.length == input.length;
            //update the element in the weight vector if the element is in the mask
			for (int i = 0; i < input.length; i++) {
				if (mask[i] > 0) {
					weights[i] = (1.0f - learningRate[i]) * weights[i] + learningRate[i] * input[i];
				} else {
					weights[i] = 0;
				}
			}
            //normalize the weight vector if needed
			if (normalize_flag) {
				weights = normalize(weights, input.length, 2);
			}
		}

		//update the weight vector
		private void updateWeights(float[] weights, float[] input, float[] learningRate, boolean normalize_flag) {

			// make sure both arrays have the same length
//			assert weights.length == input.length;

			for (int i = 0; i < input.length; i++) {
				weights[i] = (1.0f - learningRate[i]) * weights[i] + learningRate[i] * input[i];
				if(Float.isNaN(weights[i])){
					System.out.println("ihnibition meets nan: "+i);
				}
			}
			//normalize the weight vector if needed
			if (normalize_flag) {
				weights = normalize(weights, input.length, 2);
			}
		}

		//update the weight vector
		private void updateWeights(float[] weights, float[] input, float learningRate, boolean normalize_flag) {
			// make sure both arrays have the same length
			assert weights.length == input.length;
			//update the element in the weight vector
			for (int i = 0; i < input.length; i++) {
				weights[i] = (1.0f - learningRate) * weights[i] + learningRate * input[i];
			}
			//normalize the weight vector if needed
			if (normalize_flag) {
				weights = normalize(weights, input.length, 2);
			}
		}

		//calculate the learning rate
		private float[] getAmnesicLearningRate(int[] age, float[] mask) {
			//construct the result vector
			float[] result = new float[age.length];
			//calculate the learning rate for each element in the mask
			for (int i = 0; i < age.length; i++) {
				if (mask[i] > 0) {
					assert (age[i] > 0);
					result[i] = getAmnesicLearningRate(age[i]);
				}
			}
			return result;
		}

		//calculate the learning rate
		private float[] getAmnesicLearningRate(int[] age) {
			//construct the result vector
			float[] result = new float[age.length];
			//calculate the learning rate for each element
			for (int i = 0; i < age.length; i++) {
					assert (age[i] > 0);
					result[i] = getAmnesicLearningRate(age[i]);

			}
			return result;
		}

      //calculate the learning rate 		
	  private float getAmnesicLearningRate(int age){			
			float mu, learning_rate;	
			//calculate mu according age
			if(age < T1){
				mu = 0.0f;
			}			
			else if((age < T2) && (age >= T1)){
				mu = C * ((float) age - T1) / (T2-T1);
			}			
			else{
				mu = C * ((float) age - T2) / GAMMA;
			}
            //calculate the learning rate			
			learning_rate = (1+mu) / ((float) age);
			
			return learning_rate;
		}

	  //calculate the single learning rate 
	  private float getLearningRate(int age){
		  
		  return (1.0f/((float) age));
	  }
	  
	 //calculate the response for inhibitory neuron  
	  public void computeResponse(float response){
          	    newResponse = 1-response;        	   
		}
	  
	  //transfer the new response value to the old response
	  public void replaceResponse(){

				oldResponse = newResponse;		
			
	  }

	  //do inner product between 2 vectors
	  private float dotProduct(float[] a, float[] b, int size) {
			float r = 0.0f;

			for (int i = 0; i < size; i++) {
				r += (a[i] * b[i]);
			}

			return r;
		}

	  //multiply a constant for each element in a vector	  
	  private void multilyConstant(float constant, float[] array){
		  for(int i = 0; i < array.length; i++){
			  array[i] = array[i]*constant;
		  }
	  }
	  //set the neuron's state
	  public void setState(boolean state){
		  mState = state;
	  }
	  
	  //get the neuron's state
	  public boolean getState(){
		  //return the neuron's state
		  return mState;
	  }

	  //set the neuron's type
	  public void setType(int type){
		  mType = type;
	  }
	  
   	  //get the neuron's type
	  public int getType(){
		  return mType;
	  }
	  
	  //set the neuron's winner flag value
	  public void setwinnerflag(boolean flag){
		  winnerFlag=flag;
	  }
	  
	  //get the neuron's winner flag value
	  public boolean getwinnerflag(){
		  return winnerFlag;
	  }
	  
	  //set old response
	  public void setoldresponse(float response){
		  oldResponse=response;
	  }
	  
	  //get old response
	  public float getoldresponse(){
		  return oldResponse;
	  }
	  
	  //set the new response
	  public void setnewresponse(float response){
		  newResponse=response;
	  }
	
      //get the new response
	  public float getnewresponse(){
		  return newResponse;
	  }

	  //set the bottom-up reResponse
	  public void setbottomUpresponse(float response){
		  bottomUpResponse=response;
	  }

	  //get the bottom-up reResponse
	  public float getbottomUpresponse(){
		  return bottomUpResponse;
	  }
	
	  //set the top-down preResponse
	  public void settopDownresponse(float response){
		  topDownResponse=response;
	  }

	  //get the top-down preResponse
	  public float gettopDownresponse(){
		  return topDownResponse;
	  }

	  //set the lateral preResponse
	  public void setlateralExcitationresponse(float response){
		  lateralResponse=response;
	  }

	  //get the lateral preResponse
	  public float getlateralExcitationresponse(){
		  return lateralResponse;
	  }

	  //set the firing age
	  public void setinhibithage(int inage, int index){
		  mInAge[index]=inage;
	  }

	  //get the firing age
	  public int[] getinhibitionage(){
		  return mInAge;
	  }

	  //set the firing age
	  public void setfiringage(int firingage){
		  mFiringAge=firingage;
	  }

	  //get the firing age
	  public int getfiringage(){
		  return mFiringAge;
	  }

	  //set the top-down synapses' ages
	  public void settopdownages(int[] topdownages){
		  System.arraycopy(topdownages, 0, topDownAge, 0, topdownages.length);
	  }

	  //set one top-down synapse's age
	  public void settopdownage(int topdownage, int index){
		  topDownAge[index]=topdownage;
	  }

	  //get the top-down synapses' ages
	  public int[] gettopdownage(){
		  return topDownAge;
	  }

	  //set the lateral synapses' ages
	  public void setlateralages(int[] lateralages){
		  System.arraycopy(lateralages, 0, lateralAge, 0, lateralages.length);
	  }

	  //set one lateral synapse's age
	  public void setlateralage(int lateralage, int index){
		  lateralAge[index]=lateralage;
	  }

	  //set the lateral synapses' ages
	  public int[] getlateralage(){
		  return lateralAge;
	  }

	  //set the bottom-up synapses' ages
	  public void setbottomupages(int[] bottomupages){
		  System.arraycopy(bottomupages, 0, bottomUpAge, 0, bottomupages.length);

	  }

	  //set one bottom-up synapse's age
	  public void setbottomupage(int bottomupage, int index){
		  bottomUpAge[index]=bottomupage;
	  }

	  //get the bottom-up synapses' ages
	  public int[] getbottomupage(){
		  return bottomUpAge;
	  }

	  //set the neuron's index
	  public void setindex(int index){
		  mIndex=index;
	  }

	  //get the neuron's index
	  public int getindex(){
		  return mIndex;
	  }

	  //set the neuron's 3D location
	  public void setlocation(float[] location){
			System.arraycopy(location, 0, mLocation, 0, location.length);
	   }

	  //set the neuron's 3D location
	  public float[] getlocation(){
		   return mLocation;
	  }

	  //get the top-down weight vector
	  public float[] getTopDownWeights() {
			return mTopDownWeights;
		}

	  //set the top-down weight vector
	  public void setTopDownWeights(float[] topDownWeights) {
			System.arraycopy(topDownWeights, 0, mTopDownWeights, 0, topDownWeights.length);
		}

	  //set one lateral weight element
	  public void setLateralWeight(float LateralWeight, int index) {
			this.mLateralWeights[index] = LateralWeight;
		}

	  //get the lateral weight vector
	  public float[] getLateralWeights() {
			return mLateralWeights;
		}

	  //set the lateral weight vector
	  public void setLateralWeights(float[] LateralWeights) {
			System.arraycopy(LateralWeights, 0, mLateralWeights, 0, LateralWeights.length);
		}

	  //set one top-down weight element
	  public void setTopDownWeight(float topDownWeight, int index) {
			this.mTopDownWeights[index] = topDownWeight;
		}

	  //get the bottom-up weight vector
	  public float[] getBottomUpWeights() {
			return mBottomUpWeights;
		}

	  //set the bottom-up weight vector
	  public void setBottomUpWeights(float[] bottomUpWeights) {
			System.arraycopy(bottomUpWeights, 0, mBottomUpWeights, 0, bottomUpWeights.length);
		}

	  //set one bottom-up weight element
	  public void setBottomUpWeight(float bottomUpWeight, int index) {
			this.mBottomUpWeights[index] = bottomUpWeight;
		}

	  //get the top-down variance vector
	  public float[] getTopDownVariances() {
			return mTopDownVariance;
		}

	  //set the top-down variance vector
	  public void setTopDownVariances(float[] topDownVariance) {
			System.arraycopy(topDownVariance, 0, mTopDownVariance, 0, topDownVariance.length);
		}

	  //set one top-down variance vector element
	  public void setTopDownVariance(float topDownVariance, int index) {
			this.mTopDownVariance[index] = topDownVariance;
		}

	  //get the lateral variance vector
	  public float[] getLateralVariances() {
			return mLateralVariance;
		}

	  //set the lateral variance vector
	  public void setLateralVariances(float[] lateralVariance) {
			System.arraycopy(lateralVariance, 0, mLateralVariance, 0, lateralVariance.length);
		}

	  //set one lateral variance vector element
	  public void setLateralVariance(float LateralVariance, int index) {
			this.mLateralVariance[index] = LateralVariance;
		}

	  //get the bottom-up variance vector
	  public float[] getBottomUpVariances() {
			return mBottomUpVariance;
		}

	  //set the bottom-up variance vector
	  public void setBottomUpVariances(float[] bottomUpVariance) {
			System.arraycopy(bottomUpVariance, 0, mBottomUpVariance, 0, bottomUpVariance.length);
		}

	  //set one bottom-up variance vector element
	  public void setBottomUpVariance(float bottomUpVariance, int index) {
			this.mBottomUpVariance[index] = bottomUpVariance;
		}

	  //get the top-down mask vector
	  public float[] getTopDownMask() {
			return mTopDownMask;
		}

	  //set the top-down mask vector
	  public void setTopDownMasks(float[] topDownMask) {
			System.arraycopy(topDownMask, 0, mTopDownMask, 0, topDownMask.length);
		}

	  //set one top-down mask vector element
	  public void setTopDownMask(float topDownMask, int index) {
			this.mTopDownMask[index] = topDownMask;
		}
	  
	  //get the lateral mask vector
	  public float[] getLateralMask() {
			return mLateralMask;
		}

	  //set the lateral mask vector
	  public void setLateralMasks(float[] lateralMask) {
			System.arraycopy(lateralMask, 0, mLateralMask, 0, lateralMask.length);

		}

	  //set one lateral mask vector element
	  public void setLateralMask(float lateralMask, int index) {
			this.mLateralMask[index] = lateralMask;
		}


	  //get the bottom-up mask vector
	  public float[] getBottomUpMask() {
			return mBottomUpMask;
		}

	  //set the bottom-up mask vector
	  public void setBottomUpMasks(float[] bottomUpMask) {
			System.arraycopy(bottomUpMask, 0, mBottomUpMask, 0, bottomUpMask.length);
		}

	  //set one bottom-up mask vector element
	  public void setBottomUpMask(float bottomUpMask, int index) {
			this.mBottomUpMask[index] = bottomUpMask;
		}

	  //normalize the vector
	  public float[] normalize(float[] input, int size, int flag) {
			
		  float[] weight = new float[size];
			System.arraycopy(input, 0, weight, 0, size);			
			if (flag ==1){
				float min = weight[0];
				float max = weight[0];	
				//find the max element and minimum element in the vector
				for (int i = 0; i < size; i++){
					if(weight[i] < min){min = weight[i];}
					if(weight[i] > max){max = weight[i];}
				}
				//calculate the difference between max and min value
				float diff = max-min + MACHINE_FLOAT_ZERO;	
				//normalize the element form 0 to 1
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
	  
	  //update the weight vector
	  private void updateWeights(float[] weights, float[] input, float learningRate){
			
			//make sure both arrays have the same length
			assert weights.length == input.length;
			
			for (int i = 0; i < input.length; i++) {
				weights[i] = (1.0f - learningRate) * weights[i] + learningRate * input[i];
			}
		}
	  
	  //reset the responses
	  public void resetResponses(){			
			bottomUpResponse=0.0f;
			topDownResponse=0.0f;			
		}

	    //calculate the mean value for a vector
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

		//normalize the vector
		public float[] normalize1(float[] weight, int size, int flag) {
			if (flag ==1){
				float min = weight[0];
				float max = weight[0];	
				//find the max element and minimum element in the vector
				for (int i = 0; i < size; i++){
					if(weight[i] < min){min = weight[i];}
					if(weight[i] > max){max = weight[i];}
				}
				//calculate the difference between max and min value
				float diff = max-min + MACHINE_FLOAT_ZERO;	
				//normalize the element form 0 to 1
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

}