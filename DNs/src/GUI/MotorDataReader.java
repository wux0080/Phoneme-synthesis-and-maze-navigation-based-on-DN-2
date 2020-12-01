package GUI;
import java.io.FileNotFoundException;
import java.io.IOException;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLInt64;
import com.jmatio.types.MLStructure;

public class MotorDataReader {

	private MatFileReader reader; 
	private MLCell mlCell;
	
	private int streamIndex; // index of the current matlab structure object.
	//private final int LIMIT = 4029; // number of samples for the final_training_data.mat
	
	private int LIMIT; // number of samples for the final_training_data.mat
	
	private int numMotor;
	
	public MotorDataReader(String filename){
		
		try {
			reader = new MatFileReader(filename);
			
			
			mlCell = (MLCell) reader.getMLArray("combined_data_motor");
		
			// gets the width of the MLArray matrix
			LIMIT = mlCell.getN();
			//LIMIT = 40; // Length of the input sequence.

			numMotor = (int) ((MLDouble) reader.getMLArray("numMotor")).get(0).intValue();
			
			streamIndex = 0;
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	
	}
	
	public MotorDataReader(String filename, int lenSequence){
		
		try {
			reader = new MatFileReader(filename);
			
			
			mlCell = (MLCell) reader.getMLArray("combined_data_motor");
		
			// User specified sequence length.
			LIMIT = lenSequence;
			//LIMIT = 40; // Length of the input sequence.

			numMotor = (int) ((MLDouble) reader.getMLArray("numMotor")).get(0).intValue();
			
			streamIndex = 0;
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	
	}
	
	public int getLimitSize(){
		return LIMIT;
	}
	
	public boolean hasMotor(){
		return ( streamIndex < LIMIT-1);
	}
	

	public float[][][] getStreamInput(){
		
		float[][][] streamValue = null;
		
		if(hasMotor()){
			
			streamValue = new float[numMotor][][];
			
			for (int k = 0; k < numMotor; k++) {

				// get the training image matrix (38 x 38) as a MatLab double matrix.
				MLDouble mlTemp = (MLDouble) ( (MLStructure) mlCell.get(0,streamIndex)).getField("z" + (k+1));
				
				// Converts from a MatLab double matrix into Java double 2D array (38 x 38).
				//double[][] temp = mlTemp.getArray();
			
				// Cast all double values into float because the DN program 
				// does all of its computation with float values.
				streamValue[k] = new float[mlTemp.getM()][mlTemp.getN()];
				
						
				for (int i = 0; i < mlTemp.getM(); i++) {
					for (int j = 0; j < mlTemp.getN(); j++) {
						streamValue[k][i][j] = (float) ( (double) mlTemp.get(i,j) ); //(float) temp[i][j];
					}
				}

			}
						
			// Increment to read the next image.
			streamIndex++;
		}
		
		// return the image array with float precision.
		return streamValue;
	}
    public int getLength(){
    	return LIMIT;
    }
}
