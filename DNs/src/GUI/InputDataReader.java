package GUI;
import java.io.FileNotFoundException;
import java.io.IOException;

import com.jmatio.io.*;
import com.jmatio.types.*;

public class InputDataReader {
	
	private MatFileReader reader; 
	private MLCell mlCell;
	
	private int streamIndex; // index of the current matlab structure object.
	//private final int LIMIT = 4029; // number of samples for the final_training_data.mat
	
	private int LIMIT; // number of samples for the final_training_data.mat
	
	private int numInput;
	
	public InputDataReader(String filename){
		
		try {
			reader = new MatFileReader(filename);
			
			
			mlCell = (MLCell) reader.getMLArray("combined_data_input");
		
			// gets the width of the MLArray matrix
			LIMIT = mlCell.getN();
			//LIMIT = 40; // Length of the input sequence.
			
			System.out.println(LIMIT);
			
			numInput = (int) ((MLDouble) reader.getMLArray("numInput")).get(0).intValue();
			
			streamIndex = 0;
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	
	}
	
	public InputDataReader(String filename, int lenSequence){
		
		try {
			reader = new MatFileReader(filename);
			
			
			mlCell = (MLCell) reader.getMLArray("combined_data_input");
		
			// User specified sequence length.
			LIMIT = lenSequence;
			//LIMIT = 40; // Length of the input sequence.
			
			//System.out.println(reader.getMLArray("numInput"));
			
			numInput = (int) ((MLDouble) reader.getMLArray("numInput")).get(0).intValue();
			
			streamIndex = 0;
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	
	}
	
	/*
	 * Return true if the index is less than the total number of objects.
	 * Otherwise we reached the "end" of objects. 
	 */
	public boolean hasInput(){
		return ( streamIndex < LIMIT);
	}
	
	public int getLimitSize(){
		return LIMIT;
	}
	
	public float[][][] getStreamInput(){
		
		float[][][] streamValue = null;
		
		if(hasInput()){
			
			streamValue = new float[numInput][][];
			
			for (int k = 0; k < numInput; k++) {
				// get the training image matrix (38 x 38) as a MatLab double matrix.
				MLDouble mlTemp = (MLDouble) ( (MLStructure) mlCell.get(0,streamIndex)).getField("x" + (k+1));
				
				// Converts from a MatLab double matrix into Java double 2D array (38 x 38).
				//double[][] temp = mlTemp.getArray();
			
				// Cast all double values into float because the DN program 
				// does all of its computation with float values.
				streamValue[k] = new float[mlTemp.getM()][mlTemp.getN()];
				
						
				for (int i = 0; i < mlTemp.getM(); i++) {
					for (int j = 0; j < mlTemp.getN(); j++) {
						streamValue[k][i][j] =  (float) ( (double) mlTemp.get(i,j) ); //(float) temp[i][j];
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
