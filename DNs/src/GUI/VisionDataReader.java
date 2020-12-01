package GUI;
import java.io.FileNotFoundException;
import java.io.IOException;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;


public class VisionDataReader {
	private MatFileReader reader; 
	private MLCell mlCell;
	
	public VisionDataReader(String filename) {
		try {
			reader = new MatFileReader(filename);
			// TODO new_data or segment_data, need to fix this.
			mlCell = (MLCell)reader.getMLArray("segment_data");
			if (mlCell == null)
				mlCell = (MLCell)reader.getMLArray("new_data");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public int getLength(){
		return mlCell.getDimensions()[1];
	}
	
	public float[][][] getTrainingImage(int id){
		// Current VisionDataReader reads only one input image.
		float[][][] streamValue = new float[1][][];
		
		// get the training image matrix (38 x 38) as a MatLab double matrix.
		MLDouble mlTemp = (MLDouble) ( (MLStructure) mlCell.get(0,id)).getField("training_image");
		streamValue[0] = new float[mlTemp.getM()][mlTemp.getN()];
		
		for (int i = 0; i < mlTemp.getM(); i++) {
			for (int j = 0; j < mlTemp.getN(); j++) {
				streamValue[0][i][j] =  (float) ( (double) mlTemp.get(i,j) ); //(float) temp[i][j];
			}
		}
		return streamValue;
	};
	
	public int[] getWhereId(int id){
		int[] whereId; 
		MLDouble mlTemp = (MLDouble) ( (MLStructure) mlCell.get(0,id)).getField("where_id");
		whereId = new int[mlTemp.getM() * mlTemp.getN()];
		//System.out.println(Integer.toString(mlTemp.getM()) + "," + Integer.toString(mlTemp.getN()));
		for (int i = 0 ; i < mlTemp.getN(); i++){
			whereId[i] = (int)((double)mlTemp.get(0, i));
		}
		return whereId;
	}
	
	public int getWhatId(int id){
		int whatId;
		MLDouble mlTemp = (MLDouble) ( (MLStructure) mlCell.get(0,id)).getField("type");
		whatId = (int)((double)mlTemp.get(0, 0));
		return whatId;
	}
	
	public int getScaleId(int id){
		int scaleId;
		MLDouble mlTemp = (MLDouble) ( (MLStructure) mlCell.get(0,id)).getField("scale");
		scaleId = (int)((double)mlTemp.get(0, 0));
		return scaleId;
	}
}
