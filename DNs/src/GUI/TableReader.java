package GUI;
import java.io.*;
import org.apache.poi.hssf.usermodel.*;
import org.apache.poi.ss.usermodel.CellType; 
import java.util.Scanner;
import java.util.ArrayList;

public class TableReader {
	private String mFilename;
	private int type;
	private float[][] vector;
    public TableReader(String filename){
    	mFilename = filename;
    	String[] temp = mFilename.split("\\.");
    	if(temp[1].equals("xls")){
    		type = 1;
    	}
    	else if(temp[1].equals("txt")||temp[1].equals("csv")){
    		type = 2;
    	}
    	else{
    		type = 0;
    	}
    }
    
    public float[][] getTable() throws IOException {
    	if(type == 1){ 
    		vector = readXls();    		
    	}
    	if(type == 2){
    		vector = readTxt();
    	}
    	return vector;
    }
    
    public float[][] readTxt() throws IOException {
    	Scanner reader = new Scanner(new File(mFilename));
    	ArrayList<String> templist = new ArrayList<String>(); 
    	while(reader.hasNextLine()){
    		String temp = reader.nextLine();
    		if(temp.trim().matches("^[0-9].*")){
    			templist.add(temp);
    		}
    	}
    	reader.close();
    	float[][] list = new float[templist.size()][];
    	for(int i = 0; i < templist.size(); i++){
    		String[] temp2 = templist.get(i).split(","); 
    		list[i] = new float[temp2.length];
    		for(int j = 0; j < temp2.length; j++){
    			list[i][j] = Float.parseFloat(temp2[j].trim());
    		}
    	}
    	return list;
    }
    
    public float[][] readXls() throws IOException {
        InputStream is = new FileInputStream(mFilename);
        HSSFWorkbook hssfWorkbook = new HSSFWorkbook(is);

        int sheetsize = hssfWorkbook.getNumberOfSheets();

            HSSFSheet hssfSheet = hssfWorkbook.getSheetAt(0);
            
            // Row loop
            int rowsize = hssfSheet.getLastRowNum();
            float[][] list = new float[rowsize][];
            for (int rowNum = 0; rowNum < rowsize; rowNum++) {
                HSSFRow hssfRow = hssfSheet.getRow(rowNum+1);
                if (hssfRow == null) {
                    continue;
                }
                // Cell loop
                int cellsize = hssfRow.getLastCellNum();
            	list[rowNum] = new float[cellsize];
                for (int cellNum = 0; cellNum < cellsize; cellNum++) {
//                	System.out.println("row number: "+rowsize+" cell number: "+cellsize);
                	HSSFCell hssfcell = hssfRow.getCell(cellNum);
                    if (hssfcell == null) {
                        continue;
                    }
//                    System.out.println("a "+Float.parseFloat(getValue(hssfcell)));
                    list[rowNum][cellNum] = Float.parseFloat(getValue(hssfcell));
                }
            }
           
        return list;
    }
    
    @SuppressWarnings("static-access")
	private String getValue(HSSFCell hssfCell) {
        if (hssfCell.getCellTypeEnum() == CellType.BOOLEAN) {
            // return boolean type
            return String.valueOf(hssfCell.getBooleanCellValue());
        } else if (hssfCell.getCellTypeEnum() == CellType.NUMERIC) {
            // return number type
        	hssfCell.setCellType(CellType.STRING);
 //       	System.out.println("reach "+hssfCell.getStringCellValue());
            return hssfCell.getStringCellValue();
        } else {
            // return string type
            return String.valueOf(hssfCell.getStringCellValue());
        }
    }
 /*   
    public static void main(String arg[]) throws IOException{
    	TableReader t = new TableReader("002.csv");
    	float[][] x = t.getTable();
    	for(int i = 0; i < x.length; i++){
    		for(int j = 0; j < x[i].length; j++){
    				System.out.println("row"+i+" col"+j+" "+x[i][j]);
    			
    		}
    	}
    	
    }*/
}
