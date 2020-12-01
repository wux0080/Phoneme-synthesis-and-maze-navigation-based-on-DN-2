package GUI;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;  
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;  
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.MalformedURLException;  
import java.net.URL;
import java.util.zip.*;

public class JavaDownload {
	private String folder;
    private String filename;
    private String urlString;
    public JavaDownload(String name, String url){
    	folder = name;
    	urlString = url;
    	filename = urlString.substring(urlString.lastIndexOf("/") +1);
    }
    public void downloadData() {  
        URL url = null;  
        File file01 = new File(folder);
        if(file01.exists()){
        	if(file01.isDirectory()) {
        		System.out.println(folder+" Exists....");
        	}
        	else {
        		System.out.println(folder+"'s zip file Exists....");
         	    Unzip();
                if(file01.exists()){
                	System.out.println(folder+" is unzipped....");
                }
        	}
        }
        else{
        	System.out.println(folder+" not found....");
        	System.out.println("Start download "+folder+"....");
            try {  
                url = new URL(urlString);  
                DataInputStream dataInputStream = new DataInputStream(url.openStream());  
 
                FileOutputStream fileOutputStream = new FileOutputStream(new File(filename));  
  
                byte[] buffer = new byte[dataInputStream.available()];  
                int length; 
               
                while ( (length = dataInputStream.read(buffer)) > 0) {  
                    fileOutputStream.write(buffer, 0, length);                      
                }  
                fileOutputStream.flush();
  
                dataInputStream.close();  
                fileOutputStream.close(); 
         	    Unzip();
                if(file01.exists()){
                	System.out.println(folder+" download finished....");
                }
            } catch (MalformedURLException e) {  
                e.printStackTrace();  
            } catch (IOException e) {  
                e.printStackTrace();  
       }  
            
     	   System.out.println(folder+ " created....");
       }

    }
    
    public void Unzip(){
    	int BUFFER = 2048;
        try {
            BufferedOutputStream dest = null;
            FileInputStream fis = new FileInputStream(filename);
            ZipInputStream zis = new ZipInputStream(new BufferedInputStream(fis));
            ZipEntry entry;
            while((entry = zis.getNextEntry()) != null) {
            if(entry.getName().matches("^"+folder+"/[0-9,a-z,A-Z]+.*") && (! entry.getName().matches(".*/"))){
               System.out.println("Extracting: " +entry);
               int count;
               byte data[] = new byte[BUFFER];
               // write the files to the disk             
               File file3 = new File(entry.getName());
               if (entry.isDirectory()) {  
                   file3.mkdirs();  
               } else { 
               File parent = file3.getParentFile();  
                if (!parent.exists()) {  
                    parent.mkdirs();  
                } 
               }   
               FileOutputStream fos = new FileOutputStream(file3);
               dest = new BufferedOutputStream(fos, BUFFER);
               while ((count = zis.read(data, 0, BUFFER)) != -1) {
                  dest.write(data, 0, count);
               }
               dest.flush();
               dest.close();
                }
         }
            zis.close();
         } catch(Exception e) {
            e.printStackTrace();
         }
    }
    
    
}