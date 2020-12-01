package MazeInterface;
import java.awt.EventQueue;
import java.io.IOException;

import javax.swing.JFrame;

import GUI.JavaDownload;

public class DNMaze extends JFrame {

    public DNMaze() throws IOException {
        
        initUI();
    }
    
    private void initUI() throws IOException {
        
        add(new Board());
        setTitle("DN_MAZE");
        
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setSize(Commons.WIDTH, Commons.HEIGTH);
        setLocationRelativeTo(null);
        setResizable(false);
        setVisible(true);
    }

    public static void main(String[] args) {
        
		JavaDownload resource = new JavaDownload("GUI_Resource", "http://cse.msu.edu/~zhengzej/GUI_Resource.zip");
		resource.downloadData();
		
        EventQueue.invokeLater(new Runnable() {
            @Override
            public void run() {                
            	DNMaze game;
				try {
					game = new DNMaze();
	                game.setVisible(true);     
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}           
            }
        });
    }
}