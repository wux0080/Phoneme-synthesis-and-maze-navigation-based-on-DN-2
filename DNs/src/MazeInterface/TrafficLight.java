package MazeInterface;

import java.awt.Image;

import javax.swing.ImageIcon;

public class TrafficLight extends Sprite {
	private boolean state_pass;
	protected Image image_pass;
	protected Image image_stop;
	
    public TrafficLight(int x, int y) {        
        this.x = x;
        this.y = y;
        this.state_pass = false;

        ImageIcon i_stop = new ImageIcon(this.getClass().getResource("Images/TrafficLightRed.png"));
        image_stop = i_stop.getImage();
        
        ImageIcon i_pass = new ImageIcon(this.getClass().getResource("Images/TrafficLightGreen.png"));
        image_pass = i_pass.getImage();
        
        image = image_stop;

        i_width = image.getWidth(null);
        i_heigth = image.getHeight(null);
    }
    
    public void revertStatus(){
    	this.state_pass = !this.state_pass;
    	
    	if (this.state_pass == true) {
    		image = image_pass;
    	} else {
    		image = image_stop;
    	}
    }
    
    public boolean getStatus(){
    	return state_pass;
    }
}
