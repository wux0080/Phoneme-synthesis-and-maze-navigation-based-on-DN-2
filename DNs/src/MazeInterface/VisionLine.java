package MazeInterface;

import java.util.Random;

public class VisionLine implements Commons{
public int start_x;
public int start_y;
public int end_x;
public int end_y;
public float length;
public env type; // Hitting brick, obstacle or destination

public VisionLine(int x, int y){
	start_x = x; 
	start_y = y;
	end_x = x;
	end_y = y;
}
public void SetEnd(int x, int y){
	end_x = x;
	end_y = y;
}

public float getLength(){
	length = (float)Math.sqrt((start_x - end_x) * (start_x - end_x) +  
			(start_y - end_y) * (start_y - end_y)); 
	float noise_level = new Random().nextFloat() * noise; 
	if (new Random().nextFloat() > 0.5) { 
		length = (1 + noise_level) * length; 
	} else { 
		length = (1 - noise_level) * length; 
	};
	return length;
}

public void setLength(float input_length, float angle){
	length = input_length;
	end_x = start_x + (int) (input_length * Math.cos(Math.toRadians(angle)));
	end_y = start_y + (int) (input_length * Math.sin(Math.toRadians(angle)));
}

public void setType(env input_type){
	type = input_type;
}
}
