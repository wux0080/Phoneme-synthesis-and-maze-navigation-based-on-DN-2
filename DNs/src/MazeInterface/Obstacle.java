package MazeInterface;

import javax.swing.ImageIcon;

public class Obstacle extends Sprite {

    private boolean destroyed;

    public Obstacle(int x, int y) {
        
        this.x = x;
        this.y = y;

        ImageIcon ii = new ImageIcon(this.getClass().getResource("Images/Obstacle.png"));
        image = ii.getImage();

        i_width = image.getWidth(null);
        i_heigth = image.getHeight(null);
    }

    public boolean isDestroyed() {
        
        return destroyed;
    }

    public void setDestroyed(boolean val) {
        
        destroyed = val;
    }
}
