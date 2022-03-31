package MazeInterface;
import javax.swing.ImageIcon;

public class Destination extends Sprite {

    private boolean destroyed;

    public Destination(int x, int y) {
        
        this.x = x;
        this.y = y;

        ImageIcon ii = new ImageIcon(this.getClass().getResource("Images/Destination.png"));
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
