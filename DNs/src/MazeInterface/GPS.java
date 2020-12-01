package MazeInterface;
import javax.swing.ImageIcon;

public class GPS extends Sprite {

    private boolean destroyed;

    public GPS(float x, float y) {
        
        this.x = x;
        this.y = y;

        ImageIcon ii = new ImageIcon(this.getClass().getResource("Images/Arrow.png"));
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
