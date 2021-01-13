package MazeInterface;
import java.awt.Image;
import java.awt.Rectangle;

public class Sprite {

    protected float x;
    protected float y;
    protected int i_width;
    protected int i_heigth;
    protected Image image;

    public void setX(int x) {
        this.x = x;
    }

    public int getX() {
        return (int)x;
    }

    public void setY(int y) {
        this.y = y;
    }

    public int getY() {
        return (int)y;
    }

    public int getWidth() {
        return i_width;
    }

    public int getHeight() {
        return i_heigth;
    }

    Image getImage() {
        return image;
    }

    Rectangle getRect() {
        return new Rectangle((int)x, (int)y,
                image.getWidth(null), image.getHeight(null));
    }
}
