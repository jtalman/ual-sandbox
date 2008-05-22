package ual.gui;

import java.awt.*;
import java.lang.ref.*;
import javax.swing.*;

/**
 * A prtotype of the UAL splash class based on the NetBeans design.
 *
 * @author  Nikolay Malitsky
 * @version $Id: Splash.java,v 1.1 2003/02/28 20:41:40 ual Exp $
 */

public class Splash {

    /** The splash image */
    static Reference splashRef;
    
    public Splash() {
    }
    
    /** Shows the splash window. */
    public static SplashOutput showSplash() {
        
        Window splashWindow = (Window) new SplashFrame();
        splashWindow.show();
        splashWindow.toFront();
        return (SplashOutput) splashWindow;
    }
    
    /** Hides the splash window. */
    public static void hideSplash( SplashOutput xsplashWindow){
        
        final Window splashWindow = (Window) xsplashWindow;
        javax.swing.SwingUtilities.invokeLater ( new Runnable () {
            public void run() {
                splashWindow.setVisible(false);
                splashWindow.dispose();
            }
        });
        
    }
    
    static interface SplashOutput {
        public void print (String s);
    }
    
    /**
     * Standard way how to place the window to the center of the screen.
     */
    public static final void center(Window c){
        c.pack();
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
        Dimension dialogSize = c.getSize();
        c.setLocation((screenSize.width - dialogSize.width)/2,
                        (screenSize.height - dialogSize.height)/2);        
    }
    
    /** Returns the splash image */
    static Image getSplash() {
        Image ret;
        if((splashRef == null) || ((ret = (Image) splashRef.get()) == null)){
            ret = loadSplash();
            splashRef = new WeakReference(ret);
        }
        return ret;
    }
    
    /** Loads a splash image from its source */
    private static Image loadSplash() {
       
        Image image = null;
        
        ClassLoader cl = ClassLoader.getSystemClassLoader();        
        java.net.URL imageURL = cl.getResource("splash.gif");
        if(imageURL != null) image = Toolkit.getDefaultToolkit().getImage(imageURL);
        return image;

    }
    
    /**
     * This class implements double-buffered splash screen component.
     */
    static class SplashComponent extends JComponent {
        
        private Image image;
        
        public SplashComponent() {
            
            // image = new ImageIcon(getSplash()).getImage();
            
            Splash splash = new Splash();
            ClassLoader cl = splash.getClass().getClassLoader();
            java.net.URL imageURL = cl.getResource("splash.gif");
            if(imageURL != null) image = Toolkit.getDefaultToolkit().getImage(imageURL);
            image = new ImageIcon(image).getImage();
        }
        
        /**
         * Defines the single line of text this component will display.
         */
        public void setText(String text){
        }
        
        /**
         * Override update to *not* erase the background before painting.
         */
        public void update(Graphics g){
            paint(g);
        }
        
        /**
         * Renders this component to the given graphics.
         */
        public void paint(Graphics graphics){
            // draw background
            graphics.drawImage(image, 0, 0, null);
        }
        
        public Dimension getPreferredSize() {
            return new Dimension(image.getWidth(null), image.getHeight(null));
        }
        
        public boolean isOpaque() {
            return true;
        }
    }
    
    static class SplashFrame extends JFrame implements SplashOutput {
        
        private final SplashComponent splashComponent = new SplashComponent();
        
        public SplashFrame() {
            super("UAL Simulation Facility");
            setDefaultCloseOperation(DO_NOTHING_ON_CLOSE);
            setCursor(java.awt.Cursor.getPredefinedCursor(java.awt.Cursor.WAIT_CURSOR));
            
            // add splash component
            getContentPane().add(splashComponent);
            Splash.center(this);
        }
        
        /**
         * Prints the given progress message on the splash screen.
         */
        public void print(String x){
            splashComponent.setText(x);
        }
    }

}
