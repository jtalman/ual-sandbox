package ual.tibetan.jfreechart;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

import ual.gui.*;


/**
 * Ramp Window
 *
 * @author  Nikolay Malitsky
 * @version $Id: RampWindow.java,v 1.1 2003/02/28 20:49:09 ual Exp $
 */

public class RampWindow extends ual.gui.ApplicationWindow {
    
    public RampWindow() { 
    }
            
    /** Initializes window components */
    public void initWindow() {
        
        super.initWindow();
        
        // Define title
        updateTitle ();
        
        // Create a list of actions
        // actions = new AcmActions();
        
        // Create the menu bar
	// ...
        // Create the tool bar 
        // ...
        // Initialize application data
        // ...
                
        this.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                JFrame frame = (JFrame) e.getSource();
                frame.setVisible (false);
	        frame.dispose ();
                // RampPlotController.s_aw = null;
                // RampPlotController.s_counter = 0;
            }
	});
        
    }

    /** Shows window */
    public void showWindow(){
        // Add Menu Bar
        if (menuBar != null) {
            setJMenuBar(menuBar);
        }
               
        // Add Tool Bar
        if(toolBar != null){
            getContentPane().add(toolBar, BorderLayout.NORTH);
        }

        // Add Applications Panel
        // AcmNavigationPanel appsPanel = new AcmNavigationPanel();  
        // getContentPane().add(appsPanel, BorderLayout.CENTER);   
            
        super.showWindow();
        
    }
    
    /** Updates the Window's title */
    void updateTitle () {
        setTitle ("Ramp Window");
    }
    
}
