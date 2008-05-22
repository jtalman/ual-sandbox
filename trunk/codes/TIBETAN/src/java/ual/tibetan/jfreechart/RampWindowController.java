package ual.tibetan.jfreechart;

import ual.gui.*;

import com.jrefinery.data.XYDataset;
import com.jrefinery.data.XYSeries;
import com.jrefinery.data.XYSeriesCollection;
import com.jrefinery.data.Range;

import com.jrefinery.chart.ChartPanel;
import com.jrefinery.chart.JFreeChart;
import com.jrefinery.chart.plot.XYPlot;
import com.jrefinery.chart.plot.CombinedXYPlot;
import com.jrefinery.chart.renderer.XYDotRenderer;

import com.jrefinery.chart.axis.VerticalNumberAxis;
import com.jrefinery.chart.axis.HorizontalNumberAxis;

/**
 * Ramp Window Controller
 * 
 * @author  Nikolay Malitsky
 * @version $Id: RampWindowController.java,v 1.1 2003/02/28 20:49:09 ual Exp $
 */

public class RampWindowController {
    
    static int s_counter = 0;

    // Reference to the View instance 
    RampWindow s_aw;

    // Window plots
    CombinedXYPlot plot;
    XYPlot m_gammaPlot;
    XYPlot m_rfPlot;

    // Current data
    XYSeries m_gammaSeries = new XYSeries("Gamma Series");
    

    /** Initializes and shows the Ramp Plot */
    public RampWindowController() {
        /* if(s_counter == 0){       
            s_aw = createFrame();  
            s_aw.initWindow();
	    s_aw.showWindow();
            s_counter++;
	}*/				  	
    }

    /** Initializes Window */
    public void initWindow(){
        s_aw = createWindow();  
        s_aw.initWindow();
    }

    /** Shows Window */
    public void showWindow(){
	s_aw.showWindow();
    }

    /** Update data */
    public void updateData(){
	XYDataset gammaSet = new XYSeriesCollection(m_gammaSeries);
	m_gammaPlot.setDataset(gammaSet);
    }

    /** Sets the range of the time axis */
    public void setTimeRange(double minTime, double maxTime){
	plot.getHorizontalValueAxis().setRange(minTime, maxTime);
    }

    /** Sets the range of the gamma axis */
    public void setGammaRange(double minGamma, double maxGamma){
	m_gammaPlot.getVerticalValueAxis().setRange(minGamma, maxGamma);
    }

    /** Sets the range of the RF value axis */
    public void setRFRange(double minValue, double maxValue){
	m_rfPlot.getVerticalValueAxis().setRange(minValue, maxValue);
    }

    /** Adds the gamma value */
    public void addGammaValue(double t, double gamma){
	m_gammaSeries.add(t, gamma);
    }


    private RampWindow createWindow(){

	// Create plot
	plot = 
	    new CombinedXYPlot(new HorizontalNumberAxis("Time"), CombinedXYPlot.VERTICAL);
	plot.setRenderer(new XYDotRenderer());
	plot.setDomainCrosshairVisible(true);
	plot.setRangeCrosshairVisible(true);

	 // create Gamma subplot ...
	
        XYDataset data1 = new XYSeriesCollection();
        m_gammaPlot = new XYPlot(data1, null,  new VerticalNumberAxis("Gamma"));

	 // create Rf subplot ...
	XYSeries series2 = new XYSeries("XY Series 2");
	
        XYDataset data2 = new XYSeriesCollection();
        m_rfPlot = new XYPlot(data2, null,  new VerticalNumberAxis("RF"));


        // add the subplots...
        plot.add(m_gammaPlot, 1);
        plot.add(m_rfPlot, 1);

	// Create chart
	JFreeChart chart = 
	    new JFreeChart("Ramp Chart", JFreeChart.DEFAULT_TITLE_FONT, plot, true);

	// Create chart panel
	ChartPanel chartPanel = new ChartPanel(chart);

	// Create frame
	RampWindow frame = new RampWindow();
	frame.setContentPane(chartPanel);

	return frame;
    }

    /** Test program */
    public static void main(String[] args)  {

	// Get the main window
	WindowManager wm = new WindowManager();
        ApplicationWindow aw = wm.getMainWindow();       
        
        // Initialize the main window
        wm.initMainWindow();
        
        // Open main window
        wm.showMainWindow();
        
	// Create Ramp Window
        RampWindowController rampController = new RampWindowController();
	rampController.initWindow();


	// Set Axis ranges
	double t0 = 2.0;
	rampController.setTimeRange(t0, 3.0);

	double gamma0 = 22.5;
	rampController.setGammaRange(gamma0, 23.5);

	double rfv0 = 4.0;
	rampController.setRFRange(rfv0, 5.0);


	// Show window
	rampController.showWindow();


	// Start monitoring gamma 
	for(int i = 0; i < 100; i++){
	    rampController.addGammaValue(t0 + 0.01*i, gamma0 + 0.01*i);
	    try {
		Thread.sleep(1000);
	    } catch (InterruptedException e) {
	    }
	    rampController.updateData();
	}


        // finish starting
        /*if (splash != null) {
        Splash.hideSplash(splash);
        splash = null;
        }*/
    }
    
}
