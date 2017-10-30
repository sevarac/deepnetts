/**  
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation 
 *  based learning and image recognition.
 * 
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 *  This file is part of DeepNetts.
 *
 *  DeepNetts is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.package deepnetts.core;
 */
    
package deepnetts.data;

import deepnetts.util.ColorUtils;
import deepnetts.util.Tensor;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

/**
 * This class represents example image to train the network. 
 * It contains image and label information.
 */
public class ExampleImage implements DataSetItem {
    
    /**
     * Image dimensions - width and height
     */
    private final int width, height;   // dont need this here , maybe only in dataset
           
    /**
     * Image label, a concept to map to this image
     */
    private final String label; 
    
    /**
     * Desired network output - maybe its better to use  int - output index with 1 ? lesss memory for huge data sets - TODO: use int here
     */   
    private float[] targetOutput; // output vector depends on number of classes- this could be int in order to save memory
              
    /**
     * Transformed RGB values of Image pixels
     * used as an input for neural net
     */
    private float[] rgbVector; 
    
    private Tensor rgbTensor;
    
    private File file;
    

    /**
     * Creates an instance of new example image with specified image and label
     * Loads image from specified file and creates matrix structures with color information
     * 
     * @param file image file
     * @param label image label
     * @throws IOException if file is not found or reading file fails from some reason.
     */
    public ExampleImage(File file, String label) throws IOException {
        this.label = label;
        this.file = file;
        BufferedImage image = ImageIO.read(file);
        width = image.getWidth();
        height = image.getHeight();

        extractPixelColors(image);        
    }

    public ExampleImage(BufferedImage image, String label) throws IOException {
        this.label = label;
        width = image.getWidth();
        height = image.getHeight();

        extractPixelColors(image);        
    }
    
    public ExampleImage(BufferedImage image) {
        this.label = null;
        width = image.getWidth();
        height = image.getHeight();

        extractPixelColors(image);        
    }    

    // getRgbVector
    private void extractPixelColors(BufferedImage image ) {
        int[][][] pixels = new int[height][width][3]; // da li ovde mogu da koristim Raster ili DataBuffer?
        rgbVector = new float[width * height * 3];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int color = image.getRGB(x, y);

                pixels[y][x][0] = ColorUtils.getBlue(color);
                pixels[y][x][1] = ColorUtils.getGreen(color);
                pixels[y][x][2] = ColorUtils.getRed(color);

                // razvicu boju na interval [-0.1, 1.175] umesto [0,1]
                // -0.1 + 1.275 * x
                // -0.1 + 1.275 * (1-x)
                
                rgbVector[y * width + x] = pixels[y][x][0] / 255.0f;// - 0.5;   normalize & translate // TODO: proveri da li ovo radi dobro!!!
                rgbVector[width * height + y * width + x] = pixels[y][x][1] / 255.0f;// - 0.5;
                rgbVector[2 * width * height + y * width + x] = pixels[y][x][2] / 255.0f;// - 0.5;
            }
        }
        
        rgbTensor = new Tensor(height, width, 3, rgbVector);
    }    

    @Override
    public float[] getTargetOutput() {
        return targetOutput;
    }
    
    public void setRgbVector(float[] array) {
        rgbVector = array;        
    }

    public float[] getRgbVector() {
        return rgbVector;
    }
    
    public final void setTargetOutput(float[] desiredOutput) {
        this.targetOutput = desiredOutput;
    }
      
    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public String getLabel() {
        return label;
    }

    @Override
    public Tensor getInput() {
        return rgbTensor;
    }

    public File getFile() {
        return file;
    }

}