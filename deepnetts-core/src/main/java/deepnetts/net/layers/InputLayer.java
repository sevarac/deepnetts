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
    
package deepnetts.net.layers;

import deepnetts.util.Tensor;

/** 
 * Input layer for convolutional network
 * 
 * TODO: move padding to next layer!!!
 * 
 * @author Zoran Sevarac
 */
public class InputLayer extends AbstractLayer { // data Layer
   
    /**
     * Padding parameter specifies empty border around image.
     * Set from padding parameter from the following Convolutional layer
     * 
     * @see initInNetwork#init method
     */
    private int padding = 0; // mozda je bolje da se ovde odmah navode dimenzije sa paddingom ??? ovo je malo nabudzeno???


    /**
     * Creates input layer with specified width, height, and depth (number of depth)
     * 
     * @param width layer width
     * @param height layer height
     * @param depth  layer depth (number of input depth)
     */
    public InputLayer(int width, int height, int depth) { // maybe add padding here as a param?
          this.width = width;
          this.height = height;
          this.depth = depth;  // number of input depth  
          init();
    }        

    /**
     * Creates an instance with specified width and height, with depth=1 (one channel)
     * 
     * @param width layer width
     * @param height layer height
     */
    public InputLayer(int width, int height) { // padding treba da povuce iz sledeceg lejera - konvolucionig
          this.width = width;
          this.height = height;
          this.depth = 1; // using single input channel
          init();
    }
                       
    /**
     * Initialize this layer in network.
     */
    @Override
    public final void init() {        
        inputs = new Tensor(height, width, depth);
        outputs = inputs;  // for input layer outputs are pointing to the same matrix as inputs
    }    
        
    /**
     * Sets network input
     * @param in input matrix/array
     */
    public void setInput(Tensor in) {
        inputs.setValues(in.getValues());
    }
            
    /**
     * This method does nothing in input layer
     */
    @Override
    public void forward() { 
        throw new IllegalStateException("This method does nothing and should never be called");
    }

    /**
     * This method does nothing in input layer
     */
    @Override
    public void backward() {  
        throw new IllegalStateException("This method does nothing and should never be called");
    }

    /**
     * This method does nothing in input layer
     */
    @Override
    public void applyWeightChanges() { }


    
}