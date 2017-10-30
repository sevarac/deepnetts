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
    
package deepnetts.util;

/**
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class GradientChecker {

    
    // 1. instantiate network, with specified random seed, and init small e
    // save all network weights
    // feed network with some (random) inputs 
    // Calculate total error E(w)
    // calculate gradients for all weights dE/dw (and save them)
    // for all weights in the network
    //   increase current weight for + e
    //   calculate total network error
    //   decrease original weight for -e
    //   calculate total network error    
    //   calculate gradient (E(w+2) - E(w-e)) / 2e
    //   check if the estimated gradients are the same

    private void run() {
        // createNetwork();
        
    }    
    
    
    public static void main(String[] args) {
        (new GradientChecker()).run();
    }


    
    
}
