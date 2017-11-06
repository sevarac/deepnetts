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
    
package deepnetts.net.train;

import deepnetts.core.DeepNetts;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.AbstractLayer;
import deepnetts.data.DataSet;
import deepnetts.data.DataSetItem;
import deepnetts.net.loss.LossFunction;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

/**
 * This class implements training algorithm for convolutional neural network.
 * 
 * @author Zoran Sevarac <zoran.sevarac@smart4net.co>
 */
public class BackpropagationTrainer implements Trainer<DataSet<?>>, Serializable {

    /**
     * Maximum number of training iterations.
     * Training will stop when this number of iterations is reached regardless the total network error.
     */
    private long maxIterations = 100000L;
    
    /**
     * Maximum allowed error.
     * Training will stop once total error has reached this value .
     */
    private float maxError = 0.01f;
    
    /**
     * Learning rate.
     */
    float learningRate = 0.01f;
    
    /**
     * Optimizer type for all layers
     */
    OptimizerType optimizer = OptimizerType.SGD;
    
    float momentum = 0;
   
    boolean batchMode = false;
    int batchSize;    
    boolean stopTraining = false;
    
    int iteration;
    private float totalError;
      
    LossFunction lossFunction;
    
    List<TrainingListener> listeners = new ArrayList<>();
    
    
    /**
     * Network to train
     */
    private NeuralNetwork neuralNet;

    private final static Logger LOGGER = Logger.getLogger(DeepNetts.class.getName());
    static {
        try {
      //      LogManager.getLogManager().reset(); // nemoj da skidas defaultnog loggera nego mu setuj formatera. Zato jer ovo skine i loggera i dnd
            Handler fh = new FileHandler("deepnetts_training.log"); // ConsoleHandler();

            fh.setFormatter(new Formatter() {
                @Override
                public String format(LogRecord record) {
                    return record.getMessage() + System.lineSeparator();
                }});
                      
         //   Handler ch = new ConsoleHandler();
//            ch.setFormatter(new Formatter() {
//                @Override
//                public String format(LogRecord record) {
//                    return record.getMessage() + System.lineSeparator();
//                }});
            
            LOGGER.getParent().getHandlers()[0].setFormatter(new Formatter() {
                @Override
                public String format(LogRecord record) {
                    return record.getMessage() + System.lineSeparator();
                }
            });
            
            LOGGER.addHandler(fh);
            LOGGER.setLevel(Level.ALL);
        } catch (IOException ex) {
            Logger.getLogger(BackpropagationTrainer.class.getName()).log(Level.SEVERE, null, ex);
        } catch (SecurityException ex) {
            Logger.getLogger(BackpropagationTrainer.class.getName()).log(Level.SEVERE, null, ex);
        }
        
    }

    public BackpropagationTrainer(NeuralNetwork neuralNet) { 
        
        if (neuralNet == null) throw new IllegalArgumentException("Argument convNet cannot be null!");
        
        this.neuralNet = neuralNet;
    }

    
    /**
     * This method does actual training procedure.
     * 
     * Make this pure function so it can run in multithreaded - can train several nn in parallel
     * put network as param
     * 
     * @param dataSet 
     */
    @Override
    public void train(DataSet<?> dataSet) {

        if (dataSet == null) throw new IllegalArgumentException("Argument dataSet cannot be null!");
        if (dataSet.size() == 0) throw new RuntimeException("Data set is empty!");
        
        // neuralNet.setOutputLabels(dataSet.getLabels());
        //LOGGER.info(FileIO.toJson((ConvolutionalNetwork)neuralNet));
                
        int trainingSamplesCount = dataSet.size();   
        stopTraining = false;
     
        if (batchMode && (batchSize==0)) batchSize = trainingSamplesCount; 
        
        // set same lr to all layers!
        for(AbstractLayer layer : neuralNet.getLayers()) {
            layer.setLearningRate(learningRate);
            layer.setMomentum(momentum);
            layer.setBatchMode(batchMode);
            layer.setBatchSize(batchSize);
            layer.setOptimizer(optimizer);
        }
     
        lossFunction = neuralNet.getLossFunction();
                
        float[] outputError; 
        iteration = 0;                
        totalError = 0;
        float prevTotalError=0, totalErrorChange;
        long startTraining, endTraining, trainingTime, startEpoch, endEpoch, epochTime;
        
        fireTrainingEvent(TrainingEvent.Type.STARTED);    

        startTraining = System.currentTimeMillis();
        do {
            iteration++;
            totalError = 0; // lossFunction.reset();
            int sampleCounter = 0;
            
          //  LOGGER.log(Level.INFO, "Iteration:{0}", new Object[]{iteration});

            startEpoch = System.currentTimeMillis();
                      
            for (DataSetItem dataSetItem : dataSet) {       // for all items in dataset 
                sampleCounter++;
                neuralNet.setInput(dataSetItem.getInput());   // set network input     
                neuralNet.forward();                                // do forward pass / calculate network output                         
                
                outputError = lossFunction.calculateOutputError(neuralNet.getOutput(), dataSetItem.getTargetOutput()); // get output error using loss function
                neuralNet.setOutputError(outputError); //mozda bi ovo moglao da bude uvek isti niz/reference pa ne mora da se seuje                
                totalError += lossFunction.getPatternError();  // maybe sum this in loass function                              
                
                neuralNet.backward();                                 // do the backward propagation using current outputError - should I use outputError as a param here?
                             
//                if (LOGGER.getLevel().intValue() <= Level.FINE.intValue()) {
//                    LOGGER.log(Level.INFO, ConvNetLogger.getInstance().logNetwork(neuralNet));  //  log the network details (outputs, wiights deltas ... )- for debugging purposes
//                }    
                
                // weight update for online mode after each training pattern
                if (!isBatchMode()) { // for online training update weight changes after each pass
                    neuralNet.applyWeightChanges();   
                } 
                else if (sampleCounter % batchSize == 0) { // mini batch
                    //LOG.info("Weight Update after: "+sampleCounter);    // ovde logovati mini batch, mozda i bacati event
                    neuralNet.applyWeightChanges();   
                }
                
                fireTrainingEvent(TrainingEvent.Type.ITERATION_FINISHED);  // move this inside for loop so we can track each pattern                           
                
            }
            endEpoch = System.currentTimeMillis(); 
            
          //   batch weight update after entire data set - ako vrlicina dataseta nije deljiva sa batchSize - ostatak
            if (isBatchMode() && (trainingSamplesCount % batchSize !=0 )) { // full batch. zarga ovaj gore ne pokriva?
                neuralNet.applyWeightChanges();
            }           
                                    
            totalError = totalError / trainingSamplesCount; // - da li total error za ceo data set ili samo za mini  batch? lossFunction.getTotalError()
            totalErrorChange = totalError - prevTotalError; // todo: pamti istoriju ovoga i crtaj funkciju, to je brzina konvergencije na 10, 100, 1000 iteracija paterna - ovo treba meriti
            prevTotalError = totalError;
            epochTime = endEpoch-startEpoch;

            LOGGER.log(Level.INFO, "Iteration:" + iteration + ", Time:"+epochTime + "ms, TotalError:" + totalError +", ErrorChange:"+totalErrorChange); // Time:"+epochTime + "ms,
    //        LOG.log(Level.INFO, ConvNetLogger.getInstance().logNetwork(neuralNet));  
            
            fireTrainingEvent(TrainingEvent.Type.EPOCH_FINISHED);
            
            stopTraining = ((iteration == maxIterations) || (totalError <= maxError)) || stopTraining;
            
        } while (!stopTraining); // or learning slowed, or overfitting, ...    
        
        endTraining = System.currentTimeMillis();
        trainingTime = endTraining - startTraining;
        
        LOGGER.info("Total Training Time: " + trainingTime + "ms");
        
        fireTrainingEvent(TrainingEvent.Type.STOPPED);
    }

    public long getMaxIterations() {
        return maxIterations;
    }

    public BackpropagationTrainer setMaxIterations(long maxIterations) {
        this.maxIterations = maxIterations;
        return this;
    }

    public float getMaxError() {
        return maxError;
    }

    // cannot be negative
    public BackpropagationTrainer setMaxError(float maxError) {
        this.maxError = maxError;
        return this;
    }

    // cannot be negative
    public BackpropagationTrainer setLearningRate(float learningRate) {
        this.learningRate = learningRate;
        return this;
    }
    
   
    private void fireTrainingEvent(TrainingEvent.Type type) {
        for(TrainingListener l : listeners) {
            l.handleEvent(new TrainingEvent<>(this, type));
        }
    }
    
    public void addListener(TrainingListener listener) {
        if (!listeners.contains(listener)) {
            listeners.add(listener);
        }
    }

    public void removeListener(TrainingListener listener) {
        listeners.remove(listener);
    }

    public boolean isBatchMode() {
        return batchMode;
    }

    public BackpropagationTrainer setBatchMode(boolean batchMode) {
        this.batchMode = batchMode;
        return this;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public BackpropagationTrainer setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    public BackpropagationTrainer setMomentum(float momentum) {
       this.momentum = momentum;
       return this;
    }

    public float getMomentum() {
        return momentum;
    }

    public float getLearningRate() {
        return learningRate;
    }
    
    public void stop() {
        stopTraining = true;
    }

    public float getTotalError() {
        return totalError;
    }

    public int getIteration() {
        return iteration;
    }

    public OptimizerType getOptimizer() {
        return optimizer;
    }

    public BackpropagationTrainer setOptimizer(OptimizerType optimizer) {
        this.optimizer = optimizer;
        return this;
    }
    
}
