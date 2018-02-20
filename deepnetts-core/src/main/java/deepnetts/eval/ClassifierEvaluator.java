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
    
package deepnetts.eval;

import deepnetts.net.ConvolutionalNetwork;
import deepnetts.data.ExampleImage;
import deepnetts.data.ImageSet;
import java.util.HashMap;
import java.util.List;

/**
 * TODO: put in visrec.ml
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class ClassifierEvaluator implements Evaluator<ConvolutionalNetwork, ImageSet> {
   
    /**
     * Class labels
     */
    private List<String> classLabels;
    
    /**
     * Classification stats for each class
     */
    HashMap<String, ClassificationStats> resultsByLabels;
    
    /**
     * Total classification performance
     */
    private ClassificationStats total;
    
    private float threshold = 0.5f;
    
    // we could use ConfusionMatrix here to. Can we reuse it from some other lib?
    
              
    private void  init() {
        resultsByLabels = new HashMap<>();
        for(String label : classLabels) {
            resultsByLabels.put(label, new ClassificationStats());
        }
                
        resultsByLabels.put("negative", new ClassificationStats());
    }
    
    
    @Override
    public void evaluate(ConvolutionalNetwork convNet, ImageSet imageSet) {    
        classLabels =  imageSet.getOutputLabels();
        init();
        
        total = new ClassificationStats();
        
        for(ExampleImage exampleImage : imageSet) {
            convNet.setInput(exampleImage.getInput());
            convNet.forward();
            float[] output = convNet.getOutput();
            String predictedLabel = processResult(output, exampleImage.getTargetOutput());                       
            //System.out.println(exampleImage.getFile().getName() + " : "+exampleImage.getLabel()+ " : " + predictedLabel + " : " + output[0]);            
        }        
        
        calculatePercents();
    }

    private String processResult(float[] predictedOutput, float[] targetOutput) {
        
        if (predictedOutput.length == 1) { // binary classifier
            String clazz = classLabels.get(0);
            if ((predictedOutput[0] > threshold) && (targetOutput[0] ==1)) {
                total.correct++;
                resultsByLabels.get(clazz).correct++; //tp
                return clazz;
            } else if ((predictedOutput[0] <= threshold) && (targetOutput[0] == 0)) {
                total.correct++;
                resultsByLabels.get("negative").correct++;  // tn
                return "negative";
            } else if ((predictedOutput[0] > threshold) && (targetOutput[0] ==0)) {
                total.incorrect++; // fp
                resultsByLabels.get(clazz).incorrect++;
                return clazz;
            } else if ((predictedOutput[0] <= threshold) && (targetOutput[0] == 1)) {
                total.incorrect++; // fn
                resultsByLabels.get("negative").incorrect++;
                return "negative";
            } else {
                return "error";                            
            }
            
        } else { // multi class classifier
            // nadji max iz predictedOutput i vidi da li je na istoj idx poziciji kao i 1 u targetOutput
            int targetIdx = indexOfMax(targetOutput);
            String targetClass=null;
            
            if (!isNegativeTarget(targetOutput)) {
                targetClass = classLabels.get(targetIdx); // ovo ne radi za negative!!
            } else {
                targetClass = "negative";
            }
            
            int predictedIdx = indexOfMax(predictedOutput);
            String predictedClass = null;
            if (predictedOutput[predictedIdx] > threshold) {
                predictedClass = classLabels.get(predictedIdx);
            } else {
                predictedClass = "negative";
            }

            // todo: add tp, fp, fn, tn here i to za svaku klasu posebno - vidi kao rade u pythonu to
            if (predictedIdx == targetIdx && predictedOutput[predictedIdx] > threshold) {
                total.correct++;
                resultsByLabels.get(targetClass).correct++;
            } else if (predictedOutput[predictedIdx] > threshold && predictedIdx != targetIdx) {
                total.incorrect++;
                resultsByLabels.get(targetClass).incorrect++;
            } else if (targetClass.equals("negative") && predictedClass.equals("negative")) {
                resultsByLabels.get(targetClass).correct++;
            } else if (targetClass.equals("negative") && !predictedClass.equals("negative")) {
                resultsByLabels.get(targetClass).incorrect++;
            }
            
            return predictedClass;
        }
    }
    
    
    /**
     * Returns index of max element in specified array
     * @param array
     * @return 
     */
    private int indexOfMax(float[] array) {
        int maxIdx = 0;
        for(int i=0; i<array.length; i++) {
            if (array[i] > array[maxIdx]) maxIdx = i;
        }
        return maxIdx;
    }
    
    private boolean isNegativeTarget(float[] array) {
        for(int i=0; i<array.length; i++)
            if (array[i]!=0) return false;
        
        return true;
    }

    private void calculatePercents() {        
        for(ClassificationStats stats : resultsByLabels.values()) {
            float totalForLabel = stats.correct + stats.incorrect;
            stats.correctPercent = (stats.correct / totalForLabel)*100;
            stats.incorrectPercent = (stats.incorrect / totalForLabel)*100;
        }
    }
    
    public static class ClassificationStats {
        String classLabel;
        // razdvojiti ove
        // correct = tp+tn
        // incorrect = fp+fn
        int correct=0, incorrect=0;
        float correctPercent = 0, incorrectPercent = 0;
        
        int tp, tn, fp, fn;
        
        
        @Override
        public String toString() {
            return "correct = " + correct + " ("+correctPercent +"%), incorrect = " + incorrect+" ("+incorrectPercent+"%)";
        }                        
    }

    public float getThreshold() {
        return threshold;
    }

    public void setThreshold(float threshold) {
        this.threshold = threshold;
    }
    
    

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append(System.lineSeparator()).append("------------------------------------------------------------------------").append(System.lineSeparator()).
           append("CLASSIFIER EVALUATION RESULTS ").append(System.lineSeparator()).append("------------------------------------------------------------------------").append(System.lineSeparator());
        sb.append("Total classes: ").append(classLabels.size()).append(System.lineSeparator());
        sb.append("Total correct: ").append(total.correct).append(System.lineSeparator());
        sb.append("Total incorrect: ").append(total.incorrect).append(System.lineSeparator());        
        sb.append("Results by labels").append(System.lineSeparator());
        
        for(String label : resultsByLabels.keySet()) {            
            ClassificationStats result = resultsByLabels.get(label);
            if (result.correct == 0 && result.incorrect == 0) continue; // if some of them is negative or nan dont show it
            sb.append(label).append(": ");   
            sb.append(result).append(System.lineSeparator());
        }
        
        return sb.toString();
    }
               
}