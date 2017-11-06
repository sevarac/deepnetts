package deepnetts.net.layers;


import deepnetts.util.Tensor;
import org.junit.Test;
import static org.junit.Assert.*;
import org.junit.Ignore;

/**
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class ConvolutionalLayerTest {
    
    public ConvolutionalLayerTest() {
    }
    
    @Test
    public void testForward() {
        System.out.println("forward");
        // test single channel and single filter

        InputLayer inputLayer = new InputLayer(5, 5, 1);
        Tensor input = new Tensor(5, 5, 
                                new float[] {0.3f,  0.5f,   0.6f,  0.2f,  0.14f,
                                             0.1f, -0.6f,   0.51f, 0.23f, 0.14f,
                                             0.28f, 0.61f, -0.15f, 0.47f, 0.34f,
                                             0.46f, 0.72f,  0.61f, 0.43f, 0.34f, 
                                             0.62f, 0.31f, -0.25f, 0.17f, 0.53f});

        // set biases to zero
        float[] biases = new float[] {0.0f};
        
        Tensor filter = new Tensor(3, 3,
                                new float[] {0.1f,   0.2f,  0.3f,
                                            -0.11f, -0.2f, -0.3f,
                                             0.4f,  0.5f,   0.21f});        
        // use linear tf
        Tensor expectedOutputs = new Tensor(5, 5,
                                new float[] {-0.286f, -0.4659f, -0.1717f, 0.2004f, 0.112f,
                                              0.6381f, 0.6515f, 0.3927f, 0.2443f, 0.3527f,
                                             -0.0178f, 0.6073f, 0.6162f, 0.4899f, 0.2733f,
                                              0.3061f, 0.0779f, -0.1235f, 0.0222f, 0.3327f,
                                              0.091f, 0.3178f, 0.2879f, 0.0835f, -0.0137f});
                
        ConvolutionalLayer instance = new ConvolutionalLayer(3, 3, 1);
        instance.setPrevLayer(inputLayer);
        instance.setActivationType(ActivationType.LINEAR);
        instance.init();
        instance.filters[0] = filter;
        instance.biases = biases;
                
        inputLayer.setInput(input);
        instance.forward();
        Tensor actualOutputs = instance.getOutputs();
        
        boolean areEqual = actualOutputs.equals(expectedOutputs, 1e-7f); // delta je dozvoljena greska 1e-15
        
        assertTrue(areEqual); 
        
    }

    @Ignore
    public void testBackward() {
        System.out.println("backward");
        ConvolutionalLayer instance = null;
        instance.backward();
        fail("The test case is a prototype.");
    }

    @Ignore
    public void testApplyWeightChanges() {
        System.out.println("applyWeightChanges");
        ConvolutionalLayer instance = null;
        instance.applyWeightChanges();
        fail("The test case is a prototype.");
    }

}
