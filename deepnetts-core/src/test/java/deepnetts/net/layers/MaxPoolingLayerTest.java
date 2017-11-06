package deepnetts.net.layers;


import deepnetts.util.Tensor;
import org.junit.Test;
import static org.junit.Assert.*;
import org.junit.Ignore;

/**
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class MaxPoolingLayerTest {
    
    public MaxPoolingLayerTest() {
    }
    
    /**
     * Test of forward method, of class MaxPoolingLayer.
     */
    @Test
    public void testForward() {

        InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6, 
                                new float[] {0.3f,  0.5f,   0.6f,  0.2f,   0.14f,  0.1f,
                                            -0.6f,  0.51f,  0.23f, 0.14f,  0.28f,  0.61f,
                                            -0.15f, 0.47f,  0.34f, 0.46f,  0.72f,  0.61f, 
                                            0.43f,  0.34f,  0.62f, 0.31f, -0.25f,  0.17f,
                                            0.53f,  0.41f,  0.73f, 0.92f, -0.21f,  0.84f,
                                            0.18f,  0.74f,  0.28f, 0.37f,  0.15f,  0.62f});
       
        Tensor filter = new Tensor(3, 3,
                                new float[] {0.1f,   0.2f,  0.3f,
                                            -0.11f, -0.2f, -0.3f,
                                             0.4f,   0.5f,  0.21f });

        // set biases to zero
        float[] biases = new float[] {0.0f};        
        
        ConvolutionalLayer convLayer = new ConvolutionalLayer(3, 3, 1);
        convLayer.setPrevLayer(inputLayer);
        convLayer.activationType = ActivationType.LINEAR;
        convLayer.init();
        convLayer.filters[0] = filter;
        convLayer.biases = biases;
                
        inputLayer.setInput(input);
        convLayer.forward();    // vidi koliki je output i njega onda pooluj
                                 /* [-0.40289998, -0.24970004, 0.11339998, 0.072799996, 0.2441,      0.38160002,
                                      0.20070001,  0.45139998, 0.5405,     0.52190006,  0.4957,      0.4742, 
                                      0.2084,      0.4037,     0.39240003, 0.1401,     -0.08989998, -0.066199996,
                                      0.27409998,  0.45,       0.72080004, 0.99470013,  0.77730006,  0.52349997,
                                      0.2044,      0.4385,     0.29759997, 0.1762,      0.074000016, 0.23410001,
                                     -0.029000014, 0.10220002, 0.21460003, 0.044200003, 0.04530002,  0.0064999983]*/
                
        MaxPoolingLayer instance = new MaxPoolingLayer(2, 2, 2);
        instance.setPrevLayer(convLayer);
        instance.init();
        instance.forward();

        // prikazi ovde i zapamcene pocicije koje su potrebne za backward pass
        
        Tensor actualOutputs = instance.getOutputs();        
        Tensor expectedOutputs = new Tensor(3, 3,
                                new float[] {0.45139998f, 0.5405f, 0.4957f,
                                             0.45f, 0.99470013f, 0.77730006f,
                                             0.4385f, 0.29759997f, 0.23410001f});

                /* maxIdxs  1,1     2,1     4,1
                            1,3     3,3     4,3
                            1,4     2,4     5,4     */
        boolean areEqual = actualOutputs.equals(expectedOutputs, 1e-7f); //delta je dozvoljena greska 1e-15
                
        assertTrue(areEqual); 
    }

    /**
     * Test of backward method, of class MaxPoolingLayer.
     * propusti gresku iz sledeceg lejera daltu unazad samo za neurone koji su bili max (na osnovu zapamcenih pozicija)         
     */
    @Ignore
    public void testBackward() {
   InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6, 
                                new float[] {0.3f,  0.5f,   0.6f,  0.2f,   0.14f,  0.1f,
                                            -0.6f,  0.51f,  0.23f, 0.14f,  0.28f,  0.61f,
                                            -0.15f, 0.47f,  0.34f, 0.46f,  0.72f,  0.61f, 
                                            0.43f,  0.34f,  0.62f, 0.31f, -0.25f,  0.17f,
                                            0.53f,  0.41f,  0.73f, 0.92f, -0.21f,  0.84f,
                                            0.18f,  0.74f,  0.28f, 0.37f,  0.15f,  0.62f});
       
        Tensor filter = new Tensor(3, 3,
                                new float[] {0.1f,   0.2f,  0.3f,
                                            -0.11f, -0.2f, -0.3f,
                                             0.4f,   0.5f,  0.21f });

        // set biases to zero
        float[] biases = new float[] {0.0f};        
        
        ConvolutionalLayer convLayer = new ConvolutionalLayer(3, 3, 1);
        convLayer.setPrevLayer(inputLayer);
        convLayer.activationType = ActivationType.LINEAR;
        convLayer.init();
        convLayer.filters[0] = filter;
        convLayer.biases = biases;
                
        inputLayer.setInput(input);
        convLayer.forward();    // vidi koliki je output i njega onda pooluj
                                 /* [-0.40289998, -0.24970004, 0.11339998, 0.072799996, 0.2441,      0.38160002,
                                      0.20070001,  0.45139998, 0.5405,     0.52190006,  0.4957,      0.4742, 
                                      0.2084,      0.4037,     0.39240003, 0.1401,     -0.08989998, -0.066199996,
                                      0.27409998,  0.45,       0.72080004, 0.99470013,  0.77730006,  0.52349997,
                                      0.2044,      0.4385,     0.29759997, 0.1762,      0.074000016, 0.23410001,
                                     -0.029000014, 0.10220002, 0.21460003, 0.044200003, 0.04530002,  0.0064999983]*/
                
        MaxPoolingLayer instance = new MaxPoolingLayer(2, 2, 2);
        instance.setPrevLayer(convLayer);
        instance.init();
        instance.forward();
               
                /* maxIdxs  1,1     2,1     4,1
                            1,3     3,3     4,3
                            1,4     2,4     5,4     */
                                
        FullyConnectedLayer nextLayer = new FullyConnectedLayer(2);
        instance.setNextlayer(nextLayer);
        nextLayer.setPrevLayer(instance);
        nextLayer.init(); // init weights               
        nextLayer.setDeltas(new Tensor(0.2f, 0.1f));
        // dubina tezina bi trebalo da bude 2 - koliko ima neurona u fc.
        /* [0.18075174, 0.5545214, 
            0.072818756, 0.31912476, 
           -0.49894053, -0.6323205, 0.2685551, 0.4376064, -0.34319848, 0.11627263, -0.3802099, 0.60284144, 0.15107232, -0.5185875, -0.41857186, 0.7019462, -0.0617525, -0.64165723] */
        
        instance.backward();        
                
        Tensor actual = instance.getDeltas();  /* 0.09160249, 0.046476226, -0.16302016,
                                                  0.09747166, -0.05701243, -0.015757836,
                                                  -0.021644289, -0.013519749, -0.076516226 */
        Tensor expected = new Tensor(3, 3,
                                new float[] {0.45139998f, 0.5405f, 0.4957f,
                                             0.45f, 0.99470013f, 0.77730006f,
                                             0.4385f, 0.29759997f, 0.23410001f});        
        
        boolean areEqual = actual.equals(expected, 1e-7f); //delta je dozvoljena greska 1e-15        
        
        assertTrue(areEqual); 
        
        // MaxPooling needs nextLayer lets say FC in order to test propagation of deltas
        // propagate only deltas for outputs that were used in forward propagation
        // use the same network as in forward pass, only add fc layer on top
        
    }


    
}
