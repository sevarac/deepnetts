package deepnetts.net.layers;

import deepnetts.util.RandomGenerator;
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
    public void testForwardSingleChannel() {

        InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6,
                new float[]{0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                    -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                    -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                    0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                    0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                    0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f});

        Tensor filter = new Tensor(3, 3,
                new float[]{0.1f, 0.2f, 0.3f,
                    -0.11f, -0.2f, -0.3f,
                    0.4f, 0.5f, 0.21f});

        // set biases to zero
        float[] biases = new float[]{0.0f};

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

        Tensor actualOutputs = instance.getOutputs();
        Tensor expectedOutputs = new Tensor(3, 3,
                new float[]{0.45139998f, 0.5405f, 0.4957f,
                    0.45f, 0.99470013f, 0.77730006f,
                    0.4385f, 0.29759997f, 0.23410001f});

        /* maxIdxs  1,1     1,2     1,4
                            3,1     3,3     3,4
                            4,1     4,2     4,5  */
        assertArrayEquals(actualOutputs.getValues(), expectedOutputs.getValues(), 1e-8f);
    }

    @Test
    public void testForwardMultiChannel() {

        InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6,
                new float[]{0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                    -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                    -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                    0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                    0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                    0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f});

        ConvolutionalLayer convLayer = new ConvolutionalLayer(3, 3, 2);
        convLayer.setPrevLayer(inputLayer);
        convLayer.activationType = ActivationType.LINEAR;
        convLayer.init();
        convLayer.filters[0] = new Tensor(3, 3,
                new float[]{0.1f, 0.2f, 0.3f,
                    -0.11f, -0.2f, -0.3f,
                    0.4f, 0.5f, 0.21f});
        convLayer.filters[1] = new Tensor(3, 3,
                new float[]{0.11f, 0.21f, 0.31f,
                    -0.21f, -0.22f, -0.23f,
                    0.31f, 0.31f, 0.31f});
        // set biases to zero
        convLayer.biases = new float[]{0.0f, 0.0f};

        inputLayer.setInput(input);
        convLayer.forward();    // output from convolutional layer:
        /*         
                                    -0.40289998, -0.24970004, 0.11339998, 0.072799996, 0.2441,      0.38160002,
                                     0.20070001,  0.45139998, 0.5405,     0.52190006,  0.4957,      0.4742,
                                     0.2084,      0.4037,     0.39240003, 0.1401,     -0.08989998, -0.066199996,
                                     0.27409998,  0.45,       0.72080004, 0.99470013,  0.77730006,  0.52349997,
                                     0.2044,      0.4385,     0.29759997, 0.1762,      0.074000016, 0.23410001,
                                    -0.029000014, 0.10220002, 0.21460003, 0.044200003, 0.04530002,  0.0064999983,
                                    
                                    -0.20889999, -0.26760003, -0.010199998, -6.999895E-4,  0.22350001,   0.22450002,
                                     0.3319,      0.48950002,  0.44680002,   0.4791,       0.40600002,   0.25570002, 
                                     0.19569999,  0.3932,      0.2622,       0.014099985, -0.060699996, -0.15130001, 
                                     0.2328,      0.3976,      0.6252,       0.6627,       0.8222,       0.4177, 
                                     0.27,        0.31350002,  0.23630002,  -0.0035999827, 0.04750003,   0.10620001, 
                                     0.028599992, 0.105699986, 0.18150005,   0.033699997,  0.064200014, -0.014600009"        
         */

        MaxPoolingLayer instance = new MaxPoolingLayer(2, 2, 2);
        instance.setPrevLayer(convLayer);
        instance.init();
        instance.forward();

        Tensor actualOutputs = instance.getOutputs();
        Tensor expectedOutputs = new Tensor(3, 3, 2,
                new float[]{0.45139998f, 0.5405f, 0.4957f,
                    0.45f, 0.99470013f, 0.77730006f,
                    0.4385f, 0.29759997f, 0.23410001f,
                    0.48950002f, 0.4791f, 0.40600002f,
                    0.3976f, 0.6627f, 0.8222f,
                    0.31350002f, 0.23630002f, 0.10620001f});

        /* maxIdxs  1,1     1,2     1,4
                            3,1     3,3     3,4
                            4,1     4,2     4,5 
        
                            1,1     1,3     1,4        
                            3,1     3,3     3,4 
                            4,1     4,2     4,5
         */
        assertArrayEquals(actualOutputs.getValues(), expectedOutputs.getValues(), 1e-8f);
    }

    /**
     * Test of backward method, of class MaxPoolingLayer. propusti gresku iz
     * sledeceg lejera daltu unazad samo za neurone koji su bili max (na osnovu
     * zapamcenih pozicija)
     */
    @Test
    public void testBackwardSingleChannelFromFullyConnected() {
        RandomGenerator.getDefault().initSeed(123);
        InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6,
                new float[]{0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                    -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                    -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                    0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                    0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                    0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f});

        Tensor filter = new Tensor(3, 3,
                new float[]{0.1f, 0.2f, 0.3f,
                    -0.11f, -0.2f, -0.3f,
                    0.4f, 0.5f, 0.21f});

        // set biases to zero
        float[] biases = new float[]{0.0f};

        ConvolutionalLayer convLayer = new ConvolutionalLayer(3, 3, 1);
        convLayer.setPrevLayer(inputLayer);
        convLayer.activationType = ActivationType.LINEAR;
        convLayer.init();
        convLayer.filters[0] = filter;
        convLayer.biases = biases;

        inputLayer.setInput(input);
        convLayer.forward();    // vidi koliki je output i njega onda pooluj

        MaxPoolingLayer instance = new MaxPoolingLayer(2, 2, 2);
        instance.setPrevLayer(convLayer);
        instance.init();
        instance.forward();

        /* Max Pooling Output
                                new float[] {0.45139998f, 0.5405f, 0.4957f,
                                             0.45f, 0.99470013f, 0.77730006f,
                                             0.4385f, 0.29759997f, 0.23410001f});        
         */
                /* maxIdxs  1,1     1,2     1,4
                            3,1     3,3     3,4
                            4,1     4,2     4,5  */
        FullyConnectedLayer nextLayer = new FullyConnectedLayer(2);
        instance.setNextlayer(nextLayer);
        nextLayer.setPrevLayer(instance);
        nextLayer.init(); // init weights               
        nextLayer.setDeltas(new Tensor(0.1f, 0.2f));
        // poslednja dimenzija matrice tezina je 2 - koliko ima neurona u fc. - zasto je 3x3x1  X  2  (prev layer x fcCols)
        /* test samo sa jednim neuronom u fc i delta 0.1, pomnozi sve tezine sa 0.1
        
        weights sa dva neurona u fc:
            0.18075174, 0.5545214, 0.072818756,
            0.31912476, -0.49894053, -0.6323205,
            0.2685551, 0.4376064, -0.34319848, 
        
            0.11627263, -0.3802099, 0.60284144,
            0.15107232, -0.5185875, -0.41857186,
            0.7019462, -0.0617525, -0.64165723
        
         */

        instance.backward();
        Tensor actual = instance.getDeltas();

        // sum delta * weight and transpose
        // Test: 0.1 * 0.18075174 + 0.2 * 11627263 = 0.0413297 ... 
        Tensor expected = new Tensor(3, 3,
                new float[]{0.0413297f, 0.062126942f, 0.16724476f, 
                           -0.020589843f, -0.15361156f, 0.03141014f,
                            0.12785016f, -0.14694643f, -0.1626513f});

        assertArrayEquals(actual.getValues(), expected.getValues(), 1e-8f);
    }
    
    @Test
    public void testBackwardMultiChannelFromFullyConnected() {
        RandomGenerator.getDefault().initSeed(123);
        
        InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6,
                new float[]{0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                    -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                    -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                    0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                    0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                    0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f});

        Tensor filter = new Tensor(3, 3,
                new float[]{0.1f, 0.2f, 0.3f,
                    -0.11f, -0.2f, -0.3f,
                    0.4f, 0.5f, 0.21f});

        // set biases to zero
        float[] biases = new float[]{0.0f, 0.0f};

        ConvolutionalLayer convLayer = new ConvolutionalLayer(3, 3, 2);
        convLayer.setPrevLayer(inputLayer);
        convLayer.activationType = ActivationType.LINEAR;
        convLayer.init();
        convLayer.filters[0] = filter; // treba mi 2 filtera
        convLayer.filters[1] = filter;
        convLayer.biases = biases;

        inputLayer.setInput(input);
        convLayer.forward();    // vidi koliki je output i njega onda pooluj

        MaxPoolingLayer instance = new MaxPoolingLayer(2, 2, 2);
        instance.setPrevLayer(convLayer);
        instance.init();
        instance.forward();

        /* Max Pooling Output
                                new float[] {0.45139998f, 0.5405f, 0.4957f,
                                             0.45f, 0.99470013f, 0.77730006f,
                                             0.4385f, 0.29759997f, 0.23410001f});        
         */
                /* maxIdxs  1,1     1,2     1,4
                            3,1     3,3     3,4
                            4,1     4,2     4,5  */
        FullyConnectedLayer nextLayer = new FullyConnectedLayer(2);
        instance.setNextlayer(nextLayer);
        nextLayer.setPrevLayer(instance);
        nextLayer.init(); // init weights               
        nextLayer.setDeltas(new Tensor(0.1f, 0.2f));
        // poslednja dimenzija matrice tezina je 2 - koliko ima neurona u fc. - zasto je 3x3x1  X  2  (prev layer x fcCols)
        /* test samo sa jednim neuronom u fc i delta 0.1, pomnozi sve tezine sa 0.1
        
        weights sa dva neurona u fc:
            0.0862301, -0.28197122, 0.44707918,
            0.112038255, -0.38459483, -0.31042123,
            0.5205773, -0.04579687, -0.47586578, 
        
            -0.4501995, -0.4715696, -0.4138012,
            -0.44830847, -0.1773395, -0.08274263,
             0.47323978, 0.41012484, 0.19486493, 
        
            0.08227211, -0.54566085, -0.11338204, 
            -1.4930964E-4, -0.26475245, 0.52672565, 
            -0.38034722, 0.3306117, -0.26243588, 
        
            0.31195676, -0.04197538, -0.5319252,
            -0.06671178, -0.29084632, -0.31613958, 
            -0.025327086, 0.12511307, -0.3920598
        
         */

        instance.backward();
        Tensor actual = instance.getDeltas();

        // sum delta * weight and transpose
        // Test: 0.1 * 0.0862301 + 0.2 * 0.08227211 =  0.025077432   +
        // Test: 0.1 * -0.28197122 + 0.2 * -0.54566085 = −0.137329292 +
        // Test: 0.1 * -0.4501995 + 0.2 * 0.31195676 = 0.017371402  
        // Test: 0.1 * 0.19486493 + 0.2 * -0.3920598 = −0.05892546
        
        Tensor expected = new Tensor(3, 3, 2,
                new float[]{ 0.025077432f, 0.011173964f, -0.024011713f,
                            -0.1373293f, -0.091409974f, 0.06154266f,
                             0.022031512f, 0.07430301f, -0.100073755f,
                             
                             0.017371401f, -0.058173206f, 0.04225856f,
                             -0.055552036f, -0.075903215f, 0.0660351f, 
                             -0.14776516f, -0.07150218f, -0.05892546f });

        assertArrayEquals(actual.getValues(), expected.getValues(), 1e-8f);
    }    
    
    @Ignore
    public void testBackwardSingleChannelFromConvolutional() {
        RandomGenerator.getDefault().initSeed(123);
        InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6,
                new float[]{0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                    -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                    -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                    0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                    0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                    0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f});
     
        ConvolutionalLayer prevLayer = new ConvolutionalLayer(3, 3, 1);
        prevLayer.setPrevLayer(inputLayer);
        prevLayer.activationType = ActivationType.LINEAR;
        prevLayer.init();
        prevLayer.filters[0] = new Tensor(3, 3,
                                  new float[]{0.1f, 0.2f, 0.3f,
                                             -0.11f, -0.2f, -0.3f,
                                              0.4f, 0.5f, 0.21f});
        prevLayer.biases = new float[]{0.0f};

        inputLayer.setInput(input);
        prevLayer.forward();    // vidi koliki je output i njega onda pooluj

        MaxPoolingLayer instance = new MaxPoolingLayer(2, 2, 2);
        instance.setPrevLayer(prevLayer);
        instance.init();
        instance.forward();

        ConvolutionalLayer nextLayer = new ConvolutionalLayer(3, 3, 1);
        instance.setNextlayer(nextLayer);
        nextLayer.setPrevLayer(instance);
        nextLayer.activationType = ActivationType.LINEAR;
        nextLayer.init();
        nextLayer.setDeltas(new Tensor(0.1f, 0.2f));
        nextLayer.filters[0] = new Tensor(3, 3,
                                  new float[]{0.1f, 0.2f, 0.3f,
                                             -0.11f, -0.2f, -0.3f,
                                              0.4f, 0.5f, 0.21f});;
        nextLayer.biases = new float[]{0.0f};      
        
        nextLayer.setDeltas(new Tensor(3, 3)); // postavi delte iz sledeceg lejera
        
        // svaka celija iz narednog lejera treba da sa svojom deltom pomnozi sve tezine iz filtera i prepise ih u delte prethodnih lejera
        // caka je sto se pozicije filtera preklapaju i tako sabiraju - idealan trenutak da razjasnij matematiku - crtaj u nekoj svesci@@
        
        /* test samo sa jednim neuronom u fc i delta 0.1, pomnozi sve tezine sa 0.1
        
       zadaj delte - ima ih onoliko kolik oima celija/outputa u conv layery
       ispisi tezine kojih ima u 4d weights
       ispisi i izracunaj ocekivane rezultate 
        
         */

        instance.backward();
        
        Tensor actual = instance.getDeltas();

        // sum delta * weight and transpose
        // Test: 0.1 * 0.18075174 + 0.2 * 11627263 = 0.0413297 ... 
        Tensor expected = new Tensor(3, 3,
                new float[]{0.0413297f, 0.062126942f, 0.16724476f, 
                           -0.020589843f, -0.15361156f, 0.03141014f,
                            0.12785016f, -0.14694643f, -0.1626513f});

        assertArrayEquals(actual.getValues(), expected.getValues(), 1e-8f);
    }
        

}
