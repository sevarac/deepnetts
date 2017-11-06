package deepnetts.net.layers;


import deepnetts.util.RandomGenerator;
import deepnetts.util.Tensor;
import deepnetts.util.WeightsInit;
import org.junit.Test;
import static org.junit.Assert.*;
import org.junit.Ignore;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class FullyConnectedLayerTest {
    
    public FullyConnectedLayerTest() {
    }

//
//    /**
//     * Test of init method, of class FullyConnectedLayer.
//     * TODO: test when prev layer is also fc.
//     * provide some predefined weights2 from some other framework to compare
//     */
//    @Test
//    public void testInitInNetwork() {
//        System.out.println("initInNetwork");
//        
//        WeightsInit.initSeed(123);        
//                     
//        InputLayer prevLayer = new InputLayer(5, 1);
//        prevLayer.init();
//                
//        FullyConnectedLayer instance = new FullyConnectedLayer(10); // also specify weights2 initialization algorithm
//        instance.setPrevLayer(prevLayer);
//        instance.init();
//        // xavier weights2 init 0.28229551858550417, 0.6209434458719868, -0.3120622803406485, 0.13762279226641172, 0.38689773252500836, 0.4748637856568586, 0.2732821523183022, -0.5414868945866499, 0.37474378113053974, 0.09956992497369166, 0.516242609907135, -0.4440918337026123, 0.6011109059633722, -0.5494823873092433, -0.5445215933090334, -0.517661951595734, -0.09554287126707417, 0.47357137428198326, 0.09499970865657659, -0.13092225478120523, -0.3057097143015954, -0.4391870474677929, -0.30303483158040495, -0.048468928779627385, -0.07703208384167015, -0.3650464603601738, 0.1444682027729537, -0.590134249466491, 0.2389367937459569, 0.23288631762525014, -0.5372596814735492, 0.5651694629516103, -0.07984545680771227, 0.24515941530631158, 0.22312843096417012, 0.16289979231408247, 0.09449407548703148, -0.1417058602053337, -0.39322508952613106, -0.5986551040498984, -0.5925390315567799, 0.41793673223025285, 0.30368757161482407, 0.5584233152051302, 0.4122845692026037, 0.01797144260330552, 0.1189976626855681, -0.3437085917638453, -0.2489739173863732, 0.24671434014784865
//        
//        Tensor actualResult = instance.getWeights();
//        Tensor expectedResult = new Tensor(5, 10, 1, 1);
//        expectedResult.setValues(0.28229551858550417, 0.6209434458719868, -0.3120622803406485, 0.13762279226641172, 0.38689773252500836, 0.4748637856568586, 0.2732821523183022, -0.5414868945866499, 0.37474378113053974, 0.09956992497369166, 0.516242609907135, -0.4440918337026123, 0.6011109059633722, -0.5494823873092433, -0.5445215933090334, -0.517661951595734, -0.09554287126707417, 0.47357137428198326, 0.09499970865657659, -0.13092225478120523, -0.3057097143015954, -0.4391870474677929, -0.30303483158040495, -0.048468928779627385, -0.07703208384167015, -0.3650464603601738, 0.1444682027729537, -0.590134249466491, 0.2389367937459569, 0.23288631762525014, -0.5372596814735492, 0.5651694629516103, -0.07984545680771227, 0.24515941530631158, 0.22312843096417012, 0.16289979231408247, 0.09449407548703148, -0.1417058602053337, -0.39322508952613106, -0.5986551040498984, -0.5925390315567799, 0.41793673223025285, 0.30368757161482407, 0.5584233152051302, 0.4122845692026037, 0.01797144260330552, 0.1189976626855681, -0.3437085917638453, -0.2489739173863732, 0.24671434014784865);
//        
//        assertEquals(expectedResult, actualResult);                
//    }
//
    
    @Ignore
    public void testForwardWithLinearWithMock() {        
        // initialize weights2 with specified random seed
    //    WeightsInit.initSeed(123);         // using predefined weights2 values        
        RandomGenerator.getDefault().initSeed(123); // init random generator with seed that will be used for weights2 (same effect as line above)
        
        // input vector for this layer
        Tensor input = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f);
        Tensor weights = new Tensor(5, 10); // niz tezina ovde        
        WeightsInit.uniform(weights.getValues(), 5); // "[0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501]"

        Tensor expectedOutputs = new Tensor( 0.04212712f, 0.3698768f, 0.10604945f, 0.24532129f, 0.17567812f, 0.34893453f, 0.16589892f, -0.34877524f, 0.09166324f, -0.01524709f);
        
        // create prev fc layer with 5 outputs
        FullyConnectedLayer prevLayer =  mock(FullyConnectedLayer.class); //new FullyConnectedLayer(5);                        
        when(prevLayer.getWidth()).thenReturn(5);
        when(prevLayer.getOutputs()).thenReturn(input);  //prevLayer.setOutputs(input);
                              
        // create instance of layer to test
        FullyConnectedLayer instance = new FullyConnectedLayer(10, ActivationType.LINEAR);
        instance.setPrevLayer(prevLayer);        
        instance.init(); // init weights2 structure
        instance.setWeights(weights); // set weights2 values
        instance.setBiases(new float[] {0.1f, 0.2f, 0.3f, 0.11f, 0.12f, 0.13f, 0.21f, 0.22f, 0.23f, 0.24f}); // set bias values
        
        // do the forward pass
        instance.forward();        
        // get layer outpputs
        Tensor actualOutputs = instance.getOutputs();
        
        assertArrayEquals(actualOutputs.getValues(), expectedOutputs.getValues(), 1e-7f);
    }
    
    
    
    /**
     * Test of forward pass, of class FullyConnectedLayer using Linear activation function.
     * Test if matrix multiplication and bias addition works correctly
 output = inputs * weights2 + bias (where * is matrix multiplication)
     */
    @Test
    public void testForwardWithLinearActivation() {        
        // initialize weights2 with specified random seed
        RandomGenerator.getDefault().initSeed(123); // init random generator with seed that will be used for weights2 (same effect as line above)
        
        // input vector for this layer
        Tensor input = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f);
        Tensor weights = new Tensor(5, 10); // niz tezina ovde        
        WeightsInit.uniform(weights.getValues(), 5); // "[0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501]"
       
        // create prev fc layer with 5 outputs
        FullyConnectedLayer prevLayer = new FullyConnectedLayer(5);        
        prevLayer.setOutputs(input);
                
        // create instance of layer to test
        FullyConnectedLayer instance = new FullyConnectedLayer(10, ActivationType.LINEAR);
        instance.setPrevLayer(prevLayer);        
        instance.init(); // init weights2 structure
        instance.setWeights(weights); // set weights2 values
        instance.setBiases(new float[] {0.1f, 0.2f, 0.3f, 0.11f, 0.12f, 0.13f, 0.21f, 0.22f, 0.23f, 0.24f}); // set bias values
        
        // do the forward pass
        instance.forward();        
        // get layer outpputs
        Tensor actualOutputs = instance.getOutputs();   // "[0.042127118, 0.36987683, 0.10604945, 0.24532129, 0.17567813, 0.34893453, 0.16589889, -0.3487752, 0.09166323, -0.015247092]"
        Tensor expectedOutputs = new Tensor( 0.04212712f, 0.3698768f, 0.10604945f, 0.24532129f, 0.17567812f, 0.34893453f, 0.16589892f, -0.34877524f, 0.09166324f, -0.01524709f);
                                        
        assertArrayEquals(actualOutputs.getValues(), expectedOutputs.getValues(), 1e-7f);
    }
    
    @Test
    public void testForwardWithSigmoidActivation() {
        RandomGenerator.getDefault().initSeed(123); // init default random generator with seed that will be used for weights2 (same effect as line above)
        // input vector for this layer
        Tensor input = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f);
        Tensor weights = new Tensor(5, 10);  
        WeightsInit.uniform(weights.getValues(), 5); // [0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501]
       
        // create prev fc layer with 5 outputs
        FullyConnectedLayer prevLayer = new FullyConnectedLayer(5);        
        prevLayer.setOutputs(input);
                
        // create instance of layer to test
        FullyConnectedLayer instance = new FullyConnectedLayer(10, ActivationType.SIGMOID);
        instance.setPrevLayer(prevLayer);        
        instance.init(); // init weights2 structure
        instance.setWeights(weights); // set weights2 values
        instance.setBiases(new float[] {0.1f, 0.2f, 0.3f, 0.11f, 0.12f, 0.13f, 0.21f, 0.22f, 0.23f, 0.24f}); // set bias values
        
        // run forward pass
        instance.forward();        
        // get layer outputs        
        Tensor actualOutputs = instance.getOutputs();
        Tensor expectedOutputs = new Tensor( 0.51053022f, 0.59142921f, 0.52648754f, 0.56102458f, 0.54380692f, 0.58635918f, 0.54137987f, 0.41367945f, 0.52289978f, 0.4961883f );

        assertArrayEquals(actualOutputs.getValues(), expectedOutputs.getValues(), 1e-7f);
    }
        
    
    /**
     * Test of backward method, of class FullyConnectedLayer using linear activation function.
     * Checks if deltas are calculated correctly.
     * 
     * Tests only when prev layer is also FullyConnectd
     * It should test also when prev layer is maxpooling or convolutional
     * 
     *  kad ide backwrd treba dda trasponuje matricu tezina i da je pomnozi sa deltama iz sledeceg lejera 
     * d1 =d2 * WT 
     * 
     */
    @Test
    public void testBackwardLinear() {
        RandomGenerator.getDefault().initSeed(123); // init random generator with seed that will be used for weights2 (same effect as line above)                
        Tensor inputs = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f); // input vector for this layer
        Tensor weights = new Tensor(5, 10);  
        WeightsInit.uniform(weights.getValues(), 5); // 0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501
        
        Tensor nextDeltas = new Tensor(10);
        nextDeltas.setValues(0.04212712f, 0.3698768f, 0.10604945f, 0.24532129f, 0.17567812f, 0.34893453f, 0.16589892f, -0.34877524f, 0.09166324f, -0.01524709f);
                
        FullyConnectedLayer prevLayer = new FullyConnectedLayer(5); // not used for anything just dummy to prevent npe in init      
        prevLayer.setOutputs(inputs);
        
        FullyConnectedLayer instance = new FullyConnectedLayer(5, ActivationType.LINEAR);
        instance.setPrevLayer(prevLayer);               
        instance.init();
               
        Tensor ones = Tensor.ones(10);
                
        FullyConnectedLayer nextLayer = new FullyConnectedLayer(10);
        nextLayer.setPrevLayer(instance);
        instance.setNextlayer(nextLayer);        
        nextLayer.init();
        instance.setOutputs(ones);        
        nextLayer.setWeights(weights);
        nextLayer.setDeltas(nextDeltas);
        
        instance.backward();
        Tensor result = instance.deltas;
        Tensor expResult = new Tensor(5); // navedi ovde koje
        expResult.setValues( 0.02364707f,  0.09271421f,  0.04896196f,  0.29104635f,  0.37890932f);

        assertArrayEquals(expResult.getValues(), result.getValues(), 1e-7f);
    }

    /**
     * Test of backward method, of class FullyConnectedLayer using linear activation function.
     * Check if deltas are calculated correctly.
     * 
     * Tests only when prev layer is also FullyConnectd
     * It should test also when prev layer is maxpooling or convolutional
     */
    @Test
    public void testBackwardSigmoid() {
        RandomGenerator.getDefault().initSeed(123); // init random generator with seed that will be used for weights2 (same effect as line above)                
        Tensor inputs = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f); // input vector for this layer

        Tensor weights2 = new Tensor(5, 10);  
        WeightsInit.uniform(weights2.getValues(), 5); // "[0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501]"
        
        Tensor weights1 = new Tensor(5, 5);  
        WeightsInit.uniform(weights1.getValues(), 5);    // "[-0.25812685, -0.020679474, 0.102154374, -0.32011545, -0.41728795, -0.05412379, 0.16895384, -0.3470215, 0.16467547, 0.31206572, -0.37989998, -0.30708057, 0.39963514, -0.08906731, -0.056459278, 0.39290035, 0.17335385, -0.07480636, 0.15777558, 0.29387093, 0.115187526, 0.14577365, 0.0668174, -0.4196531, -0.10020122]"    
        
        Tensor nextDeltas = new Tensor(10);
        nextDeltas.setValues(0.04212712f, 0.3698768f, 0.10604945f, 0.24532129f, 0.17567812f, 0.34893453f, 0.16589892f, -0.34877524f, 0.09166324f, -0.01524709f);
                
        FullyConnectedLayer prevLayer = new FullyConnectedLayer(5); // not used for anything just dummy to prevent npe in init      
        prevLayer.setOutputs(inputs);
        
        FullyConnectedLayer instance = new FullyConnectedLayer(5, ActivationType.SIGMOID);
        instance.setPrevLayer(prevLayer);               
        instance.init();    
        instance.setWeights(weights1);
        instance.setBiases(new float[] {0.1f, 0.2f, 0.3f, 0.11f, 0.12f}); // set bias values
                
        FullyConnectedLayer nextLayer = new FullyConnectedLayer(10);
        nextLayer.setPrevLayer(instance);
        instance.setNextlayer(nextLayer);        
        nextLayer.init();    
        nextLayer.setWeights(weights2);
        nextLayer.setDeltas(nextDeltas);
        
        instance.forward();
        
        instance.backward();
        Tensor result = instance.deltas;
        Tensor expResult = new Tensor(5); // "[0.005872122, 0.022724332, 0.011843424, 0.07269054, 0.093866885]"
        expResult.setValues( 0.00587212f,  0.02272433f,  0.01184342f,  0.07269055f,  0.09386688f);

        assertArrayEquals(expResult.getValues(), result.getValues(), 1e-8f);
    }    
    
    /**
     * Test of applyWeightChanges method, of class FullyConnectedLayer. in online modee
 todo: test batch mode, da li pamti prev weights2, da li resetuje deltaWeights i deltaBiases na nulu
     */
    @Test
    public void testApplyWeightChanges() {           
        RandomGenerator.getDefault().initSeed(123);
        
        FullyConnectedLayer prevLayer = new FullyConnectedLayer(5);
        FullyConnectedLayer instance = new FullyConnectedLayer(10);
        instance.setPrevLayer(prevLayer);
        instance.init();
        
        Tensor weights = new Tensor(5, 10);
        WeightsInit.initSeed(123);
            
        Tensor biases = new Tensor(10); //  ? zasto ovde isod opet randomize kad gore imam uniform
        WeightsInit.uniform(weights.getValues(), 5); //  0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501
        WeightsInit.randomize(biases.getValues()); // "[-0.2885946, -0.023120344, 0.114212096, -0.35789996, -0.46654212, -0.060512245, 0.18889612, -0.38798183, 0.18411279, 0.34890008]"
        instance.setWeights(weights);
        instance.setBiases(biases.getValues());        
        instance.deltaBiases = new float[10];
        instance.deltaWeights = new Tensor(5, 10);
        WeightsInit.randomize(instance.deltaWeights.getValues()); // "[-0.4247411, -0.3433265, 0.44680566, -0.09958029, -0.063123405, 0.43927592, 0.19381553, -0.083636045, 0.17639846, 0.32855767, 0.12878358, 0.1629799, 0.07470411, -0.46918643, -0.11202836, 0.2753712, -0.31087178, -0.08334535, -0.47327846, -0.3146028, -0.46844327, 0.112583816, 0.33040798, -0.10281438, 0.24008608, 0.32558167, 0.44147235, 0.32505035, 0.32593954, -0.17731398, 0.014207661, 0.28614485, 0.09407586, 0.123054445, -0.27172554, -0.14735949, -0.19683117, -0.33119786, 0.19504476, 0.23377019, -0.07173675, -0.06975782, 0.46547735, -0.06449604, 0.22543085, -0.25612664, -0.16484225, -0.21565866, 0.45828927, -0.13396758]"
        WeightsInit.randomize(instance.deltaBiases); // "[-0.014675736, -0.20824462, -0.2544266, -0.29433697, 0.19522274, -0.042135, -0.2805665, 0.44587213, -0.38881636, 0.2882418]"
                
        instance.applyWeightChanges();        
        Tensor expectedWeights = new Tensor(-0.22512805f, -0.57834274f, 0.8858789f, -0.27705812f, -0.28378475f, 0.50557935f, 0.29112953f, -0.29929897f, 0.44997644f, 0.4380083f, 0.46456295f, 0.20707384f, 0.26794374f, -0.7713099f, -0.49491742f, 0.4379894f, -0.045887947f, -0.29116237f, -0.4028719f, -0.5448313f, -0.10340464f, 0.20406264f, 0.016387641f, -0.35627222f, 0.6651356f, 0.28818867f, 0.05292958f, -0.04253599f, -0.059095383f, -0.5151812f, -0.35183465f, 0.14134777f, 0.026516795f, 0.5094531f, 0.063139975f, 0.011747062f, -0.12965626f, -0.77672803f, 0.1024687f, 0.23364827f, -0.2879062f, 0.36031187f, 0.15492517f, 0.20544726f, 0.011152849f, -0.0014150143f, -0.19911501f, -0.64997375f, 0.40381932f, -0.3714426f);
        Tensor expectedBiases = new Tensor(10);
        expectedBiases.setValues(-0.30327034f, -0.23136496f, -0.1402145f,  -0.65223693f, -0.27131938f, -0.10264724f,  -0.09167038f,  0.0578903f,  -0.20470357f,  0.63714188f); // 0.4507922, 0.35359812, 0.25770348, -0.39445835, 0.7433155, 0.7244545, -0.07723093, 0.26560563, -0.23044652, 0.2704906
                
        assertArrayEquals(weights.getValues(), expectedWeights.getValues(), 1e-7f);
        assertArrayEquals(biases.getValues(), expectedBiases.getValues(), 1e-7f);                        
    }
    

   
    
}
