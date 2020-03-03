import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;

public class LinearPerceptron extends AbstractClassifier
{
    private double[] weights;
    private int numAttributes;

    public double learningRate = 1;
    public double maxIterations = 100;


    @Override
    public void buildClassifier(Instances instances) throws Exception
    {
        //Ensure only valid training data is used
        testCapabilities(instances);

        numAttributes = instances.numAttributes() - 1;
        weights = new double[numAttributes];

        //Initialise weights
        for(int i = 0; i < numAttributes; i++)
        {
            weights[i] = 1;
        }

        int correctCount = 0;   //Number of instances correctly classified
        int index = 0;          //Current instance index
        int iterations = 0;     //Number of iterations completed

        //While all instances have not been classified without any failures or reached max iterations
        while(correctCount < instances.numInstances() && iterations < maxIterations)
        {
            correctCount++;
            iterations++;

            //Evaluate instance
            Instance currentInstance  = instances.get(index);
            double eval = 0;
            for(int i = 0; i < numAttributes; i++)
            {
                eval += weights[i] * currentInstance.value(i);
            }

            //Classify instance using evaluation
            int classification = eval > 0? 1 : -1;
            //Replace class index value with -1 or 1
            int trueClassification = currentInstance.classValue() == 0? -1 : 1;

            //If not correctly classified
            if(classification != trueClassification)
            {
                //Reset correct counter
                correctCount = 0;
                //Increment weights
                for(int i = 0; i < numAttributes; i++)
                {
                    weights[i] += 0.5 * learningRate * (trueClassification - classification) * currentInstance.value(i);
                }
            }

            index = (index + 1 ) % instances.numInstances();
        }

        System.out.println(iterations);
        System.out.println(Arrays.toString(weights));
    }

    @Override
    public double classifyInstance(Instance instance)
    {
        return 1.0;
    }


    public void testCapabilities(Instances instances) throws Exception {
        Capabilities cap = getCapabilities();
        //Enable only numeric and binary classes
        cap.disableAll();
        cap.enable(Capabilities.Capability.NUMERIC_CLASS);
        cap.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        cap.enable(Capabilities.Capability.BINARY_CLASS);

        //Test instances
        cap.testWithFail(instances);
    }
}
