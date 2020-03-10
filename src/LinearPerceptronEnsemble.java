import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Copy;

import java.util.Arrays;


public class LinearPerceptronEnsemble extends AbstractClassifier
{
    public double attributeProportion = 0.5;
    public int ensembleSize = 50;

    private EnhancedLinearPerceptron[] perceptrons;
    private boolean[][] attributesForEachClassifier;
    private int numAttributes;

    @Override
    public void buildClassifier(Instances instances) throws Exception
    {
        //Ensure only valid training data is used
        testCapabilities(instances);

        numAttributes = instances.numAttributes() - 1;
        attributesForEachClassifier = new boolean[ensembleSize][numAttributes];
        perceptrons = new EnhancedLinearPerceptron[ensembleSize];

        for(int i = 0; i < ensembleSize; i++)
        {
            perceptrons[i] = new EnhancedLinearPerceptron();

            Instances newInstances = new Instances(instances);
            int numAttributesToRemove = (int)Math.ceil(numAttributes * (1.0-attributeProportion));
            while(numAttributesToRemove > 0)
            {
                int attributeToRemove = (int)(Math.random()*numAttributes);
                if(!attributesForEachClassifier[i][attributeToRemove])
                {
                    attributesForEachClassifier[i][attributeToRemove] = true;
                    numAttributesToRemove--;
                }
            }

            for(int j = numAttributes - 1; j >= 0; j--)
            {
                if(attributesForEachClassifier[i][j])
                {
                    newInstances.deleteAttributeAt(j);
                }
            }

            perceptrons[i].buildClassifier(newInstances);
        }
    }

    @Override
    public double classifyInstance(Instance instance)
    {
        int highestIndex = 0;
        double highestProbability = 0;

        double[] probabilities = distributionForInstance(instance);

        for(int i = 0; i < instance.numClasses(); i++)
        {
            if(probabilities[i] > highestProbability)
            {
                highestProbability = probabilities[i];
                highestIndex = i;
            }
        }

        return highestIndex;
    }

    @Override
    public double[] distributionForInstance(Instance instance)
    {
        double[] probabilities = new double[instance.numClasses()];

        for(int i = 0; i < ensembleSize; i++)
        {
            Instance newInstance = new DenseInstance(instance);
            for(int j = numAttributes - 1; j >= 0; j--)
            {
                if(attributesForEachClassifier[i][j])
                {
                    newInstance.deleteAttributeAt(j);
                }
            }

            probabilities[(int)perceptrons[i].classifyInstance(instance)] += 1.0/ensembleSize;
        }

        return probabilities;
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
