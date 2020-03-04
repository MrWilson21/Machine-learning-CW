import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;


public class EnhancedLinearPerceptron extends AbstractClassifier
{
    private double[] weights;
    private int numAttributes;

    public double learningRate = 1;
    public double maxIterations = 100;
    public double biasTerm = 0;

    public boolean normaliseAttributes = true;
    public boolean modelSelection = false;
    public int numFolds = 10;

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

        //Normalise attribute vectors
        if(normaliseAttributes)
        {
            for(int i = 0; i < instances.numInstances(); i++)
            {
                normaliseInstance(instances.instance(i));
            }
        }

        if(modelSelection)
        {
            chooseMethod(instances);
        }
        else
        {
            buildOnLine(instances);
        }
    }

    @Override
    public double classifyInstance(Instance instance)
    {
        //Normalise attribute vector
        if(normaliseAttributes)
        {
            normaliseInstance(instance);
        }

        //Assign evaluation to the value of bias term then add each weighted component
        double eval = biasTerm;
        for(int i = 0; i < numAttributes; i++)
        {
            eval += weights[i] * instance.value(i);
        }

        return eval > 0? 1 : 0;
    }

    //Choose to build with either on-line or off-line methods depending on which is more accurate
    private void chooseMethod(Instances instances)
    {
        double onLineAccuracies = 0;
        double offLineAccuracies = 0;

        for(int i = 0; i < numFolds; i++)
        {
            Instances train = instances.trainCV(10, 1);
            Instances test = instances.testCV(10, 1);

            onLineAccuracies += accuracy(test, train, false);
            offLineAccuracies += accuracy(test, train, true);
        }

        //Build with which ever method has the best mean accuracy
        if(onLineAccuracies > offLineAccuracies)
        {
            System.out.println("Building with on-line method");
            buildOnLine(instances);
        }
        else
        {
            System.out.println("Building with off-line method");
            buildOffLine(instances);
        }
    }

    //Test accuracy of either off-line or on-line methods with given test and training data
    private double accuracy(Instances test, Instances train, boolean offline)
    {
        if(offline)
        {
            buildOffLine(train);
        }
        else
        {
            buildOnLine(train);
        }

        double correct = 0;
        for(int i = 0; i < test.numInstances(); i++)
        {
            Instance instance = test.get(i);
            if(classifyInstance(instance) == instance.value(instance.numAttributes() - 1))
            {
                correct++;
            }
        }

        return correct / (double)test.numInstances();
    }

    private void buildOnLine(Instances instances)
    {
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
            //Assign evaluation to the value of bias term then add each weighted component
            double eval = biasTerm;
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
    }

    private void buildOffLine(Instances instances)
    {

        double[] deltaWeights = new double[numAttributes];

        //Initialise delta weights
        for(int i = 0; i < numAttributes; i++)
        {
            weights[i] = 0;
        }

        int correctCount = 0;   //Number of instances correctly classified
        int iterations = 0;     //Number of iterations completed

        //While all instances have not been classified without any failures or reached max iterations
        while(correctCount < instances.numInstances() && iterations < maxIterations)
        {
            correctCount = 0;

            for(int index = 0; index < instances.numInstances(); index++)
            {
                iterations++;
                correctCount++;

                //Evaluate instance
                Instance currentInstance  = instances.get(index);
                //Assign evaluation to the value of bias term then add each weighted component
                double eval = biasTerm;
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
                    //Increment delta weights
                    for(int i = 0; i < numAttributes; i++)
                    {
                        deltaWeights[i] = 0.5 * learningRate * (trueClassification - classification) * currentInstance.value(i);
                    }
                }
            }
            //Increment real weights using delta weights
            for(int index = 0; index < numAttributes; index++)
            {
                weights[index] += deltaWeights[index];
            }
        }
    }

    //Normalise instance attributes to 0 mean and unit standard deviation
    private void normaliseInstance(Instance instance)
    {
        //Get total value of attributes and create array of attribute values
        double total = 0;
        double[] values = new double[numAttributes];
        for(int i = 0; i < numAttributes; i++)
        {
            total += instance.value(i);
            values[i] = instance.value(i);
        }

        //Calculate mean from total
        double mean = total / numAttributes;
        //Get standard deviation from attribute values
        double sd = calculateSD(values);
        //Scale each attribute by subtracting mean and dividing by standard deviation
        for(int i = 0; i < numAttributes; i++)
        {
            double scaledAttribute = (instance.value(i) - mean) / sd;
            instance.setValue(i, scaledAttribute);
        }
    }

    //Calculate standard deviation of an array of doubles
    private double calculateSD(double[] numArray)
    {
        double sum = 0.0, standardDeviation = 0.0;
        int length = numArray.length;

        //Calculate mean
        for(double num : numArray) {
            sum += num;
        }
        double mean = sum/length;

        //Sum squared distances between each value and mean
        for(double num: numArray) {
            standardDeviation += Math.pow(num - mean, 2);
        }

        //Return square root of mean of squared distances
        return Math.sqrt(standardDeviation/length);
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

