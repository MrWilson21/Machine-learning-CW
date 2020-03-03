import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;
import java.util.Random;

public class WekaTools
{
    public static double accuracy(Classifier classifier, Instances test, Instances train)
    {
        try
        {
            classifier.buildClassifier(train);
        } catch (Exception e)
        {
            e.printStackTrace();
        }

        double correct = 0;
        for(int i = 0; i < test.numInstances(); i++)
        {
            Instance instance = test.get(i);
            try
            {
                if(classifier.classifyInstance(instance) == instance.value(instance.numAttributes() - 1))
                {
                    correct++;
                }
            } catch (Exception e)
            {
                e.printStackTrace();
            }
        }

        return correct / (double)test.numInstances();
    }

    public static Instances loadClassificationData(String path)
    {
        Instances data = null;
        try
        {
            FileReader reader = new FileReader(path);
            data = new Instances(reader);
        } catch(Exception e)
        {
            System.out.println("Exception caught: "+e + " in " + path);
        }

        data.setClassIndex(data.numAttributes() - 1);

        return data;
    }

    public static void printDistribution(Instances instances, Classifier classifier)
    {
        for (int i = 0; i < instances.numInstances(); i++)
        {
            try
            {
                System.out.print("Classification: " + classifier.classifyInstance(instances.instance(i)) + ", Distribution: ");
                for (Double d : classifier.distributionForInstance(instances.instance(i)))
                {
                    System.out.print(d + " ");
                }
                System.out.print("\n");
            } catch (Exception e)
            {
                e.printStackTrace();
            }
        }
    }

    public static Instances[] splitData(Instances instances, Double testProportion)
    {
        Instances[] split = new Instances[2];
        instances.randomize(new Random());

        int splitIndex = (int)Math.round(testProportion * (double)instances.numInstances());

        split[0] = new Instances(instances);            //Train
        split[1] = new Instances(instances, 0); //Test

        for(int i = 0; i < splitIndex; i++)
        {
            Instance instanceToMove = split[0].remove(0);
            split[1].add(instanceToMove);
        }

        return split;
    }

    public static double[] classDistribution(Instances data)
    {
        double[] classDistribution = new double[data.numClasses()];

        for(int i = 0; i < data.numInstances(); i++)
        {
            classDistribution[(int)data.get(i).value(data.numAttributes() - 1)]++;
        }

        return classDistribution;
    }
}
