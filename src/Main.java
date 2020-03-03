import weka.classifiers.lazy.IBk;
import weka.core.Instances;

public class Main
{

    public static void main(String[] args)
    {
        Instances train = WekaTools.loadClassificationData("test.arff");

         nnClassifier = new IBk();
        try
        {
            nnClassifier.buildClassifier(train);
        } catch (Exception e)
        {
            e.printStackTrace();
        }

        WekaTools.printDistribution(test, nnClassifier);
        System.out.println(WekaTools.accuracy(nnClassifier, train, train));
    }
}
