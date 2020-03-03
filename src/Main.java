import weka.classifiers.lazy.IBk;
import weka.core.Instances;

public class Main
{

    public static void main(String[] args)
    {
        Instances train = WekaTools.loadClassificationData("test.arff");

        LinearPerceptron lp = new LinearPerceptron();
        try
        {
            lp.buildClassifier(train);
        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }
}
