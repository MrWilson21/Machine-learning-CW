import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.misc.

public class LinearPerceptron implements AbstractClassifier
{
    private double[] weights;
    private int numAttributes;

    public double learningRate;


    @Override
    public void buildClassifier(Instances instances) throws Exception
    {
        numAttributes = instances.numAttributes();
        weights = new double[numAttributes];

        for(int i = 0; i < numAttributes; i++)
        {
            weights[i] = 1;
        }

        boolean failed = true;
        int index = 0;
        while(failed = true)
        {
            Instance currentInstance  = instances.get(index);
            double eval = 0;
            for(int i = 0; i < numAttributes; i++)
            {
                eval += weights[i] * currentInstance.value(i);
            }

            int classification = eval > 0? 1 : -1;

            if(classification != currentInstance.classValue())
            {
                for(int i = 0; i < numAttributes; i++)
                {
                    weights[i] += learningRate * (currentInstance.classValue() - eval) * currentInstance.value(i);
                }
            }
        }

        System.out.println(weights);
    }
}
