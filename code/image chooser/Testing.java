package FinalYearProject;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.swing.*;
import java.util.Random;
import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;

public class Testing
{
    private static final int INPUT_NEURONS = 784;
    private static final int HIDDEN_NEURONS1 = 400;
    private static final int HIDDEN_NEURONS2 = 300;
    private static final int HIDDEN_NEURONS3 = 200;
    private static final int OUTPUT_NEURONS = 10;
    static int epoch = 0;
    private static String matrix1[];


    // Input to Hidden Weights (with Biases).
    private static double wih1[][] = new double[INPUT_NEURONS + 1][HIDDEN_NEURONS1];
    private static double whh2[][] = new double[HIDDEN_NEURONS1 + 1][HIDDEN_NEURONS2];
    private static double whh3[][] = new double[HIDDEN_NEURONS2 + 1][HIDDEN_NEURONS3];

    // Hidden to Output Weights (with Biases).
    private static double who[][] = new double[HIDDEN_NEURONS3 + 1][OUTPUT_NEURONS];

    // Activations.
    private static double inputs[] = new double[INPUT_NEURONS];
    private static double hidden1[] = new double[HIDDEN_NEURONS1];
    private static double hidden2[] = new double[HIDDEN_NEURONS2];
    private static double hidden3[] = new double[HIDDEN_NEURONS3];
    // private static double target[] = new double[OUTPUT_NEURONS];
    private static double actual[] = new double[OUTPUT_NEURONS];





    /*private static double trainInputs1[][] = new double[][]{{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.11372549,0.509803922,0.509803922,0.882352941,1,1,0.42745098,0.02745098,0.454901961,0.952941176,0.784313725,0.023529412,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.031372549,0.37254902,0.850980392,0.992156863,0.992156863,0.992156863,0.992156863,0.992156863,0.992156863,0.992156863,0.992156863,0.992156863,0.97254902,0.043137255,0,0,0,0,0,0,0,0,0,0,0,0,0,0.105882353,0.835294118,0.992156863,0.992156863,0.941176471,0.560784314,0.435294118,0.596078431,0.992156863,0.992156863,0.992156863,0.992156863,0.760784314,0.243137255,0,0,0,0,0,0,0,0,0,0,0,0,0,0.105882353,0.835294118,0.992156863,0.843137255,0.411764706,0.121568627,0,0.02745098,0.6,0.992156863,0.992156863,0.992156863,0.956862745,0.28627451,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.584313725,0.992156863,0.901960784,0.133333333,0,0,0,0.270588235,0.992156863,0.992156863,0.992156863,0.992156863,0.309803922,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.584313725,0.992156863,0.866666667,0.050980392,0,0,0.109803922,0.611764706,0.992156863,0.992156863,0.992156863,0.690196078,0.043137255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.247058824,0.925490196,0.992156863,0.68627451,0.054901961,0,0.729411765,0.992156863,0.992156863,0.992156863,0.768627451,0.050980392,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.278431373,0.992156863,0.992156863,0.682352941,0.196078431,0.780392157,0.992156863,0.992156863,0.929411765,0.290196078,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.035294118,0.360784314,0.917647059,0.992156863,0.992156863,0.992156863,0.992156863,0.929411765,0.290196078,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.301960784,0.992156863,0.992156863,0.992156863,0.925490196,0.294117647,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.043137255,0.631372549,0.992156863,0.992156863,0.992156863,0.592156863,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.039215686,0.396078431,0.992156863,0.992156863,0.929411765,0.91372549,0.619607843,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.51372549,0.992156863,0.992156863,0.678431373,0.149019608,0.729411765,0.725490196,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.490196078,0.929411765,0.992156863,0.447058824,0.054901961,0,0.729411765,0.725490196,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.243137255,0.933333333,0.992156863,0.690196078,0.050980392,0,0.082352941,0.82745098,0.725490196,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.043137255,0.941176471,0.992156863,0.88627451,0.043137255,0,0,0.647058824,0.992156863,0.533333333,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.494117647,0.992156863,0.952941176,0.270588235,0,0,0.380392157,0.964705882,0.890196078,0.141176471,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.533333333,0.992156863,0.698039216,0.439215686,0.439215686,0.760784314,0.97254902,0.992156863,0.298039216,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.533333333,0.992156863,0.992156863,0.992156863,0.992156863,0.992156863,0.992156863,0.556862745,0.011764706,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.058823529,0.992156863,0.992156863,0.992156863,0.921568627,0.505882353,0.176470588,0.007843137,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    }};*/

    private static double trainInputs1[][] = new double[783][0];
    private static String filechose;

    private static void NeuralNetwork()
    {


        System.out.println("\nTest network against original input:");
        testNetworkTraining();



        return;
    }






    private static void testNetworkTraining()
    {

        // This function simply tests the training vectors against network.
        for(int i = 0; i < 1; i++)
        {
            for(int j = 0; j < INPUT_NEURONS; j++)
            {
                //inputs[j] = trainInputs1[i][j];
                inputs[j] =  Double.parseDouble(matrix1[j])/255;
            } // j

            feedForward();

            for(int j = 0; j < INPUT_NEURONS; j++)
            {
                //  System.out.print(inputs[j] + "\t");
            } // j

            System.out.print("Output: " + maximum(actual) + "\n");

        } // i

        return;
    }


    private static int maximum(final double[] vector)
    {
        // This function returns the index of the maximum of vector().
        int sel = 0;
        double max = vector[sel];

        for(int index = 0; index < OUTPUT_NEURONS; index++)
        {
            if(vector[index] > max){
                max = vector[index];
                sel = index;
            }
        }
        return sel;
    }

    private static void feedForward()
    {
        double sum = 0.0;

        // Calculate input to hidden layer1.
        for(int hid = 0; hid < HIDDEN_NEURONS1; hid++)
        {
            sum = 0.0;
            for(int inp = 0; inp < INPUT_NEURONS; inp++)
            {
                sum += inputs[inp] * wih1[inp][hid];
            } // inp

            sum += wih1[INPUT_NEURONS][hid]; // Add in bias.
            hidden1[hid] = sigmoid(sum);
        } // hid


        // Calculate input to hidden layer2.
        for(int hid = 0; hid < HIDDEN_NEURONS2; hid++)
        {
            sum = 0.0;
            for(int inp = 0; inp < HIDDEN_NEURONS1; inp++)
            {
                sum += hidden1[inp] * whh2[inp][hid];
            } // inp

            sum += whh2[HIDDEN_NEURONS1][hid]; // Add in bias.
            hidden2[hid] = sigmoid(sum);
        } // hid

        // Calculate input to hidden layer3.
        for(int hid = 0; hid < HIDDEN_NEURONS3; hid++)
        {
            sum = 0.0;
            for(int inp = 0; inp < HIDDEN_NEURONS2; inp++)
            {
                sum += hidden2[inp] * whh3[inp][hid];
            } // inp

            sum += whh3[HIDDEN_NEURONS2][hid]; // Add in bias.
            hidden3[hid] = sigmoid(sum);
        } // hid

        // Calculate the hidden to output layer.
        for(int out = 0; out < OUTPUT_NEURONS; out++)
        {
            sum = 0.0;
            for(int hid = 0; hid < HIDDEN_NEURONS3; hid++)
            {
                sum += hidden3[hid] * who[hid][out];
            } // hid

            sum += who[HIDDEN_NEURONS3][out]; // Add in bias.
            actual[out] = sigmoid(sum);
        } // out
        return;
    }




    private static double sigmoid(final double val)
    {
        return (1.0 / (1.0 + Math.exp(-val)));
    }



    public void read () throws FileNotFoundException{

        String fileName = "vishid-0-10.txt";
        Scanner inputStream = null;
        //System.out.println("The file " + fileName + "\ncontains the following lines:\n");
        try
        {
            inputStream = new Scanner(new File("D:\\weights\\wih1.txt"));//The txt file is being read correctly.
        }
        catch(FileNotFoundException e)
        {
            System.out.println("Error opening the file " + fileName);
            System.exit(0);
        }





        for(int hid = 0; hid < HIDDEN_NEURONS1; hid++)
        {
            for(int inp = 0; inp <= INPUT_NEURONS; inp++)
            {

                String line = inputStream.nextLine();
                wih1[inp][hid] = Double.parseDouble(line);

                if(inp == 784){
                    wih1[INPUT_NEURONS][hid] = wih1[inp][hid];
                }
            } // inp
            //wih1[INPUT_NEURONS][hid] = Double.parseDouble(line);

        }


        inputStream.close();


    }

    public void read1 () throws FileNotFoundException{

        String fileName = "vishid-1-10.txt";
        Scanner inputStream = null;
        //System.out.println("The file " + fileName + "\ncontains the following lines:\n");
        try
        {
            inputStream = new Scanner(new File("D:\\weights\\whh2.txt"));//The txt file is being read correctly.
        }
        catch(FileNotFoundException e)
        {
            System.out.println("Error opening the file " + fileName);
            System.exit(0);
        }




        for(int hid = 0; hid < HIDDEN_NEURONS2; hid++)
        {
            for(int inp = 0; inp <= HIDDEN_NEURONS1; inp++)
            {
                String line = inputStream.nextLine();
                whh2[inp][hid] = Double.parseDouble(line);

                if(inp == 400){
                    whh2[HIDDEN_NEURONS1][hid] = whh2[inp][hid];
                }


            } // inp


        } // hid
        inputStream.close();


    }
    public void read2 () throws FileNotFoundException{

        String fileName = "vishid-2-10.txt";
        Scanner inputStream = null;
        //System.out.println("The file " + fileName + "\ncontains the following lines:\n");
        try
        {
            inputStream = new Scanner(new File("D:\\weights\\whh3.txt"));//The txt file is being read correctly.
        }
        catch(FileNotFoundException e)
        {
            System.out.println("Error opening the file " + fileName);
            System.exit(0);
        }




        for(int hid = 0; hid < HIDDEN_NEURONS3; hid++)
        {
            for(int inp = 0; inp <= HIDDEN_NEURONS2; inp++)
            {
                String line = inputStream.nextLine();
                whh3[inp][hid] = Double.parseDouble(line);

                if(inp == 300){
                    whh3[HIDDEN_NEURONS2][hid] = whh3[inp][hid];
                }

            } // inp


        } // hid
        inputStream.close();


    }

    public void read3 () throws FileNotFoundException{

        String fileName = "vishid-3-10.txt";
        Scanner inputStream = null;
        //System.out.println("The file " + fileName + "\ncontains the following lines:\n");
        try
        {
            inputStream = new Scanner(new File("D:\\weights\\who.txt"));//The txt file is being read correctly.
        }
        catch(FileNotFoundException e)
        {
            System.out.println("Error opening the file " + fileName);
            System.exit(0);
        }



        for(int out = 0; out < OUTPUT_NEURONS; out++)
        {
            for(int hid = 0; hid <= HIDDEN_NEURONS3; hid++)
            {

                String line = inputStream.nextLine();
                who[hid][out] = Double.parseDouble(line);

                if(hid == 200){
                    who[HIDDEN_NEURONS3][out] = who[hid][out];
                }
            } // hid

        } // out
        inputStream.close();


    }

    private static String fileChose(){
        JFileChooser fc = new JFileChooser();
        int ret = fc.showOpenDialog(null);
        if (ret == JFileChooser.APPROVE_OPTION)
        {
            File file = fc.getSelectedFile();
            String filename = file.getAbsolutePath();
            return filename;
        }
        else {
            return null;
        }
    }

    private static void ImageChooser() throws IOException {
        int height = 28;
        int width = 28;
        int channels = 1;
        float a=1;
        String matrix;

         filechose = fileChose().toString();

        // FileChose is a string we will need a file
        File file = new File(filechose);

        // Use NativeImageLoader to convert to numerical matrix
        NativeImageLoader loader = new NativeImageLoader(height, width, channels);

        // Get the image into an INDarray
        INDArray image = loader.asMatrix(file);
        matrix = image.toString();
        matrix = matrix.replaceAll("\\[", "").replaceAll("\\]","");
        matrix1 = matrix.split(",");
       // for(int i=0; i<784; i++) {

         //   trainInputs1
            //System.out.println("input " + i + matrix1[i]);
           // matrix1[i];
       // }
    }



    public static void main(String[] args) throws IOException
    {
        Testing r0 = new Testing();
        Testing r1 = new Testing();
        Testing r2 = new Testing();
        Testing r3 = new Testing();

        r0.read();
        r1.read1();
        r2.read2();
        r3.read3();

        ImageChooser();
        NeuralNetwork();
        //JOptionPane.showMessageDialog(null, "The number is  : " + maximum(actual));
        final ImageIcon icon = new ImageIcon(filechose);
        JOptionPane.showMessageDialog(null, "The number is  : " + maximum(actual), "About", JOptionPane.INFORMATION_MESSAGE, icon);

        return;
    }

}
