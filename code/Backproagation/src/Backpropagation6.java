





import java.util.Random;
import java.util.Scanner;



import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;

public class Backpropagation6
{
    private static final int INPUT_NEURONS = 784;
    private static final int HIDDEN_NEURONS1 = 400;
    private static final int HIDDEN_NEURONS2 = 300;
    private static final int HIDDEN_NEURONS3 = 200;
    private static final int OUTPUT_NEURONS = 10;
    static int epoch = 0;

    private static final double LEARN_RATE = 0.1;    // Rho.
    private static final int TRAINING_REPS = 50000;

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
    private static double target[] = new double[OUTPUT_NEURONS];
    private static double actual[] = new double[OUTPUT_NEURONS];

    // Unit errors.
    private static double erro[] = new double[OUTPUT_NEURONS];
    private static double errh3[] = new double[HIDDEN_NEURONS3];
    private static double errh2[] = new double[HIDDEN_NEURONS2];
    private static double errh1[] = new double[HIDDEN_NEURONS1];

    //private static final int MAX_SAMPLES = 300;
    
    //private static int trainInputs[][] = new int[300][784];

                                                          
                                                          
private static final int MAX_SAMPLES = 2000;
    
    private static double trainInputs[][] = new double[2000][784];

                                                          
                                                          
    private static double trainInputs1[][] = new double[][]{{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1608,0.6392,0.7961,0.7961,0.7961,0,0,0,0,0,0,0,0.3216,0.4039,0.4,0.3216,0,0,0,0,0,0,0,0,0,0,0,0.6784,0.9922,0.8784,0.6353,0.3216,0,0,0,0,0,0,0,0.0824,0.4,0.5569,0.7176,0.9176,0.2,0,0,0,0,0,0,0,0,0,0.3216,0.9922,0.5098,0.0784,0,0,0,0,0,0,0,0,0,0,0,0,0,0.6745,0.9098,0,0,0,0,0,0,0,0,0.2,0.9922,0.7961,0.0784,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0824,0.8392,0.7529,0,0,0,0,0,0,0,0,0.5176,0.9882,0.0784,0,0,0,0,0,0,0,0,0,0,0,0,0,0.2431,0.7961,0.9137,0.1961,0,0,0,0,0,0,0,0,0.8392,0.9922,0,0,0,0,0,0,0,0,0,0,0,0,0.2,0.5176,1,0.9137,0.4824,0,0,0,0,0,0,0,0,0,0.8353,0.9882,0.1608,0,0,0,0,0,0,0,0,0,0.4,0.7176,0.9922,0.9882,0.7529,0.1961,0,0,0,0,0,0,0,0,0,0,0.3608,0.9922,0.9961,0.6745,0.3608,0.3608,0.2,0.2,0.2824,0.5961,0.5961,0.9137,0.9961,0.9922,0.7176,0.2392,0,0,0,0,0,0,0,0,0,0,0,0,0.0392,0.5137,0.9922,0.9882,0.9922,0.9882,0.9922,0.9882,0.9922,0.9882,0.9922,0.6706,0.5137,0.1961,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0824,0.4,0.4,0.4,0.4824,0.6353,0.4,0.4,0.3216,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
}};

        private static double trainOutput[][] = new double[][] 
                                            {{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
                                             {0, 1, 0, 0, 0, 0, 0, 0, 0, 0}, 
                                             {0, 0, 1, 0, 0, 0, 0, 0, 0, 0}, 
                                             {0, 0, 0, 1, 0, 0, 0, 0, 0, 0}, 
                                             {0, 0, 0, 0, 1, 0, 0, 0, 0, 0}, 
                                             {0, 0, 0, 0, 0, 1, 0, 0, 0, 0}, 
                                             {0, 0, 0, 0, 0, 0, 1, 0, 0, 0}, 
                                             {0, 0, 0, 0, 0, 0, 0, 1, 0, 0}, 
                                             {0, 0, 0, 0, 0, 0, 0, 0, 1, 0}, 
                                             {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};

    private static void NeuralNetwork()
    {
        int sample = 0;
        int sample1 = 0;

        assignRandomWeights();

        // Train the network.
        for(epoch = 0; epoch < TRAINING_REPS; epoch++)
        {
            
            if(sample == MAX_SAMPLES){
                sample = 0;
            }
            
            if(sample<200){
            	sample1 = 0;
            }else if(sample>199 && sample<400){
            	sample1 = 1;
            }else if(sample>399 && sample<600){
            	sample1 = 2;
            }else if(sample>599 && sample<800){
            	sample1 = 3;
            }else if(sample>799 && sample<1000){
            	sample1 = 4;
            }else if(sample>999 && sample<1200){
            	sample1 = 5;
            }else if(sample>1199 && sample<1400){
            	sample1 = 6;
            }else if(sample>1399 && sample<1600){
            	sample1 = 7;
            }else if(sample>1599 && sample<1800){
            	sample1 = 8;
            }else{
            	sample1 = 9;
            }

            for(int i = 0; i < INPUT_NEURONS; i++)
            {
                inputs[i] = trainInputs[sample][i];
            } // i

            for(int i = 0; i < OUTPUT_NEURONS; i++)
            {
                target[i] = trainOutput[sample1][i];
            } // i

            feedForward();

            backPropagate();
            
            sample += 1;

        } // epoch

       // getTrainingStats();
		
        System.out.println("\nTest network against original input:");
        testNetworkTraining();
		
       
		
        return;
    }

  
   

   

    private static void testNetworkTraining()
    {
    	int count = 1;
        // This function simply tests the training vectors against network.
        for(int i = 0; i < MAX_SAMPLES; i++)
        {
            for(int j = 0; j < INPUT_NEURONS; j++)
            {
                inputs[j] = trainInputs[i][j];
            } // j
            
            feedForward();
            
            for(int j = 0; j < INPUT_NEURONS; j++)
            {
              //  System.out.print(inputs[j] + "\t");
            } // j
            
            System.out.print("Output: " + count + " : " + maximum(actual) + "\n");
            count++;
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

    private static void backPropagate()
    {
        // Calculate the output layer error (step 3 for output cell).
        for(int out = 0; out < OUTPUT_NEURONS; out++)
        {
            erro[out] = (target[out] - actual[out]) * sigmoidDerivative(actual[out]);
        }

        // Calculate the hidden layer3 error (step 3 for hidden cell).
        for(int hid = 0; hid < HIDDEN_NEURONS3; hid++)
        {
            errh3[hid] = 0.0;
            for(int out = 0; out < OUTPUT_NEURONS; out++)
            {
                errh3[hid] += erro[out] * who[hid][out];
            }
            errh3[hid] *= sigmoidDerivative(hidden3[hid]);
        }
        
     // Calculate the hidden layer2 error (step 3 for hidden cell).
        for(int hid = 0; hid < HIDDEN_NEURONS2; hid++)
        {
            errh2[hid] = 0.0;
            for(int out = 0; out < HIDDEN_NEURONS3; out++)
            {
                errh2[hid] += errh3[out] * whh3[hid][out];
            }
            errh2[hid] *= sigmoidDerivative(hidden2[hid]);
        }
        
     // Calculate the hidden layer1 error (step 3 for hidden cell).
        for(int hid = 0; hid < HIDDEN_NEURONS1; hid++)
        {
            errh1[hid] = 0.0;
            for(int out = 0; out < HIDDEN_NEURONS2; out++)
            {
                errh1[hid] += errh2[out] * whh2[hid][out];
            }
            errh1[hid] *= sigmoidDerivative(hidden1[hid]);
        }
        
        
        

        // Update the weights for the output layer (step 4).
        for(int out = 0; out < OUTPUT_NEURONS; out++)
        {
            for(int hid = 0; hid < HIDDEN_NEURONS3; hid++)
            {
                who[hid][out] += (LEARN_RATE * erro[out] * hidden3[hid]);
                
                /*if(epoch == (TRAINING_REPS-1)){}*/
            } // hid
            who[HIDDEN_NEURONS3][out] += (LEARN_RATE * erro[out]); // Update the bias.
        } // out

        // Update the weights for the hidden layer3 (step 4).
        for(int hid = 0; hid < HIDDEN_NEURONS3; hid++)
        {
            for(int inp = 0; inp < HIDDEN_NEURONS2; inp++)
            {
                whh3[inp][hid] += (LEARN_RATE * errh3[hid] * hidden2[inp]);
            } // inp
            whh3[HIDDEN_NEURONS2][hid] += (LEARN_RATE * errh3[hid]); // Update the bias.
        } // hid
        
     // Update the weights for the hidden layer2 (step 4).
        for(int hid = 0; hid < HIDDEN_NEURONS2; hid++)
        {
            for(int inp = 0; inp < HIDDEN_NEURONS1; inp++)
            {
                whh2[inp][hid] += (LEARN_RATE * errh2[hid] * hidden1[inp]);
            } // inp
            whh2[HIDDEN_NEURONS1][hid] += (LEARN_RATE * errh2[hid]); // Update the bias.
        } // hid
        
     // Update the weights for the hidden layer1 (step 4).
        for(int hid = 0; hid < HIDDEN_NEURONS1; hid++)
        {
            for(int inp = 0; inp < INPUT_NEURONS; inp++)
            {
                wih1[inp][hid] += (LEARN_RATE * errh1[hid] * inputs[inp]);
            } // inp
            wih1[INPUT_NEURONS][hid] += (LEARN_RATE * errh1[hid]); // Update the bias.
        } // hid
        return;
    }
    
    private static void assignRandomWeights()
    {
        for(int inp = 0; inp <= INPUT_NEURONS; inp++) // Do not subtract 1 here.
        {
            for(int hid = 0; hid < HIDDEN_NEURONS1; hid++)
            {
                // Assign a random weight value between -0.5 and 0.5
                wih1[inp][hid] = new Random().nextDouble() - 0.5;
            } // hid
        } // inp
        
        for(int inp = 0; inp <= HIDDEN_NEURONS1; inp++) // Do not subtract 1 here.
        {
            for(int hid = 0; hid < HIDDEN_NEURONS2; hid++)
            {
                // Assign a random weight value between -0.5 and 0.5
                whh2[inp][hid] = new Random().nextDouble() - 0.5;
            } // hid
        } // inp
        
        for(int inp = 0; inp <= HIDDEN_NEURONS2; inp++) // Do not subtract 1 here.
        {
            for(int hid = 0; hid < HIDDEN_NEURONS3; hid++)
            {
                // Assign a random weight value between -0.5 and 0.5
                whh3[inp][hid] = new Random().nextDouble() - 0.5;
            } // hid
        } // inp

        for(int hid = 0; hid <= HIDDEN_NEURONS3; hid++) // Do not subtract 1 here.
        {
            for(int out = 0; out < OUTPUT_NEURONS; out++)
            {
                // Assign a random weight value between -0.5 and 0.5
                who[hid][out] = new Random().nextDouble() - 0.5;
            } // out
        } // hid
        return;
    }

   

    private static double sigmoid(final double val)
    {
        return (1.0 / (1.0 + Math.exp(-val)));
    }

    private static double sigmoidDerivative(final double val)
    {
        return (val * (1.0 - val));
    }
    
    public void read () throws FileNotFoundException{
		
		String fileName = "vishid-0-10.txt";
		Scanner inputStream = null;
		//System.out.println("The file " + fileName + "\ncontains the following lines:\n");
		try
		{
		  inputStream = new Scanner(new File("D:\\5layer\\vishid-0-10.txt"));//The txt file is being read correctly.
		}
		catch(FileNotFoundException e)
		{
		  System.out.println("Error opening the file " + fileName);
		  System.exit(0);
		}

		
		
		for(int inp = 0; inp <= INPUT_NEURONS; inp++) // Do not subtract 1 here.
        {
            for(int hid = 0; hid < HIDDEN_NEURONS1; hid++)
            {
               if(inputStream.hasNextLine()){
            	   
                String line = inputStream.nextLine(); 
                wih1[inp][hid] = Double.parseDouble(line);}
              // System.out.println(wih[inp][hid]);
            } // out
        } // hid
		
		inputStream.close();
	
	
}
    
	public void read1 () throws FileNotFoundException{
		
		String fileName = "vishid-1-10.txt";
		Scanner inputStream = null;
		//System.out.println("The file " + fileName + "\ncontains the following lines:\n");
		try
		{
		  inputStream = new Scanner(new File("D:\\5layer\\vishid-1-10.txt"));//The txt file is being read correctly.
		}
		catch(FileNotFoundException e)
		{
		  System.out.println("Error opening the file " + fileName);
		  System.exit(0);
		}

		
		for(int hid = 0; hid <= HIDDEN_NEURONS1; hid++) // Do not subtract 1 here.
        {
            for(int out = 0; out < HIDDEN_NEURONS2; out++)
            {
            	if(inputStream.hasNextLine()){
                String line = inputStream.nextLine();
                whh2[hid][out] = Double.parseDouble(line);}
             //  System.out.println(wih[hid][out]);
            } // out
        } // hid
		
		inputStream.close();
	
	
}
	public void read2 () throws FileNotFoundException{
		
		String fileName = "vishid-2-10.txt";
		Scanner inputStream = null;
		//System.out.println("The file " + fileName + "\ncontains the following lines:\n");
		try
		{
		  inputStream = new Scanner(new File("D:\\5layer\\vishid-2-10.txt"));//The txt file is being read correctly.
		}
		catch(FileNotFoundException e)
		{
		  System.out.println("Error opening the file " + fileName);
		  System.exit(0);
		}

		
		for(int hid = 0; hid <= HIDDEN_NEURONS2; hid++) // Do not subtract 1 here.
        {
            for(int out = 0; out < HIDDEN_NEURONS3; out++)
            {
            	if(inputStream.hasNextLine()){
                String line = inputStream.nextLine();
                whh3[hid][out] = Double.parseDouble(line);}
             //  System.out.println(wih[hid][out]);
            } // out
        } // hid
		
		inputStream.close();
	
	
}

	public void read3 () throws FileNotFoundException{
	
	String fileName = "vishid-3-10.txt";
	Scanner inputStream = null;
	//System.out.println("The file " + fileName + "\ncontains the following lines:\n");
	try
	{
	  inputStream = new Scanner(new File("D:\\5layer\\vishid-3-10.txt"));//The txt file is being read correctly.
	}
	catch(FileNotFoundException e)
	{
	  System.out.println("Error opening the file " + fileName);
	  System.exit(0);
	}

	
	for(int hid = 0; hid <= HIDDEN_NEURONS3; hid++) // Do not subtract 1 here.
    {
        for(int out = 0; out < OUTPUT_NEURONS; out++)
        {
        	
            String line = inputStream.nextLine();
            who[hid][out] = Double.parseDouble(line);
        /*System.out.println(who[hid][out]);*/
        } // out
    } // hid
	
	inputStream.close();


}

    
    public static void main(String[] args) throws IOException
    {
    	/*Backpropagation6 r0 = new Backpropagation6();
    	Backpropagation6 r1 = new Backpropagation6();
    	Backpropagation6 r2 = new Backpropagation6();
    	Backpropagation6 r3 = new Backpropagation6();
    	
    	r0.read();
    	r1.read1();
    	r2.read2();
    	r3.read3();*/
    	
    	
    	
    	double [][] tInputs=new double[2000][784];
		ReadCSV1 r = new ReadCSV1();
		r.read_X(tInputs);
		
	
			
		      for(int i=0;i<2000;i++)
		     {
			     for(int j=0;j<784;j++)
			     {
				      //  trainInputs[i][j] = (int) tInputs[i][j];
				      //  System.out.print(tInputs[i][j]);
			    	 trainInputs[i][j] = tInputs[i][j];
			    	// System.out.println(trainInputs[i][j]);
			     }
			     
			    
		     }
    	
    	
    	NeuralNetwork();
    	
    	PrintWriter pw1 = new PrintWriter(new FileWriter("D:\\5layer\\weights\\who.txt"));
    	PrintWriter pw2 = new PrintWriter(new FileWriter("D:\\5layer\\weights\\whh3.txt"));
    	PrintWriter pw3 = new PrintWriter(new FileWriter("D:\\5layer\\weights\\whh2.txt"));
    	PrintWriter pw4 = new PrintWriter(new FileWriter("D:\\5layer\\weights\\wih1.txt"));
		 
    	for(int out = 0; out < OUTPUT_NEURONS; out++)
        {
            for(int hid = 0; hid < HIDDEN_NEURONS3; hid++)
            {
               pw1.println(who[hid][out]); 
            } // hid
            pw1.println(who[HIDDEN_NEURONS3][out]);
        } // out
		pw1.close();
		
		for(int hid = 0; hid < HIDDEN_NEURONS3; hid++)
        {
            for(int inp = 0; inp < HIDDEN_NEURONS2; inp++)
            {
               
                pw2.println(whh3[inp][hid]); 
            } // inp
           
            pw2.println(whh3[HIDDEN_NEURONS2][hid]);
        } // hid
        pw2.close();
        
        
     
        for(int hid = 0; hid < HIDDEN_NEURONS2; hid++)
        {
            for(int inp = 0; inp < HIDDEN_NEURONS1; inp++)
            {
               
            	pw3.println(whh2[inp][hid]);
            } // inp
           
            pw3.println(whh2[HIDDEN_NEURONS1][hid]);
        } // hid
        pw3.close();
        
     // Update the weights for the hidden layer1 (step 4).
        for(int hid = 0; hid < HIDDEN_NEURONS1; hid++)
        {
            for(int inp = 0; inp < INPUT_NEURONS; inp++)
            {
               
            	pw4.println(wih1[inp][hid]);
            } // inp
            
            pw4.println(wih1[INPUT_NEURONS][hid]);
        } // hid
		pw4.close();
        return;
    }

}
