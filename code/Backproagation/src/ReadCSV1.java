


import java.io.*;
import java.util.Arrays;
import java.util.Scanner;

public class ReadCSV1 {

	public ReadCSV1(){
		
	}
	
	//private static int trainInputs[][] = new int[10][784];
	
	public static void main(String[] args) throws FileNotFoundException {
		ReadCSV r =new ReadCSV();
		
		
		
	}
	public void read_X(double[][]train_X) throws FileNotFoundException
	{
		//int[][] train_X = new int[1007][881];    
		//int[][] train_X = new int[1007][775];   

        String delimiter = ",";
        int c=0;
        int row=0;
      Scanner sc = new Scanner(new File("D:\\sorted200.csv"));
     //   Scanner sc = new Scanner(new File("/home/om/Music/data/drug_protein.csv"));
        while (sc.hasNextLine())
        {
            String line = sc.nextLine();
            String[]testStr = line.split(delimiter);
            for (int x=0;x<testStr.length;x++){
                
            	train_X[row][x]=Double.parseDouble(testStr[x]);
            	
            }       
            row++;
        }
        //System.out.println();
        
        }
       
	}
	


