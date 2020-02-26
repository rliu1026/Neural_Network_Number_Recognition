import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.*;

/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 */

public class NNImpl {
    private ArrayList<Node> inputNodes; //list of the output layer nodes.
    private ArrayList<Node> hiddenNodes;    //list of the hidden layer nodes
    private ArrayList<Node> outputNodes;    // list of the output layer nodes

    private ArrayList<Instance> trainingSet;    //the training set

    private double learningRate;    // variable to store the learning rate
    private int maxEpoch;   // variable to store the maximum number of epochs
    private Random random;  // random number generator to shuffle the training set

    /**
     * This constructor creates the nodes necessary for the neural network
     * Also connects the nodes of different layers
     * After calling the constructor the last node of both inputNodes and
     * hiddenNodes will be bias nodes.
     */
    NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Random random, Double[][] hiddenWeights, Double[][] outputWeights) {
        this.trainingSet = trainingSet;
        this.learningRate = learningRate;
        this.maxEpoch = maxEpoch;
        this.random = random;

        //input layer nodes
        inputNodes = new ArrayList<>();
        int inputNodeCount = trainingSet.get(0).attributes.size();
        int outputNodeCount = trainingSet.get(0).classValues.size(); // 3 in our case
        for (int i = 0; i < inputNodeCount; i++) {
            Node node = new Node(0);
            inputNodes.add(node);
        }

        //bias node from input layer to hidden
        Node biasToHidden = new Node(1);
        inputNodes.add(biasToHidden);

        //hidden layer nodes
        hiddenNodes = new ArrayList<>();
        for (int i = 0; i < hiddenNodeCount; i++) {
            Node node = new Node(2);
            //Connecting hidden layer nodes with input layer nodes
            for (int j = 0; j < inputNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j), hiddenWeights[i][j]);
                node.parents.add(nwp);
            }
            hiddenNodes.add(node);
        }

        //bias node from hidden layer to output
        Node biasToOutput = new Node(3);
        hiddenNodes.add(biasToOutput);

        //Output node layer
        outputNodes = new ArrayList<>();
        for (int i = 0; i < outputNodeCount; i++) {
            Node node = new Node(4);
            //Connecting output layer nodes with hidden layer nodes
            for (int j = 0; j < hiddenNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
                node.parents.add(nwp);
            }
            outputNodes.add(node);
        }
    }

    /**
     * This method runs a forward check and calculate outputs on one instance.
     * No classification or learning (weight updates) are done here.
     *
     * @param example
     */
    private void forward(Instance example) {

    	// Load inputs into each input nodes:
    	for (int i=0; i<inputNodes.size()-1; i++) { // Not including bias node
			Node inNode = inputNodes.get(i);
			inNode.setInput(example.attributes.get(i));
		}

    	// Calculate outputs for each node in the hidden layer:
    	for (Node hNode : hiddenNodes) {
			hNode.calculateOutput();
		}

    	// Calculate outputs for each output node:
		double sum = 0;
			// sum of all output nodes' natural exponential
		for (Node outNode : outputNodes) {
			outNode.calculateOutput(); // calculate numerator in softmax
			sum += outNode.getOutput(); // calculate denominator in softmax
		}
		for (Node outNode : outputNodes) {
			outNode.divide(sum); // calculate softmax, the final output
		}
    }

    /**
     * Get the prediction from the neural network for a single instance
     * Return the idx with highest output values. For example if the outputs
     * of the outputNodes are [0.1, 0.5, 0.2], it should return 1.
     * The parameter is a single instance
     */
    public int predict(Instance instance) {
        // TODO: add code here

    	forward(instance); // Run forward check on this instance

    	// Collect all outputs by output nodes:
    	ArrayList<Double> possibilities= new ArrayList<Double>();
    		// list of outputs
    	for (Node outNode : outputNodes) {
    		possibilities.add(outNode.getOutput());
		}

    	// Find the largest output:
    	double max = possibilities.get(0);
    	int prediction = 0;
    	for (int i=0; i<possibilities.size(); i++) {
    		if (max < possibilities.get(i)) {
    			max = possibilities.get(i);
    			prediction = i; // 1, 2, or 3
    		}
    	}
        return prediction;
    }


    /**
     * Train the neural networks with the given parameters
     * <p> training set, learning rate, epoch limit
     * The parameters are stored as attributes of this class
     */
    public void train() {
        // TODO: add code here
    	//System.out.println("Start training: \n");
    	//boolean firstInstance = true;

    	for (int epoch = 0; epoch<this.maxEpoch; epoch++ ) {

    		// Shuffle orders of instances in the trainingSet
    		// to prevent over-fitting
    		Collections.shuffle(trainingSet, random);

    		for (Instance example : trainingSet) {

    			forward(example); // Run forward check on this instance
/*
    			for (Node outNode : outputNodes) {
    				System.out.println("o:" + outNode.getOutput());
    			}
*/
    			// Calculate delta values for weight update:
    			for (int i=0; i<outputNodes.size(); i++) {
    				outputNodes.get(i).calculateDelta(example.classValues.get(i), new ArrayList<Node>());
    			}
    			for (Node hNode : hiddenNodes) {
    				hNode.calculateDelta(0, outputNodes);
    			}

    			// Update each weights among layers:
    			for (Node outNode : outputNodes) {
    				outNode.updateWeight(learningRate);
    			}
    			for (Node hNode : hiddenNodes ) {
    				hNode.updateWeight(learningRate);
    			}

    			//setSiblings();

    			/*
    			if (firstInstance) {
    				visualizeWeights();
    				firstInstance = false;
    			}
    			*/
    		}
    		//printWeights(epoch);

    		// Calculate loss entropy:
    		double sumLoss = 0.0;
    		for(Instance example: trainingSet)
    			sumLoss += loss(example);
    		double meanLoss = sumLoss / trainingSet.size();

    		System.out.printf("Epoch: %s, Loss: %.3e\n", epoch, meanLoss);
    	}
    }

    /**
     * Calculate the cross entropy loss from the neural network for
     * a single instance.
     * The parameter is a single instance
     */
    private double loss(Instance example) {
        // TODO: add code here

    	forward(example);

    	double loss = 0.0;
    	for (int k=0; k<outputNodes.size(); k++) {
    		loss += example.classValues.get(k)
    				* Math.log(outputNodes.get(k).getOutput());
    	}
    	return -loss;
    }


    /**
     * This method prints out all weights among layers.
     * For the use of debugging.
     */
    private void visualizeWeights() {
    	System.out.println("\nWeights visualization: ");
    	System.out.println("\nWeights between input and hidden layer: ");
    	for (Node hNode : hiddenNodes) {
    		if (hNode.parents != null) {
    			for (NodeWeightPair nw : hNode.parents) {
        			System.out.println(nw.weight);
        		}
    		}
    	}
    	System.out.println("\nWeights between hidden and output layer: ");
    	for (Node outNode : outputNodes) {
    		for (NodeWeightPair nw : outNode.parents) {
    			System.out.println(nw.weight);
    		}
    	}
    	System.out.println("\nWeights visualization ends\n");
    }

    /**
     * This is another method that prints out all weights among layers.
     * For the use of debugging.
     */
    public void printWeights(int epoch) {
    	PrintWriter out = null;
    	try {
    		out = new PrintWriter(new FileOutputStream("result_weights.txt", true));
    		out.print("EPOCH: " + epoch + "\n");
    		out.print("**************************\n");
    		out.print("Updated Weights between the Hidden and Input Layers after Epoch " + epoch + "\n");
    		out.print("-------------------------------------------------------------------------------------------\n");
    		for (Node hidden : hiddenNodes) {
    			if (hidden.parents != null) {
    				for (NodeWeightPair nwp : hidden.parents) {
    					out.print(nwp.weight + "\n");
    				}
    			}
    		}
    		out.print("\nUpdated Weights between the Output and Hidden Layers after Epoch " + epoch	+ "\n");
    		out.print("-------------------------------------------------------------------------------------------\n");
    		for (Node output : outputNodes) {
    			for (NodeWeightPair nwp : output.parents) {
    				out.print(nwp.weight + "\n");
    			}
    		}
    		out.println();
    	} catch (FileNotFoundException e) {
    		System.out.println("no file found for output");
    	}
    	if (out != null)
    		out.close();
    }

    /**
     * This method was intended to store all nodes of the layer in each node,
     * eventually not used.
     */
    private void setSiblings() {
    	for (int i=0; i<hiddenNodes.size(); i++) {
        	hiddenNodes.get(i).siblings = hiddenNodes;
        	hiddenNodes.get(i).indexAmongSiblings = i;
        }
        for (int i=0; i<outputNodes.size(); i++) {
        	outputNodes.get(i).siblings = outputNodes;
        	outputNodes.get(i).indexAmongSiblings = i;
        }
    }


}
