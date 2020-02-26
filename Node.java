import java.util.*;

/**
 * Class for internal organization of a Neural Network.
 * There are 5 types of nodes. Check the type attribute of the node for details.
 * Feel free to modify the provided function signatures to fit your own implementation
 */

public class Node {
    private int type = 0;
    	//0 = input, 1 = biasToHidden, 2 = hidden, 3 = biasToOutput, 4 = Output
    public ArrayList<NodeWeightPair> parents = null;
    	//Array List that will contain the parents (including the bias node) with weights if applicable

    private double inputValue = 0.0;
    private double outputValue = 0.0;
    private double outputGradient = 0.0;
    private double delta = 0.0; //input gradient

    // The two fields below was intended to serve the softmax function,
    // but eventually not used:
    public ArrayList<Node> siblings = null; // all nodes in the same layer
    public int indexAmongSiblings = 0; // position of this node in its layer

    /**
     * Constructor: Create a node with a specific type
     * @param type
     */
    Node(int type) {
        if (type > 4 || type < 0) {
            System.out.println("Incorrect value for node type");
            System.exit(1);

        } else {
            this.type = type;
        }

        if (type == 2 || type == 4) {
            parents = new ArrayList<>();
            	// parents are filled in NNImpl constructor
        }
        if (type == 4)
        	siblings = new ArrayList<>();
    }

    /**
     * For an input node sets the input value which will be the value of a particular attribute
     * @param inputValue
     */
    public void setInput(double inputValue) {
        if (type == 0) {    //If input node
            this.inputValue = inputValue;
        }
    }

    /**
     * Calculate the output of a node.
     * You can get this value by using getOutput()
     * i.e. store the value into this.outputValue
     */
    public void calculateOutput() {
        if (type == 2 || type == 4) {   //Not an input or bias node
            // TODO: add code here

        	double sum = 0;
    		for (NodeWeightPair nw : parents) {
    			sum += (nw.node.getOutput() * nw.weight);
    		}

        	if (type == 2) { // hidden node, ReLU function:
        		this.outputValue = relu(sum);

        	} else if (type == 4) { // output node, Softmax function
        		//this.outputValue = softmax(this.indexAmongSiblings);
        		this.outputValue = Math.exp(sum);
        	}
        }
    }

    /**
     * Gets the output value.
     * Should always use this method to get the value,
     * especially for the input layer (instead of using this.inputValue),
     * otherwise bias nodes will be ignored.
     *
     * @return value of this node
     */
    public double getOutput() {

        if (type == 0) { //Input node
            return inputValue;
        } else if (type == 1 || type == 3) { //Bias node
            return 1.00;
        } else {
            return outputValue;
        }

    }

    /**
     * Calculate the delta value of a node.
     * @param teacher
     * @param outputNodesLinked
     */
    public void calculateDelta(int teacher, ArrayList<Node> outputNodesLinked) {
    	// TODO: add code here

        if (type == 2) {

        	double sum = 0.0;
    		for (NodeWeightPair nw : parents) {
    			sum += (nw.node.getOutput() * nw.weight);
    		}
    		double factor = 0.0;
    		for (Node outNode : outputNodesLinked) {

    			for (NodeWeightPair nw : outNode.parents) {
    				if (nw.node == this) {
    					factor += nw.weight * outNode.delta;
    					break;
    				}
    			}
    		}
    		this.delta = reluPrime(sum) * factor;
    		//System.out.println("hidden delta = " +this.delta);

        } else if (type == 4)  {
        	this.delta = softmaxPrime((double)teacher);
        	//System.out.println("output delta = " +this.delta);
        }
    }

    /**
     * Update the weights between parents node and current node
     * @param learningRate
     */
    public void updateWeight(double learningRate) {
        if (type == 2 || type == 4) {
            // TODO: add code here

        	for (NodeWeightPair nw : parents) {
        		nw.weight += learningRate *nw.node.getOutput() * this.delta;
        	}
        }
    }

    /**
     * This is a helper method that finalize the output
     * through Softmax function.
     * @param sum, sum of all output nodes' natural exponential
     */
    public void divide(double sum) {
    	if(type == 4)
    		this.outputValue /= sum;
    }

    /**
     * This is a helper method to calculate ReLU function.
     * @param input
     * @return
     */
    public double relu(double input) {
    	return Math.max(0, input);
    }

    /**
     * This is a helper method to calculate ReLU derivative.
     * @param input
     * @return
     */
    public double reluPrime(double input) {
    	if (input <= 0)
    		return 0.0;
    	else
    		return 1.0;
    }

    /**
     * This function to calculate the softmax function was not used.
     * Softmax was eventually implemented in NNImpl.java
     * @param index
     * @param siblingsWS the list of input (weighted sum) into every output unit.
     * 			If there are K output units, siblingsWS.length = K.
     * @return
     */
    public double softmax(int index) {

    	ArrayList<Double> siblingsWS = new ArrayList<Double>();
    	for (int i = 0; i < siblings.size(); i++) {
        	double sum = 0.0;
        	for (NodeWeightPair nw : siblings.get(i).parents) {
        		// Since the parent nodes are all in hidden layer,
        		// their outputs are already calculated
    			sum += (nw.node.getOutput() * nw.weight);
    		}
        	siblingsWS.add(sum);
        }

    	double activation;
		double numer = Math.exp(siblingsWS.get(index));
		double denom = 0.0;
		for (int i=0; i<siblingsWS.size(); i++) {
			denom += Math.exp(siblingsWS.get(i));
		}
		activation = numer / denom;
		return activation;
    }

    /**
     * This is a helper method to calculate Softmax derivative.
     * @param teacher
     * @return
     */
    public double softmaxPrime(double teacher) {
    	return teacher - this.outputValue;
    }
}


