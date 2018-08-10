//---------------------------------------------------------------------------

#ifndef BackProb2004H
#define BackProb2004H

// The layer dimensions are set per use of this library in the following
// way:

#define MAX_INPUT_LAYER_SIZE   30
#define MAX_HIDDEN_LAYER_SIZE  60
#define MAX_OUTPUT_LAYER_SIZE  20

#define INPUT_TO_HIDDEN  0
#define HIDDEN_TO_OUTPUT 1

#define DEFAULT_EPSILON        1.0f
#define DEFAULT_LEARNING_RATE  0.5f

//---------------------------------------------------------------------------
// Backpropagation Neural Network
//---------------------------------------------------------------------------

// A simple form of a single hidden layer feed forward network

class feedForwardNetwork
{
  private:

   double InputLayer[MAX_INPUT_LAYER_SIZE+1];
   double HiddenLayer[MAX_HIDDEN_LAYER_SIZE+1];
   double OutputLayer[MAX_OUTPUT_LAYER_SIZE];

   double weightsToHidden[MAX_INPUT_LAYER_SIZE+1][MAX_HIDDEN_LAYER_SIZE];
   double weightsToOutput[MAX_HIDDEN_LAYER_SIZE+1][MAX_OUTPUT_LAYER_SIZE];

   int inNeurons;
   int hiddenNeurons;
   int outNeurons;

   double epsilon;        // accepted error
   double learningRate;

  public:

   feedForwardNetwork() { configure(1,1,1); };
   feedForwardNetwork(int in, int hidden, int out) { configure(in,hidden,out);};

   void configure(int in, int hidden, int out);
   void init();
   void setInput(int x, double value);
   void setOutput(int x, double value);
   void apply();

   double getWeight(int layer, int x, int y);
   double getOutput(int x);
   double getInput(int x);
   double getHidden(int x);
   void  setEpsilon(double eps);
   void  setLearningRate(double mu);
   void  setWeights(double w1[MAX_INPUT_LAYER_SIZE+1][MAX_HIDDEN_LAYER_SIZE], double w2[MAX_HIDDEN_LAYER_SIZE+1][MAX_OUTPUT_LAYER_SIZE]);
   void setWeight(int level, int i, int j, double w);
   void  getWeights(double w1[MAX_INPUT_LAYER_SIZE+1][MAX_HIDDEN_LAYER_SIZE], double w2[MAX_HIDDEN_LAYER_SIZE+1][MAX_OUTPUT_LAYER_SIZE]);
   double getEpsilon();
   double getLearningRate();
   void backpropagate(double t[MAX_OUTPUT_LAYER_SIZE]);


   double t(double x);  // transfer function
   double energy(double *t, double *y, int num);

};
//---------------------------------------------------------------------------
#endif
