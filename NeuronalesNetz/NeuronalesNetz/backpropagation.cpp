//---------------------------------------------------------------------------
// Backpropagation Neural Network
//---------------------------------------------------------------------------

// A simple form of a single hidden layer feed forward network

// This implementation serves as an easy example of basic backpropagation
// techniques.

// It shall not be used as a module in a students' project !

//---------------------------------------------------------------------------
#include "stdafx.h"
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <math.h>
#include <time.h>
#include "BackPropagation.h"

#define RANDOMIZER

#ifdef RANDOMIZER

/* this part is needed if your C-Environment does not provide
   the two functions randomize() and random(int x)             */
   
void randomize()
{
  srand(unsigned(time(NULL)));
}

int random(int x)
{
  return (rand()%(x+1));   
}

#endif

void feedForwardNetwork::configure(int in, int hidden, int out)
{
  if ((in > 0) && (in < MAX_INPUT_LAYER_SIZE))
    inNeurons = in;
  else
    inNeurons = 1;
  if ((hidden > 0) && (hidden < MAX_HIDDEN_LAYER_SIZE))
    hiddenNeurons = hidden;
  else
    hiddenNeurons = 1;
  if ((out > 0) && (out < MAX_OUTPUT_LAYER_SIZE))
    outNeurons = out;
  else
    outNeurons = 1;

  epsilon = DEFAULT_EPSILON;
  learningRate = DEFAULT_LEARNING_RATE;

  // to be added: error handling !

  randomize();
}

void feedForwardNetwork::init()
{
  // initialize weights

  int i,j;

  // all neuron activations set to 0

  for (i=0;i<MAX_INPUT_LAYER_SIZE+1; i++)
    InputLayer[i] = 0.0f;

  InputLayer[inNeurons] = 1.0f;  // threshold activation (common trick)

  for (i=0;i<MAX_HIDDEN_LAYER_SIZE+1; i++)
    HiddenLayer[i] = 0.0f;

  HiddenLayer[hiddenNeurons] = 1.0f;  // threshold activation (common trick)

  for (i=0;i<MAX_OUTPUT_LAYER_SIZE; i++)
    OutputLayer[i] = 0.0f;


  // all weights are set to 0

  for (i=0;i<MAX_INPUT_LAYER_SIZE+1;i++)
    for (j=0;j<MAX_HIDDEN_LAYER_SIZE;j++)
      weightsToHidden[i][j] = 0.0f;

  for (i=0;i<MAX_HIDDEN_LAYER_SIZE+1;i++)
    for (j=0;j<MAX_OUTPUT_LAYER_SIZE;j++)
      weightsToOutput[i][j] = 0.0f;

  // the weights of the configured net (node subset)
  // are set to a random number between -0.5 and 0.5

  for (i=0;i<MAX_INPUT_LAYER_SIZE+1;i++)
    for (j=0;j<MAX_HIDDEN_LAYER_SIZE;j++)
    {
      weightsToHidden[i][j] = (double)((random(100)-50)/100.0f);
    }

  for (i=0;i<MAX_HIDDEN_LAYER_SIZE+1;i++)
    for (j=0;j<MAX_OUTPUT_LAYER_SIZE;j++)
      weightsToOutput[i][j] = (double)((random(100)-50)/100.0f);

}

double feedForwardNetwork::t(double x)
{
  return (double)(1.0f/(1.0f + exp((double) -x)));
}

void feedForwardNetwork::setInput(int x, double value)
{
  if ((x >= 0) && (x < inNeurons)&& (value >= 0.0f) && (value <= 1.0f))
    InputLayer[x] = value;

  // add error handling !
}

void feedForwardNetwork::setOutput(int x, double value)
{
  if ((x >= 0) && (x < outNeurons) && (value >= 0.0f) && (value <= 1.0f))
    OutputLayer[x] = value;

  // add error handling !
}

double feedForwardNetwork::getInput(int x)
{
  double ret = -1.0f;

  if ((x >= 0) && (x < inNeurons))
    ret = InputLayer[x];

  return ret;

  // add error handling !
}

double feedForwardNetwork::getOutput(int x)
{
  double ret = -1.0f;

  if ((x >= 0) && (x < outNeurons))
    ret = OutputLayer[x];

  return ret;

  // add error handling !
}

double feedForwardNetwork::getHidden(int x)
{
  double ret = -1.0f;

  if ((x >= 0) && (x < hiddenNeurons))
    ret = HiddenLayer[x];

  return ret;

  // add error handling !
}

double feedForwardNetwork::getWeight(int layer, int x, int y)
{
  double ret = -1.0f;

  if (layer == INPUT_TO_HIDDEN) // from input to hidden
  {
    if ((x >= 0) && (x < inNeurons+1) &&   //includes threshold
        (y >= 0) && (y < hiddenNeurons))
    {
       ret = weightsToHidden[x][y];
    }
  }


  if (layer == HIDDEN_TO_OUTPUT) // from hidden layer to output
  {
    if ((x >= 0) && (x < hiddenNeurons+1) &&   //includes threshold
        (y >= 0) && (y < outNeurons))
    {
       ret = weightsToOutput[x][y];
    }
  }

  return ret;

  // add error handling !
}

void feedForwardNetwork::apply()
{
  int i,j;
  double net;

  // add input check !

  // propagate activation through the net.

  // compute hidden layer activation

  InputLayer[inNeurons]= 1.0f;  // for threshold computation

  for (j=0; j<hiddenNeurons; j++)
  {
    net = 0.0f; // netto input of a neuron

    for (i=0;i<inNeurons+1;i++)
    {
      net += weightsToHidden[i][j]*InputLayer[i];
    }

    HiddenLayer[j] = t(net);  // using transfer function (sigmoid)
  }

  for (j=0; j<outNeurons; j++)
  {
    net = 0.0f; // netto input of a neuron

    for (i=0;i<hiddenNeurons+1;i++)
    {
      net += weightsToOutput[i][j]*HiddenLayer[i];
    }

    OutputLayer[j] = t(net);  // using transfer function (sigmoid)
  }


}

void feedForwardNetwork::backpropagate(double t[MAX_OUTPUT_LAYER_SIZE])
{

  double deltaH[MAX_HIDDEN_LAYER_SIZE];
  int i,j;
  double e,y;
  double delta;

  // neural network learning step

  e = energy(t,OutputLayer,outNeurons);

  if (epsilon < e)
  {
   // backpropagation

   // update weights to output layer
   // Formula :  delta_wij = lernrate dj hiddenlayer_i
   //                   dj = (tj-yj)yj(1-yj)

   for (i=0; i< hiddenNeurons+1; i++)
     deltaH[i] = 0.0f;

   for (j=0; j< outNeurons; j++)
   {
     y = OutputLayer[j];
     delta = (t[j]-y)*y*(1.0f-y);

     for (i=0; i< hiddenNeurons+1; i++)
     {
       deltaH[i] += delta * weightsToOutput[i][j];
       weightsToOutput[i][j] += learningRate * delta * HiddenLayer[i];
     }
   }

   for (i=0;i<hiddenNeurons;i++)
   {
     delta = deltaH[i]*HiddenLayer[i]*(1.0f-HiddenLayer[i]);

     for (j=0;j<inNeurons+1;j++)
     {
       weightsToHidden[j][i] += learningRate * delta * InputLayer[j];
     }
   }
  }
}

double feedForwardNetwork::getEpsilon()
{
  return epsilon;
}

double feedForwardNetwork::getLearningRate()
{
  return learningRate;
}

void feedForwardNetwork::setEpsilon(double eps)
{
  if (eps > 0.0f)
  {
    epsilon = eps;
  }
}

void feedForwardNetwork::setLearningRate(double mu)
{
  if ((mu > 0.0f) && (mu <= 10.0f))
  {
    learningRate = mu;
  }
}

double feedForwardNetwork::energy(double *t, double *y, int num)
{
   // no range checks !!

   double energy = 0.0f;
   int   i;

   for (i=0; i< num; i++)
   {
      energy += (t[i]-y[i])*(t[i]-y[i]);
   }

   energy /= 2;

   return energy;

}

void  feedForwardNetwork::setWeights(double w1[MAX_INPUT_LAYER_SIZE+1][MAX_HIDDEN_LAYER_SIZE], double w2[MAX_HIDDEN_LAYER_SIZE+1][MAX_OUTPUT_LAYER_SIZE])
{

   int i,j;

   for (i=0;i<inNeurons+1;i++)
     for (j=0;j<hiddenNeurons;j++)
       weightsToHidden[i][j] = w1[i][j];

   for (i=0;i<hiddenNeurons+1;i++)
     for (j=0;j<outNeurons;j++)
       weightsToOutput[i][j] = w2[i][j];

}

void  feedForwardNetwork::getWeights(double w1[MAX_INPUT_LAYER_SIZE+1][MAX_HIDDEN_LAYER_SIZE], double w2[MAX_HIDDEN_LAYER_SIZE+1][MAX_OUTPUT_LAYER_SIZE])
{

   int i,j;

   for (i=0;i<inNeurons+1;i++)
     for (j=0;j<hiddenNeurons;j++)
       w1[i][j] = weightsToHidden[i][j] ;

   for (i=0;i<hiddenNeurons+1;i++)
     for (j=0;j<outNeurons;j++)
       w2[i][j] = weightsToOutput[i][j];

}

void feedForwardNetwork::setWeight(int level,  int i, int j, double w)
{

  /* check correct weight position */

  if ((level > 1) || (level < 0))
  {
        /* error - handling ! */
  }
  else
  {
    if (level == 0)
    {
      if ((i >= 0) && (i < inNeurons+1) &&
          (j >= 0) && (j < hiddenNeurons))
      {
        weightsToHidden[i][j] = w;
      }
    }
    if (level == 1)
    {
      if ((i >= 0) && (i < hiddenNeurons+1) &&
          (j >= 0) && (j < outNeurons))
      {
        weightsToOutput[i][j] = w;
      }
    }
  }
}

