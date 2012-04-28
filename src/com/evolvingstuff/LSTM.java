package com.evolvingstuff;

import java.util.*;

public class LSTM implements IAgent, IAgentSupervised {
	private double init_weight_range = 0.1;
	private int full_input_dimension;
	private int full_hidden_dimension;
	private int output_dimension;
	private int cell_blocks;
	private Neuron neuronNetInput;
	private Neuron neuronInputGate;
	private Neuron neuronForgetGate;
	private Neuron neuronOutputGate;
	private Neuron neuronCECSquash;
	
	private double [] CEC;
	private double [] context;
	
	private double [] peepInputGate;
	private double [] peepForgetGate;
	private double [] peepOutputGate;
	private double [][] weightsNetInput;
	private double [][] weightsInputGate;
	private double [][] weightsForgetGate;
	private double [][] weightsOutputGate;
	private double [][] weightsGlobalOutput;
	
	private double [][] dSdwWeightsNetInput;
	private double [][] dSdwWeightsInputGate;
	private double [][] dSdwWeightsForgetGate;
	
	private double biasInputGate = 2;
	private double biasForgetGate = -2;
	private double biasOutputGate = 2;
	
	private double learningRate = 0.07;
	
	public double[] GetHiddenState() {
		return context.clone();
	}
	
	public void SetHiddenState(double[] new_state) {
		context = new_state.clone();
	}
	
	public LSTM(Random r, int input_dimension, int output_dimension, int cell_blocks) {
		this.output_dimension = output_dimension;
		this.cell_blocks = cell_blocks;
		
		CEC = new double[cell_blocks];
		context = new double[cell_blocks];
		
		full_input_dimension = input_dimension + cell_blocks + 1; //+1 for bias
		full_hidden_dimension = cell_blocks + 1; //+1 for bias
		
		neuronNetInput = Neuron.Factory(NeuronType.Tanh);
		neuronInputGate = Neuron.Factory(NeuronType.Sigmoid);
		neuronForgetGate = Neuron.Factory(NeuronType.Sigmoid);
		neuronOutputGate = Neuron.Factory(NeuronType.Sigmoid);
		neuronCECSquash= Neuron.Factory(NeuronType.Identity);
		
		weightsNetInput = new double[cell_blocks][full_input_dimension];
		weightsInputGate = new double[cell_blocks][full_input_dimension];
		weightsForgetGate = new double[cell_blocks][full_input_dimension];
		
		dSdwWeightsNetInput = new double[cell_blocks][full_input_dimension];
		dSdwWeightsInputGate = new double[cell_blocks][full_input_dimension];
		dSdwWeightsForgetGate = new double[cell_blocks][full_input_dimension];
		
		weightsOutputGate = new double[cell_blocks][full_input_dimension];
		
		for (int i = 0; i < full_input_dimension; i++) {
			for (int j = 0; j < cell_blocks; j++) {
				weightsNetInput[j][i] = (r.nextDouble() * 2 - 1) * init_weight_range;
				weightsInputGate[j][i] = (r.nextDouble() * 2 - 1) * init_weight_range;
				weightsForgetGate[j][i] = (r.nextDouble() * 2 - 1) * init_weight_range;
				weightsOutputGate[j][i] = (r.nextDouble() * 2 - 1) * init_weight_range;
			}
		}
		
		for (int j = 0; j < cell_blocks; j++) {
			weightsInputGate[j][full_input_dimension-1] += biasInputGate;
			weightsForgetGate[j][full_input_dimension-1] += biasForgetGate;
			weightsOutputGate[j][full_input_dimension-1] += biasOutputGate;
		}
		
		peepInputGate = new double[cell_blocks];
		peepForgetGate = new double[cell_blocks];
		peepOutputGate = new double[cell_blocks];
		
		for (int j = 0; j < cell_blocks; j++) {
			peepInputGate[j] = (r.nextDouble() * 2 - 1) * init_weight_range;
			peepForgetGate[j] = (r.nextDouble() * 2 - 1) * init_weight_range;
			peepOutputGate[j] = (r.nextDouble() * 2 - 1) * init_weight_range;
		}
		
		weightsGlobalOutput = new double[output_dimension][full_hidden_dimension];
		
		for (int j = 0; j < full_hidden_dimension; j++) {
			for (int k = 0; k < output_dimension; k++)
				weightsGlobalOutput[k][j] = (r.nextDouble() * 2 - 1) * init_weight_range;
		}
		
	}
	
	public void Reset() {
		//TODO: reset deltas here?
		for (int c = 0; c < CEC.length; c++)
			CEC[c] = 0.0;
		for (int c = 0; c < context.length; c++)
			context[c] = 0.0;
		//reset accumulated partials
		for (int c = 0; c < cell_blocks; c++) {
			for (int i = 0; i < full_input_dimension; i++) {
				this.dSdwWeightsForgetGate[c][i] = 0;
				this.dSdwWeightsInputGate[c][i] = 0;
				this.dSdwWeightsNetInput[c][i] = 0;
			}
		}
	}
	
	public double[] Next(double[] input) throws Exception {
		return Next(input, null);
	}
	
	
	
	public void Display() {
		System.out.println("==============================");
		System.out.println("LSTM: todo...");
		System.out.println("\n==============================");
	}
	
	public double[] GetParameters() {
		double[] params = new double[(full_input_dimension) * cell_blocks * 4 + 3 * cell_blocks + full_hidden_dimension * output_dimension];
		int loc = 0;
		for (int j = 0; j < cell_blocks; j++) {
			for (int i = 0; i < full_input_dimension; i++) {
				params[loc++] = weightsNetInput[j][i];
				params[loc++] = weightsInputGate[j][i];
				params[loc++] = weightsForgetGate[j][i];
				params[loc++] = weightsOutputGate[j][i];
			}
			params[loc++] = peepInputGate[j];
			params[loc++] = peepForgetGate[j];
			params[loc++] = peepOutputGate[j];
		}
		
		for (int j = 0; j < full_hidden_dimension; j++) {
			for (int k = 0; k < output_dimension; k++)
				params[loc++] = weightsGlobalOutput[k][j];
		}
		if (loc != params.length)
			System.out.println("ERROR in LSTM.GetParameters() " + loc + " vs " + params.length);
		return params;
	}
	
	public void SetParameters(double[] params) {
		int loc = 0;
		for (int j = 0; j < cell_blocks; j++) {
			for (int i = 0; i < full_input_dimension; i++) {
				weightsNetInput[j][i] = params[loc++];
				weightsInputGate[j][i] = params[loc++];
				weightsForgetGate[j][i] = params[loc++];
				weightsOutputGate[j][i] = params[loc++];
			}
			peepInputGate[j] = params[loc++];
			peepForgetGate[j] = params[loc++];
			peepOutputGate[j] = params[loc++];
		}
		
		for (int j = 0; j < full_hidden_dimension; j++) {
			for (int k = 0; k < output_dimension; k++)
				weightsGlobalOutput[k][j] = params[loc++];
		}
		if (loc != params.length)
			System.out.println("ERROR in LSTM.SetParameters() " + loc + " vs " + params.length);
	}
	
	public int GetHiddenDimension() {
		return cell_blocks;
	}

	public double[] Next(double[] input, double[] target_output) {
		
		//setup input vector
		double[] full_input = new double[full_input_dimension];
		int loc = 0;
		for (int i = 0; i < input.length; i++)
			full_input[loc++] = input[i];
		for (int c = 0; c < context.length; c++)
			full_input[loc++] = context[c];
		full_input[loc++] = 1.0; //bias

		//cell block arrays
		double[] NetInputSum = new double[cell_blocks];
		double[] InputGateSum = new double[cell_blocks];
		double[] ForgetGateSum = new double[cell_blocks];
		double[] OutputGateSum = new double[cell_blocks];
		
		double[] NetInputAct = new double[cell_blocks];
		double[] InputGateAct = new double[cell_blocks];
		double[] ForgetGateAct = new double[cell_blocks];
		double[] OutputGateAct = new double[cell_blocks];
		
		double[] CECSquashAct = new double[cell_blocks];
		
		double[] NetOutputAct = new double[cell_blocks];
		
		//inputs to cell blocks
		for (int i = 0; i < full_input_dimension; i++) {
			for (int j = 0; j < cell_blocks; j++) {
				NetInputSum[j] += weightsNetInput[j][i] * full_input[i];
				InputGateSum[j] += weightsInputGate[j][i] * full_input[i];
				ForgetGateSum[j] += weightsForgetGate[j][i] * full_input[i];
				OutputGateSum[j] += weightsOutputGate[j][i] * full_input[i];
			}
		}
		
		double[] CEC1 = new double[cell_blocks];
		double[] CEC2 = new double[cell_blocks];
		double[] CEC3 = new double[cell_blocks];
		
		//internals of cell blocks
		for (int j = 0; j < cell_blocks; j++) {
			CEC1[j] = CEC[j];
			
			NetInputAct[j] = neuronNetInput.Activate(NetInputSum[j]);
			
			ForgetGateSum[j] += peepForgetGate[j] * CEC1[j];
			ForgetGateAct[j] = neuronForgetGate.Activate(ForgetGateSum[j]);

			CEC2[j] = CEC1[j] * ForgetGateAct[j];
			
			InputGateSum[j] += peepInputGate[j] * CEC2[j];
			InputGateAct[j] = neuronInputGate.Activate(InputGateSum[j]);
			
			CEC3[j] = CEC2[j] + NetInputAct[j] * InputGateAct[j];

			OutputGateSum[j] += peepOutputGate[j] * CEC3[j]; //TODO: this versus squashed?
			OutputGateAct[j] = neuronOutputGate.Activate(OutputGateSum[j]);
			
			CECSquashAct[j] = neuronCECSquash.Activate(CEC3[j]);
			
			NetOutputAct[j] = CECSquashAct[j] * OutputGateAct[j];
		}
		
		//prepare hidden layer plus bias
		double [] full_hidden = new double[full_hidden_dimension];
		loc = 0;
		for (int j = 0; j < cell_blocks; j++)
			full_hidden[loc++] = NetOutputAct[j];
		full_hidden[loc++] = 1.0; //bias
		
		//calculate output
		double[] output = new double[output_dimension];
		for (int k = 0; k < output_dimension; k++) {
			for (int j = 0; j < full_hidden_dimension; j++)
				output[k] += weightsGlobalOutput[k][j] * full_hidden[j];
			//output not squashed
		}

		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		//BACKPROP
		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		
		//scale partials
		for (int c = 0; c < cell_blocks; c++) {
			for (int i = 0; i < full_input_dimension; i++) {
				this.dSdwWeightsInputGate[c][i] *= ForgetGateAct[c];
				this.dSdwWeightsForgetGate[c][i] *= ForgetGateAct[c];
				this.dSdwWeightsNetInput[c][i] *= ForgetGateAct[c];
				
				dSdwWeightsInputGate[c][i] += full_input[i] * neuronInputGate.Derivative(InputGateSum[c]) * NetInputAct[c];
				dSdwWeightsForgetGate[c][i] += full_input[i] * neuronForgetGate.Derivative(ForgetGateSum[c]) * CEC1[c];
				dSdwWeightsNetInput[c][i] += full_input[i] * neuronNetInput.Derivative(NetInputSum[c]) * InputGateAct[c];
			}
		}
		
		if (target_output != null) {
			double[] deltaGlobalOutputPre = new double[output_dimension];
			for (int k = 0; k < output_dimension; k++) {
				deltaGlobalOutputPre[k] = target_output[k] - output[k];
			}
	
			//output to hidden
			double[] deltaNetOutput = new double[cell_blocks];
			for (int k = 0; k < output_dimension; k++) {
				//links
				for (int c = 0; c < cell_blocks; c++) {
					deltaNetOutput[c] += deltaGlobalOutputPre[k] * weightsGlobalOutput[k][c];
					weightsGlobalOutput[k][c] += deltaGlobalOutputPre[k] * NetOutputAct[c] * learningRate;
				}
				//bias
				weightsGlobalOutput[k][cell_blocks] += deltaGlobalOutputPre[k] * 1.0 * learningRate;
			}
	
			for (int c = 0; c < cell_blocks; c++) {
				
				//update output gates
				double deltaOutputGatePost = deltaNetOutput[c] * CECSquashAct[c];
				double deltaOutputGatePre = neuronOutputGate.Derivative(OutputGateSum[c]) * deltaOutputGatePost;
				for (int i = 0; i < full_input_dimension; i++) {
					weightsOutputGate[c][i] += full_input[i] * deltaOutputGatePre * learningRate;
				}
				peepOutputGate[c] += CEC3[c] * deltaOutputGatePre * learningRate;
				
				//before outgate
				double deltaCEC3 = deltaNetOutput[c] * OutputGateAct[c] * neuronCECSquash.Derivative(CEC3[c]);
				
				//update input gates
				double deltaInputGatePost = deltaCEC3 * NetInputAct[c];
				double deltaInputGatePre = neuronInputGate.Derivative(InputGateSum[c]) * deltaInputGatePost;
				for (int i = 0; i < full_input_dimension; i++) {
					weightsInputGate[c][i] += dSdwWeightsInputGate[c][i] * deltaCEC3 * learningRate;
				}
				peepInputGate[c] += CEC2[c] * deltaInputGatePre * learningRate;
				
				//before ingate
				double deltaCEC2 = deltaCEC3;
				
				//update forget gates
				double deltaForgetGatePost = deltaCEC2 * CEC1[c];
				double deltaForgetGatePre = neuronForgetGate.Derivative(ForgetGateSum[c]) * deltaForgetGatePost;
				for (int i = 0; i < full_input_dimension; i++) {
					weightsForgetGate[c][i] += dSdwWeightsForgetGate[c][i] * deltaCEC2 * learningRate;
				}
				peepForgetGate[c] += CEC1[c] * deltaForgetGatePre * learningRate;
					
				//update cell inputs
				for (int i = 0; i < full_input_dimension; i++) {
					weightsNetInput[c][i] += dSdwWeightsNetInput[c][i] * deltaCEC3 * learningRate;
				}
				//no peeps for cell inputs
			}
		}
		
		//////////////////////////////////////////////////////////////
		
		//roll-over context to next time step
		for (int j = 0; j < cell_blocks; j++) {
			context[j] = NetOutputAct[j];
			CEC[j] = CEC3[j];
		}
		
		//give results
		return output;
	}
}


