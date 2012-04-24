package com.evolvingstuff;

public interface IAgentSupervised 
{
	void Reset();
	double[] Next(double[] input, double[] target_output) throws Exception;
	double[] Next(double[] input) throws Exception;
}
