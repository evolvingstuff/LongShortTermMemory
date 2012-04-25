package com.evolvingstuff;

import java.util.*;

public class EmbeddedReberGrammar
{
	int tests = 5000;
	
	public class State
	{
		public State(Transition[] transitions)
		{
			this.transitions = transitions;
		}
		public Transition[] transitions;
	}
	
	public class Transition
	{
		public Transition(int next_state_id, int token)
		{
			this.next_state_id = next_state_id;
			this.token = token;
		}
		
		public int next_state_id;
		public int token;
	}
	
	private Random r;
	private State[] states;
	private static String[] num_to_string = {"B","T","P","S","X","V","E"};
	private int B = 0;
	private int T = 1;
	private int P = 2;
	private int S = 3;
	private int X = 4;
	private int V = 5;
	private int E = 6;
	public static boolean reset_at_begining = true;//false
	public static boolean super_discrete_feedback = false;
	public static boolean discrete_feedback = true;
	private boolean error_squared = false;
	public static boolean ignore_short_transitions = false; //Partial vs Complete version
	public static boolean deterministic_evaluation = false; //TODO
	private boolean validation_mode = false;
	
	public EmbeddedReberGrammar(Random r)
	{
		
		this.r = r;
		states = new State[19];
		states[0] = new State(new Transition[] {new Transition(1,B)});
		states[1] = new State(new Transition[] {new Transition(2,T), new Transition(11,P)});
		states[2] = new State(new Transition[] {new Transition(3,B)});
		states[3] = new State(new Transition[] {new Transition(4,T), new Transition(9,P)});
		states[4] = new State(new Transition[] {new Transition(4,S), new Transition(5,X)});
		states[5] = new State(new Transition[] {new Transition(6,S), new Transition(9,X)});
		states[6] = new State(new Transition[] {new Transition(7,E)});
		states[7] = new State(new Transition[] {new Transition(8,T)});
		states[8] = new State(new Transition[] {new Transition(0,E)});
		states[9] = new State(new Transition[] {new Transition(9,T), new Transition(10,V)});
		states[10] = new State(new Transition[] {new Transition(5,P), new Transition(6,V)});
		states[11] = new State(new Transition[] {new Transition(12,B)});
		states[12] = new State(new Transition[] {new Transition(13,T), new Transition(17,P)});
		states[13] = new State(new Transition[] {new Transition(13,S), new Transition(14,X)});
		states[14] = new State(new Transition[] {new Transition(15,S), new Transition(17,X)});
		states[15] = new State(new Transition[] {new Transition(16,E)});
		states[16] = new State(new Transition[] {new Transition(8,P)});
		states[17] = new State(new Transition[] {new Transition(17,T), new Transition(18,V)});
		states[18] = new State(new Transition[] {new Transition(14,P), new Transition(15,V)});
	}

	public int GetActionDimension() 
	{
		return 7;
	}

	public int GetObservationDimension() 
	{
		return 7;
	}

	public void SetValidationMode(boolean validation) {
		this.validation_mode = validation;
	}

	public double EvaluateFitnessSupervised(IAgentSupervised agent) throws Exception {
		if (deterministic_evaluation == true)
			r = new Random(1);
		
		int state_id = 0;
		agent.Reset();
		double tot_fit = 0;
		double tot_long_transitions = 0;
		double incorrect_long_transitions = 0;
		for (int t = 0; t < tests; t++)
		{
			int transition = -1;
			if (states[state_id].transitions.length == 1)
				transition = 0;
			else if (states[state_id].transitions.length == 2)
				transition = r.nextInt(2);
			else
				System.out.println("ERROR: more that 2 transitions");
			if (transition == -1)
				System.out.println("ERROR! no transition selected");
			
			double[] agent_input = new double[7];
			agent_input[states[state_id].transitions[transition].token] = 1.0;
			
			state_id = states[state_id].transitions[transition].next_state_id;
			
			double[] target = new double[7];
			for (int i = 0; i < states[state_id].transitions.length; i++)
				target[states[state_id].transitions[i].token] = 1.0;
			
			double[] agent_output;
			if (!validation_mode)
				agent_output = agent.Next(agent_input, target);
			else
				agent_output = agent.Next(agent_input);
			
			if (state_id == 7 || state_id == 16)
				tot_long_transitions++;
			
			boolean missed_long = false;
			
			if (super_discrete_feedback == true)
			{
				boolean all_correct = true;
				for (int i = 0; i < 7; i++)
				{
					if (Math.abs(target[i] - agent_output[i]) >= 0.5) {
						all_correct = false;
						break;
					}
				}
				if (all_correct)
					tot_fit += 1/(double)tests;
			}
			else
			{
				for (int i = 0; i < 7; i++)
				{
					if (discrete_feedback == true)
					{
	
						if (Math.abs(target[i] - agent_output[i]) < 0.5)
							tot_fit += 1/(7*(double)tests);
						else
						{
							if (state_id == 7 || state_id == 16)
								missed_long = true;
						}
					}
					else
					{
						if (error_squared == true)
							tot_fit += (1 - (target[i] - agent_output[i])*(target[i] - agent_output[i]))/(7*(double)tests);
						else
							tot_fit += (1 - Math.abs(target[i] - agent_output[i]))/(7*(double)tests);
					}
				}
			}
			
			if (missed_long == true)
				incorrect_long_transitions += 1;
				
		}
		
		System.out.println("\t\t\tLong-transition error (validation:"+validation_mode+") = " + (incorrect_long_transitions/tot_long_transitions));
		
		return tot_fit;
	}

}
