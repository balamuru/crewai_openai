import unittest
import sys
import os
from crewai import Agent, Task, Crew, Process

# Add src to path to ensure imports work if needed, though we are using direct crewai imports here
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

class TestCrewAllInclusive(unittest.TestCase):
    def test_run_crew(self):
        """Test the crew execution with self-contained agents and tasks."""
        
        # 1. Define Agents
        ai_power_analyst = Agent(
            role='AI Power Usage Analyst',
            goal='Analyze power demand trends specifically driven by AI adoption',
            backstory="""You are a specialist in analyzing the energy footprint of artificial intelligence models. 
            You track the computational requirements of large language models and training clusters to forecast future energy needs.""",
            verbose=True,
            allow_delegation=False
        )

        data_center_specialist = Agent(
            role='Data Center Infrastructure Specialist',
            goal='Investigate energy supply and demand for data centers',
            backstory="""You are an expert in data center infrastructure and energy grid requirements. 
            You understand the physical constraints of power distribution and cooling systems needed to support massive computing operations.""",
            verbose=True,
            allow_delegation=False
        )

        writer = Agent(
            role='Technical Report Writer',
            goal='Summarize research into a report',
            backstory="""You are a skilled technical writer capable of synthesizing complex information into clear, concise reports. 
            You can take technical findings from analysts and engineers and turn them into readable documents for stakeholders.""",
            verbose=True
        )

        # 2. Define Tasks
        task1 = Task(
            description='Conduct research on current and projected power demand trends specifically driven by the adoption of AI technologies.',
            expected_output='A summary of AI power demand trends.',
            agent=ai_power_analyst
        )

        task2 = Task(
            description='Research the current state of data center energy supply and demand, including challenges in grid connection and cooling.',
            expected_output='A summary of data center energy supply and demand.',
            agent=data_center_specialist
        )

        task3 = Task(
            description='Collect the information from the previous tasks and format it into a comprehensive report on AI and Data Center Energy Trends.',
            expected_output='A comprehensive markdown report covering AI power demand and data center energy supply.',
            agent=writer,
            context=[task1, task2] # Explicitly provide context from previous tasks
        )

        # 3. Instantiate Crew
        crew = Crew(
            agents=[ai_power_analyst, data_center_specialist, writer],
            tasks=[task1, task2, task3],
            process=Process.sequential,
            verbose=True
        )

        # 4. Kickoff
        result = crew.kickoff()
        
        # 5. Verify
        self.assertIsNotNone(result)
        print("\n\n########################")
        print("## Here is the Report ##")
        print("########################\n")
        print(result)

if __name__ == '__main__':
    unittest.main()
