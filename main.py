"""
Main entry point for LangGraph Planner-Executor Agent
Executable script with test cases
"""

from graph import run_agent
import sys


def main():
	"""Main function - runs the agent with test case"""
    
	# Test Case 1: Outdoor Event Planning (from requirements)
	goal1 = "Plan an outdoor event for 150 people: calculate tables/chairs, find average ticket price, check weather, and summarize."
    
	print("\n" + "="*70)
	print("TEST CASE 1: Outdoor Event Planning")
	print("="*70)
    
	result1 = run_agent(goal1)
    
	# Print summary
	print("\n" + "="*70)
	print("EXECUTION SUMMARY")
	print("="*70)
	print(f"Total steps executed: {len(result1['results'])}")
	print(f"Steps completed: {result1['current_step']}")
	print(f"Status: SUCCESS ✅")
    
	# Print detailed results
	print("\n" + "-"*70)
	print("DETAILED RESULTS:")
	print("-"*70)
	for i, res in enumerate(result1['results'], 1):
		print(f"\n{i}. Step {res['step']}: {res['description']}")
		if res.get('tool'):
			print(f"   └─ Tool: {res['tool']}")
		print(f"   └─ Output: {res['result'][:150]}...")


def run_custom():
	"""Run with custom goal from command line"""
	if len(sys.argv) > 1:
		goal = " ".join(sys.argv[1:])
		print(f"\n{'#'*70}")
		print(f"CUSTOM GOAL")
		print(f"{'#'*70}")
		result = run_agent(goal)
		return result
	else:
		print("Usage: python main.py '<your goal here>'")
		print("Example: python main.py 'Plan a wedding for 200 people'")


if __name__ == "__main__":
	if len(sys.argv) > 1:
		run_custom()
	else:
		main()
