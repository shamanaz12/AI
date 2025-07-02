import os
from dotenv import load_dotenv
from agent import HealthWellnessAgent
from context import ContextManager
from guardrails import SafetyGuardrails
from hooks import setup_hooks
from tools.goal_analyzer import GoalAnalyzer
from tools.meal_planner import MealPlanner
from tools.workout_recommender import WorkoutRecommender
from tools.scheduler import Scheduler
from tools.tracker import ProgressTracker
from agents.escalation_agent import EscalationAgent
from agents.nutrition_expert_agent import NutritionExpertAgent
from agents.injury_support_agent import InjurySupportAgent
from utils.streaming import StreamingChat

# Load environment variables
load_dotenv()

class HealthWellnessApp:
    def __init__(self):
        self.setup_configuration()
        self.initialize_components()
        self.setup_agents()
        self.setup_tools()
        
    def setup_configuration(self):
        """Initialize API configuration"""
        self.model_name = "gemini-2.0-flash"
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        self.base_config = {
            "api_key": self.api_key,
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "model": self.model_name
        }
    
    def initialize_components(self):
        """Initialize core components"""
        self.context_manager = ContextManager()
        self.safety_guardrails = SafetyGuardrails()
        self.streaming_chat = StreamingChat(self.base_config)
        
        # Setup hooks
        setup_hooks(self.context_manager, self.safety_guardrails)
    
    def setup_agents(self):
        """Initialize all agents"""
        self.main_agent = HealthWellnessAgent(self.base_config)
        self.escalation_agent = EscalationAgent(self.base_config)
        self.nutrition_agent = NutritionExpertAgent(self.base_config)
        self.injury_agent = InjurySupportAgent(self.base_config)
    
    def setup_tools(self):
        """Initialize all tools"""
        self.goal_analyzer = GoalAnalyzer(self.base_config)
        self.meal_planner = MealPlanner(self.base_config)
        self.workout_recommender = WorkoutRecommender(self.base_config)
        self.scheduler = Scheduler()
        self.progress_tracker = ProgressTracker()
    
    def process_user_input(self, user_input, user_context=None):
        """Main processing function"""
        try:
            # Apply safety guardrails
            if not self.safety_guardrails.is_safe_input(user_input):
                return "I can't help with that request. Please ask something related to health and wellness."
            
            # Update context
            if user_context:
                self.context_manager.update_context(user_context)
            
            # Process through main agent
            response = self.main_agent.process_query(
                user_input, 
                self.context_manager.get_context()
            )
            
            return response
            
        except Exception as e:
            return f"Error processing your request: {str(e)}"
    
    def run_interactive_mode(self):
        """Run interactive chat mode"""
        print("🏥 Health & Wellness AI Agent")
        print("=" * 40)
        print("Ask me anything about health, fitness, nutrition, or wellness!")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Take care! Stay healthy!")
                    break
                
                if not user_input:
                    continue
                
                print("\n🤖 Assistant: ", end="")
                response = self.process_user_input(user_input)
                print(response)
                print("\n" + "-" * 40 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Take care! Stay healthy!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")

def main():
    """Main application entry point"""
    try:
        app = HealthWellnessApp()
        app.run_interactive_mode()
    except Exception as e:
        print(f"Failed to start application: {str(e)}")

if __name__ == "__main__":
    main()