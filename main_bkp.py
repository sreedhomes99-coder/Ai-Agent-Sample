from dotenv import load_dotenv
import os

load_dotenv()

# Check if API key is set
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("Error: ANTHROPIC_API_KEY environment variable is not set.")
    print("Please set your API key in the .env file or environment variables.")
    print("\nExample .env file:")
    print("ANTHROPIC_API_KEY=your_api_key_here")
else:
    from langchain_anthropic import ChatAnthropic
    
    # Initialize the LLM
    llm = ChatAnthropic(model="claude-opus-4-5-20251101")
    
    # Create a simple query
    query = "What is the weather report now in India hyderabad Provide a brief summary."
    
    # Invoke the LLM
    response = llm.invoke(query)
    
    # Print the response
    print("Response from Claude:")
    print("=" * 50)
    print(response.content)
