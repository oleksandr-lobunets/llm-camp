import json
import os
import types # Import os to access environment variables
from IPython.display import display, HTML
import markdown
import google.generativeai as genai # Import the Gemini API client library
from google.genai import types
from google.genai.types import FunctionDeclaration


is_html: bool = False

class ToolManager:
    """Manages the registration and execution of tools/functions."""
    def __init__(self):
        self.tools_descriptions = []
        self.functions = {}

    def register_tool(self, func, description):
        """Registers a function as a tool with its description."""
        self.tools_descriptions.append(description)
        self.functions[func.__name__] = func
        print(f"Registered tool: {func.__name__}")
        print(f"Description: {description}")

    def get_tool_descriptions(self):
        """Returns a list of descriptions for all registered tools."""
        return self.tools_descriptions

    def execute_tool_call(self, tool_call_response):
        """Executes a tool function based on the model's tool call response."""
        function_name = tool_call_response.name
        try:
            arguments = json.loads(tool_call_response.arguments)
        except json.JSONDecodeError:
            return {
                "type": "function_call_output",
                "call_id": tool_call_response.call_id,
                "output": json.dumps({"error": "Invalid JSON arguments"}, indent=2),
            }

        func_to_call = self.functions.get(function_name)
        if not func_to_call:
            return {
                "type": "function_call_output",
                "call_id": tool_call_response.call_id,
                "output": json.dumps({"error": f"Function '{function_name}' not found"}, indent=2),
            }

        try:
            result = func_to_call(**arguments)
            return {
                "type": "function_call_output",
                "call_id": tool_call_response.call_id,
                "output": json.dumps(result, indent=2),
            }
        except Exception as e:
            return {
                "type": "function_call_output",
                "call_id": tool_call_response.call_id,
                "output": json.dumps({"error": str(e)}, indent=2),
            }


def _shorten_text(text, max_length=50):
    """Shortens a string to a maximum length, appending '...' if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


class ChatUserInterface:
    """Handles input/output interactions with the user."""
    def get_user_input(self):
        """Prompts the user for input and returns it."""
        return input("You: ")
    
    def display_message(self, message):
        """Prints a simple message to the console."""
        print(message)

    def display_function_call_details(self, entry, result, is_html=True):
        """Displays formatted details of a function call and its output."""
        
        if is_html:
            call_html = f"""
            <details>
            <summary>Function call: <tt>{entry.name}({_shorten_text(entry.arguments)})</tt></summary>
            <div>
                <b>Call</b>
                <pre>{entry}</pre>
            </div>
            <div>
                <b>Output</b>
                <pre>{result['output']}</pre>
            </div>
            </details>
            """
            display(HTML(call_html))
        else:
            call_makedown = call_html = f"""
            <details>
            <summary>Function call: <tt>{entry.name}({_shorten_text(entry.arguments)})</tt></summary>
            <div>
                <b>Call</b>
                <pre>{entry}</pre>
            </div>
            <div>
                <b>Output</b>
                <pre>{result['output']}</pre>
            </div>
            </details>
            """
            display(markdown.markdown(call_makedown))



    def display_assistant_response(self, entry, is_html=True):
        """Displays the assistant's message, formatted with Markdown."""
        # For Gemini API, direct text access might be through entry.text
        # If entry.content is still used, adapt accordingly.
        # This assumes entry.text is the primary way for simple text responses.
        response_text = entry.text if hasattr(entry, 'text') else entry.content[0].text
        response_html = markdown.markdown(response_text)
        if is_html:
            html = f"""
                <div>
                    <div><b>Assistant:</b></div>
                    <div>{response_html}</div>
                </div>
            """
            display(HTML(html))
        else:
            response_markdown = f"""
            **Assistant:**
            {response_html}
            """
            # Display the response in Markdown format
                      
            display(markdown.markdown(response_markdown))





class AIConversationAssistant:
    """Manages the conversation flow with the AI model."""
    def __init__(self, tool_manager: ToolManager, developer_prompt: str, ui_interface: ChatUserInterface, api_key: str = None, model_name: str = 'gemini-2.5-flash'):
        self.tool_manager = tool_manager
        self.developer_prompt = developer_prompt
        self.ui_interface = ui_interface
        self.model_name = model_name

        # Configure the Gemini API client
        if api_key:
            genai.configure(api_key=api_key)
        else:
            # Attempt to load from environment variable by default
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            if not os.environ.get("GEMINI_API_KEY"):
                print("Error: GEMINI_API_KEY environment variable not set. Please provide it or set the env var.")
                raise ValueError("GEMINI_API_KEY environment variable not set")


        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            tools=self.tool_manager.get_tool_descriptions()
        )
        
        # Initialize chat history. Gemini models use a `start_chat` method.
        # The history will be managed internally by the ChatSession object.
        self.chat_session = self.model.start_chat(history=[
            {"role": "user", "parts": [self.developer_prompt]}, # Developer prompt goes as a user message
            {"role": "model", "parts": ["Understood. I'm ready to assist."]} # Model acknowledges
        ])
        
        # We'll keep a simplified chat messages list for display consistency,
        # but the actual history for the model is within self.chat_session
        self._display_chat_messages = []
        self._display_chat_messages.append({"role": "developer", "content": self.developer_prompt})


    def _get_model_response(self, user_message_parts):
        """Sends chat messages to the AI model and returns its response."""
        # For Gemini, we send messages through the chat session
        response = self.chat_session.send_message(user_message_parts)
        return response

    def start_chat(self):
        """Initiates and manages the main chat loop."""
        while True:
            user_question = self.ui_interface.get_user_input()
            if user_question.strip().lower() == 'stop':  
                self.ui_interface.display_message("Chat ended.")
                break

            # Append user message to our display history
            self._display_chat_messages.append({"role": "user", "content": user_question})

            # Prepare message parts for Gemini API (can be text, images, etc.)
            user_message_parts = [user_question]

            response = self._get_model_response(user_message_parts)

            # Process Gemini's response
            # Gemini's response object might have .candidates which contains parts
            # or directly .text for simple text responses, or .parts for tool calls
            
            # The structure of the response and tool calls needs careful handling
            # as it differs from OpenAI.
            
            # For Gemini, responses can contain parts, including function calls
            # The `parts` attribute of a content object will contain all elements.

            # Iterate through the parts of the response
            # Note: The structure of function calls from Gemini API (especially when
            # using the `google.generativeai` library) is different from the OpenAI-like
            # `tool_call_response` you were expecting. You'll need to adapt `ToolManager`
            # or how you process these to match Gemini's output.

            # A more robust way to handle Gemini's responses:
            # Typically, direct text content is in `response.text` for simple answers.
            # For complex responses with tool calls, you'll iterate through `response.parts`.

            # Let's re-think the response processing for Gemini's native format.
            # The `response.parts` will contain `Part` objects, which can be `TextPart`,
            # `FunctionCallPart`, `FunctionResponsePart`, etc.

            processed_response = False
            for candidate in response.candidates: # A response can have multiple candidates
                for part in candidate.content.parts:
                    print (f"Processing part: {part}") # Debugging output to see part structure
                    # if hasattr(part, 'function_call') and (part.function_call is not None and len(part.parts) > 0):
                    if hasattr(part, 'function_call') and part.function_call:

                        tool_call = part.function_call
                        # Construct a dictionary similar to your old tool_call_response
                        # for compatibility with `execute_tool_call`
                        dummy_tool_call_response = type('ToolCallResponse', (object,), {
                            'name': tool_call.name,
                            'arguments': json.dumps(dict(tool_call.args)), # Gemini returns args as a structure
                            'call_id': 'gemini_call_' + tool_call.name # Gemini doesn't always provide a direct call_id
                        })()
                        
                        tool_output = self.tool_manager.execute_tool_call(dummy_tool_call_response)
                        
                        # Append the tool output to chat history for the model
                        # Gemini expects tool outputs in a specific format for next turn
                        function_response = genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=tool_call.name,
                                response={"result": tool_output["output"]}
                            )
                        )
                        next_response = self.chat_session.send_message([function_response])

                        self.ui_interface.display_function_call_details(dummy_tool_call_response, tool_output, is_html=False)
                        self._display_chat_messages.append(tool_output)
                        
                        # Display the model's response after the function call
                        if next_response.text:
                            class GeminiMessage:
                                def __init__(self, text):
                                    self.text = text
                                    self.content = [type('ContentPart', (object,), {'text': text})()]
                            
                            gemini_message_entry = GeminiMessage(next_response.text)
                            self.ui_interface.display_assistant_response(gemini_message_entry, is_html=is_html)
                            self._display_chat_messages.append({"role": "model", "content": next_response.text})
                        processed_response = True
                        break

                    elif hasattr(part, 'text'): # This is a text message from the model
                        # For display purposes, we treat this as the model's message
                        class GeminiMessage: # Create a dummy class to match `entry.content[0].text` structure
                            def __init__(self, text):
                                self.text = text
                                self.content = [type('ContentPart', (object,), {'text': text})()]
                        
                        gemini_message_entry = GeminiMessage(part.text)
                        self.ui_interface.display_assistant_response(gemini_message_entry, is_html=is_html)
                        self._display_chat_messages.append({"role": "model", "content": part.text}) # Add to display history
                        processed_response = True
                        break # Processed a message, can break from this inner loop

                if processed_response:
                    break # Break the outer loop if a message was displayed

            # If no direct message was processed and only tool calls happened,
            # the inner loop will continue and get the next model response.
            # If the model gives no output or only tool calls without a final message,
            # the loop will run until a message is generated or a limit is hit.

# --- Example Usage (How to instantiate and run) ---
# You'll need to define some dummy tools and an API key.

# Example dummy tool
def get_current_weather(location: str):
    """Gets the current weather for a specified location."""
    if "Odesa" in location:
        return {"temperature": "25°C", "conditions": "Sunny"}
    return {"temperature": "Unknown", "conditions": "Cannot retrieve for this location"}

if __name__ == "__main__":
    # Set your Gemini API Key.
    # It's highly recommended to set this as an environment variable (e.g., GEMINI_API_KEY)
    # or pass it securely.
    # For local testing, you can uncomment and set it directly:
    # GEMINI_API_KEY = "YOUR_GEMINI_API_KEY" 
    # Or ensure it's set in your environment: export GEMINI_API_KEY="your_api_key_here"

    # Initialize ToolManager and register tools
    tool_manager = ToolManager()
    tool_manager.register_tool(
        get_current_weather, 
        genai.types.FunctionDeclaration(
            name="get_current_weather",
            description="Gets the current weather for a specified location.",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}
                },
                "required": ["location"],
            },
        )
    )

    # tool_manager.register_tool(
    #     get_current_weather, 
    #     FunctionDeclaration(
    #         name="_current_weather",
    #         description="Gets the current weather for a specified location.",
    #         parameters={
    #             "type": "object",
    #             "properties": {
    #                 "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}
    #             },
    #             "required": ["location"],
    #         },
    #     )
    # )

    # Initialize ChatUserInterface
    ui = ChatUserInterface()

    # Define your developer prompt
    developer_prompt = "You are a helpful assistant that can answer questions and use tools. If the user asks about weather, use the `get_current_weather` tool."

    assistant = AIConversationAssistant(
        tool_manager=tool_manager,
        developer_prompt=developer_prompt,
        ui_interface=ui,
        model_name='gemini-2.5-flash-lite-preview-06-17'
    )

    assistant.start_chat()