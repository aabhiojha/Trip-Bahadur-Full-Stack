import json
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain_core.exceptions import LangChainException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize LLMs with error handling
try:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("WARNING: GROQ_API_KEY not set or using placeholder value")
        main_llm = None
    else:
        main_llm = ChatGroq(
            model="llama3-70b-8192",
            api_key=api_key,
            # temperature=2
        )
        print("GROQ LLM initialized successfully")
except Exception as e:
    print(f"Error initializing GROQ LLM: {e}")
    main_llm = None


@tool
def generate_itinerary(destination: str) -> dict:
    """Generates a structured multi-day itinerary with geographical data for a given destination."""
    prompt = f"""
    Create an itinerary for {destination} including:
    - Real elevation data in meters.
    - Ensure the latitude and longitude of a location do not repeat; make small adjustments (even 0.0001) if necessary.
    - Include notable landmarks.
    - Make sure the coordinates are as accurate as possible
    - Start the itinerary from Kathmandu if the destination is in Nepal.
    
    Format as JSON with this structure:
    {{
      "itinerary": [
        {{
          "day": 1,
          "location": "Name",
          "elevation": 1000,
          "coordinates": {{
            "latitude": 28.1234,
            "longitude": 83.5678
          }},
          "highlight": "Main feature",
          "description": {{
            "trekking_duration": "Approximate trekking duration.",
            "key_highlights": [
              "Scenic views, cultural experiences, or notable landmarks."
            ],
            "permits": "Information about required permits.",
            "best_time_to_visit": "Best time to visit.",
            "difficulty_level": "Brief overview of difficulty level.",
            "tips": [
              "Additional tips or recommendations for trekkers (e.g., packing essentials, acclimatization advice, pass)."
            ]
          }}
        }}
      ]
    }}
    Do not write anything else, not even ```json or ```.
    """
    try:
        response = main_llm.invoke(prompt)

        # Ensure the response content is not empty before attempting to parse
        if not response.content.strip():
            return {"error": "Empty response received from model."}

        json_str = (
            response.content.replace("```json", "")  # Remove ```json
            .replace("```", "")  # Remove ```
            .replace("<tool-use></tool-use>", "")
            .strip()  # Remove leading/trailing whitespace
        )
        return json.loads(json_str)
    except Exception as e:
        return {"error": str(e)}


# Available tools
tools = [generate_itinerary]


def handle_tool_calls(response: AIMessage) -> list:
    """Processes and executes tool calls in the LLM response."""
    if not response.additional_kwargs.get("tool_calls"):
        return [{"status": "no_tool_calls"}]

    tool_map = {t.name: t for t in tools}
    results = []

    for tc in response.additional_kwargs["tool_calls"]:
        try:
            func_name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])

            if func_name not in tool_map:
                results.append({"tool": func_name, "status": "not_found"})
                continue

            tool_result = tool_map[func_name].invoke(args)
            results.append(
                {"tool": func_name, "arguments": args, "result": tool_result}
            )
        except json.JSONDecodeError:
            results.append({"tool": func_name, "status": "invalid_json"})
        except Exception as e:
            results.append({"tool": func_name, "error": str(e)})

    return results


# Flask API endpoint
@app.route("/generate-itinerary", methods=["POST"])
def generate_itinerary_api():
    try:
        # Check if LLM is properly initialized
        if main_llm is None:
            return (
                jsonify(
                    {
                        "error": "Service temporarily unavailable. Please configure GROQ_API_KEY environment variable."
                    }
                ),
                503,
            )

        print(request.get_data())
        # Get user input from the request JSON
        data = request.get_json()
        user_input = data.get("query", "").strip()

        if not user_input:
            return jsonify({"error": "No query provided"}), 400

        # Process the user input with the LLM
        response = main_llm.bind_tools(tools).invoke(
            [
                SystemMessage(
                    content="You are a friendly travel expert. Use tools only when asked for specific itineraries."
                ),
                HumanMessage(content=user_input),
            ]
        )

        if isinstance(response, AIMessage):
            # Check for tool calls first
            if response.tool_calls:
                tool_results = handle_tool_calls(response)
                for result in tool_results:
                    if "result" in result:
                        return jsonify({"itinerary": result["result"]})
                    elif "error" in result:
                        return jsonify({"error": result["error"]}), 400
            else:
                print(response.content)
                return jsonify({"response": response.content})

        else:
            return jsonify({"error": "Unexpected response format"}), 500

    except LangChainException as e:
        return jsonify({"error": f"System error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
