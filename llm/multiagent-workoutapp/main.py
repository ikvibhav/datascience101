# Note: Replace **<YOUR_APPLICATION_TOKEN>** with your actual Application token
import requests

# The complete API endpoint URL for this flow
url = f"https://api.langflow.astra.datastax.com/lf/412d79b2-20d3-43f2-b8aa-caa1ef05c2e8/api/v1/run/592e5797-f89c-44ed-b046-0d3edd6d66a7"

# Request payload configuration
payload = {
    "input_value": "hello world!",  # The input value to be processed by the flow
    "output_type": "text",  # Specifies the expected output format
    "input_type": "text",  # Specifies the input format
}

# Request headers
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer <YOUR_APPLICATION_TOKEN>",  # Authentication key from environment variable'}
}

try:
    # Send API request
    response = requests.request("POST", url, json=payload, headers=headers)
    response.raise_for_status()  # Raise exception for bad status codes

    # Print response
    print(response.text)

except requests.exceptions.RequestException as e:
    print(f"Error making API request: {e}")
except ValueError as e:
    print(f"Error parsing response: {e}")
