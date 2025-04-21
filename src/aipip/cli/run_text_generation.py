import argparse
import json
import sys
from typing import Dict, List, Optional

# Load .env file if present, BEFORE importing local modules that use settings
# This allows placing a .env file in the project root for local development
# without needing to export environment variables.
from dotenv import load_dotenv
load_dotenv()

from aipip.config.loader import load_settings
from aipip.providers.registry import ProviderRegistry
from aipip.services.text_generation_service import TextGenerationService


def parse_messages(message_args: List[str]) -> Optional[List[Dict[str, str]]]:
    """Parses messages from command line format [role] [content] [role] [content]..."""
    if not message_args:
        return None
    if len(message_args) % 2 != 0:
        raise ValueError("Messages must be provided in pairs of [role] [content]")

    messages = []
    for i in range(0, len(message_args), 2):
        messages.append({"role": message_args[i], "content": message_args[i+1]})
    return messages

def main():
    parser = argparse.ArgumentParser(description="Generate text using AIPIP.")
    parser.add_argument("--provider", required=True, help="Name of the provider (e.g., openai, google, anthropic)")
    parser.add_argument("--model", help="Specific model name to use (optional)")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--prompt", help="Single text prompt.")
    input_group.add_argument("--messages", nargs='+', help="List of messages: role1 content1 role2 content2 ...")

    parser.add_argument("--temperature", type=float, help="Sampling temperature (optional)")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate (optional)")
    # Placeholder for potentially passing other kwargs via JSON string or similar
    # parser.add_argument("--kwargs", type=json.loads, help='JSON string of extra provider arguments')

    args = parser.parse_args()

    try:
        # Load settings (reads from environment / .env)
        settings = load_settings()

        # Initialize registry and service
        registry = ProviderRegistry(settings=settings)
        service = TextGenerationService(registry=registry)

        # Parse messages if provided
        messages_data = parse_messages(args.messages) if args.messages else None

        # Prepare kwargs for service call, filtering out None values
        service_kwargs = {
            key: value for key, value in {
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                # Add other args here
            }.items() if value is not None
        }
        # Add any extra kwargs from a potential --kwargs argument later
        # if args.kwargs:
        #    service_kwargs.update(args.kwargs)

        # Call the service
        response = service.generate(
            provider_name=args.provider,
            prompt=args.prompt,
            messages=messages_data,
            model=args.model,
            **service_kwargs
        )

        # Print the result
        print("--- Generated Text ---")
        print(response.text)
        print("\n--- Metadata ---")
        # Pretty print metadata
        print(json.dumps(response.metadata, indent=2))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 