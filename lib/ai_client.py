import ollama


class OllamaChatClient:
    """Generic reusable client for interacting with a local chat model service."""

    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name
        self.history = []

    def send(self, prompt):
        # Validate prompt
        if prompt is None:
            raise ValueError("Prompt cannot be None")

        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")

        prompt = prompt.strip()

        if not prompt:
            raise ValueError("Prompt cannot be empty")

        # Create and store user message
        user_message = {
            "role": "user",
            "content": prompt
        }

        self.history.append(user_message)

        try:
            # Call Ollama
            response = ollama.chat(
                model=self.model_name,
                messages=self.history
            )

            # Dictionary-style response
            if isinstance(response, dict):
                content = response.get("message", {}).get("content")

            # Object-style response
            else:
                message = getattr(response, "message", None)
                content = getattr(message, "content", None)

            # Validate assistant content
            if not isinstance(content, str):
                raise ValueError("Invalid assistant response")

            content = content.strip()

            if not content:
                raise ValueError("Blank assistant response")

            # Save assistant response
            self.history.append({
                "role": "assistant",
                "content": content
            })

            return content

        except Exception as exc:
            # Remove failed user message
            if self.history and self.history[-1] == user_message:
                self.history.pop()

            raise RuntimeError(
                f"Service error: {exc}"
            ) from exc

    def reset(self):
        """Clear the conversation history."""
        self.history.clear()

    def message_count(self):
        """Return the number of stored messages."""
        return len(self.history)

    def get_transcript(self):
        """Return a safe copy of the conversation history."""
        return [message.copy() for message in self.history]