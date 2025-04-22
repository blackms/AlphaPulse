# Error Handling

*   Handle failures reported by delegated tasks gracefully: Log the failure in your task log and report the issue clearly back to the Commander in your final `attempt_completion` message.
*   Handle potential errors during file/directory operations (e.g., `mkdir`, `git init`, `write_to_file`) and log them.