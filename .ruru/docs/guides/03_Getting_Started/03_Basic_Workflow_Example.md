+++
# --- Basic Metadata ---
id = "KB-RC-SETUP-BASIC-WORKFLOW"
title = "Getting Started: Basic Workflow Example"
status = "draft"
difficulty = "beginner"
estimated_time = "~10 minutes"
target_audience = ["users"]
prerequisites = ["Roo Commander installed", "Basic understanding of Roo Commander concepts (Modes, Delegation)"]
learning_objectives = ["Understand a typical interaction flow with Roo Commander", "See how delegation to specialist modes works", "Recognize the communication pattern (User -> Commander -> Specialist -> Commander -> User)"]
template_schema_doc = ".ruru/templates/toml-md/10_guide_tutorial.README.md"
tags = ["roo-commander", "getting-started", "workflow", "example", "tutorial", "delegation", "modes"]
related_docs = [
    "../README.md", # Link to the KB README
    "01_Installation_Setup.md",
    "02_Initial_Interaction.md",
    "../../02_Core_Concepts/03_MDTM_Explained.md", # Mention MDTM vs simple task
    "../../04_Understanding_Modes/02_Mode_Selection_Guide.md" # Mention specialist selection
    ]
+++

# Getting Started: Basic Workflow Example

## 1. Introduction / Goal 🎯

This guide illustrates a simple, common workflow when interacting with **👑 Roo Commander**. It shows how you state a goal, how Commander analyzes it and delegates to a specialist mode, and how the result is reported back to you.

The goal of this example is to implement a basic Python utility function to demonstrate the delegation process.

## 2. Prerequisites Checklist ✅

*   [ ] Roo Commander is installed in your workspace.
*   [ ] You have activated the "👑 Roo Commander" mode in the Roo Code chat interface.
*   [ ] You have a basic understanding of modes and delegation (covered in previous guides).

## 3. The Workflow Steps 🪜

Let's imagine you want to create a simple Python function to add two numbers.

*   **Step 1: State Your Goal (User -> Commander)**
    *   You provide a clear instruction to Roo Commander in the chat:
        ```prompt
        Please create a Python function called `add_numbers` in a new file named `src/utils/math_utils.py`. This function should take two arguments (a, b) and return their sum. Also, add a simple docstring explaining what it does.
        ```

*   **Step 2: Commander Analyzes & Plans (Commander Internal)**
    *   Commander receives your request.
    *   It analyzes the intent: "Create a Python function" -> This is a coding task.
    *   It identifies the language: Python.
    *   It consults the Mode Selection Guide (or its internal knowledge) and determines the best specialist is likely `dev-python` (or potentially `util-senior-dev` if a specific Python mode isn't available/configured).
    *   It determines the task is relatively simple and self-contained, suitable for direct delegation via `new_task` rather than the full MDTM workflow.
    *   It prepares the instructions for the specialist.

*   **Step 3: Commander Delegates (Commander -> Specialist via `new_task`)**
    *   Commander uses the `new_task` tool to delegate the work:
        ```xml
        <thinking>
        Goal is to create a simple Python function.
        Task is straightforward, direct delegation is appropriate.
        Selected specialist: `dev-python`.
        Preparing delegation message.
        </thinking>
        <new_task>
          <mode>dev-python</mode>
          <message>
          🐍 Create Python Function Task:
          Goal: Create a function `add_numbers(a, b)` that returns the sum of a and b.
          File: Create a new file at `src/utils/math_utils.py`.
          Content: Include the function definition and a simple docstring.
          Example:
          ```python
          def add_numbers(a, b):
            \"\"\"Adds two numbers and returns the result.\"\"\"
            return a + b
          ```
          Ensure the file is created with this content. Report success and the file path.
          Coordinator Task ID: TASK-CMD-20250428-110000
          </message>
        </new_task>
        ```
    *   Commander then informs you (optional, depends on verbosity): "Okay, I've asked the Python Developer (`dev-python`) to create the `add_numbers` function in `src/utils/math_utils.py`."

*   **Step 4: Specialist Executes (Specialist Internal)**
    *   The `dev-python` mode receives the task via `new_task`.
    *   It reads the instructions.
    *   It consults its own rules and potentially its KB (e.g., for Python style guidelines).
    *   It uses the `write_to_file` tool to create `src/utils/math_utils.py` with the specified content.
    *   It verifies the file was written successfully.

*   **Step 5: Specialist Reports Completion (Specialist -> Commander via `attempt_completion`)**
    *   The `dev-python` mode signals completion back to Commander:
        ```xml
        <thinking>
        Task received from TASK-CMD-20250428-110000.
        Goal: Create `add_numbers` function in `src/utils/math_utils.py`.
        Action: Used `write_to_file` to create the file with the requested function and docstring.
        Verification: Tool reported success.
        Outcome: Success.
        </thinking>
        <attempt_completion>
          <result>
          ✅ Successfully created the function `add_numbers` in the new file `src/utils/math_utils.py`.
          File Content:
          ```python
          def add_numbers(a, b):
            \"\"\"Adds two numbers and returns the result.\"\"\"
            return a + b
          ```
          </result>
        </attempt_completion>
        ```

*   **Step 6: Commander Reports to User (Commander -> User)**
    *   Commander receives the successful completion signal from `dev-python`.
    *   It processes the result and informs you:
        ```prompt
        ✅ The Python Developer (`dev-python`) has successfully created the function `add_numbers` in the new file `src/utils/math_utils.py`.

        You can review the file to see the result.
        ```

## 4. Key Takeaways 💡

*   **Clear Goal:** Providing a clear goal helps Commander choose the right approach and specialist.
*   **Delegation:** Commander orchestrates by delegating implementation details to specialized modes.
*   **Tools:** Modes use tools like `new_task`, `read_file`, `write_to_file`, `apply_diff` to perform actions.
*   **Communication Loop:** The flow typically involves User -> Commander -> Specialist -> Commander -> User.
*   **MDTM vs. Simple Task:** For simple tasks like this, Commander might delegate directly. For more complex features involving multiple steps or specialists, it would likely initiate the MDTM workflow (creating a `.ruru/tasks/TASK-...md` file).

This basic example illustrates the core interaction pattern. As tasks become more complex, Commander will leverage more sophisticated planning, coordination, and tracking mechanisms like MDTM.