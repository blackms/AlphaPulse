+++
# --- Basic Metadata ---
id = "KB-RC-BESTPRACTICE-REVIEWING"
title = "Best Practices: Reviewing AI-Generated Output"
status = "draft"
doc_version = "1.0" # Version of this guide
content_version = 1.0
audience = ["users", "developers", "project_managers", "leads"]
last_reviewed = "2025-04-28" # Use current date
template_schema_doc = ".ruru/templates/toml-md/10_guide_tutorial.README.md" # Using guide template schema
tags = ["roo-commander", "intellimanage", "best-practices", "review", "quality-assurance", "qa", "ai-output", "code-review", "acqa", "guide"]
related_docs = [
    "../README.md", # Link to the KB README
    "../../../../.ruru/processes/acqa-process.md", # Link to the ACQA process
    "01_Effective_Prompting_Delegation.md",
    "03_Troubleshooting_Common_Issues.md"
    ]
difficulty = "intermediate"
estimated_time = "~15 minutes"
prerequisites = ["Receiving output (code, docs, plans) from Roo Commander modes"]
learning_objectives = ["Understand the importance of reviewing AI-generated content", "Learn specific checks for reviewing code, documentation, and plans", "Know how the ACQA process influences review requirements", "Understand how to provide effective feedback for revisions"]
+++

# Best Practices: Reviewing AI-Generated Output

## 1. Introduction / Goal 🎯

Roo Commander modes can generate various outputs, including code, documentation, configuration files, project plans (IntelliManage artifacts), and analysis reports. While AI significantly accelerates development, it's crucial to remember that **AI-generated output requires human review and validation.**

This guide provides best practices for effectively reviewing different types of AI output within the Roo Commander workflow, ensuring quality, correctness, and alignment with project standards.

**Goal:** To equip users with strategies for critically evaluating AI-generated content and providing constructive feedback, ensuring the final output meets project requirements and quality standards.

## 2. Why Review AI Output? 🤔

AI models, even advanced ones, are not infallible. They can:

*   **Hallucinate:** Generate plausible-sounding but incorrect or nonsensical information.
*   **Introduce Bugs:** Write code that contains logical errors, security vulnerabilities, or performance issues.
*   **Misinterpret Requirements:** Fail to fully grasp the nuances of your request or the provided context.
*   **Deviate from Standards:** Produce code or documentation that doesn't adhere to project-specific style guides or conventions.
*   **Lack Context:** Miss broader implications or dependencies not explicitly provided in their immediate context.

**You, the user or designated reviewer (e.g., a Lead mode), are ultimately responsible for the quality and correctness of the final work integrated into the project.**

## 3. The ACQA Process & Confidence Scores 📊

Roo Commander incorporates the **Adaptive Confidence-based Quality Assurance (ACQA)** process (`.ruru/processes/acqa-process.md`).

*   **Confidence Score:** When a specialist mode completes a task, it often provides a confidence score (e.g., Low, Medium, High) indicating how certain it is about the quality and correctness of its output.
*   **Review Level:** The ACQA process uses this score, combined with user preferences (caution level), to suggest an appropriate level of review (e.g., quick scan, detailed review, rigorous testing).
*   **Coordinator Role:** The coordinating mode (`roo-commander`, `roo-dispatch`) typically presents the output along with the confidence score and may suggest involving a Lead mode (e.g., `lead-qa`, `lead-security`) for review based on ACQA guidelines.

**Even with high confidence scores, a baseline level of human review is always recommended.**

## 4. Reviewing Code 💻

When reviewing AI-generated code (e.g., from `dev-*`, `framework-*`, `util-senior-dev`):

*   **Functionality:** Does the code actually do what was requested? Test it against the requirements and acceptance criteria (from the MDTM task file).
*   **Correctness:** Are there logical errors, off-by-one errors, incorrect assumptions, or race conditions?
*   **Edge Cases:** Does it handle edge cases, invalid inputs, and potential errors gracefully?
*   **Style & Conventions:** Does it adhere to project coding standards (linting rules, naming conventions, formatting)?
*   **Security:** Are there any obvious security vulnerabilities (e.g., SQL injection, cross-site scripting, improper handling of secrets)? Consider involving `lead-security` if unsure.
*   **Performance:** Are there inefficient algorithms, unnecessary loops, or potential performance bottlenecks?
*   **Readability & Maintainability:** Is the code clear, well-structured, and easy for other developers to understand and modify?
*   **Tests:** Were appropriate unit, integration, or E2E tests generated (if requested)? Do they pass and provide adequate coverage?
*   **Comments & Documentation:** Is the code adequately commented? Is accompanying documentation (e.g., docstrings) accurate?

## 5. Reviewing Documentation ✍️

When reviewing AI-generated documentation (e.g., from `util-writer`, or docstrings/comments from developers):

*   **Accuracy:** Does the documentation correctly describe the code, feature, or process? Is it technically sound?
*   **Clarity:** Is the language clear, concise, and easy to understand for the target audience? Is jargon used appropriately?
*   **Completeness:** Does it cover all necessary aspects? Are there missing steps, parameters, or explanations?
*   **Consistency:** Is the terminology, tone, and formatting consistent with other project documentation?
*   **Examples:** Are code examples (if included) correct, relevant, and easy to follow?
*   **Formatting:** Is the Markdown (or other format) rendered correctly? Are links working?

## 6. Reviewing Plans & Artifacts 🗺️

When reviewing AI-generated project plans or IntelliManage artifacts (e.g., Epics, Features, Tasks created by `manager-project` or coordinators):

*   **Alignment:** Does the plan or artifact align with the higher-level goals (e.g., does the Feature contribute to the Epic)?
*   **Feasibility:** Is the scope realistic? Are the proposed steps achievable?
*   **Completeness:** Are requirements, descriptions, and acceptance criteria sufficiently detailed? Are dependencies identified?
*   **Clarity:** Is the artifact's purpose and scope clear and unambiguous?
*   **Links:** Are hierarchical links (`epic_id`, `feature_id`) correct? Are necessary `depends_on` links present?
*   **Metadata:** Are fields like `status`, `type`, `priority`, and `tags` appropriate?

## 7. Providing Feedback for Revisions 🗣️

If the AI output requires changes:

*   **Be Specific:** Clearly state *what* is wrong and *why*. Reference specific lines of code, sections of text, or requirements.
*   **Suggest Corrections:** If possible, suggest the desired change or provide an example of the correct approach.
*   **Reference Standards:** Point to specific project guidelines, style guides, or documentation the AI should adhere to.
*   **Instruct the Coordinator:** Provide the feedback to the coordinating mode (`session-manager`, `roo-commander`) and ask it to delegate the revision back to the appropriate specialist, including your specific feedback in the delegation message.
    *   *Example:* "Please ask `dev-react` to revise the component in `src/components/Login.tsx`. The error handling for invalid passwords needs to use the `handleApiError` utility function as per ADR-003, instead of the current `console.error`."

## 8. Conclusion ✅

Reviewing AI-generated output is an integral part of the AI-assisted development workflow. By applying critical thinking, following structured checks based on the output type, understanding the ACQA process, and providing clear feedback, you ensure that the final deliverables meet the required standards of quality, correctness, and security. Treat AI output as a draft or a starting point, not necessarily the final product.