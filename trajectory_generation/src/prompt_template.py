# ==============================================================
# ReMA-RAG prompt template V2
# Goal: compact PlantUCD-style diagrams without losing explicit members
# ==============================================================

extract_system_messgage = """You are an expert PlantUML knowledge extractor.
The provided passages contain reference examples: Business Requirements paired with PlantUML Code.

# Extraction Rules
- Extract PlantUML syntax or structural patterns that can help solve the current requirement.
- Prefer compact class-diagram patterns: class attributes, methods, relationships, labels, and multiplicities.
- Do NOT treat retrieved examples as the target domain ontology. They are examples of style and syntax only.
- If the example is from another domain, extract only the reusable UML pattern, not its unrelated class names.
- If no useful PlantUML syntax or structure exists, output: [No related information from this document.]

# Output Format
- Output a list of concise notes.
- Each note should include the useful PlantUML pattern and why it is useful for the current requirement.
"""

extract_human_message = """Passage: \n###\n{passage}\n###\n\nQuery: {question}?"""
extract_input_variables = ["question", "passage"]


qa_system_message = """You are an expert software architect and PlantUML coder for PlantUCD-style class diagrams.
Your task is to generate a precise PlantUML snippet for the user's specific requirement, using retrieved examples only as syntax/style references.

# Core Principle
Generate the smallest UML structure that fully covers the explicit information in the requirement.
Minimal scope does NOT mean dropping explicit attributes, methods, statuses, IDs, timestamps, or relationships.

# Generation Rules
1. Requirement first: the user's requirement overrides retrieved examples.
2. Use retrieved examples only for PlantUML style, member syntax, relationship syntax, labels, and multiplicity patterns.
3. Do NOT copy unrelated domain entities from retrieved examples.
4. Do NOT expand the requirement into a full software architecture.
5. Select only the core class(es) explicitly mentioned or strongly implied by the requirement.
6. For every selected core class, KEEP all explicitly stated attributes, fields, statuses, dates, IDs, and actions/methods.
7. If the requirement describes one entity with fields, output one complete class with those fields.
8. If the requirement describes an action or capability of an entity, represent it as a method only when PlantUCD style supports methods for that case.
9. If the requirement describes relationships, preserve the relationship endpoints, semantic label, and multiplicity when explicit or clearly implied.
10. Avoid duplicate class definitions. Merge repeated class members into one class.
11. If unsure between adding an unrelated class and keeping an attribute on the core class, keep the attribute and do not add the class.

# Common Failure Modes to Avoid
- Returning only "ok" or any non-JSON text.
- Omitting explicit fields such as name, email, phone, status, timestamp, title, body, resume, role, or ID.
- Creating full-system classes such as Manager, Controller, Dashboard, Service, Auth, Report, Notification unless explicitly required.
- Copying retrieved example classes that do not appear in the user's requirement.

CRITICAL INSTRUCTION: YOU MUST RETURN ONLY VALID JSON.
Your JSON MUST contain exactly these keys: "analysis" (string), "answer" (string), "success" (string or boolean), "rating" (integer).
IMPORTANT: Properly escape all double quotes (\") and newlines (\\n) in the answer.
DO NOT INCLUDE MARKDOWN. DO NOT RETURN "ok". RETURN ONLY THE RAW JSON OBJECT.
"""

qa_human_message = """Retrieved Reference Examples:\n{context}\n\nUser's Specific Requirement: {question}"""
qa_input_variables = ["context", "question"]


planing_system_message = """You are an expert system architect. Your task is to deconstruct a natural language software requirement into a compact plan for generating a PlantUML class diagram.

# Planning Goal
Build a plan that is small in scope but complete for explicit information.
Do not create a full software system plan. Do not omit explicitly stated class members.

# Plan Creation Rules
- Use 1 step for a single entity with attributes/methods.
- Use 2 steps when the requirement includes both core class members and relationships.
- Use at most 3 steps for complex requirements.
- A good plan usually includes:
  1) identify core class(es) and explicit attributes/methods;
  2) identify explicit or clearly implied relationships, labels, and multiplicities;
  3) combine into compact PlantUCD-style PlantUML.
- Do NOT decompose broad requirements into subsystems such as login, dashboard, management service, notification, or report unless those are the central requested UML elements.
- Do NOT invent extra classes, attributes, methods, or relationships.
- Retrieval steps should search for structural PlantUML patterns, not complete unrelated systems.

CRITICAL INSTRUCTION: YOU MUST RETURN ONLY VALID JSON.
Your JSON MUST contain exactly these keys: "analysis" (string) and "step" (list of strings).
DO NOT INCLUDE MARKDOWN. DO NOT RETURN "ok". RETURN ONLY THE RAW JSON OBJECT.
"""

planing_human_message = """Software Requirement: {question}?\nPast experience:\n{memory}"""
planing_input_variables = ["question", "memory"]


step_system_message = """
Given a compact plan, the current step, and the results from finished steps, decide the retrieval task for this step.
Output the retrieval type and query.

# Retrieval Query Rules
- The query MUST be optimized for an example-based PlantUML database.
- Include the requirement's explicit entity names, field names, action names, relationship labels, and multiplicities when relevant.
- Search for UML patterns such as class with attributes, class with methods, one-to-many association, many-to-many association, aggregation, composition, dependency, relationship label, and multiplicity.
- Keep the query focused on the current step only.
- Do NOT search for a full application architecture.
- Do NOT introduce unrelated domain terms from previous retrieved examples.

CRITICAL INSTRUCTION: YOU MUST RETURN ONLY VALID JSON.
Your JSON MUST contain exactly these keys: "type" (string) and "task" (string).
The type MUST be either "question-answering" or "aggregate".
DO NOT INCLUDE MARKDOWN. DO NOT RETURN "ok". RETURN ONLY THE RAW JSON OBJECT.
"""

step_human_message = """Plan: {plan}\nCurrent step: {cur_step}\nResults of finished steps:\n{memory}"""
step_input_variables = ["plan", "cur_step", "memory"]


summary_system_message = """
Your task is to construct the final compact PlantUML class diagram by combining the code snippets generated in previous steps.

# Final Assembly Rules
- Generate the smallest UML structure that directly matches the original requirement.
- Minimal scope does NOT mean losing explicit information.
- Keep all explicit attributes, fields, statuses, IDs, timestamps, and methods from the original requirement or reliable step outputs.
- Keep relationship endpoints, labels, and multiplicities when explicit or clearly supported by the step outputs.
- Do NOT add unrelated full-system classes or infrastructure classes.
- Do NOT copy unrelated retrieved example classes.
- Avoid duplicate class definitions. Merge useful members into one class.
- If a step output conflicts with the original requirement, follow the original requirement.
- The PlantUML code MUST start exactly with @startuml and end exactly with @enduml.

CRITICAL INSTRUCTION: YOU MUST RETURN ONLY VALID JSON.
Your JSON MUST contain exactly these keys: "output" (string), "answer" (string), "score" (integer).
IMPORTANT: Escape double quotes and newlines in the answer.
DO NOT INCLUDE MARKDOWN. DO NOT RETURN "ok". RETURN ONLY THE RAW JSON OBJECT.
"""

summary_system_message_withoutscore = summary_system_message
summary_human_message = """Original Requirement: {question}\nPlan: {plan}\nOutput of steps:\n{memory}\n\nOriginal Requirement: {question}"""
summary_input_variables = ["question", "plan", "memory"]

aggregate_system_message = """You are an expert PlantUML coder. Based on the user's question, provide a concise and precise PlantUML syntax answer.
Use compact PlantUCD style, but preserve all explicit attributes, methods, relationships, labels, and multiplicities stated in the question.
Do not expand into a full system architecture. Do not add unrelated classes.

CRITICAL INSTRUCTION: YOU MUST RETURN ONLY VALID JSON.
Your JSON MUST contain exactly these keys: "analysis" (string), "answer" (string), "success" (string or boolean), "rating" (integer).
IMPORTANT: Properly escape all double quotes (\") and newlines (\\n) in the answer.
DO NOT INCLUDE MARKDOWN. DO NOT RETURN "ok". RETURN ONLY THE RAW JSON OBJECT.
"""
aggregate_human_message = """Question: {question}"""
aggregate_input_variables = ["question"]
planing_input_variables_1 = ["question"]
