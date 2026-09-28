# Synthetic Training Data Generation Prompt

Generate high-quality synthetic training examples across the following domains:

* General Knowledge
* Mathematics & Reasoning
* Programming & Software Engineering
* Tool Use & Agentic Tasks
* Science & Engineering
* Translation
* Summarization
* Reading Comprehension & QA
* Danish Culture & Society
* Instruction Following

For each example:

* Choose one domain and a specific subtopic.
* Create a realistic, diverse user instruction or question.
* Provide a correct, useful, self-contained assistant response.
* Vary task difficulty from simple to advanced.
* Vary response length, format, tone, and reasoning requirements.
* Prefer examples that require understanding, problem solving, or instruction following rather than trivial memorization.
* Include both Danish and English examples where appropriate.
* Avoid repetitive templates, near-duplicate questions, and overly artificial wording.
* Ensure facts, calculations, code, and reasoning are accurate.
* Do not mention that the example is synthetic or part of a training dataset.

Output each example as:

```json
{
  "domain": "<domain>",
  "instruction": "<user instruction>",
  "response": "<ideal assistant response>"
}
```

Generate diverse examples with balanced coverage across the specified domains.
