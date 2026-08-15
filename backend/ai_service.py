import os
import asyncio
import time
import re
from google import genai
from google.genai import types

_clients = {}

def get_client(api_key: str = None):
    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Please configure it in the application or backend/.env"
        )
    print(f"DEBUG: Using API key starting with {key[:10]}...")
    if key not in _clients:
        _clients[key] = genai.Client(api_key=key)
    return _clients[key]

MODEL_NAME = "gemini-2.5-flash"


# ── Vision OCR ─────────────────────────────────────────────────────────────

async def extract_code_from_image(image_bytes: bytes, media_type: str = "image/png", api_key: str = None) -> str:
    async def _call():
        client = get_client(api_key)
        prompt = "Extract ALL code from this screenshot exactly as written. Output ONLY raw code."
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type=media_type)]
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error in extract_code_from_image: {e}")
            raise e
    return await _call()

# ── Language auto-detect ────────────────────────────────────────────────────

async def detect_language(code: str, api_key: str = None) -> str:
    async def _call():
        client = get_client(api_key)
        prompt = "What programming language is this? Reply with ONLY the language name.\n\n" + code[:1000]
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error in detect_language: {e}")
            raise e
    return await _call()


def get_explain_prompt(code: str, lang: str) -> str:
    lines = "\n".join(f"Line {i+1}: {l}" for i, l in enumerate(code.split("\n")))
    engineer_role = f"senior {lang} engineer" if lang and lang.lower() != "auto" else "senior software engineer"
    return f"""You are a {engineer_role}. Analyze this code.
Output your analysis using ONLY these tagged sections. You MUST include every single section block from DETECTED_LANGUAGE to VIVA_QUESTIONS in your output. Do not omit, merge, rename, or shorten any section. If a section is not applicable to the input code, write a brief explanation of why inside its tags.

CODE:
{lines}

DETECTED_LANGUAGE_START
[Output ONLY the lowercase identifier of the programming language, e.g. python, javascript, typescript, cpp, c, java, rust, go, html, css, ruby, php, etc.]
DETECTED_LANGUAGE_END

LINE_EXPLANATIONS_START
[For EVERY line: Line N: code | Explanation: 1 sentence]
LINE_EXPLANATIONS_END

BUG_DETECTION_START
[List bugs as BUG_1: Line N - ISSUE | FIX. If none, write BUG_NONE]
BUG_DETECTION_END

CORRECTED_CODE_START
[Full corrected code or CLEAN]
CORRECTED_CODE_END

TIME_COMPLEXITY_START
[Big-O and 1 sentence]
TIME_COMPLEXITY_END

SPACE_COMPLEXITY_START
[Big-O and 1 sentence]
SPACE_COMPLEXITY_END

SUGGESTIONS_START
[1-5 actionable tips]
SUGGESTIONS_END

DSA_PATTERN_START
[Identify the primary DSA patterns, algorithms, or data structures used in this code, e.g. Sliding Window, DFS, Hash Table, Stack, Binary Search. Provide 1-2 sentences explaining why this pattern fits the problem.]
DSA_PATTERN_END

PLATFORM_PROBLEMS_START
[Suggest 4-5 related practice problems for this pattern. CRITICAL MANDATE: You MUST include problems from AT LEAST 3 DIFFERENT PLATFORMS (mix of GeeksforGeeks, HackerRank, Codeforces, Coding Ninjas, Codolio, and LeetCode). DO NOT output only LeetCode links. For each problem, output strictly on a new line: Platform | Title | Link (use actual valid problem URLs like https://www.geeksforgeeks.org/..., https://www.hackerrank.com/..., https://codeforces.com/..., https://leetcode.com/...).]
PLATFORM_PROBLEMS_END

OPTIMIZED_CODE_START
[Provide the absolute most optimal version of the analyzed code in terms of time and space complexity. Do NOT use markdown code fences inside this tag.]
OPTIMIZED_CODE_END

OPTIMIZED_TIME_COMPLEXITY_START
[State the exact Big-O time complexity of the optimized code, e.g. O(N) linear time.]
OPTIMIZED_TIME_COMPLEXITY_END

OPTIMIZED_SPACE_COMPLEXITY_START
[State the exact Big-O space complexity of the optimized code, e.g. O(1) constant space.]
OPTIMIZED_SPACE_COMPLEXITY_END

PRACTICE_EXERCISES_START
[Provide 3 practice questions (Beginner, Intermediate, Advanced) that build on this concept. For each, output the difficulty, aim, and a brief sample input/output. Use clean Markdown structure.]
PRACTICE_EXERCISES_END

INTERVIEW_QUESTIONS_START
[Generate 3-5 technical interview questions an interviewer would ask about this code. Focus on edge cases, scaling limits, and structural design choices. List questions clearly. For each question, also generate a detailed, clear answer. Format them strictly as Q: <question> followed by A: <answer> on newlines, just like in VIVA_QUESTIONS.]
INTERVIEW_QUESTIONS_END

ALGORITHM_START
[Provide a formal, step-by-step academic algorithm in English describing how this program executes (e.g. Step 1: Start, Step 2: Initialize variables, etc.).]
ALGORITHM_END

VIVA_QUESTIONS_START
[Generate 5 classic oral exam (Viva Voce) questions that a college professor or lab examiner would ask about this code. For each question, output the Question and the Answer. Format them clearly with Q: and A: on newlines.]
VIVA_QUESTIONS_END"""


def get_dry_run_prompt(code: str, lang: str, test_case: str = None) -> str:
    lines = "\n".join(f"Line {i+1}: {l}" for i, l in enumerate(code.split("\n")))
    engineer_role = f"senior {lang} engineer" if lang and lang.lower() != "auto" else "senior software engineer"
    test_case_instruction = ""
    if test_case:
        test_case_instruction = f"\nTrace the code execution specifically using the following user-provided sample test case/input: {test_case}\n"
        
    return f"""You are a {engineer_role}. Trace the execution of this code.{test_case_instruction}
Output your response using ONLY these tagged sections. You MUST include every single section block from DRY_RUN to RECURSION_TREE in your output.

CODE:
{lines}

DRY_RUN_START
[Provide a step-by-step trace execution of the code as a clean JSON array (no markdown code blocks, output raw valid JSON only). Each object in the array represents a single step in execution and must follow this structure:
{{
  "step": number (1-based index),
  "line": number (line number currently executing),
  "action": "clear humanoid explanation of what happens on this line in 1 short sentence",
  "variables": {{ "variable_name": value, ... }},
  "ds_type": "none" | "array" | "stack" | "queue" | "linkedlist" | "tree" | "graph" | "recursion" | "map" | "set",
  "ds_data": [
     // For array/stack/queue: current list of values, e.g. [12, 45, 9] or ["a", "b"]
     // For linkedlist: nodes in sequence, e.g. [10, 20, 30]
     // For recursion: current function call stack, e.g. ["fib(3)", "fib(2)"]
     // For tree/graph: list of active parent-child relationships, e.g. [["A", "B"], ["A", "C"]]
     // For map: current key-value mapping as a dictionary/object, e.g. {{ "2": 0, "7": 1 }}
     // For set: current unique set elements as a list, e.g. [2, 7]
  ]
}}

CRITICAL VISUALIZATION RULES:
1. If the code uses, traverses, or modifies any array, vector, list, or string (like search, sort, two-sum, reverse, sub-array, etc.), you MUST set "ds_type": "array" and "ds_data" to the current state of that array (e.g. [2, 7, 11, 15] or [0, 1, 1, 0, 1]). You MUST track the pointer indices (like i, j, k, low, mid, high, left, right, slow, fast) inside the "variables" object (e.g. {{"i": 1, "target": 9}}) so the frontend can float pointer arrows above/below the array boxes.
2. If the code uses a map, dictionary, or set (like std::map, unordered_map, set, dict), set "ds_type": "map" or "ds_type": "set". Set "ds_data" to the key-value dictionary (e.g. {{ "2": 0, "7": 1 }}) or set element array (e.g. [2, 7]).
3. End the array with a step representing the final return/output value.]
DRY_RUN_END

FLOWCHART_START
[Generate a clean, valid Mermaid.js control flow diagram (graph TD) showing the execution flow of the code. 
CRITICAL MERMAID SYNTAX RULES:
1. ALWAYS wrap ALL node label texts inside double quotes, e.g. `A["Start"]:::startEnd` or `B["Initialize maxi = 0"]:::default`.
2. NEVER put spaces before `:::` (write `:::default`, NOT ` ::: default`).
3. DO NOT use semicolons `;` at the end of lines.
4. If using subgraphs, ALWAYS wrap the subgraph name in double quotes, e.g. `subgraph "findi(root, &maxi)"`.
5. Label decision paths strictly using `-->|"Yes"|`.

Include these class definitions in the flowchart output to apply our custom color theme:
classDef default fill:#122858,stroke:#3ecfb2,stroke-width:1.5px,color:#f3f5ed;
classDef decision fill:#0f2348,stroke:#c9a84c,stroke-width:1.5px,color:#c9a84c;
classDef startEnd fill:#0c1e3a,stroke:#4a9eff,stroke-width:1.5px,color:#4a9eff;
]
FLOWCHART_END

RECURSION_TREE_START
[If the code uses recursion, generate a clean, valid Mermaid.js graph TD diagram representing the recursion tree of the execution with the actual arguments and values. Use custom color classes. If the code does not use recursion, write RECURSION_NONE.
CRITICAL MERMAID SYNTAX RULES:
1. ALWAYS wrap ALL node label texts in double quotes, e.g. `A["solve(5)"]:::default`.
2. NEVER put spaces before `:::`.
3. If using subgraphs, ALWAYS wrap the title in double quotes, e.g. `subgraph "solve(n)"`.

Include these class definitions in the flowchart output to apply our custom color theme:
classDef default fill:#122858,stroke:#3ecfb2,stroke-width:1.5px,color:#f3f5ed;
classDef decision fill:#0f2348,stroke:#c9a84c,stroke-width:1.5px,color:#c9a84c;
classDef startEnd fill:#0c1e3a,stroke:#4a9eff,stroke-width:1.5px,color:#4a9eff;
]
RECURSION_TREE_END"""


def _extract(text: str, start: str, end: str) -> str:
    try:
        s = text.index(start) + len(start)
        e = text.index(end)
        return text[s:e].strip()
    except Exception as ex:
        import sys
        import traceback
        print(f"Extraction failed for tags {start} -> {end}: {ex}", file=sys.stderr)
        return ""


async def analyze_code(code: str, language: str, test_case: str = None, api_key: str = None) -> dict:
    client = get_client(api_key)
    
    prompt1 = get_explain_prompt(code, language)
    prompt2 = get_dry_run_prompt(code, language, test_case)
    
    async def call_gemini(prompt: str):
        for attempt in range(3):
            try:
                response = await asyncio.to_thread(client.models.generate_content, model=MODEL_NAME, contents=prompt)
                return response.text
            except Exception as e:
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 2:
                    await asyncio.sleep(4 * (attempt + 1))
                else:
                    raise e

    r1, r2 = await asyncio.gather(call_gemini(prompt1), call_gemini(prompt2))
    
    return {
        "detected_language": _extract(r1, "DETECTED_LANGUAGE_START", "DETECTED_LANGUAGE_END"),
        "line_explanations": _extract(r1, "LINE_EXPLANATIONS_START", "LINE_EXPLANATIONS_END"),
        "bug_detection":     _extract(r1, "BUG_DETECTION_START",     "BUG_DETECTION_END"),
        "corrected_code":    _extract(r1, "CORRECTED_CODE_START",    "CORRECTED_CODE_END"),
        "dry_run":           _extract(r2, "DRY_RUN_START",           "DRY_RUN_END"),
        "flowchart":         _extract(r2, "FLOWCHART_START",         "FLOWCHART_END"),
        "recursion_tree":    _extract(r2, "RECURSION_TREE_START",    "RECURSION_TREE_END"),
        "time_complexity":   _extract(r1, "TIME_COMPLEXITY_START",   "TIME_COMPLEXITY_END"),
        "space_complexity":  _extract(r1, "SPACE_COMPLEXITY_START",  "SPACE_COMPLEXITY_END"),
        "suggestions":       _extract(r1, "SUGGESTIONS_START",       "SUGGESTIONS_END"),
        "dsa_pattern":       _extract(r1, "DSA_PATTERN_START",       "DSA_PATTERN_END"),
        "platform_problems": _extract(r1, "PLATFORM_PROBLEMS_START", "PLATFORM_PROBLEMS_END"),
        "optimized_code":    _extract(r1, "OPTIMIZED_CODE_START",    "OPTIMIZED_CODE_END"),
        "optimized_time_complexity": _extract(r1, "OPTIMIZED_TIME_COMPLEXITY_START", "OPTIMIZED_TIME_COMPLEXITY_END"),
        "optimized_space_complexity": _extract(r1, "OPTIMIZED_SPACE_COMPLEXITY_START", "OPTIMIZED_SPACE_COMPLEXITY_END"),
        "practice_exercises":_extract(r1, "PRACTICE_EXERCISES_START", "PRACTICE_EXERCISES_END"),
        "interview_questions":_extract(r1, "INTERVIEW_QUESTIONS_START", "INTERVIEW_QUESTIONS_END"),
        "algorithm":         _extract(r1, "ALGORITHM_START",         "ALGORITHM_END"),
        "viva_questions":    _extract(r1, "VIVA_QUESTIONS_START",    "VIVA_QUESTIONS_END"),
    }


async def refactor_code(code: str, language: str, target_complexity: str, api_key: str = None) -> dict:
    async def _call():
        client = get_client(api_key)
        prompt = f"""You are a senior {language} engineer. Refactor the following code to achieve a target complexity of: {target_complexity}.
If it's already at that complexity or cannot be refactored further, optimize it as much as possible while explaining why.

Output your response using ONLY these tagged sections:

REFACTORED_CODE_START
[Output ONLY the complete, optimized/refactored code. No Markdown code block fences.]
REFACTORED_CODE_END

EXPLANATION_START
[Provide a clear, brief explanation of the changes made and how they achieve the target complexity, or why they could not.]
EXPLANATION_END

Code to refactor:
{code}"""
        try:
            response = await asyncio.to_thread(client.models.generate_content, model=MODEL_NAME, contents=prompt)
            r = response.text
            return {
                "refactored_code": _extract(r, "REFACTORED_CODE_START", "REFACTORED_CODE_END"),
                "explanation": _extract(r, "EXPLANATION_START", "EXPLANATION_END")
            }
        except Exception as e:
            print(f"Error in refactor_code: {e}")
            raise e
    return await _call()


async def generate_comments_for_code(code: str, language: str, api_key: str = None) -> dict:
    async def _call():
        client = get_client(api_key)
        prompt = f"""You are a senior {language} software engineer. Your task is to add sparse, extremely natural, humanoid inline comments and brief docstrings to the code.
Follow these strict rules to keep the comments clean and professional:
1. Do NOT explain what the language syntax does (e.g., do not write `# loop over items` for a loop, or `# initialize variables`).
2. Only add inline comments to explain the "WHY" behind non-obvious logic, tricky mathematical calculations, complex regex, edge case handling, or subtle algorithm choices.
3. Use a casual, expert human developer tone. Write like a real programmer writing a note to their teammate.
4. Keep comments brief and minimal. Limit comments to at most 5-10 short lines across the entire file.
5. The code structure, variables, and behavior must remain 100% identical. Do not add markdown code fences around the output code.

Output your response using ONLY these tagged sections:

COMMENTED_CODE_START
[Output ONLY the complete commented code. Do not use Markdown code block fences.]
COMMENTED_CODE_END

EXPLANATION_START
[Briefly summarize the comments and documentation standards added.]
EXPLANATION_END

Code:
{code}"""
        try:
            response = await asyncio.to_thread(client.models.generate_content, model=MODEL_NAME, contents=prompt)
            r = response.text
            return {
                "commented_code": _extract(r, "COMMENTED_CODE_START", "COMMENTED_CODE_END"),
                "explanation": _extract(r, "EXPLANATION_START", "EXPLANATION_END")
            }
        except Exception as e:
            print(f"Error in generate_comments_for_code: {e}")
            raise e
    return await _call()


async def ask_followup(question: str, code: str, language: str, history: list, api_key: str = None, interview_mode: bool = False) -> str:
    client = get_client(api_key)
    if interview_mode:
        ctx = (
            f"You are a strict, professional technical interviewer from a top-tier tech company.\n"
            f"The candidate has uploaded this code context:\n{code}\n\n"
            "Follow these rules for the conversation:\n"
            "1. Conduct a realistic coding mock interview. Ask technical questions about their choices, edge cases, or potential bottlenecks.\n"
            "2. Keep your questions and responses extremely short and focused (1-2 sentences max). Do not explain solutions or help them unless they ask for hints.\n"
            "3. Evaluate the candidate's answers critically. If they answer incorrectly, guide them with small hints.\n"
            "4. If they ask to wrap up or finish, give them a final constructive score and detailed feedback."
        )
    else:
        ctx = (
            f"You are an expert {language} software engineer assisting a developer.\n"
            f"Here is the active context code:\n{code}\n\n"
            "Follow these rules for your response:\n"
            "1. Be extremely direct, concise, and to-the-point. Answer the user's query immediately.\n"
            "2. Do NOT include conversational filler, greetings, or polite transitions (e.g. 'Sure, I can explain that...', 'Here is the code:').\n"
            "3. Focus on explaining ONLY what was asked. Do not give general tutorials or explain basic syntax unless explicitly requested.\n"
            "4. Keep formatting clean and readable using standard Markdown. Keep paragraphs brief."
        )
    chats = []
    for t in history[-4:]:
        if t.get("question"): chats.append({"role": "user", "parts": [{"text": t["question"]}]})
        if t.get("answer"): chats.append({"role": "model", "parts": [{"text": t["answer"]}]})
    
    try:
        res = client.models.generate_content(
            model=MODEL_NAME,
            contents=chats + [{"role": "user", "parts": [{"text": question}]}],
            config=types.GenerateContentConfig(system_instruction=ctx)
        )
        return res.text.strip()
    except Exception as e:
        print(f"Error in ask_followup: {e}")
        raise e