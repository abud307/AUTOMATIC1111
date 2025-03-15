"""
Txt2Img Prompt Optimizer (Multilingual)

This script optimizes text prompts for Stable Diffusion image generation.
It can detect non-English prompts, translate them to English, and then optimize them
for better image generation results.

The script uses a LangGraph workflow to manage the optimization process, with nodes for
language detection, translation, and optimization. If LangGraph is not available,
it falls back to a simplified workflow.
"""

from modules import scripts
from modules.processing import StableDiffusionProcessingTxt2Img
import os
from dotenv import load_dotenv
import requests
from typing import Dict, Literal, TypedDict, Optional, Any

# Try to import LangGraph related libraries
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("Warning: LangGraph library not installed, using simplified implementation")
    print("Can be installed via 'pip install langgraph'")

# Try to import Pydantic
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("Warning: Pydantic library not installed, using simplified implementation")
    print("Can be installed via 'pip install pydantic'")

# Load environment variables
load_dotenv()

# Get DeepSeek API key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Define state type
class PromptState(TypedDict):
    original_prompt: str
    language: str
    translated_prompt: Optional[str]
    optimized_prompt: Optional[str]
    error: Optional[str]

class PromptTemplate(BaseModel):
    """Prompt template for specific tasks"""
    name: str = Field(..., description="Template name")
    content: str = Field(..., description="Template content")

    def __str__(self) -> str:
        return self.content.strip()

class PromptTemplates(BaseModel):
    """Collection of prompt templates"""
    txt2img_optimizer: PromptTemplate = Field(
        default=PromptTemplate(
            name="Stable Diffusion Prompt Optimizer",
            content="""\
            You are an expert prompt engineer for Stable Diffusion image generation with deep knowledge of how SD models interpret text.

            Your task is to transform standard prompts into highly optimized versions that produce exceptional quality images. Follow these guidelines:

            1. Maintain the original subject and core concept
            2. Enhance with precise descriptive adjectives and specific details
            3. Add appropriate artistic style references (artists, movements, platforms)
            4. Incorporate quality-boosting terms (masterpiece, best quality, highly detailed)
            5. Apply technical enhancements through brackets for emphasis:
            - Use (term) for 1.1x emphasis
            - Use ((term)) for 1.2x emphasis
            - Use [term] for 0.9x emphasis
            - Use [[term]] for 0.8x emphasis
            - Use :1.x for specific weighting

            6. Structure prompts effectively:
            - Main subject first with strongest emphasis
            - Scene details and environment
            - Style, quality, and technical terms last

            Return ONLY the optimized prompt without explanations or commentary. Preserve all special formatting like (), [], {}, :1.2, etc. from the original prompt.
            """
        ),
        description="Stable Diffusion prompt optimization template"
    )

    language_detector: PromptTemplate = Field(
        default=PromptTemplate(
            name="Language Detector",
            content="""\
            You are a language detection expert. Your task is to identify if the given text is in English or not.

            Analyze the provided text and determine if it's in English. Return ONLY 'yes' if the text is primarily in English, or 'no' if it's primarily in another language.

            If the text is primarily in English or contains mostly English words with a few non-English terms, return 'yes'.
            If the text is primarily in another language, return 'no'.

            Return ONLY 'yes' or 'no' without any explanations or additional text.
            """
        ),
        description="Language detection template"
    )

    universal_translator: PromptTemplate = Field(
        default=PromptTemplate(
            name="Universal Translator",
            content="""\
            You are a professional translator specializing in translating text to English for image generation.

            Your task is to accurately translate prompts from any language to English while preserving the original meaning and intent. Follow these guidelines:

            1. Maintain the core subject and concept of the original prompt
            2. Preserve any special formatting like (), [], {}, :1.2, etc.
            3. Translate cultural-specific terms appropriately for an international audience
            4. Keep artistic style references intact
            5. Ensure the translation is natural and fluent in English

            Return ONLY the translated English prompt without explanations or commentary.
            """
        ),
        description="Universal translation template"
    )

    def get(self, template_name: str) -> PromptTemplate:
        """Get template by name"""
        if hasattr(self, template_name):
            return getattr(self, template_name)
        raise ValueError(f"Template not found: {template_name}")

# Create template instance
TEMPLATES = PromptTemplates()


# Helper function for simple language detection
def simple_language_detection(prompt: str) -> str:
    """Simple language detection based on ASCII character ratio"""
    if not prompt:
        return "unknown"

    non_ascii_chars = 0
    for char in prompt:
        if ord(char) > 127:
            non_ascii_chars += 1

    language = "english" if (non_ascii_chars / len(prompt) < 0.3) else "other"
    print(f"Simple language detection: Prompt '{prompt}' detected as '{'English' if language == 'english' else 'Non-English'}'")
    return language

# Agent functions
def router_agent(state: PromptState) -> Dict[str, Any]:
    """Determine the language of the prompt"""
    prompt = state["original_prompt"]

    if not prompt:
        return {"language": "unknown"}

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }

        # Use predefined language detection template
        detector_template = TEMPLATES.get("language_detector")

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": detector_template.content},
                {"role": "user", "content": f"Is this text in English? {prompt}"}
            ],
            "temperature": 0.1,
            "max_tokens": 10
        }

        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            result = response.json()
            is_english = result["choices"][0]["message"]["content"].strip().lower() == "yes"
            language = "english" if is_english else "other"
            print(f"RouterAgent: Prompt '{prompt}' detected as '{'English' if language == 'english' else 'Non-English'}'")
            return {"language": language}
        else:
            print(f"RouterAgent: Language detection failed - {response.status_code} - {response.text}")
            # Fallback to simple detection
            language = simple_language_detection(prompt)
            return {"language": language}
    except Exception as e:
        print(f"RouterAgent: Language detection failed - {str(e)}")
        # Fallback to simple detection
        language = simple_language_detection(prompt)
        return {"language": language}

def translator_agent(state: PromptState) -> Dict[str, Any]:
    """Translate non-English prompts to English"""
    prompt = state["original_prompt"]
    language = state["language"]

    if language == "english":
        print("TranslatorAgent: Prompt is already in English, no translation needed")
        return {"translated_prompt": prompt}

    if not DEEPSEEK_API_KEY:
        print("TranslatorAgent: Warning - DEEPSEEK_API_KEY not set, using simplified translation")
        return {"error": "DEEPSEEK_API_KEY not set", "translated_prompt": prompt}

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }

        # Use predefined universal translation template
        translator_template = TEMPLATES.get("universal_translator")

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": translator_template.content},
                {"role": "user", "content": f"Translate this prompt from {language} to English: {prompt}"}
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }

        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            result = response.json()
            translated_text = result["choices"][0]["message"]["content"].strip()
            print(f"TranslatorAgent: Translation result - '{translated_text}'")
            return {"translated_prompt": translated_text}
        else:
            print(f"TranslatorAgent: Translation failed - {response.status_code} - {response.text}")
            return {"error": f"Translation API error: {response.status_code}", "translated_prompt": prompt}
    except Exception as e:
        print(f"TranslatorAgent: Translation failed - {str(e)}")
        return {"error": f"Translation error: {str(e)}", "translated_prompt": prompt}

def optimizer_agent(state: PromptState) -> Dict[str, Any]:
    """Optimize English prompts"""
    # Determine the prompt to optimize
    prompt_to_optimize = state.get("translated_prompt") or state["original_prompt"]

    if not DEEPSEEK_API_KEY:
        print("OptimizerAgent: Warning - DEEPSEEK_API_KEY not set, using local optimization")
        optimized = local_optimize(prompt_to_optimize)
        return {"optimized_prompt": optimized}

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }

        # Use predefined optimization template
        optimizer_template = TEMPLATES.get("txt2img_optimizer")

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": optimizer_template.content},
                {"role": "user", "content": f"Optimize this prompt: {prompt_to_optimize}"}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }

        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            result = response.json()
            enhanced_text = result["choices"][0]["message"]["content"].strip()
            print(f"OptimizerAgent: Optimization result - '{enhanced_text}'")
            return {"optimized_prompt": enhanced_text}
        else:
            print(f"OptimizerAgent: Optimization failed - {response.status_code} - {response.text}")
            optimized = local_optimize(prompt_to_optimize)
            return {"error": f"Optimization API error: {response.status_code}", "optimized_prompt": optimized}
    except Exception as e:
        print(f"OptimizerAgent: Optimization failed - {str(e)}")
        optimized = local_optimize(prompt_to_optimize)
        return {"error": f"Optimization error: {str(e)}", "optimized_prompt": optimized}

def local_optimize(prompt: str) -> str:
    """Local prompt optimization method (used when API is unavailable)"""
    # Example optimization: add quality-boosting keywords
    quality_terms = ["high quality", "detailed", "sharp focus"]
    style_terms = ["masterpiece", "best quality"]

    # Check if prompt already contains these terms
    optimized = prompt

    # Add quality terms
    for term in quality_terms:
        if term.lower() not in optimized.lower():
            if optimized.strip().endswith(('，', '。', ',', '.')):
                optimized = f"{optimized} {term}"
            else:
                optimized = f"{optimized}, {term}"

    # Add style terms (at the beginning)
    for term in reversed(style_terms):
        if term.lower() not in optimized.lower():
            optimized = f"{term}, {optimized}"

    print(f"OptimizerAgent: Local optimization result - '{optimized}'")
    return optimized

# Define routing logic
def should_translate(state: PromptState) -> Literal["translator", "optimizer"]:
    """Determine if translation is needed"""
    if state.get("language", "") != "english":
        return "translator"
    else:
        return "optimizer"

# Create LangGraph workflow
def create_prompt_optimization_graph():
    """Create prompt optimization workflow graph"""
    # If LangGraph is not available, return None
    if not LANGGRAPH_AVAILABLE:
        return None

    # Create state graph
    graph = StateGraph(PromptState)

    # Add nodes
    graph.add_node("router", router_agent)
    graph.add_node("translator", translator_agent)
    graph.add_node("optimizer", optimizer_agent)

    # Add edges
    # From start to router
    graph.set_entry_point("router")

    # From router to translator or optimizer (based on language)
    graph.add_conditional_edges(
        "router",
        should_translate,
        {
            "translator": "translator",
            "optimizer": "optimizer"
        }
    )

    # From translator to optimizer
    graph.add_edge("translator", "optimizer")

    # From optimizer to end
    graph.add_edge("optimizer", END)

    # Compile workflow
    return graph.compile()

# Simplified workflow (used when LangGraph is not available)
def simple_prompt_optimization_workflow(prompt: str) -> str:
    """Simplified prompt optimization workflow"""
    print("\n--- Simplified workflow started ---")
    print(f"Original prompt: '{prompt}'")

    # Initialize state
    state = PromptState(
        original_prompt=prompt,
        language="unknown",
        translated_prompt=None,
        optimized_prompt=None,
        error=None
    )

    # Step 1: Router - determine language
    router_result = router_agent(state)
    state["language"] = router_result["language"]

    # Step 2: Translator - translate if not English
    if state["language"] != "english":
        translator_result = translator_agent(state)
        state["translated_prompt"] = translator_result.get("translated_prompt")
        if "error" in translator_result:
            state["error"] = translator_result["error"]

    # Step 3: Optimizer - optimize prompt
    optimizer_result = optimizer_agent(state)
    state["optimized_prompt"] = optimizer_result.get("optimized_prompt")
    if "error" in optimizer_result and not state["error"]:
        state["error"] = optimizer_result["error"]

    print(f"Final optimized prompt: '{state['optimized_prompt']}'")
    print("--- Simplified workflow finished ---\n")

    return state["optimized_prompt"] or prompt

class PromptOptimizer(scripts.Script):
    # Class-level flag to track if initialization message has been shown
    _init_message_shown = False

    def __init__(self):
        super().__init__()
        # Show initialization message only once
        if not PromptOptimizer._init_message_shown:
            print("\n\n=== Txt2Img Prompt Optimizer (Multilingual) script loaded ===\n\n")
            PromptOptimizer._init_message_shown = True

        # Try to create LangGraph workflow
        self.graph = create_prompt_optimization_graph()

        # If LangGraph is not available, use simplified workflow
        if self.graph is None and not PromptOptimizer._init_message_shown:
            print("Using simplified prompt optimization workflow")

        # Track processed prompts to avoid duplicates
        self.processed_prompts = set()

    def title(self):
        return "Txt2Img Prompt Optimizer (Multilingual)"

    # Return AlwaysVisible to show script in UI
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # No UI elements needed
    def ui(self, is_img2img):
        return []

    # Optimize prompt before processing
    def process(self, p):
        # Only optimize Txt2Img processing objects
        if not isinstance(p, StableDiffusionProcessingTxt2Img):
            return p

        # Record original prompt
        original_prompt = p.prompt
        print(f"\n=== Original prompt ===\n{original_prompt}\n")

        # Optimize main prompt (if not already processed)
        if p.prompt not in self.processed_prompts:
            optimized_prompt = self.optimize_prompt(p.prompt)
            p.prompt = optimized_prompt
            # Ensure all_prompts also uses optimized prompt
            if hasattr(p, 'all_prompts') and p.all_prompts:
                p.all_prompts = [optimized_prompt] * len(p.all_prompts)
            # Ensure main_prompt also uses optimized prompt
            if hasattr(p, 'main_prompt'):
                p.main_prompt = optimized_prompt
            self.processed_prompts.add(optimized_prompt)

        # Record optimization information (optional, for verification)
        if not hasattr(p, 'extra_generation_params'):
            p.extra_generation_params = {}
        p.extra_generation_params['Prompt optimized'] = True

        # Record final prompt sent to model
        print(f"\n=== Final prompt sent to model ===\n{p.prompt}\n")

        # Add post-processing hook to ensure prompt remains optimized
        original_setup_prompts = p.setup_prompts

        def patched_setup_prompts():
            # Call original method
            original_setup_prompts()
            # Ensure prompt remains optimized
            if p.prompt in self.processed_prompts:
                p.all_prompts = [p.prompt] * len(p.all_prompts)
                p.main_prompt = p.prompt

        # Replace method
        p.setup_prompts = patched_setup_prompts

        return p

    def postprocess(self, p, processed):
        """Post-process after image generation"""
        # Add original prompt to extra generation params
        if hasattr(self, 'extra_generation_params') and hasattr(self, 'main_prompt'):
            processed.infotexts[0] = processed.infotexts[0].replace(
                "Prompt: ", f"Prompt: {self.extra_generation_params.get('Original prompt', '')}\nOptimized: "
            )
        # Nothing to do here
        return processed

    def optimize_prompt(self, prompt: str) -> str:
        """Optimize a prompt using the workflow"""
        if not prompt:
            return prompt

        # Use LangGraph workflow or simplified workflow
        if self.graph is not None:
            # Use LangGraph workflow
            try:
                print("\n--- LangGraph started ---")
                print(f"Original prompt: '{prompt}'")

                # Create initial state
                initial_state = PromptState(
                    original_prompt=prompt,
                    language="unknown",
                    translated_prompt=None,
                    optimized_prompt=None,
                    error=None
                )

                # Execute workflow
                final_state = self.graph.invoke(initial_state)

                optimized = final_state.get("optimized_prompt") or prompt
                print(f"Final optimized prompt: '{optimized}'")
                print("--- LangGraph finished ---\n")
                return optimized
            except Exception as e:
                print(f"LangGraph workflow error: {str(e)}")
                print("Falling back to simplified workflow")
                return simple_prompt_optimization_workflow(prompt)
        else:
            # Use simplified workflow
            return simple_prompt_optimization_workflow(prompt)

# For standalone testing
if __name__ == "__main__":
    # Test the prompt optimization workflow
    test_prompts = [
        "a beautiful landscape with mountains",  # English
        "美丽的山水画",  # Chinese: "beautiful landscape painting"
    ]

    print("Testing prompt optimization workflow...")

    # Initialize optimizer
    optimizer = PromptOptimizer()

    # Test each prompt
    for prompt in test_prompts:
        print(f"\nTesting prompt: '{prompt}'")
        optimized = optimizer.optimize_prompt(prompt)
        print(f"Optimized: '{optimized}'")
