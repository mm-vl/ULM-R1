import re

import torch
import torch.nn.functional as F
from word2number import w2n
from difflib import SequenceMatcher


def extract_answer_text_from_qa(problem: str, ground_truth_letter: str):
    # Match all choices with format "A. text" (case-insensitive, handles extra spaces)
    choices = re.findall(r"^\s*([A-Na-n])\.\s*(.+?)\s*$", problem, re.MULTILINE)

    for letter, text in choices:
        if letter == ground_truth_letter:
            return text.strip()
    return None

def safe_string_equal(prediction, answer):
    """
    Check if the prediction is equal to the answer, even if they are of different types.
    """
    # Convert inputs to lowercase strings for consistent comparison
    prediction_str = str(prediction.rstrip('.')).lower().strip()
    answer_str = str(answer).lower().strip()

    # Direct equality check
    if prediction_str == answer_str:
        return True

    # Check both prediction and answer for word-to-number conversion
    try:
        if prediction_str in w2n.american_number_system:
            if str(w2n.word_to_num(prediction_str)) == answer_str:
                return True

        if answer_str in w2n.american_number_system:
            if prediction_str == str(w2n.word_to_num(answer_str)):
                return True
    except (ImportError, ValueError):
        pass  # Module not available or conversion failed

    return False

def extract_answer_text_from_response(response):
    patterns = [
        # Text answers (for open-ended responses)
        r'(?:answer|choice)(?:\s+is|\s*=|\s*:)\s*"([^"]+)"',  # "answer is "text""
        r'(?:answer|choice)(?:\s+is|\s*=|\s*:)\s*\'([^\']+)\'',  # "answer is 'text'"
        r'(?:correct|right)(?:\s+answer|\s+choice)(?:\s+is|\s*=|\s*:)\s*"([^"]+)"',
        # "correct answer is "text""
        r'(?:correct|right)(?:\s+answer|\s+choice)(?:\s+is|\s*=|\s*:)\s*\'([^\']+)\'',
        # "correct answer is 'text'"
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[0].strip()

    # If no patterns match, return the original response
    return response

def extract_answer_letter_from_response(response):
    patterns = [
        # 选项字母开头或独立形式
        # r'(?:^|\s)([A-Z])(?:\.\s|\.\)|:\s|\)|\s(?:is|as)\s)',  # A. or A) or A: or just A
        r'(?:answer|choice|option|select|choose|pick)\s(?:is|as|being|would be|should be|:)?\s*(?:option\s|choice\s|letter\s)?([A-Z])(?:\b|\.|\))',
        # "The answer is B"
        r'\b(?:select|choose|pick)\s(?:option\s|choice\s|letter\s)?([A-Z])(?:\b|\.|\))',
        # "Choose B"
        # r'(?<=\s)([A-Z])(?:\s|$)',  # Standalone uppercase letter

        # 明确指出答案的模式
        r'[Tt]he answer is ([A-Z])',
        r'[Tt]he correct answer is ([A-Z])',
        r'[Aa]nswer is ([A-Z])',
        r'[Oo]ption ([A-Z])',
        r'[Cc]hoice ([A-Z])'
    ]
    prediction = response

    compiled_patterns = [re.compile(p) for p in patterns]

    for pattern in compiled_patterns:
        matches = pattern.findall(response)
        if matches:
            return matches[0]  # 返回第一个匹配项

    return prediction


# ******************* ******************* *******************
def word_jaccard(phrase1, phrase2):
    words1 = set(phrase1.lower().split())
    words2 = set(phrase2.lower().split())

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    if len(union) == 0:
        return 0
    return len(intersection) / len(union)

def soft_jaccard(phrase1, phrase2, threshold=0.8):
    words1 = set(phrase1.lower().split())
    words2 = set(phrase2.lower().split())

    # hard match
    exact_matches = words1.intersection(words2)

    # soft match
    soft_matches = 0
    matched_words2 = set()

    for w1 in words1 - exact_matches:
        best_ratio = 0
        best_match = None

        for w2 in words2 - exact_matches - matched_words2:
            ratio = SequenceMatcher(None, w1, w2).ratio()
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = w2

        if best_match:
            soft_matches += best_ratio
            matched_words2.add(best_match)

    # Soft Jaccard
    intersection_size = len(exact_matches) + soft_matches
    union_size = len(words1) + len(words2) - intersection_size

    return intersection_size / union_size if union_size else 0


def token_level_max_match_similarity(
        image_tokens: torch.Tensor, text_tokens: torch.Tensor
) -> torch.Tensor:
    """
    Compute symmetric average of max cosine similarity between image and text tokens.

    Args:
        image_tokens: [L_i, d] - token-level features from image
        text_tokens: [L_t, d] - token-level features from text

    Returns:
        scalar tensor: symmetric average max cosine similarity
    """
    # Normalize token features
    image_tokens = F.normalize(image_tokens, p=2, dim=-1)  # [L_i, d]
    text_tokens  = F.normalize(text_tokens,  p=2, dim=-1)  # [L_t, d]

    # Compute cosine similarity matrix: [L_i, L_t]
    sim_matrix = image_tokens @ text_tokens.T

    # max over text tokens for each image token → [L_i]
    max_sim_i2t = sim_matrix.max(dim=1).values

    # max over image tokens for each text token → [L_t]
    max_sim_t2i = sim_matrix.max(dim=0).values

    # symmetric mean
    return 0.5 * (max_sim_i2t.mean() + max_sim_t2i.mean())


def batch_feature_similarity(A, B, aggregated_method="max", top_k=5):
    A_norm = F.normalize(A, p=2, dim=-1)       # [bs, n1, dim]
    B_norm = F.normalize(B, p=2, dim=-1)       # [bs, n2, dim]

    sim_matrix = torch.bmm(A_norm, B_norm.transpose(1, 2))  # [bs, n1, n2]

    sim = None
    if aggregated_method == "max":
        sim, _ = sim_matrix.view(sim_matrix.size(0), -1).max(dim=1)  # [bs]
    elif aggregated_method == "mean":
        sim = sim_matrix.mean(dim=[1, 2])
    elif aggregated_method == "top_k" and top_k is not None:
        topk_sim, _ = torch.topk(sim_matrix.view(A.shape[0], -1), k=top_k, dim=-1)
        sim = topk_sim.mean(dim=-1)

    return sim


def feature_similarity(A, B):
    A_norm = F.normalize(A, p=2, dim=-1)       # [n1, dim]
    B_norm = F.normalize(B, p=2, dim=-1)       # [n2, dim]

    sim_matrix = A_norm @ B_norm.T  # [n1, n2]
    sim = sim_matrix.mean(dim=[1])
    return sim


def compute_attention_similarity(tensor1, tensor2):
    bs, seq_len1, dim = tensor1.shape

    # [bs, 30, 576]
    attention_scores = torch.bmm(tensor2, tensor1.transpose(1, 2)) / (dim ** 0.5)
    attention_weights = F.softmax(attention_scores, dim=-1)

    # [bs, 30, dim]
    context_vectors = torch.bmm(attention_weights, tensor1)

    similarity = F.cosine_similarity(
        tensor2.view(bs, -1),
        context_vectors.view(bs, -1),
        dim=1
    )
    return similarity


# https://github.com/google-research/google-research/blob/master/cmmd/distance.py
def mmd(x, y, _SIGMA=10, _SCALE=1000):
    """
    Args:
      x: The first set of embeddings of shape (n, embedding_dim).
      y: The second set of embeddings of shape (n, embedding_dim).

    Returns:
      The MMD distance between x and y embedding sets.
    """
    # x = torch.from_numpy(x)
    # y = torch.from_numpy(y)

    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))

    gamma = 1 / (2 * _SIGMA**2)
    k_xx = torch.mean(
        torch.exp(-gamma * (
                -2 * torch.matmul(x, x.T)
                + torch.unsqueeze(x_sqnorms, 1)
                + torch.unsqueeze(x_sqnorms, 0)
        ))
    )
    k_xy = torch.mean(
        torch.exp(-gamma * (
                -2 * torch.matmul(x, y.T)
                + torch.unsqueeze(x_sqnorms, 1)
                + torch.unsqueeze(y_sqnorms, 0)
        ))
    )
    k_yy = torch.mean(
        torch.exp(-gamma * (
                -2 * torch.matmul(y, y.T)
                + torch.unsqueeze(y_sqnorms, 1)
                + torch.unsqueeze(y_sqnorms, 0)
        ))
    )
    return _SCALE * (k_xx + k_yy - 2 * k_xy)


# ******************* Task Instruction *******************
OPTIONS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]
MCQ_PROMPT = (
    "{Question}\n\n"
    "Answer the question based on the image and your world knowledge.\n\n"
    "Please write your thinking process inside <think> </think> tags, and provide your final answer (option letter, e.g., A/B/C/D) inside <answer> </answer> tags.\n"
    "Your response MUST strictly follow this format: <think> ... </think><answer>option letter</answer>"
)
VQA_PROMPT = (
    "{Question}\n\n"
    "Answer the question based on the image and your world knowledge.\n\n"
    "Please write your thinking process inside <think> </think> tags, and provide your final answer (only 1–3 words) inside <answer> </answer> tags.\n"
    "Your response MUST strictly follow this format: <think> ... </think><answer>concise answer</answer>"
)
CLS_PROMPT = (
    "Identify the species of the most prominent {Category} in the image.\n\n"
    "Please write your thinking process inside <think> </think> tags, and provide your final answer inside <answer> </answer> tags.\n"
    "Your response MUST strictly follow this format: <think> ... </think> <answer>species name</answer>"
)
OD_PROMPT = (
    "Detect all objects belonging to the category '{Category}' in the image, and provide the bounding boxes (between 0 and 384, integer) and confidence (between 0 and 1, with two decimal places).\n\n"
    "Please write your thinking process inside <think> </think> tags, and provide your final answer inside <answer> </answer> tags.\n"
    "Your response MUST strictly follow this format: <think> ... </think> <answer>[{{'Position': [x1, y1, x2, y2], 'Confidence': number}}, ...]</answer>"
)

SYSTEM_PROMPT_Janus = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language."
)
# ******************* ******************* *******************


from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import math
def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward

