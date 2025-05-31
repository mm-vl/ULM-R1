from .r_utils import *


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # completion_contents = [completion[0]["content"] for completion in completions]

    # pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    pattern = re.compile(r"<think>.*?</think>[.\s]*<answer>.*?</answer>", re.DOTALL)

    matches = [pattern.fullmatch(content) for content in completions]
    return [1.0 if match else 0.0 for match in matches]


def format_tag_reward(completions, **kwargs):
    # responses = [completion[0]["content"] for completion in completions]
    return [think_mark_num(response) for response in completions]


def think_mark_num(text):
    reward = 0
    if text.count("<think>") == 1:
        reward += 0.125

    if text.count("</think>") == 1:
        reward += 0.125

    if text.count("<answer>") == 1:
        reward += 0.125

    if text.count("</answer>") == 1:
        reward += 0.125
    return reward


# ******************* ******************* *******************
def common_qa_accuracy_reward(completions, qa_solution, **kwargs):
    # contents = [completion[0]["content"] for completion in completions]

    rewards = []
    for idx, (content, sol) in enumerate(zip(completions, qa_solution)):
        content = str(content)
        sol = str(sol)

        reward = 0.0
        # Extract GT answer from solution if it has think/answer tags [always satisfy]
        sol_match = re.search(r"<answer>(.*?)</answer>", sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

        # Extract predicted answers from content if it has think/answer tags
        content_match = re.search(r"<answer>(.*?)</answer>", content)
        if content_match:
            student_answer = content_match.group(1).strip()
        else:
            # remove <think>.*?</think>
            student_answer = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            # student_answer = re.sub(r'<think>.*', '', student_answer, flags=re.DOTALL)
            # student_answer = re.sub(r'.*</think>', '', student_answer, flags=re.DOTALL)
            student_answer = student_answer.replace(
                '<think>', '').replace('</think>', '').replace(
                '<answer>', '').replace('</answer>', '').replace(
                'Answer:', '').replace('Answer', '').strip()

        if safe_string_equal(student_answer, ground_truth):
            reward = 1.0
        else:
            if kwargs['qa_type'] == 'MC':  # FIXME: for multi-choice qa
                if len(student_answer) <= 2:  # prediction is option letter (A | A. | A:)
                    if safe_string_equal(student_answer, ground_truth):
                        reward = 1.0
                else:  # e.g., <option letter + answer text>, <answer text>, <The answer is xx>
                    # correct letter in prediction, but others may also be in
                    if bool(re.search(rf'\b{re.escape(ground_truth)}\b', student_answer)):
                        student_answer = extract_answer_letter_from_response(student_answer)
                        if len(student_answer) <= 2:  # option letter
                            if safe_string_equal(student_answer, ground_truth):
                                reward = 1.0
                    else:
                        gt_text = extract_answer_text_from_qa(kwargs['qa_problem'][idx], ground_truth)
                        if bool(re.search(rf'\b{re.escape(gt_text)}\b', student_answer)):
                            student_answer = extract_answer_text_from_response(student_answer)
                            if safe_string_equal(student_answer, gt_text):
                                reward = 1.0
            else:
                # reward = word_jaccard(student_answer.rstrip('.,!?'), ground_truth)
                reward = soft_jaccard(student_answer.rstrip('.,!?'), ground_truth)

        rewards.append(reward)
        # print(f"{'=' * 160}")
        # print(f"------------- Accuracy reward: {reward} -------------\n")
        # print(f"Prompt: {kwargs['prompts'][idx]}\n")
        # print(f"Content: {content}\n")
        # print(f"Solution: {sol}\n")
        # print(f"{'=' * 160}")

    return rewards


def common_cls_accuracy_reward(completions, cls_solution, **kwargs):
    rewards = []
    for idx, (content, sol) in enumerate(zip(completions, cls_solution)):
        content = str(content)
        sol = str(sol)

        reward = 0.0

        # Extract GT answer from solution if it has think/answer tags [always have]
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()  # word

        # Extract answer from content if it has think/answer tags
        content_match = re.search(r'<answer>(.*?)</answer>', content)
        student_answer = content_match.group(1).strip() if content_match else content.strip()
        student_answer = student_answer.lower()

        # Compare the extracted answers
        if student_answer == ground_truth:
            reward = 1.0

        elif ground_truth in student_answer or student_answer in ground_truth:
            reward = 0.9

        # if content_match:
        #     student_answer = content_match.group(1).strip()
        #     student_answer = student_answer.lower()
        #     # Compare the extracted answers
        #     if student_answer == ground_truth:
        #         reward = 1.0
        #
        #     elif ground_truth in student_answer:
        #         reward = 0.8
        # else:
        #     student_answer = content.strip()
        #     student_answer = student_answer.replace(
        #         '<think>', '').replace('</think>', '').replace(
        #         '<answer>', '').replace('</answer>', '').replace(
        #         'Answer:', '').replace('Answer', '').strip()
        #     student_answer = student_answer.lower()
        #
        #     # rouge = metric_F.text.rouge.rouge_score(
        #     #     student_answer, ground_truth, use_stemmer=True, rouge_keys='rougeL'
        #     # )["recall"]
        #
        #     if student_answer == ground_truth:
        #         reward = 1.0
        #     elif ground_truth in student_answer:
        #         reward = 0.8

        rewards.append(reward)
    return rewards



