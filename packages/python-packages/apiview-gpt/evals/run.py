import os
import json
import pathlib
import json
import argparse
from typing import Dict, Set, Tuple
import copy

import dotenv
from azure.ai.evaluation import evaluate, SimilarityEvaluator, GroundednessEvaluator

dotenv.load_dotenv()

NUM_RUNS: int = 3

class CustomAPIViewEvaluator:

    def __init__(self):
        pass

    def _get_violation_matches(self, expected: Dict, actual: Dict) -> Tuple[Set, Set, Set]:
        """Compare violations based on both line numbers and rule IDs."""
        exact_matches = set()
        rule_matches_wrong_line = set()
        line_matches_wrong_rule = set()

        violations_left = copy.deepcopy(actual["violations"])
        for expected_violation in expected["violations"]:
            e_line = expected_violation["line_no"]
            e_rules = frozenset(expected_violation["rule_ids"])

            for actual_violation in violations_left:
                a_line = actual_violation["line_no"]
                a_rules = frozenset(actual_violation["rule_ids"])

                rule_match = any(rule for rule in a_rules if rule in e_rules)
                if e_line == a_line and rule_match:
                    exact_matches.add((e_line, tuple(sorted(e_rules))))
                    # Remove the matched actual violation to avoid double counting
                    violations_left.remove(actual_violation)
                    break
                if rule_match:
                    if abs(e_line - a_line) <= 5:
                        # If the line numbers are close, consider it a match
                        rule_matches_wrong_line.add((tuple(sorted(e_rules)), e_line, a_line))
                elif e_line == a_line:
                    line_matches_wrong_rule.add((e_line, tuple(sorted(e_rules)), tuple(sorted(a_rules))))

        return exact_matches, rule_matches_wrong_line, line_matches_wrong_rule

    def __call__(self, *, response: str, query: str, language: str, output: str, **kwargs):
        expected = json.loads(response)
        actual = json.loads(output)

        exact_matches, rule_matches_wrong_line, line_matches_wrong_rule = self._get_violation_matches(expected, actual)

        review_eval = {
            "total_violations": len(expected["violations"]),
            "violations_found": len(actual["violations"]),
            "exact_matches": len(exact_matches),
            "rule_matches_wrong_line": len(rule_matches_wrong_line),
            "line_matches_wrong_rule": len(line_matches_wrong_rule),
            "true_positives": len(exact_matches),
            "false_positives": len(actual["violations"]) - (len(exact_matches) + len(rule_matches_wrong_line)),
            "false_negatives": len(expected["violations"]) - (len(exact_matches) + len(rule_matches_wrong_line)),
            "percent_coverage": (len(exact_matches) / len(expected["violations"]) * 100) if expected["violations"] else 0,
            "wrong_line_details": list(rule_matches_wrong_line),
            "wrong_rule_details": list(line_matches_wrong_rule)
        }
        return review_eval


def review_apiview(query: str, language: str):
    from src._gpt_reviewer_openai import GptReviewer  # pylint: disable=import-error,no-name-in-module
    rg = GptReviewer()
    review = rg.get_response(query, language)
    return {"response": review.model_dump_json()}


def calculate_overall_score(row: dict) -> float:
    """Calculate weighted score based on various metrics.
    """
    weights = {
        'exact_match_weight': 0.6,     # Highest weight - perfect matches
        'rule_match_weight': 0.2,      # Lower weight - right rule, wrong line
        'false_positive_penalty': 0.2, # Penalty for incorrect violations
        'false_negative_penalty': 0.3, # High penalty for missing violations
        'groundedness_weight': 0.15,   # Weight for staying grounded in guidelines
        'similarity_weight': 0.05      # Smaller weight for similarity in responses
    }

    exact_match_score = (row["outputs.custom_eval.exact_matches"] / 
                        row["outputs.custom_eval.total_violations"] 
                        if row["outputs.custom_eval.total_violations"] > 0 else 0.0)

    # Only consider rule matches if there are remaining unmatched violations
    remaining_violations = row["outputs.custom_eval.total_violations"] - row["outputs.custom_eval.exact_matches"]
    rule_match_score = (row["outputs.custom_eval.rule_matches_wrong_line"] / 
                       remaining_violations
                       if remaining_violations > 0 else 0.0)

    # Perfect match case - give full credit for exact + rule match weights
    if exact_match_score == 1.0:
        rule_match_score = 1.0

    false_positive_rate = (row["outputs.custom_eval.false_positives"] / 
                          row["outputs.custom_eval.violations_found"]
                          if row["outputs.custom_eval.violations_found"] > 0 else 0.0)
    
    false_negative_rate = (row["outputs.custom_eval.false_negatives"] / 
                          row["outputs.custom_eval.total_violations"]
                          if row["outputs.custom_eval.total_violations"] > 0 else 0.0)

    groundedness_normalized = (row["outputs.groundedness.groundedness"] - 1) / 4
    similarity_normalized = (row["outputs.similarity.similarity"] - 1) / 4

    score = (
        weights['exact_match_weight'] * exact_match_score +
        weights['rule_match_weight'] * rule_match_score -
        weights['false_positive_penalty'] * false_positive_rate -
        weights['false_negative_penalty'] * false_negative_rate +
        weights['groundedness_weight'] * groundedness_normalized +
        weights['similarity_weight'] * similarity_normalized
    )

    normalized_score = max(0, min(100, score * 100))
    return normalized_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evals.")
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="The language to run evals for.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=NUM_RUNS,
        help="The number of runs to perform, with the best run results kept.",
    )

    parser.add_argument(
        "--test-case",
        type=str,
        default="all",
        help="Only run a particular test case.",
    )
    args = parser.parse_args()

    # needed for AI-assisted evaluation
    model_config: dict[str, str] = {
        "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
        "api_key": os.environ["AZURE_OPENAI_API_KEY"],
        "azure_deployment": "gpt-4o",
        "api_version": "2025-01-01-preview",
    }

    custom_eval = CustomAPIViewEvaluator()
    groundedness = GroundednessEvaluator(model_config=model_config)
    similarity_eval = SimilarityEvaluator(model_config=model_config)

    eval_results = []
    for run in range(args.num):
        # TODO this should look under the path and find all test files, should not be hardcoded
        path = pathlib.Path(__file__).parent / "tests" / args.language / "small.jsonl"
        print(f"Running evals {run + 1}/{args.num}...")

        result = evaluate(
            data=str(path),
            evaluators={
                "custom_eval": custom_eval,
                "similarity": similarity_eval,
                "groundedness": groundedness,
            },
            evaluator_config={
                "similarity": {
                    "column_mapping": {
                        "response": "${target.response}",
                        "query": "${data.query}",
                        "language": "${data.language}",
                        "ground_truth": "${data.response}",
                    },
                },
                "groundedness": {
                    "column_mapping": {
                        "response": "${target.response}",
                        "query": "${data.query}",
                        "language": "${data.language}",
                        "context": "${data.response}",
                    },
                },
                "custom_eval": {
                    "column_mapping": {
                        "response": "${data.response}",
                        "query": "${data.query}",
                        "language": "${data.language}",
                        "output": "${target.response}",
                    },
                },
            },
            target=review_apiview,
            # TODO we can send data to our foundry project for history / more graphical insights
            # azure_ai_project={
            #     "subscription_id": os.environ["AZURE_SUBSCRIPTION_ID"],
            #     "resource_group_name": os.environ["AZURE_FOUNDRY_RESOURCE_GROUP"],
            #     "project_name": os.environ["AZURE_FOUNDRY_PROJECT_NAME"],
            # }
        )

        results_list = []
        total_score = 0

        for row in result["rows"]:
            score = calculate_overall_score(row)
            total_score += score
            
            results_list.append({
                "expected": json.loads(row["inputs.response"]),
                "actual": json.loads(row["outputs.response"]),
                "total_violations": row["outputs.custom_eval.total_violations"],
                "violations_found": row["outputs.custom_eval.violations_found"],
                "exact_matches": row["outputs.custom_eval.exact_matches"],
                "true_positives": row["outputs.custom_eval.true_positives"],
                "false_positives": row["outputs.custom_eval.false_positives"],
                "false_negatives": row["outputs.custom_eval.false_negatives"],
                "percent_coverage": row["outputs.custom_eval.percent_coverage"],
                "rule_matches_wrong_line": row["outputs.custom_eval.rule_matches_wrong_line"],
                "wrong_rule_details": row["outputs.custom_eval.wrong_rule_details"],
                "line_matches_wrong_rule": row["outputs.custom_eval.line_matches_wrong_rule"],
                "wrong_line_details": row["outputs.custom_eval.wrong_line_details"],
                "similarity": row["outputs.similarity.similarity"],
                "groundedness": row["outputs.groundedness.groundedness"],
                "groundedness_reason": row["outputs.groundedness.groundedness_reason"],
                "overall_score": score,
            })

        average_score = total_score / len(result["rows"])
        results_list.append({
            "average_score": average_score,
            "total_evals": len(result["rows"])
        })
        if not eval_results:
            eval_results = results_list
        elif average_score > eval_results[-1]["average_score"]:
            eval_results = results_list

    output_path = pathlib.Path(__file__).parent / "results" / "python" / "small.json"
    with open(str(output_path), "w") as f:
        json.dump(eval_results, indent=4, fp=f)
