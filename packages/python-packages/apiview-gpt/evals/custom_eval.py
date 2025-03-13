import json


class CustomAPIViewEvaluator:

    def __init__(self):
        pass

    def __call__(self, *, response: str, query: str, language: str, output: str, **kwargs):
        expected = json.loads(response)
        actual = json.loads(output)

        expected_violations = {v for violation in expected["violations"] for v in violation["rule_ids"]}
        actual_violations = {v for violation in actual["violations"] for v in violation["rule_ids"]}

        matching_violations = len(actual_violations.intersection(expected_violations))

        review_eval = {
            "total_violations": len(expected["violations"]),
            "violations_found": len(actual["violations"]),
            "true_positives": matching_violations,
            "false_positives": len(actual["violations"]) - matching_violations,
            "false_negatives": len(expected["violations"]) - matching_violations,
            "percent_coverage": matching_violations / len(expected["violations"]) * 100,
        }
        return review_eval


def review_apiview(query: str, language: str):
    from src._gpt_reviewer_openai import GptReviewer  # pylint: disable=import-error,no-name-in-module
    rg = GptReviewer()
    review = rg.get_response(query, language)
    return {"response": review.model_dump_json()}
