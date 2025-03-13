import os
import json
import pathlib

import dotenv
from azure.ai.evaluation import evaluate, SimilarityEvaluator

from custom_eval import CustomAPIViewEvaluator, review_apiview

dotenv.load_dotenv()

# needed for SimilarityEvaluator which is an AI-assisted evaluation
model_config: dict[str, str] = {
    "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
    "api_key": os.environ["AZURE_OPENAI_API_KEY"],
    "azure_deployment": "gpt-4o",
    "api_version": "2025-01-01-preview",
}

custom_eval = CustomAPIViewEvaluator()
similarity_eval = SimilarityEvaluator(model_config=model_config)


if __name__ == "__main__":
    path = pathlib.Path(__file__).parent / "tests" / "python.jsonl"
    result = evaluate(
        data=str(path),
        evaluators={
            "custom_eval": custom_eval,
            "similarity": similarity_eval,
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

    with open("./results/python_result.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "expected": json.loads(result["rows"][0]["inputs.response"]),
                    "actual": json.loads(result["rows"][0]["outputs.response"]),
                    "total_violations": result["rows"][0]["outputs.custom_eval.total_violations"],
                    "violations_found": result["rows"][0]["outputs.custom_eval.violations_found"],
                    "true_positives": result["rows"][0]["outputs.custom_eval.true_positives"],
                    "false_positives": result["rows"][0]["outputs.custom_eval.false_positives"],
                    "false_negatives": result["rows"][0]["outputs.custom_eval.false_negatives"],
                    "percent_coverage": result["rows"][0]["outputs.custom_eval.percent_coverage"],
                    "similarity": result["rows"][0]["outputs.similarity.similarity"],
                },
                indent=4,
            )
        )
