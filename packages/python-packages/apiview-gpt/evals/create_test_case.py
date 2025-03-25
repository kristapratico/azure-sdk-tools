import argparse
import json

def check_apiview():
    with open('check.json', 'r') as f:
        content = f.read()

    formatted_content = content.replace('\\n', '\n')

    with open('check_formatted.json', 'w') as f:
        f.write(formatted_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a test case for the given function.")
    parser.add_argument(
        "--apiview_path",
        type=str,
        required=True,
        help="The function name to create a test case for.",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="The arguments to pass to the function.",
    )
    parser.add_argument(
        "--expected_path",
        type=str,
        required=True,
        help="The expected output of the function.",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="The expected output of the function.",
    )

    args = parser.parse_args()

    with open(args.apiview_path, "r") as f:
        apiview_contents = f.read()

    with open(args.expected_path, "r") as f:
        expected_contents = json.loads(f.read())

    test_case = {"query": apiview_contents, "language": args.language, "response": json.dumps(expected_contents)}

    with open(args.file_path, "a") as f:
        f.write("\n")
        json.dump(test_case, f)
