"""
Interactive console loan-approval predictor (frontend).

Reads applicant details from the user with robust validation and outputs an
approval decision. The actual prediction logic lives in backend/src/predict_core.py
so it is shared with the LangGraph agent and any future API.
"""

import os
import sys

# backend/src holds predict_core / common; add it to the import path.
_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend", "src"))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from predict_core import predict_applicant  # noqa: E402


def check_quit(user_input):
    cleaned = user_input.strip().lower()
    if cleaned in ["q", "quit", "exit"]:
        print("\nExiting Interactive Console Predictor. Goodbye!")
        sys.exit(0)


def prompt_validated(prompt_text, validator, parser, error_message):
    while True:
        try:
            raw = input(prompt_text)
        except EOFError:
            # stdin closed (e.g. piped input exhausted) — exit cleanly, no crash.
            print("\nInput stream closed. Exiting.")
            raise SystemExit(0)
        try:
            check_quit(raw)
            parsed = parser(raw)
            if validator(parsed):
                return parsed
            print(f"Error: {error_message}")
        except ValueError:
            print(f"Error: Invalid format. {error_message}")
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"Error: {e}")


def parse_employment(val):
    val = val.strip().upper()
    if val in ["E", "EMPLOYED"]:
        return "Employed"
    if val in ["S", "SELF-EMPLOYED", "SELF EMPLOYED", "SE"]:
        return "Self-Employed"
    if val in ["U", "UNEMPLOYED"]:
        return "Unemployed"
    raise ValueError("Invalid Employment Status.")


def parse_property(val):
    val = val.strip().upper()
    if val in ["U", "URBAN"]:
        return "Urban"
    if val in ["SU", "SEMI-URBAN", "SEMI URBAN"]:
        return "Semi-Urban"
    if val in ["R", "RURAL"]:
        return "Rural"
    raise ValueError("Invalid Property Area.")


def run_interactive_predictor():
    print("=" * 60)
    print("LOAN APPROVAL INTERACTIVE PREDICTOR")
    print("=" * 60)
    print("Type 'q' or 'exit' at any prompt to quit.\n")

    while True:
        print("-" * 40)
        print("Please enter applicant details:")
        print("-" * 40)

        age = prompt_validated("Age (18-100): ", lambda x: 18 <= x <= 100,
                               lambda x: int(x), "Age must be an integer 18-100.")
        income = prompt_validated("Monthly Income ($): ", lambda x: x > 0,
                                  lambda x: float(x), "Positive number required.")
        amount = prompt_validated("Loan Amount ($): ", lambda x: x > 0,
                                  lambda x: float(x), "Positive number required.")
        credit = prompt_validated("Credit Score (300-900): ", lambda x: 300 <= x <= 900,
                                  lambda x: int(x), "Integer 300-900 required.")
        employment = prompt_validated(
            "Employment Status (Employed/Self-Employed/Unemployed or E/S/U): ",
            lambda x: x in ["Employed", "Self-Employed", "Unemployed"],
            parse_employment, "Must be Employed (E), Self-Employed (S) or Unemployed (U).")
        existing = prompt_validated("Number of Existing Loans: ", lambda x: x >= 0,
                                    lambda x: int(x), "Non-negative integer required.")
        term = prompt_validated("Loan Term in months (6-360): ", lambda x: 6 <= x <= 360,
                                lambda x: int(x), "Integer 6-360 required.")
        area = prompt_validated(
            "Property Area (Urban/Semi-Urban/Rural or U/SU/R): ",
            lambda x: x in ["Urban", "Semi-Urban", "Rural"],
            parse_property, "Must be Urban (U), Semi-Urban (SU) or Rural (R).")

        applicant = {
            "Age": age, "MonthlyIncome": income, "LoanAmount": amount,
            "CreditScore": credit, "EmploymentStatus": employment,
            "ExistingLoans": existing, "LoanTerm": term, "PropertyArea": area,
        }

        result = predict_applicant(applicant)
        print("\n" + "=" * 40)
        print("PREDICTION RESULT")
        print("=" * 40)
        print(f"Status:      {result['status']}")
        print(f"Probability: {result['probability']*100:.2f}%")
        print("=" * 40 + "\n")


if __name__ == "__main__":
    try:
        run_interactive_predictor()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
        sys.exit(0)
