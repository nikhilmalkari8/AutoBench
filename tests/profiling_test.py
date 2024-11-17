import pandas as pd
from sklearn.datasets import fetch_california_housing
from autobench.profiling import generate_profile

def test_generate_profile():
    """
    Test the generate_profile function using the California Housing dataset.
    """
    # Load California Housing dataset
    housing = fetch_california_housing(as_frame=True)
    data = housing.frame
    data["target"] = housing.target

    # Run profiling
    print("\n=== Running Profiling Test ===")
    generate_profile(data, target_column="target", save_report=True, report_filename="test_profiling_report.json")
    print("\nProfiling test completed. Check the 'test_profiling_report.json' for details.")


if __name__ == "__main__":
    test_generate_profile()

