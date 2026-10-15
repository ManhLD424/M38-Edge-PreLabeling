"""Test suite runner for M38: Model Compression for Edge Pre-Labeling."""
import sys
import unittest
from pathlib import Path

def run_all_tests():
    test_dir = Path(__file__).resolve().parent
    repo_root = test_dir.parent
    sys.path.insert(0, str(repo_root))

    print("==================================================")
    print("      M38 SYSTEM TEST SUITE: EXECUTION            ")
    print("==================================================")
    print(f"Repository Root: {repo_root}")
    print(f"Test Directory:  {test_dir}\n")

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(test_dir), pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n==================================================")
    print(f"Tests run:      {result.testsRun}")
    print(f"Errors:         {len(result.errors)}")
    print(f"Failures:       {len(result.failures)}")
    print(f"Skipped:        {len(result.skipped)}")
    print("==================================================")

    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
