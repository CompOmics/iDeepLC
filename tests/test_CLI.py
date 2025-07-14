# Python
import unittest
from ideeplc.__main__ import _argument_parser

class TestCLI(unittest.TestCase):
    def test_argument_parser(self):
        """Test the CLI argument parser."""
        parser = _argument_parser()
        args = parser.parse_args(["--input", "test.csv", "--save", "--calibrate"])

        # Assertions to verify parsed arguments
        self.assertEqual(args.input, "test.csv", "Input argument should match")
        self.assertTrue(args.save, "Save argument should be True")
        self.assertTrue(args.calibrate, "Calibrate argument should be True")

if __name__ == "__main__":
    unittest.main()