import unittest

from checker import (
    check_answer,
    extract_number,
    is_contains_correct,
    is_correct,
    is_exact_correct,
    is_letter_correct,
)


class CheckerTests(unittest.TestCase):
    def test_numeric_checker_handles_commas_decimals_signs_and_plain_numbers(self):
        self.assertEqual(extract_number("Final answer: $1,234.50"), "1234.50")
        self.assertTrue(is_correct("The result is 1,234.50", "1,234.5"))
        self.assertTrue(is_correct("Final answer: -$1,234.50", "-1,234.5"))
        self.assertTrue(is_correct("Final answer: +12.75", "+12.750"))
        self.assertTrue(is_correct("The result is 42", 42))
        self.assertTrue(is_correct("Final answer: 0.125", "0.1250001"))
        self.assertFalse(is_correct("Final answer: -12.75", "12.75"))
        self.assertFalse(is_correct("No numeric answer", "5"))

    def test_letter_checker(self):
        self.assertTrue(is_letter_correct("B", "b"))
        self.assertTrue(is_letter_correct("The answer is (C).", "C"))
        self.assertTrue(is_letter_correct("Option D is correct.", "d"))
        self.assertFalse(is_letter_correct("The answer is A.", "B"))
        self.assertFalse(is_letter_correct("AB", "A"))

    def test_normalized_exact_checker(self):
        self.assertTrue(is_exact_correct("  North\t Star\n", "north star"))
        self.assertTrue(is_exact_correct("Café", "cafe\u0301"))
        self.assertFalse(is_exact_correct("northstar", "north star"))

    def test_note_prefix_is_not_an_exact_match(self):
        expected = "SHIPMENT ARRIVES NEXT WEEK."
        generated = "NOTE 0059: SHIPMENT ARRIVES NEXT WEEK."
        self.assertFalse(is_exact_correct(generated, expected))
        self.assertFalse(check_answer(generated, expected, "exact"))

    def test_ground_truth_dispatch_uses_requested_checker(self):
        self.assertTrue(check_answer("B", "B", "letter"))
        self.assertTrue(check_answer("4", "4", "numeric"))
        self.assertTrue(check_answer("billing", "billing", "contains"))
        self.assertFalse(check_answer("anything", "anything", "unknown"))

    def test_contains_checker(self):
        self.assertTrue(is_contains_correct("BILLING", "billing"))
        self.assertTrue(is_contains_correct(" technical ", "technical"))
        self.assertFalse(is_contains_correct("The category is billing", "billing"))
        self.assertFalse(is_contains_correct("billing and account", "billing"))
        self.assertFalse(is_contains_correct("account", "billing"))
        self.assertFalse(is_contains_correct("invalid", "  "))
        self.assertFalse(is_contains_correct("accounting", "account"))
        self.assertFalse(is_contains_correct("Not account; billing is correct", "account"))


if __name__ == "__main__":
    unittest.main()
