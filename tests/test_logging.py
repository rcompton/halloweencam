import logging
import unittest
from unittest.mock import patch, MagicMock

from ghoulfluids.logging import setup_logging, get_logger

class TestLogging(unittest.TestCase):
    @patch("logging.basicConfig")
    def test_setup_logging(self, mock_basic_config: MagicMock):
        """Verify that setup_logging configures the root logger correctly."""
        setup_logging("DEBUG")
        mock_basic_config.assert_called_once_with(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=unittest.mock.ANY,
        )

        mock_basic_config.reset_mock()

        setup_logging("INFO")
        mock_basic_config.assert_called_once_with(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=unittest.mock.ANY,
        )

    def test_get_logger(self):
        """Verify that get_logger returns a logger and that it logs messages."""
        logger = get_logger("test_logger")
        self.assertIsInstance(logger, logging.Logger)

        with self.assertLogs("test_logger", level="INFO") as cm:
            logger.info("This is an info message.")
            logger.warning("This is a warning message.")
        self.assertEqual(len(cm.output), 2)
        self.assertIn("INFO:test_logger:This is an info message.", cm.output[0])
        self.assertIn("WARNING:test_logger:This is a warning message.", cm.output[1])

if __name__ == "__main__":
    unittest.main()
