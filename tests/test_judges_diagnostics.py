"""Проверяемые сведения для журнала судей, без строк датасета и секретов."""
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiment_store import Journal
from fixtures import make_task


class JudgesDiagnosticsTests(unittest.TestCase):
    def test_stdout_exposes_safe_error_stage_and_filters_request_body(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as output:
            journal = Journal(Path(tmp))
            journal.event('llm_error', error_type='AuthenticationError', http_status=401,
                          stage='request', call=1, request_body={'private': 'not-for-stdout'})
            message = json.loads(output.getvalue())
        self.assertEqual(message, {'event': 'llm_error', 'error_type': 'AuthenticationError',
                                  'http_status': 401, 'stage': 'request', 'call': 1})
        self.assertNotIn('not-for-stdout', output.getvalue())

    def test_only_ml_error_message_is_visible_and_bounded(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as output:
            journal = Journal(Path(tmp))
            journal.event('experiment_result', status='failed', error_type='LightGBMError',
                          error='Do not support special JSON characters in feature name.\n'+'x'*500)
            journal.event('llm_error', error_type='APIStatusError', error='private reflected header')
            events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertTrue(events[0]['error_message'].startswith('Do not support special JSON'))
        self.assertEqual(len(events[0]['error_message']), 400)
        self.assertNotIn('\n', events[0]['error_message'])
        self.assertNotIn('error_message', events[1])
        self.assertNotIn('private reflected header', output.getvalue())

    def test_finished_event_proves_csv_and_reports_fallback_and_llm_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = make_task(root, n=6)
            environment = os.environ.copy()
            environment.pop('OPENAI_API_KEY', None)
            environment['PYTHONIOENCODING'] = 'utf-8'
            script = Path(__file__).resolve().parents[1]/'solution.py'
            process = subprocess.run([sys.executable, '-B', str(script), str(task),
                                      '--time-limit', '30', '--no-llm'], cwd=root,
                                     env=environment, capture_output=True, encoding='utf-8', timeout=35)
            self.assertEqual(process.returncode, 0, process.stderr)
            events = [json.loads(line) for line in process.stdout.splitlines() if line.startswith('{')]
            final = next(event for event in events if event['event'] == 'finished')
            self.assertEqual(final['status'], 'fallback_small_or_constant_task')
            self.assertEqual(final['prediction_source'], 'fallback')
            self.assertTrue(final['fallback_only'])
            self.assertFalse(final['refit_completed'])
            self.assertEqual(final['llm_calls'], 0)
            self.assertEqual(final['llm_tokens'], 0)
            self.assertEqual(final['submission_rows'], 50)
            self.assertEqual(final['submission_sha256'], hashlib.sha256((root/'submission.csv').read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
