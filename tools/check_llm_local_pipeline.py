"""Локальный HTTP-контракт SDK и настоящий CLI/ML из заданного ZIP; без шлюза."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import subprocess
import sys
import threading
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
from fixtures import make_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True)
    args = parser.parse_args()
    archive = Path(args.zip).resolve()
    run = ROOT/'runs'/'judges_audit'/('llm_http_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    run.mkdir(parents=True)
    results = []
    for mode in ('proposal', 'unauthorized', 'invalid_metric', 'invalid_env_no_llm'):
        case = run/mode
        task = make_task(case, n=6 if mode == 'invalid_env_no_llm' else 240)
        if mode == 'invalid_metric':
            task.write_text('Регрессия. Целевая колонка response. Метрика не определена. '
                            'Сохраните submission.csv с id и response.', encoding='utf-8')
        with zipfile.ZipFile(archive) as bundle:
            if any(Path(n).name != n for n in bundle.namelist()):
                raise ValueError('Неплоский ZIP')
            bundle.extractall(case)
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append({'path': self.path, 'model': payload.get('model'),
                                 'max_tokens': payload.get('max_tokens'),
                                 'temperature': payload.get('temperature'),
                                 'thinking_disabled': payload.get('enable_thinking') is False,
                                 'authorization_present': bool(self.headers.get('Authorization'))})
                if mode == 'unauthorized':
                    response = {'error': {'message': 'local-private-error-marker',
                                          'type': 'invalid_api_key', 'code': 'invalid_api_key'}}
                    code = 401
                else:
                    content = {'metric': []} if mode == 'invalid_metric' else {
                        'model': 'linear', 'regularization': 7, 'iterations': 50,
                        'hypothesis': 'Проверка локального контракта API'}
                    response = {'id': 'local-only', 'object': 'chat.completion', 'created': 0,
                                'model': payload['model'], 'choices': [{'index': 0,
                                'message': {'role': 'assistant', 'content': json.dumps(content)},
                                'finish_reason': 'stop'}],
                                'usage': {'prompt_tokens': 80, 'completion_tokens': 20, 'total_tokens': 100}}
                    code = 200
                data = json.dumps(response).encode('utf-8')
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        environment = os.environ.copy()
        environment.update(OPENAI_API_KEY='local-test-only-not-a-real-key',
                           OPENAI_MODEL='qwen/qwen3.8-27b',
                           OPENAI_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1',
                           ALFA_LLM_TOKEN_LIMIT='invalid' if mode == 'invalid_env_no_llm' else '40000',
                           PYTHONIOENCODING='utf-8')
        command = [sys.executable, '-B', str(case/'solution.py'), str(task),
                   '--time-limit', '120', '--max-experiments', '3', '--max-iterations', '50',
                   '--threads', '1', '--llm-calls', '1']
        if mode == 'invalid_env_no_llm':
            command.append('--no-llm')
        try:
            process = subprocess.run(command, cwd=case, env=environment, capture_output=True,
                                     encoding='utf-8', timeout=130)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        (case/'cli_stdout.log').write_text(process.stdout, encoding='utf-8')
        (case/'cli_stderr.log').write_text(process.stderr, encoding='utf-8')
        if process.returncode != 0:
            raise RuntimeError(f'{mode}: CLI exit {process.returncode}')
        work = sorted(case.glob('.alfa_runs/*/summary.json'))[-1].parent
        summary = json.loads((work/'summary.json').read_text(encoding='utf-8'))
        events = [json.loads(s) for s in (work/'experiments.jsonl').read_text(encoding='utf-8').splitlines()]
        expected = 'fallback_unsupported_metric' if mode == 'invalid_metric' else (
            'fallback_small_or_constant_task' if mode == 'invalid_env_no_llm' else 'completed')
        if summary['status'] != expected or summary['submission_audit']['status'] != 'ok':
            raise ValueError(f'{mode}: unexpected status or invalid CSV')
        expected_calls = 0 if mode == 'invalid_env_no_llm' else 1
        if len(requests) != expected_calls or summary['llm_calls'] != expected_calls:
            raise ValueError(f'{mode}: лишние запросы/retry')
        if any(r['path'] != '/v1/chat/completions' or not r['thinking_disabled']
               or r['max_tokens'] != 1000 or not r['authorization_present'] for r in requests):
            raise ValueError('Неверно сериализован HTTP-контракт')
        if mode == 'proposal' and not any(e.get('event') == 'experiment_result'
                and e.get('status') == 'ok' and e.get('phase') == 'search'
                and e.get('config', {}).get('model') == 'linear' for e in events):
            raise ValueError('Предложение локального API не дошло до реального fit')
        if mode == 'unauthorized' and not any(e.get('event') == 'llm_error'
                and e.get('http_status') == 401 and e.get('stage') == 'request' for e in events):
            raise ValueError('Не видна причина ошибки API')
        if 'local-private-error-marker' in process.stdout+process.stderr+(work/'experiments.jsonl').read_text(encoding='utf-8'):
            raise ValueError('Тело ошибки API попало в журнал')
        results.append({'case': mode, 'status': summary['status'], 'exit_code': process.returncode,
                        'local_http_calls': len(requests), 'external_api_calls': 0,
                        'csv_rows': summary['submission_audit']['rows'],
                        'csv_sha256': summary['submission_audit']['sha256'],
                        'refit_completed': summary.get('refit_completed', False), 'requests': requests})
        print(mode, expected, 'CSV ok; local calls', len(requests), flush=True)
    report = {'status': 'passed', 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'external_api_calls': 0, 'local_http_calls': 3, 'cases': results}
    (run/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(run, flush=True)


if __name__ == '__main__':
    main()
