"""Проверка SDK wire-контракта точного ZIP: MockTransport, никаких сетевых вызовов."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import httpx
import openai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True)
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    archive = Path(args.zip).resolve()
    with zipfile.ZipFile(archive) as bundle:
        source = bundle.read('llm_client.py')
        assert bundle.read('contracts.py') == (ROOT/'contracts.py').read_bytes()
    module = ModuleType('llm_checked_zip')
    exec(compile(source.decode('utf-8'), str(archive)+'::llm_client.py', 'exec'), module.__dict__)
    real_constructor = openai.OpenAI
    scenarios = [
        {'name': 'default_normal'},
        {'name': 'explicit_endpoint', 'env': {'OPENAI_BASE_URL': 'https://provided.example.invalid/custom/v1'}},
        {'name': 'trailing_slash', 'env': {'OPENAI_BASE_URL': 'https://provided.example.invalid/v1/'}},
        {'name': 'blank_defaults', 'env': {'OPENAI_MODEL': '  ', 'OPENAI_BASE_URL': '  '}},
        {'name': 'trimmed_key', 'env': {'OPENAI_API_KEY': '  contract-dummy-not-a-real-key  '}},
        {'name': 'explicit_non_qwen', 'env': {'OPENAI_MODEL': 'vendor/plain-model'}},
        {'name': 'reduced_time_budget', 'seconds': 9},
        {'name': 'json_schema', 'schema': True},
        {'name': 'usage_missing', 'usage': None},
        {'name': 'usage_partial', 'usage': {'prompt_tokens': 80, 'completion_tokens': 20}},
        {'name': 'usage_null_total', 'usage': {'total_tokens': None}},
        {'name': 'empty_content', 'content': '', 'none': True, 'stage': 'parse'},
        {'name': 'null_content', 'content': None, 'none': True, 'stage': 'parse'},
        {'name': 'invalid_json', 'content': '{bad', 'none': True, 'stage': 'parse'},
        {'name': 'truncated_json', 'content': '{"model":', 'finish': 'length', 'none': True, 'stage': 'parse'},
        {'name': 'non_object_json', 'content': '[]', 'none': True, 'stage': 'parse'},
        {'name': 'no_choices', 'empty_choices': True, 'none': True, 'stage': 'response'},
        {'name': 'invalid_metric_type', 'content': '{"metric": []}', 'metric': True, 'none': True},
        *[{'name': 'http_'+str(code), 'http': code, 'none': True, 'stage': 'request'}
          for code in (400, 401, 403, 404, 429, 500, 503)],
        *[{'name': kind, 'raise': kind, 'none': True, 'stage': 'request'}
          for kind in ('connect_timeout', 'read_timeout', 'connect_error')],
        {'name': 'missing_key', 'env': {'OPENAI_API_KEY': ''}, 'none': True, 'requests': 0},
        {'name': 'disabled', 'max_calls': 0, 'none': True, 'requests': 0},
        {'name': 'token_reserve', 'env': {'ALFA_LLM_TOKEN_LIMIT': '1000'}, 'none': True, 'requests': 0},
        {'name': 'context_limit', 'large_context': True, 'none': True, 'requests': 0},
        {'name': 'error_limit', 'http': 401, 'attempts': 4, 'requests': 3, 'none': True, 'stage': 'request'},
    ]
    results = []
    for case in scenarios:
        events, requests = [], []
        environment = {'OPENAI_API_KEY': 'contract-dummy-not-a-real-key', **case.get('env', {})}
        expected_model = environment.get('OPENAI_MODEL', '').strip() or module.DEFAULT_MODEL
        expected_base = environment.get('OPENAI_BASE_URL', '').strip() or module.DEFAULT_BASE_URL
        expected_url = expected_base.rstrip('/')+'/chat/completions'
        expected_key = environment['OPENAI_API_KEY'].strip()
        seconds = case.get('seconds', 100)
        budget = SimpleNamespace(until=lambda fraction: seconds, remaining=lambda: 150)
        journal = SimpleNamespace(event=lambda event, **data: events.append({'event': event, **data}))

        def handler(request):
            body = json.loads(request.content)
            assert str(request.url) == expected_url
            assert request.method == 'POST'
            assert request.headers['authorization'] == 'Bearer '+expected_key
            assert body['model'] == expected_model
            assert body['temperature'] == .15 and body['max_tokens'] == 1000
            assert [m['role'] for m in body['messages']] == ['system', 'user']
            expected_timeout = min(35, seconds-2)
            timeout = request.extensions['timeout']
            assert all(timeout[k] == expected_timeout for k in ('read', 'write', 'connect', 'pool'))
            qwen = 'qwen' in expected_model.lower()
            if qwen:
                assert body['enable_thinking'] is False
                assert body['chat_template_kwargs'] == {'enable_thinking': False}
                assert body['reasoning'] == {'enabled': False}
                assert body['messages'][1]['content'].endswith('\n/no_think')
            else:
                assert not any(k in body for k in ('enable_thinking', 'chat_template_kwargs', 'reasoning'))
                assert not body['messages'][1]['content'].endswith('/no_think')
            if case.get('schema'):
                assert body['response_format']['type'] == 'json_schema'
                assert body['response_format']['json_schema']['strict'] is True
                assert body['response_format']['json_schema']['schema'] == schema
            else:
                assert 'response_format' not in body
            requests.append({'method': request.method, 'endpoint_correct': True,
                             'authorization_correct': True, 'timeout_seconds': expected_timeout,
                             'body_correct': True})
            kind = case.get('raise')
            if kind:
                error = {'connect_timeout': httpx.ConnectTimeout, 'read_timeout': httpx.ReadTimeout,
                         'connect_error': httpx.ConnectError}[kind]
                raise error('private-upstream-error-marker', request=request)
            if case.get('http', 200) != 200:
                return httpx.Response(case['http'], json={'error': {'message': 'private-upstream-error-marker',
                                                                     'type': 'invalid_request_error'}})
            choices = [] if case.get('empty_choices') else [{'index': 0,
                'message': {'role': 'assistant', 'content': case.get('content', '{"model":"linear"}')},
                'finish_reason': case.get('finish', 'stop')}]
            response = {'id': 'wire-only', 'object': 'chat.completion', 'created': 0,
                        'model': expected_model, 'choices': choices,
                        'usage': case.get('usage', {'prompt_tokens': 80, 'completion_tokens': 20, 'total_tokens': 100})}
            return httpx.Response(200, json=response)

        def factory(**kwargs):
            assert kwargs['max_retries'] == 0
            return real_constructor(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

        schema = {'type': 'object', 'properties': {'model': {'type': 'string', 'enum': ['linear']}},
                  'required': ['model'], 'additionalProperties': False}
        with patch.dict(os.environ, environment, clear=True), patch('openai.OpenAI', side_effect=factory):
            planner = module.LLMPlanner(journal, budget, case.get('max_calls', 6))
            for _ in range(case.get('attempts', 1)):
                if case.get('metric'):
                    answer = planner.resolve_metric(SimpleNamespace(target='y'), 'task')
                else:
                    payload = {'data': 'x'*24001} if case.get('large_context') else {'history': []}
                    answer = planner.request('system', payload, response_schema=schema if case.get('schema') else None)
                assert (answer is None) == case.get('none', False), case['name']
        assert len(requests) == case.get('requests', 1), case['name']
        if case.get('stage'):
            assert any(e['event'] == 'llm_error' and e['stage'] == case['stage'] for e in events), case['name']
        serialized = json.dumps(events, ensure_ascii=False)
        assert expected_key == '' or expected_key not in serialized
        assert 'private-upstream-error-marker' not in serialized
        results.append({'scenario': case['name'], 'passed': True, 'sdk_requests': len(requests),
                        'planner_calls': planner.calls, 'accounted_tokens': planner.tokens,
                        'error_types': [e['error_type'] for e in events if e['event'] == 'llm_error'],
                        'skip_reasons': [e['reason'] for e in events if e['event'] == 'llm_skipped'],
                        'wire_checks': requests})
    run = (Path(args.output_dir).resolve() if args.output_dir else
           ROOT/'runs'/'llm_final_check'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    run.mkdir(parents=True, exist_ok=False)
    result = {'status': 'passed', 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'llm_client_sha256': hashlib.sha256(source).hexdigest(),
              'openai_version': importlib.metadata.version('openai'),
              'httpx_version': importlib.metadata.version('httpx'),
              'external_api_calls': 0, 'network_connections': 0,
              'scenarios': results, 'scenario_count': len(results)}
    (run/'wire_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'passed', 'scenarios': len(results), 'result': str(run/'wire_results.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
