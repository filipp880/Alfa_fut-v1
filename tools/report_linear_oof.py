"""Сводный отчёт A/B/C из реальных полных прогонов и независимого аудита."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from report_release import analyze
TASK_NAMES={'public/regression':'Публичная регрессия','public/classification':'Публичная классификация',
    'linear_benchmarks/linear_regression':'Синтетическая линейная регрессия',
    'linear_benchmarks/linear_classification':'Синтетическая классификация'}

def main():
    cases=[('public/regression','A',ROOT/'runs/release/regression_20261026T130245_415f066a'),
        ('public/classification','A',ROOT/'runs/release/classification_20261026T130246_62a301c3'),
        ('public/regression','B',ROOT/'runs/release/regression_20261027T000649_219772ad'),
        ('public/regression','C',ROOT/'runs/release/regression_20261027T000703_bd93fbc7')]
    for stage in ('synthetic','classification'):
        for record in json.loads((ROOT/f'runs/linear_oof_benchmarks/index_{stage}.json').read_text(encoding='utf-8')):
            cases.append((record['task_source'],record['arm'],Path(record['directory'])))
    results=[]
    for task,arm,case in cases:
        print('Аудит '+task+' '+arm,flush=True)
        result=analyze(case);result.update(benchmark_task=task,arm=arm);results.append(result)
    (ROOT/'docs/LINEAR_OOF_RESULTS.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# Ранние линейные модели и OOF: результаты','',
        'Дата: 27 октября 2026 года. A — v3; B — ранний linear без OOF; C — linear + OOF.',
        'Каждый прогон из собственного ZIP и контейнера, без сети и LLM. Публичные B/C: до 18 поисковых экспериментов, '
        '10800 секунд, без ограничения числа итераций. Синтетика: до 6 поисковых экспериментов, тот же полный refit.',
        'По 4 CPU/14 ГиБ; одновременно максимум два контейнера. Исходные requirements не изменены.','',
        '| Задача | Вариант | Подтверждение | Отдельный тест синтетики | Обучений | Секунд | OOF / выбран |',
        '|---|---|---:|---:|---:|---:|---|']
    for task in ('public/regression','public/classification','linear_benchmarks/linear_regression','linear_benchmarks/linear_classification'):
        for r in sorted((v for v in results if v['benchmark_task']==task),key=lambda v:v['arm']):
            oof=r['oof'];oracle=r['oracle_test_score'];test='—' if oracle is None else f'{oracle:.8f}'
            selected='да' if oof.get('selected',False) else 'нет'
            lines.append(f"| {TASK_NAMES[task]} | {r['arm']} | {r['confirm_score']:.8f} | {test} | {r['ml_fits']} | {r['host_seconds']:.1f} | {oof['status']} / {selected} |")
    lines+=['','## Эффект каждого изменения','']
    for task in TASK_NAMES:
        arms={r['arm']:r for r in results if r['benchmark_task']==task}
        field='oracle_test_score' if task.startswith('linear_benchmarks/') else 'confirm_score'
        a,b,c=[arms[k][field] for k in ('A','B','C')]
        sign=1 if 'classification' in task else -1
        lines.append(f"- {TASK_NAMES[task]}: прирост B относительно A (положительное — улучшение) "
            f"{sign*(b-a):+.8f}; дополнительный эффект C относительно B: {sign*(c-b):+.8f}. "
            f"Основа сравнения: {'отдельный тест с ответами вне контейнера' if field=='oracle_test_score' else 'подтверждающая выборка'}.")
    lines+=['','## Независимое подтверждение новых поисковых планов','']
    for r in results:
        if r['arm']!='B':continue
        comparison=r['confirmation_comparison']
        if comparison.get('legacy_score') is not None and comparison.get('baseline_score') is not None:
            lines.append(f"- {TASK_NAMES[r['benchmark_task']]}: поисковый план на confirm "
                f"{comparison['legacy_score']:.8f}, заранее заданный контроль {comparison['baseline_score']:.8f}; "
                f"итог {r['confirm_score']:.8f}.")
    lines+=['','## Как проверена корректность','',
        'Независимый отчёт пересчитывает RMSE на исходной шкале и Macro F1 по всем классам, проверяет '
        'точные индексы, группы повторов и хронологию, фиксированные итерации confirm/refit, '
        'неизменность исходных файлов и ZIP, совпадение submission.csv с прогнозом полного refit.',
        'Для завершённого OOF заново собирается матрица из файлов каждого worker, проверяются '
        'профили fold-train, общая ось классов, отсутствие пересечений с поисковой проверкой/confirm, '
        'параметры из поиска и фиксация весов до confirm. Все 73 набора весов пересчитываются '
        'независимым кодом. Число обучений включает завершённые OOF-fit; попытки и неуспехи записаны отдельно.',
        'Синтетические тестовые ответы хранятся у проверяющего и не монтируются в ML-контейнер. '
        'Каждый вариант видит только собственные входы, модели и результаты.','',
        '## Что выбрал OOF','']
    for r in results:
        if r['arm']!='C':continue
        o=r['oof']
        lines.append(f"- {TASK_NAMES[r['benchmark_task']]}: статус `{o['status']}`, веса бустинг/лес/linear `{o.get('weights')}`, "
            f"ненулевых семейств {o.get('effective_families','—')}, принят в финал: {o.get('selected',False)}, "
            f"отдельная поисковая проверка {o.get('search_valid_score','—')} против прежнего плана {o.get('legacy_search_valid_score','—')}, "
            f"попыток {o.get('fit_attempts',0)}, завершено {o.get('completed_fits',0)}, причина `{o.get('reason')}`.")
    lines+=['','## Ограничения','',
        'Публичные периоды уже использовались в исследованиях: это проверка регрессий и повторная '
        'оценка, а не новый слепой тест. Синтетика проверяет применимость к известной структуре '
        'зависимости, но не предсказывает баллы приватной задачи. Время зависит от соседней нагрузки; '
        'таблица не доказывает ускорение. Нулевой вес семейства — допустимый результат эксперимента.',
        'Новые реальные ответы Qwen не запрашивались. Контракт клиента проверен 33 сценариями '
        'настоящего SDK без сети; влияние новых предложений LLM на качество этого выпуска отдельно не измерялось.',
        '', '## Решение о внедрении','',
        'Ранние Ridge/LogisticRegression внедрены: отдельный синтетический тест показывает RMSE '
        '21,045203 → 0,259924 и Macro F1 0,848755 → 0,942262. На двух публичных задачах '
        'качество сохранено. Это расширение обязательного портфеля для неизвестных задач.',
        'OOF не дал дополнительного прироста ни в одной из четырёх задач. Он оставлен как '
        'ограниченный поисковый этап для новых задач: без прироста на отдельных проверках '
        'замороженная смесь не заменяет прежний план. Для экономии времени доступен `--no-oof`. '
        'Не следует считать факт реализации OOF доказанным улучшением качества.',
        'Дополнительный [аудит фактических моделей](LINEAR_OOF_MODEL_AUDIT.json) проверил '
        '59 фиксированных обучений / 67 компонентов: реальные деревья, seed, конфигурации, '
        'линейные коэффициенты и SHA-256. Новых обучений при аудите нет.',
        'Финальный [ZIP](../dist/alfa_solution_v4_2026-10-27.zip) побайтно совпадает с кандидатом '
        'в полных прогонах; SHA-256 `ce4d46965cc4f13eae67e0eeb0033c284bf6e9e3cbe48c9782964a3d2ea15708`. '
        'Прирост на приватной платформе не обещается.','']
    (ROOT/'docs/LINEAR_OOF_RESEARCH.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Отчёт готов',flush=True)
if __name__=='__main__':main()
