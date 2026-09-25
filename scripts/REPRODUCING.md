# Воспроизведение анализа без новых расчётов

Команды ниже запускаются из корня checkout в отдельном Python-окружении с уже установленными
`numpy`, `ase` и `scipy`. MLIP-пакеты, веса моделей и Quantum ESPRESSO этим проверкам не нужны.
Не запускать GPU training или DFT для проверки сохранённых таблиц.

```text
python -m unittest discover -s tests -v
python scripts/per_image_stats.py
```

Первая команда проверяет статистику обеих сохранённых ladder, доступность только трёх NEB pairs
против восьми single-point pairs, отсутствие подмены недостающих NEB, выбор из пяти schedules
по архивным validation logs и возврат ошибки shell-wrapper. Последний тест использует заглушку
Python, а не MACE. На Windows используется Git Bash либо путь из `BASH_EXECUTABLE`; без Bash
только shell-тест помечается skipped. Временные файлы находятся в игнорируемой папке `tmp/` checkout.

Вторая команда читает пять DFT полос из `data/structures/` и девять MLIP profiles на полосу из
`mlip/r22_2026-09/`. Выход по умолчанию: `tmp/per_image_stats.json`; опубликованный JSON не заменяется.
Пути можно задать через `DEPOSIT`, `MLIP_RUN`, `MLIP_STATS_OUTPUT`. Неполный набор profiles
завершает анализ с ошибкой, а не создаёт внешне успешную пустую таблицу.

## R23 aggregation и schedule selection

`aggregate2.py <holdout>` читает отдельные `eval_<tag>.json` из `R23_OUT`. В архиве они объединены
в `mlip/r23_2026-09/evaluations_<holdout>.json`: ключ объекта соответствует `<tag>`. Тест разворачивает
эти bundles в отдельную временную папку и сравнивает восстановленные числовые поля с опубликованными
`ladder_<holdout>.json`. Сам агрегатор пишет `ladder_<holdout>.json` в `R23_OUT`, поэтому для проверки
следует выбирать отдельную папку. `chosen.json`, если присутствует, передаёт флаг selection gate;
он не доказывает, с каким schedule действительно обучалась данная ladder. Агрегатор оставляет
`run_schedule_verified=false`; отсутствие файла также не трактуется как пройденный gate.
Причина ошибки модели автоматически не выводится.

`pick_schedule.py` читает `R23_OUT/tune_v2.json` и `R23_RUNS/<tag>/train.log`. Gate: training-band
barrier MAE ≤10 meV; среди прошедших выбирается минимальный последний validation force RMSE из лога.
Варианты tA–tC имеют 200 epochs, tD/tE — 4000. Если ни один не проходит, исторический least-bad
выбор сохраняется с `gate_passed=false`; он не становится подтверждённым результатом. Если у прошедшего
варианта нет validation RMSE, выбор прекращается. Архивный sweep выбирает tD для длинного control;
это не меняет факт, что первоначальная ladder обучалась с tC/200 epochs.

## Дорогие entrypoints: только для намеренного повторного вычисления

`train4.sh` запускает настоящее обучение MACE на CUDA. Его позиционные аргументы:
`<rung> <seed> <epochs> <weight_E> <weight_F> <lr> <dtype> <tag>`.
Нужны заранее подготовленные `train_<rung>_s<seed>.xyz` / `valid_<rung>_s<seed>.xyz` и foundation checkpoint.
Настройки: `R23_OUT`, `R23_RUNS`, `R23_WORK`, `R23_FOUNDATION`, `R23_PYTHON`.
По умолчанию сохранены исторические `/work` и cache path; для новой машины их следует задать явно.
Выход включает модели, checkpoints и `train.log`. Повтор с тем же `<tag>` использует ту же папку
и перезаписывает лог; для независимого запуска нужен новый tag или новый `R23_RUNS`.
Ошибка обучения теперь возвращается вызывающей программе, а не превращается в exit 0.

`zeroshot.py`, `evaluate2.py`, `tune_report2.py` выполняют MLIP inference, последний использует
исторические `/work/structures`, `/work/out`, `/work/runs`. DFT/NEB/Hessian drivers также требуют
внешнего исполняемого кода и pseudopotentials. Это provenance-скрипты, а не единая переносимая команда
полной регенерации всех расчётов. Полные окружения исторических запусков, все weights и Docker build
definitions в этом checkout отсутствуют; ограничения описаны в `mlip/environments/README.md`.

## Восстановленный GRACE driver

`grace_band_multi_historical.py` добавлен 2026-09-25 как побайтовая копия сохранившегося
`tmp/gr_band_multi.py` из авторского workspace. Локальный launcher `tmp/gr5_all_minerals.sh`
ссылается именно на этот файл и staging-папки; сами remote/deploy команды не входят в этот
entrypoint и не выполнялись при восстановлении. SHA256 копии:
`96d806903a4963f2dd8f1198812e16ee076f704bb284734ebb870590ca11d34b`.
Это provenance восстановления существующего файла, не доказательство immutable snapshot
всего исторического контейнера. Новый inference для проверки не запускался.

Driver использует `tensorpotential.calculator.grace_fm` и три имени моделей, соответствующие
сохранённым `grace_*.json`. Исторические входы: `/data/bands/<band>/`, где `<band>` — mackinawite,
pyrite_VS2, marcasite, greigite_channel, greigite_cation. Первые три папки содержали одну
многокадровую `band.extxyz`, две последние — `final_00.xyz`…`final_08.xyz`. Выходы
`/data/grace_<band>.json` при повторе перезаписываются. Для запуска нужны соответствующее
GRACE-окружение и намеренно подготовленный отдельный staging volume; возможны загрузки weights.
Исторический driver пропускает failed models с сообщением в stdout; полноту выходных profiles
следует отдельно проверять `per_image_stats.py`.

## Отказ при неполных входах

Агрегатор требует `S0.P1_selftest` для всех пяти reference bands; пустой или частичный self-test
не считается PASS. Profile statistics проверяет число моделей и длину каждого массива images:
укороченная полоса не пропускается с внешне успешной неполной таблицей. Selector до чтения новых
входов атомарно помечает собственный `chosen.json` как `status=invalid`, `gate_passed=false`;
успех заменяет его текущим выбором. Поэтому неудачный повтор не оставляет старый passing choice.
Другие файлы selector не удаляет. Эти три негативных сценария входят в десять offline tests.
