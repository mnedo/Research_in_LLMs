# Learning from “Unsolvable” Tasks (RL + Supervised Signal)

### 1) Цель

Обучить LLM решать класс задач, которые **в baseline модель не решает вообще**.

**Критерий “нерешаемости”**: на фиксированном подмножестве задач (**Hard subset**) модель имеет

- **pass@128 = 0**
(то есть среди 128 независимых сэмплов не находится ни одного корректного решения).

Ваша задача — добиться, чтобы после обучения на этом же Hard subset было **pass@128 > 0** (лучше — заметно > 0), и показать это экспериментально.

### 2) Что нужно реализовать

Выберите **один метод** (можно больше — это плюс, но не обязательно):

- **LUFFY**: [https://arxiv.org/pdf/2504.14945](https://arxiv.org/pdf/2504.14945)
- **RL-PLUS**: [https://arxiv.org/pdf/2508.00222v2](https://arxiv.org/pdf/2508.00222v2)
- **SRFT**: [https://www.arxiv.org/pdf/2506.19767](https://www.arxiv.org/pdf/2506.19767)

### 3) Данные / среда и “gold trajectories”

Вам нужны задачи, где:

1. есть **верифицируемая** проверка корректности (желательно программная),
2. можно получить **правильные траектории / решения** (gold).

Выберите один источник.

### Вариант A: ваша среда из ДЗ2 (рекомендуется)

1. Берёте свою среду из ДЗ2.
2. Подбираете сложность/гиперпараметры так, чтобы для **Qwen2.5-0.5B-Instruct** существовал **Hard subset**, на котором baseline **pass@128 = 0**:
    - **Hard-train** ⊂ train
    - **Hard-val** ⊂ val
    Важно: *pass@128=0 требуется именно на hard subset’ах*, а не на всём train/val (иначе RL вообще может не взлететь или будет слишком “мертвый” сигнал).
3. Получаете **gold trajectories**:
    - либо более сильной моделью (larger open-source / API — на ваш выбор),
    - либо любым другим способом,
    - но каждую gold-траекторию нужно **проверять вашим verifier’ом**.

### Вариант B: готовый датасет с траекториями

Можно взять датасет, где есть “задача → рассуждение/решение → ответ” (например OpenThoughts-114k: [https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k/](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k/)).

Требования:

- траектория **влезает** в контекст и память при обучении,
- корректность **верифицируема** (идеально — скриптом; если нет — объясните выбранный способ верификации и почему он приемлем),
- вы явно формируете **Hard subset** по baseline pass@128=0.

### 4) Модель, формат вывода, инференс-параметры

Рекомендуемая модель:

- `Qwen/Qwen2.5-0.5B-Instruct` (можно больше, если позволяют ресурсы)

Можно использовать:

- unsloth / LoRA
- vLLM для быстрого инференса при pass@k

**Один фиксированный system prompt** для всех сравнений (baseline и все обученные варианты).

Если математика:

```
You are helpful assistant. Please reason step by step, and put your final answer within \\boxed{}.

```

Если общая задача с вашим верификатором:

```jsx
You are a helpful assistant. You always first think about the reasoning process in the mind and then provides the user with the answer.\nThe reasoning process and answer are enclosed within ‘<think>’ ‘</think>’ and ‘<answer>’ ‘</answer>’ tags, respectively, e.g.,\n<think>\nA detailed reasoning process here, with possible reflections including but not limited to reviewing previous steps for errors, exploring alternative approaches, and considering possible refinements.\n</think>\n<answer>\nReply to user here.\n</answer>. Please reason step by step, and put your final answer within <answer> </answer> tags.
```

**Важно:** в отчёте зафиксируйте единые параметры генерации для всех сравнений:

- temperature
- top_p (или top_k)
- max_tokens

### 5) Разбиение данных

Данные делятся на **train / val**.

Требования к воспроизводимости:

- **val фиксированная**.
- Вы явно выделяете:
    - **Hard-train**: подмножество train, где baseline pass@128=0
    - **Hard-val**: подмножество val, где baseline pass@128=0

На практике удобно держать ещё и “Mixed” (где что-то решается), но hard subset обязателен.

### 6) Метрики

Основная метрика: **pass@k** на **val**, отдельно на **Hard-val**.

  $\text{pass@k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$

- Для каждого задания делаете `n` независимых сэмплов (например `n=128`).
- Считаете `pass@k` (k ≤ n). Можно показывать кривые pass@k для нескольких k (например 1, 4, 8, 16, 32, 64, 128).

Дополнительно:

- динамика reward (train),
- энтропия
- средняя длина генерации,

### 7) Экспериментальный протокол

### Шаг 0. Подготовка данных

- Сформируйте train/val и hard subset’ы.
- Докажите, что baseline действительно даёт **pass@128=0** на Hard-train и Hard-val (табличкой/логами).

### Шаг 1. Baseline оценка

На **val** и **Hard-val** измерьте:

- pass@k (желательно кривая до 128, либо до 64 при ограничениях),
- среднюю длину генерации,

### Шаг 2. GRPO-only (reward-only, без gold)

Обучите на train с GRPO, используя только reward через verifier, **без** правильных траекторий.

Логируйте:

- reward curve,
- энтропию,
- длину генераций.

Оцените на val / Hard-val: pass@k и длины.

### Шаг 3. SFT → GRPO (используя gold)

1. **SFT** на gold trajectories (train).
2. Затем **GRPO** (train) с тем же reward.

Сравните с шагом 2:

- pass@k на val / Hard-val,
- динамика reward,
- энтропия,
- длина генераций.

*(Если хотите — добавьте отдельную оценку “SFT-only”, это полезно для декомпозиции эффектов.)*

### Шаг 4. Выбранный метод (LUFFY / RL-PLUS / SRFT)

Реализуйте выбранный метод так, как он предполагает совмещение:

- on-policy генераций модели,
- и gold trajectories / supervised сигнала.

Сравнение минимум с:

- baseline,
- GRPO-only,
- SFT-only (если делали),
- SFT→GRPO.

### 8) Что должно быть в отчёте

**Обязательное:**

1. **Качество (главное сравнение)**
- Графики **pass@k (k=1…128 или до 64)** на:
    - **val (full)**
    - **Hard-val**
- Сравнить минимум 3 модели: **baseline**, **GRPO-only**, **SFT→GRPO** и **ваш метод** (LUFFY/RL-PLUS/SRFT).
1. **Динамика во время обучения**
- Графики по шагам обучения: **reward curve**, **длина генераций**.
- Энтропия (если хватает ресурсов)
1. **Сломали ли “ноль”?**
- Явно показать, получилось ли **pass@128 на Hard-val: 0 → > 0** после обучения.
1. **Длина генераций**
- Средняя/медианная длина (tokens) на **val** и **Hard-val**: baseline vs обученные.
1. **Trade-off качество vs diversity**
- Ответить: **падает ли pass@k на val (full)** по сравнению с baseline после RL?
- Если падает — как сильно и почему вы думаете, что так произошло.