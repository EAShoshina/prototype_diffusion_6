# Диффузионная модель оценки хвостовых финансовых рисков

[![CI](https://github.com/EAShoshina/ddpm-tail-risk/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/ddpm-tail-risk/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

> **НИР — РУДН, 2026**  
> Тема: «Применение архитектуры трансформеров в диффузионных моделях для оценки хвостовых финансовых рисков»  
> Студент: Шошина Евгения Александровна, НФИ-01-22  
> Научный руководитель: Шорохов С.Г., к.ф.-м.н., доцент

---

## Описание

Проект реализует генеративную модель для оценки **VaR(99%)** и **CVaR(99%)** портфеля российских акций (SBER, LKOH, GAZP) на основе **DDPM с трансформерным денойзером**.

**Ключевая идея:** вместо стандартного MLP-денойзера используется Transformer Encoder с механизмом multi-head self-attention, который улавливает долгосрочные зависимости в финансовом временном ряду. Модель генерирует 2000 сценариев следующей дневной доходности, из которых эмпирически вычисляются VaR и CVaR.

### Архитектура денойзера

```
Вход: x_t (1,)  +  t (скаляр)  +  context (30,)
         │               │               │
    token_proj      step_emb       token_proj
         └───────────────┴───────────────┘
                         │
              [W+1 токенов, d_model=32]
              + sinusoidal pos_enc
                         │
              ┌──────────────────────┐
              │  Pre-LN Transformer  │  × 2 блока
              │  MHA (4 головы)      │
              │  FFN (128)  GELU     │
              └──────────────────────┘
                         │
                 last token [:, -1, :]
                         │
               LayerNorm → Linear(32→1)
                         │
                     ε̂ (1,)  — предсказанный шум
```

---

## Результаты (тестовая выборка 2022–2024, 744 дня)

| Метод | VaR(99%) mean | Нарушений, % | Tick loss | Тест Купика |
|---|---|---|---|---|
| Нормальное распределение | 4.769% | 3.09% | 0.04775 |  отклоняется |
| Историческое моделирование | 5.720% | 1.75% | 0.05707 |  не отклоняется |
| GARCH(1,1)-t | 4.649% | 1.08% | 0.04629 |  не отклоняется |
| **DDPM + Трансформер** | **3.263%** | 3.36% | **0.03194 ** |  отклоняется |

**Наилучший Tick loss** — DDPM превосходит GARCH на 31% по квантильной точности.

**Интерпретация:** модель-прототип занижает VaR в экстремальном стресс-периоде 2022–2024 (kurt=71.90 vs 8.46 на train). Tick loss демонстрирует потенциал подхода; калибровка будет улучшена в финальной версии.

---

## Структура репозитория

```
ddpm-tail-risk/
├── prototype.ipynb          # Основной notebook — полный pipeline
├── requirements.txt         # Зависимости
├── assets/
│   ├── torchinfo_summary.txt  # Вывод torchinfo
│   ├── model.onnx             # Граф архитектуры для Netron
│   └── netron_screenshot.png  # Скриншот из Netron
├── results/
│   ├── comparison_table.csv
│   ├── var_backtest.png
│   └── violations_by_method.png
└── .github/
    └── workflows/
        └── ci.yml           # CI: lint + smoke tests
```

---

## Установка и запуск

### 1. Клонировать репозиторий

```bash
git clone https://github.com/YOUR_USERNAME/ddpm-tail-risk.git
cd ddpm-tail-risk
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

### 3. Запустить notebook

```bash
jupyter notebook prototype.ipynb
```

Запустить все ячейки последовательно: **Kernel → Restart & Run All**

> **Примечание:** данные загружаются автоматически с MOEX ISS API.  
> Если файл `portfolio_returns.csv` уже есть — используется кэш.

### 4. Параметры конфигурации (ячейка 5 notebook)

| Параметр | Значение | Описание |
|---|---|---|
| `STEP` | 5 | Частота DDPM-инференса в бэктесте (1 = каждый день, ~6ч на CPU) |
| `MAX_EPOCHS` | 100 | Максимум эпох (early stopping обычно срабатывает раньше) |
| `N_SCENARIOS` | 2000 | Число генерируемых сценариев |
| `T_DIFF` | 200 | Число шагов диффузии |
| `D_MODEL` | 32 | Размерность трансформера |

---

## Архитектура модели — torchinfo

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
TransformerDenoiser                      [1, 1]                    --
├── Linear: token_proj                   [1, 31, 32]               64
├── Embedding: step_emb                  [1, 1, 32]                6,400
├── TransformerEncoder                   [1, 31, 32]               --
│   └── TransformerEncoderLayer (×2)     [1, 31, 32]               --
│       ├── MultiheadAttention           [1, 31, 32]               4,224
│       ├── Linear (FFN expand)          [1, 31, 128]              4,224
│       └── Linear (FFN project)         [1, 31, 32]               4,128
├── LayerNorm: head.0                    [1, 32]                   64
└── Linear: head.1                       [1, 1]                    33
==========================================================================================
Total params: 31,969
Trainable params: 31,969
Non-trainable params: 0
==========================================================================================
```

Полный вывод: [`assets/torchinfo_summary.txt`](assets/torchinfo_summary.txt)

---

## Граф архитектуры — Netron

Граф модели сгенерирован через ONNX-экспорт и визуализирован в [Netron](https://netron.app):

> **Как воспроизвести:**
> ```python
> # В notebook (последняя ячейка) или отдельным скриптом:
> import torch
> torch.onnx.export(
>     model, (xt, t, ctx), "assets/model.onnx",
>     input_names=["x_t", "diffusion_step", "context"],
>     output_names=["predicted_noise"],
>     opset_version=17,
> )
> # Открыть assets/model.onnx на https://netron.app
> ```

📎 Файл графа: [`assets/model.onnx`](assets/model.onnx)

![Netron graph](assets/model_onnx.png)

---

## Описание метода

### Прямой процесс (зашумление)

$$q(x_t \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar\alpha_t}\,x_0,\;(1-\bar\alpha_t)\,I\right)$$

### Обратный процесс (денойзинг)

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t, t, \text{ctx})\right) + \sqrt{\beta_t}\,z$$

### Функция потерь

$$\mathcal{L} = \mathbb{E}_{x_0, t, \varepsilon}\!\left[\|\varepsilon - \varepsilon_\theta(x_t, t, \text{ctx})\|^2\right]$$

### Оценка VaR / CVaR

$$\text{VaR}_\alpha = \inf\{l : P(X \le l) \ge \alpha\}, \quad
\text{CVaR}_\alpha = \mathbb{E}[X \mid X \le \text{VaR}_\alpha]$$

Вычисляется эмпирически по 2000 сгенерированным сценариям.

---

## CI/CD

GitHub Actions запускает при каждом push:

1. **Lint** — проверка стиля через `ruff`
2. **Smoke test: model** — forward pass `TransformerDenoiser`, проверка формы и отсутствия NaN
3. **Smoke test: scheduler** — монотонность $\bar\alpha_t$, корректность `add_noise`
4. **Smoke test: metrics** — VaR > 0, тест Купика при 1% нарушений

Конфигурация: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

---

## Зависимости

| Библиотека | Назначение |
|---|---|
| `torch` | Обучение и инференс DDPM + Transformer |
| `numpy`, `pandas` | Обработка временных рядов |
| `scipy` | Статистические тесты, MLE для GARCH |
| `matplotlib`, `seaborn` | Визуализация |
| `requests` | Загрузка данных с MOEX ISS API |
| `torchinfo` | Вывод архитектуры модели |

---

## Ссылки

1. Ho et al. (2020) — Denoising Diffusion Probabilistic Models
2. Vaswani et al. (2017) — Attention Is All You Need
3. Peebles & Xie (2023) — Scalable Diffusion Models with Transformers
4. Embrechts et al. (1997) — Modelling Extremal Events
