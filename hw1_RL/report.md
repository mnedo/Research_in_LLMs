## Исходный код ##
[Notebook with calculations]()

## Границы экспеирементов ##
В рамках домашнего задания был поставлен ряд экспериментов с алгоритмами RL в окружении CartPole из gymnasium. Модель должна делать выбор между движениями тележки, не уронив шеста. Каждый эпизод ограничен 500 командами.

Политика - FCN модель. На вход подается 4 числа - параметры состояния среды, на выходе модель выдает оценку каждому действию - логит вероятности выбора каждого действия.
![Архитектра PolicyModel]()

## Алгоритмы RL ##
При обучении алгоритма использовались:
* Vanila Policy Gardient алгоритм. $loss = log{\pi (a_t | s_t )} \cdot G_t$
* Policy Gardient with baseline. $loss = log{\pi (a_t | s_t )} \cdot (G_t- b)$
  *  $Baseline = mean(G_t)$
  *  $Baseline = ValueModel(s_t)$
  *  $Baseline = mean(G_i), i \in [1, ..., t-1, t+1, ...]$

Для начального сравнения каждая из модель начинала учиться с одинаковым seed и просматривала $5000$ эпизизодов, с $learinig \ rate = 10^{-3}$ и $\gamma = 1 - 10^{-2}$:

![Результаты]()
По резльутатам запуска:
| RL Алгоритм  | Количество эпизода до плато | Средняя награда за последине 100 эпизодов |
| :-: | :-: | :-: |
| VPG |  |  |
| PG, b = mean |  |  |
| PG, b = ValueModel |  |  |
| PG, RLOO |  |  |

Выводы: ...

### Также были проведены исследования с изменением гипераметров и добавлением регулярзиации на энтропию ### 

Таблица с результатами добавления энтропии
| VPG  | PG, b = mean | PG, b = ValueModel | PG, RLOO |
| :-: | :-: | :-: |  :-: |
| ![]() | ![]() | ![]() | ![]() |

Выводы: ...

Таблица с изменением $learing \ rate$
| VPG  | PG, b = mean | PG, b = ValueModel | PG, RLOO |
| :-: | :-: | :-: |  :-: |
| ![]() | ![]() | ![]() | ![]() |

Выводы: ...

Таблица с изменением $\gamma$
| VPG  | PG, b = mean | PG, b = ValueModel | PG, RLOO |
| :-: | :-: | :-: |  :-: |
| ![]() | ![]() | ![]() | ![]() |

Выводы: ...

## BEHAVIOUR CLONING ##

Для экспертиментов BC был взят эксперт - PG-RLOO, обученный на $2000$ эпизодах с разными seed. В среднем expert получает 100% награду на первых 100 действиях. Новая модель была обучена при помощи кросс-энтропии на выводах эксперта о полученном state.

Целю экспериментов было отобразить уязвимость такого алгоритма обучения. Главная уязвимость - при диситиляции модель сталкивается с меньшим количеством ситауции и получает генерализацю хуже.

После обучения на всех эпизодах без ограничений, модели удавалось выжить в 90% случаев
  ![Результаты обучения без ограничений](https://github.com/mnedo/Research_in_LLMs/blob/56d320f10854d567c2f1bedd265877cbe014b381/hw1_RL/images/ResAll.png)

В целях показать уязвимость неполного покрытия пространства возможных собятий, было приянто решение искуственно ограничить данные в датасете. Не были взяты хвосты распредение параметров среды

Таблица с распределниями

| ![](https://github.com/mnedo/Research_in_LLMs/blob/56d320f10854d567c2f1bedd265877cbe014b381/hw1_RL/images/DistOfPosition.png) | ![](https://github.com/mnedo/Research_in_LLMs/blob/56d320f10854d567c2f1bedd265877cbe014b381/hw1_RL/images/DistOfVelocity.png) | 
| :-: | :-: |
| ![](https://github.com/mnedo/Research_in_LLMs/blob/56d320f10854d567c2f1bedd265877cbe014b381/hw1_RL/images/DistOfPoleAngle.png) | ![](https://github.com/mnedo/Research_in_LLMs/blob/56d320f10854d567c2f1bedd265877cbe014b381/hw1_RL/images/DistOfAngVelocity.png) | 

Результаты обчения с ограничениями
| ![](https://github.com/mnedo/Research_in_LLMs/blob/56d320f10854d567c2f1bedd265877cbe014b381/hw1_RL/images/ResPos.png) | ![](https://github.com/mnedo/Research_in_LLMs/blob/56d320f10854d567c2f1bedd265877cbe014b381/hw1_RL/images/ResVel.png) | 
| :-: | :-: |
| ![](https://github.com/mnedo/Research_in_LLMs/blob/56d320f10854d567c2f1bedd265877cbe014b381/hw1_RL/images/ResAng.png) | ![](https://github.com/mnedo/Research_in_LLMs/blob/56d320f10854d567c2f1bedd265877cbe014b381/hw1_RL/images/ResAngVel.png) | 

Выводы: ...
