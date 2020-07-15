Оригинальный README [здесь](README.orig.md). Original README [here](README.orig.md).

Ветка `uniform-dynamic-buffers` содержит незаконченную интеграцию ST DCN в Vulkan post-processing layer.

Изменения в vkBasalt включают:

1. Новый эффект [AIST](src/effect_aist.cpp).
1. Составляющие его [слои](src/aist).
1. Сопутствующие [шейдеры](src/shader/aist).
1. [Генератор весов](config/prepare_test_weights.ipynb) для преобразования 1-в-1.

На данный момент ввиду нагрузки, создаваемой неоптимизированным [Instance Norm 2D](src/shader/aist/in_2d.comp.glsl),
не рекомендуется запускать данный эффект, т.к. вероятна потеря контроля над системой.

Детали построения сети доступны в [Jupyter-блокноте](vk-aist.ipynb).

Обновлено 2020-07-11: [коду стилизации видео](style-video.py) требуется порядка 0.4мс
на обработку одного кадра формата FullHD. Далеко до 60к/с, но, вероятно, значительно сказывается перенос данных
между типами памяти и перестановка размерностей входных и выходных тензоров
(один кадр из 32-разрядных чисел с п.т. занимает около 24МиБ).
Используется немного оптимизированная архитектура:
нет нормализации между оконечными слоями,
в них и начальных слоях применено другое распределение по группам для предотвращения потери информации.

`uniform-dynamic-buffers` branch contains an attempt on integration of ST DCN into Vulkan post-processing layer.

Changes to vkBasalt comprise:

1. New [AIST](src/effect_aist.cpp) effect.
1. Its 'neural' [layers](src/aist).
1. Supporting [shaders](src/shader/aist).
1. [Weights generator](config/prepare_test_weights.ipynb) for identity transform.

At the moment, due to enormous load caused by inoptimal [Instance Norm 2D](src/shader/aist/in_2d.comp.glsl),
one should avoid launching this effect, or they might lose control of their system.

2020-07-11 update: [style video code](style-video.py) takes around 0.4ms to process one FullHD frame.
Yes, far from 60fps, but I blame CPU-GPU-CPU transitions and tensor relayouts (one fp32 frame is about 24MiB).
Architecture has been slightly optimized:
no normalization in between upconv layers,
rearranged groupos in downconv layers to avoid information loss.