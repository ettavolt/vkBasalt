Оригинальный README [здесь](README.orig.md). Original README [here](README.orig.md).

Ветка `uniform-dynamic-buffers` содержит незаконченную интеграцию ST DCN в Vulkan post-processing layer.

Изменения в vkBasalt включают:

1. Новый эффект [AIST](src/effect_aist.cpp).
1. Составляющие его [слои](src/aist/nn_shaders.h).
1. Сопутствующие [шейдеры](src/shader/aist).
1. [Генератор весов](config/prepare_test_weights.ipynb) для преобразования 1-в-1.

Ввиду неоптимизированного доступа к Storage buffer всего лишь шесть свёрточных слоёв и один нормирующий
(3×15 down, 15×16 shuffle, IN2D, 16×64 down, 64×16 up, 16×15 shuffle, 15×3 up)
создают нагрузку, с трудом обрабатываемую GTX 1060 6G:
vulkanscene из Sascha Williems'es samples (HD, 1280×720) обновляется с частотой около сорока кадров в секунду.

Детали построения сети доступны в [Jupyter-блокноте](vk-aist.ipynb).

Обновлено 2020-07-11: [коду стилизации видео](style-video.py) требуется порядка 0.4мс
на обработку одного кадра формата FullHD. Далеко до 60к/с, но, вероятно, значительно сказывается перенос данных
между типами памяти и перестановка размерностей входных и выходных тензоров
(хотя один кадр из 32-разрядных чисел с п.т. занимает около 24МиБ).

`uniform-dynamic-buffers` branch contains an attempt on integration of ST DCN into Vulkan post-processing layer.

Changes to vkBasalt comprise:

1. New [AIST](src/effect_aist.cpp) effect.
1. Its 'neural' [layers](src/aist/nn_shaders.h).
1. Supporting [shaders](src/shader/aist).
1. [Weights generator](config/prepare_test_weights.ipynb) for identity transform.

Due to inoptimal Storage Buffer access just six convolutional and one normalizing layers
(3×15 down, 15×16 shuffle, IN2D, 16×64 down, 64×16 up, 16×15 shuffle, 15×3 up)
cause GTX 1060 6G to output just 40fps of Sascha Williems'es vulkanscene sample (HD, 1280×720).

2020-07-11 update: [style video code](style-video.py) takes around 0.4ms to process one FullHD frame.
Yes, far from 60fps, but I blame CPU-GPU-CPU transitions and tensor relayouts (though, one fp32 frame is about 24MiB).