# Исследование моделей удаления фона

Актуальность обзора: **18 августа 2026 года**. Область применения — веб-сервис автоматического удаления фона с фотографий. Реализация сервиса, собственное тестирование и обучение моделей в этот этап не входят.

## 1. Executive Summary

Единого SOTA для «удаления фона» нет: binary/soft segmentation, dichotomous image segmentation (DIS) и alpha matting оцениваются на разных данных и отвечают на разные вопросы. Высокий F-measure на DIS5K не гарантирует корректную полупрозрачность стекла или отдельных волос, а низкий SAD на matting benchmark обычно предполагает trimap, prompt либо ограничение на портреты.

Ключевые выводы:

- **Research SOTA для полностью автоматического DIS:** **PDFNet-L + Depth Anything V2** (CVPR 2026), `Fβmax=0.915` на DIS-VD и `0.915` на объединённом DIS-TE. Это наиболее сильный опубликованный результат среди рассмотренных автоматических DIS-подходов на одном и том же benchmark, но двухмодельный pipeline тяжёлый: около `94M + 335M` параметров, 3.9 FPS на оборудовании авторов и около минуты на CPU в официальном demo.
- **Research SOTA для alpha matting:** свежий trimap-guided **RenderMatte** заявляет SOTA сразу на нескольких matting benchmark, особенно на strand-level границах, но это arXiv от августа 2026 года на базе full fine-tune FLUX.1 Kontext; открытых production-весов и зрелого deployment-пути на дату обзора нет. Это направление для наблюдения, а не кандидат на запуск.
- **Practical SOTA:** семейство **BiRefNet**, прежде всего официальный general checkpoint и его `lite`/`HR-matting` варианты. Оно сочетает сильные тонкие границы, 1024/2048 inference, PyTorch/Hugging Face/ONNX, MIT для официального кода и весов и возможность подобрать размер под GPU или CPU.
- **Модель для проекта:** **`ZhengPeng7/BiRefNet` (general, полный вариант, ~0.2B параметров)** как основной GPU inference. Это не абсолютный research SOTA, зато существенно проще PDFNet/DiffDIS, коммерчески разрешён по MIT и лучше соответствует универсальным фотографиям, чем portrait-only MODNet.
- **Запасные варианты:** `ZhengPeng7/BiRefNet_lite` (44.4M) при жёстком бюджете latency/RAM/CPU и **InSPyReNet** как зрелый MIT-пакет с удобным CLI/API. Если допустима платная коммерческая лицензия/API, **BRIA RMBG-2.0** — сильная альтернатива с более разнообразным proprietary training set.

Перед окончательной фиксацией production-модели обязателен собственный bake-off на целевом наборе: люди и волосы, животные и шерсть, товары, тонкие конструкции, полупрозрачные/отражающие объекты, несколько объектов, сложный фон и изображения без очевидного foreground. Публичные benchmark не моделируют весь продуктовый трафик.

## 2. Background removal vs segmentation vs matting

Для композитинга наблюдаемый цвет пикселя описывают как `I = αF + (1-α)B`, где `F` — foreground, `B` — background, `α∈[0,1]` — прозрачность. Задачи различаются выходом и постановкой:

- **Semantic/instance segmentation** выдаёт класс или объект для каждого пикселя, обычно бинарную маску. Она хорошо отделяет предмет, но не обязана восстанавливать физически осмысленный alpha на волосах, motion blur, дыме или стекле.
- **Salient object detection (SOD)** автоматически находит визуально заметный foreground. U²-Net и InSPyReNet относятся именно сюда. Это удобная постановка для one-click removal, но «заметный объект» не всегда совпадает с тем, что пользователь хочет сохранить.
- **Dichotomous image segmentation (DIS)** — высокоточная class-agnostic бинарная сегментация объектов со сложными деталями. DIS5K специально содержит тонкие и сложные структуры. IS-Net, BiRefNet, MVANet, DiffDIS, BEN и PDFNet принадлежат к этому направлению.
- **Image matting** оценивает непрерывный alpha, поэтому лучше соответствует волосам, шерсти, расфокусу и полупрозрачности. Классические модели требуют trimap; trimap-free модели обычно ограничены доменом (например, портреты) либо сначала получают грубую semantic mask.
- **Promptable segmentation/matting** принимает click, box, mask или text. SAM/SAM2 хорошо выбирают нужный объект, а MAM/Matte Anything превращают mask в alpha. Это сильный режим ручной коррекции, но не эквивалент полностью автоматическому удалению фона.

Основные метрики также нельзя смешивать. В SOD/DIS используют `Fβmax`, weighted F-measure, MAE (`M`), S-measure, E-measure и HCE (human correction effort). В matting — SAD, MSE/MAD, Gradient и Connectivity; ниже обычно лучше. SAM2 для видео публикует J&F. Значения ниже сравниваются только внутри одного benchmark и протокола.

## 3. Современные подходы

1. **Nested CNN encoder-decoder.** U²-Net строит residual U-blocks внутри U-образной сети и получает большой receptive field без внешнего backbone. Это простой и переносимый baseline, но его SOD-маска не является настоящим alpha matte.
2. **Feature supervision и high-resolution pyramids.** IS-Net обучает промежуточные признаки student-сети через ground-truth encoder; InSPyReNet смешивает low/high-resolution пирамиды. Оба подхода заметно лучше сохраняют тонкие детали старых SOD-сетей.
3. **Trimap-free portrait matting.** MODNet разлагает задачу на semantics, details и fusion и работает в real time, но prior «на изображении человек» ограничивает универсальный сервис.
4. **Transformer DIS с локально-глобальным refinement.** MVANet обрабатывает distant/close-up views, BiRefNet использует bilateral reference и reconstruction-informed refinement. Это сильный класс автоматических универсальных remover-моделей.
5. **Segmentation + confidence-guided matting.** BEN/BEN2 сначала строит маску, затем уточняет неуверенные области. Идея прямо оптимизирует слабое место DIS — границы, однако публичные артефакты и benchmark менее зрелые, чем у BiRefNet.
6. **Foundation-model priors.** DiffDIS использует one-step diffusion U-Net; PDFNet добавляет pseudo-depth от Depth Anything V2. Качество высокое, но стоимость и число зависимостей возрастают.
7. **Promptable pipelines.** SAM/SAM2 решают выбор объекта; MAM или Matte Anything добавляют mask-to-matte/ViTMatte refinement. Это лучший кандидат для будущего режима «поправить результат кликом», но сложнее и тяжелее one-click endpoint.
8. **Generative matting.** SDMatte и RenderMatte используют priors diffusion/flow-моделей и достигают очень сильных matting-границ. Пока это преимущественно research/interactive класс с высокой стоимостью inference.

## 4. Кандидаты

### 4.1 U²-Net

- **Год/архитектура:** 2020, двухуровневая nested U-structure из ReSidual U-blocks; версии U²-Net и U²-NetP.
- **Задача:** SOD, не alpha matting.
- **Качество:** на момент публикации SOTA/competitive на DUTS-TE, ECSSD, DUT-OMRON, HKU-IS, PASCAL-S и SOD. Paper сообщает 30 FPS для 176.3 MB модели и 40 FPS для 4.7 MB U²-NetP, но это старый протокол и не сопоставимо с современными 1024² моделями.
- **Границы:** хорошо сохраняет структуры относительно старых SOD, но binary saliency и типичный input 320² теряют отдельные волосы; официальный human checkpoint прямо предупреждает, что не даёт hair-level accuracy.
- **Размер/ресурсы:** веса 176.3 MB или 4.7 MB. Опубликованных peak RAM/VRAM нет.
- **Runtime/deployment:** PyTorch, многочисленные ONNX/CoreML-порты, CPU и GPU; очень простой deployment, но официальный код основан на старом stack.
- **Лицензия:** официальный репозиторий и распространяемые веса — Apache-2.0; коммерческое использование допускается с выполнением условий attribution/notice.
- **Вердикт:** отличный baseline и малый fallback, но качество ниже современных DIS-моделей.

### 4.2 IS-Net / DIS

- **Год/архитектура:** ECCV 2022; intermediate supervision между segmentation student и GT encoder.
- **Задача:** DIS.
- **Качество:** официальный paper на общем DIS-TE: `Fβmax 0.799`, weighted F `0.726`, MAE `0.070`, S-measure `0.819`, E-measure `0.858`, HCE `1016`. Эти числа можно сравнивать с последующими работами, использующими тот же DIS5K protocol.
- **Границы:** DIS5K включает сложные и тонкие структуры; заметный шаг вперёд относительно U²-Net. Это всё ещё soft segmentation, не физическое matting полупрозрачных материалов.
- **Размер/скорость:** 176.6 MB; paper приводит 19.49 ms при 1024² в своей таблице, но hardware/context нужно читать вместе с paper и не переносить как production SLA. Peak RAM/VRAM не опубликованы.
- **Runtime/deployment:** PyTorch, CPU/GPU, существуют ONNX-порты. Официальный general-use checkpoint удобен, но официальный DIS checkpoint обучен на наборе с малым числом людей/животных/машин; DIS v2 weights не опубликованы.
- **Лицензия:** Apache-2.0 для официального repo; коммерческое использование допускается. Для конкретного зеркала весов лицензию надо проверять отдельно.
- **Вердикт:** зрелый сильный baseline, уже обойдён MVANet/BiRefNet/PDFNet.

### 4.3 MODNet

- **Год/архитектура:** AAAI 2022 (arXiv 2020); lightweight CNN, objective decomposition на semantic, detail и fusion branches, Self-Supervised Sub-Objectives Consistency adaptation.
- **Задача:** trimap-free **portrait matting**.
- **Качество/latency:** PPM-100 и Adobe Matting; paper заявляет real-time inference, около 67 FPS при 512² на GTX 1080 Ti и менее 1M параметров. Это нельзя сравнивать с DIS5K или SAM2 FPS.
- **Границы:** хороший непрерывный alpha для волос и мягких границ человека; существенно хуже как universal remover для товаров, животных, нескольких неизвестных объектов и transparent products.
- **Размер/ресурсы:** официальный production-like «7M» demo-model не опубликован; публичный research checkpoint порядка нескольких MB. Peak RAM/VRAM не опубликованы.
- **Runtime/deployment:** очень лёгкий CPU/GPU; есть официальные ONNX/TorchScript материалы и сторонний TensorRT. Deployment простой.
- **Лицензия:** README сейчас говорит Apache-2.0 для code, models и demos. Однако ранние сторонние интеграции цитируют CC BY-NC-SA для pretrained model; перед коммерческим включением конкретного checkpoint нужен file-level legal audit и фиксация commit/hash.
- **Вердикт:** practical SOTA для узкого portrait-only/real-time сценария, но не основной универсальный сервис.

### 4.4 InSPyReNet

- **Год/архитектура:** ACCV 2022; Inverse Saliency Pyramid Reconstruction Network, low/high-resolution pyramid blending.
- **Задача:** high-resolution SOD; есть checkpoint, обученный на DIS5K.
- **Качество:** paper показывает SOTA на LR/HR SOD и улучшение boundary accuracy; DIS checkpoint конкурентен IS-Net. Метрики SOD и DIS следует читать в соответствующих таблицах, а не переносить между наборами.
- **Границы:** high-resolution pyramid помогает волосам, шерсти и тонким структурам; всё же результат — saliency alpha-like mask, а не гарантированно корректная прозрачность.
- **Размер/ресурсы:** зависит от Res2Net/Swin backbone и режима; единый официальный peak RAM/VRAM/latency для web inference не опубликован.
- **Runtime/deployment:** PyTorch, CPU/CUDA/MPS; зрелый пакет `transparent-background`, CLI и Python API. CPU заметно медленнее GPU, но эксплуатационный путь проще многих research-моделей.
- **Лицензия:** MIT для кода и официального проекта; коммерческое использование допускается, но hash и provenance checkpoint следует задокументировать.
- **Вердикт:** хороший запасной production-вариант и CPU-friendly operational baseline.

### 4.5 MVANet

- **Год/архитектура:** CVPR 2024 Highlight; Swin-based single encoder-decoder с distant view, четырьмя close-up patches, MCLM и MCRM.
- **Задача:** DIS.
- **Качество:** на DIS5K paper сообщает `Fβmax≈0.904` на DIS-VD и `≈0.908` на DIS-TE(all) в исправленном протоколе; 93M параметров. Авторы подчёркивают выигрыш и в точности, и в скорости относительно предшественников.
- **Границы:** close-up views специально восстанавливают slender structures, поэтому модель сильна на волосах/шерсти/тонких деталях; transparency остаётся неявной.
- **Размер/ресурсы:** ~93M параметров (~372 MB только FP32 weights, расчёт по числу параметров); официального универсального peak VRAM/RAM нет.
- **Runtime/deployment:** официальный код привязан к старому PyTorch 1.10/CUDA 10.2/mmcv-full, custom multi-view pipeline; CPU теоретически возможен, практически тяжёлый. Deployment сложнее BiRefNet/HF.
- **Лицензия:** MIT repo; коммерческое использование кода разрешено. Лицензию скачиваемого Google Drive checkpoint стоит архивировать отдельно.
- **Вердикт:** сильный research-кандидат, но integration debt выше BiRefNet.

### 4.6 BiRefNet

- **Год/архитектура:** CAAI AIR 2024; transformer/CNN encoder-decoder с bilateral reference: semantic localization и reconstruction-informed detail refinement.
- **Задача:** DIS/SOD/COD и отдельные matting fine-tunes; автоматическая one-click mask.
- **Качество:** исходная работа показывает SOTA на DIS5K, HRSOD и COD; последующие одинаково протоколированные таблицы дают для full model около `Fβmax 0.897` на DIS-VD и `0.900` на DIS-TE(all). Это чуть ниже MVANet/PDFNet на DIS5K, но официальный general checkpoint дообучен для реальных изображений, а не только benchmark.
- **Границы:** high-resolution branches и reconstruction loss хорошо сохраняют волосы, шерсть и тонкие контуры; `HR-matting` вариант лучше для soft alpha. Стекло, дым и foreground/background одинакового цвета остаются сложными.
- **Размер/ресурсы:** full около 0.2B/≈220M параметров (примерно 0.88 GB FP32 weights); `lite` 44.4M (примерно 178 MB FP32). Есть 1024², 2048² и dynamic checkpoints. Официальный inference peak RAM/VRAM и единая latency не опубликованы; training README сообщает десятки GB, но это нельзя выдавать за inference requirement.
- **Runtime/deployment:** PyTorch/Transformers одной строкой, CUDA и CPU, официальные ONNX exports; full на CPU медленный, lite реалистичнее. `trust_remote_code=True` — supply-chain риск: в production нужно pin revision и vendor/audit model code.
- **Лицензия:** официальный GitHub и HF model cards — MIT, включая официальные weights; коммерческое использование разрешено с сохранением notice.
- **Вердикт:** лучший общий production-компромисс в этом обзоре.

### 4.7 BRIA RMBG-2.0

- **Год/архитектура:** 2024; BiRefNet (~0.2B) с proprietary curated dataset/training scheme для stock, e-commerce, gaming и advertising.
- **Задача:** automatic background removal / DIS-like soft mask.
- **Качество:** model card даёт в основном qualitative comparison; BRIA заявляет 90% против 85% у BiRefNet на собственной оценке. Без опубликованного набора, разметки и полного протокола это маркетинговое свидетельство, не сопоставимый public benchmark.
- **Границы:** визуально сильна на волосах, шерсти и товарах; soft output полезен для antialiasing. Независимой открытой проверки transparent materials недостаточно.
- **Размер/ресурсы:** 0.2B/около 221M параметров; ONNX-файл около 1.02 GB. Официальной воспроизводимой latency и peak RAM/VRAM нет. PyTorch поддерживает CPU/GPU, рекомендуемый input 1024².
- **Runtime/deployment:** удобный Transformers API, но gated download и `trust_remote_code`; доступны API/ONNX. Self-hosting технически среднее по сложности.
- **Лицензия:** weights на HF доступны **только для non-commercial evaluation/use** по custom BRIA license; коммерческий self-hosting требует отдельного договора, API включает коммерческие права. Open-source формулировка на странице не означает OSI-compatible license.
- **Вердикт:** вероятно очень сильное качество, но не подходит как бесплатная self-hosted основа коммерческого сервиса.

### 4.8 BEN / BEN2

- **Год/архитектура:** BEN paper — arXiv 2025; BEN Base (~94M) + Confidence-Guided Matting refiner. Публичный `BEN2` упаковывает обновлённый base/inference и optional foreground refinement; архитектура использует Swin-like attention, multi-scale contextual modules и локальные 1024² patches.
- **Задача:** DIS с matting-guided refinement.
- **Качество:** paper заявляет первое место на DIS5K validation относительно тогдашних методов; поскольку главный результат дан на validation и работа не является принятой peer-reviewed публикацией, его нельзя объявлять общим SOTA поверх CVPR 2026 PDFNet. Независимого стандартизованного benchmark BEN2 мало.
- **Границы:** confidence refinement концептуально полезен именно для волос, шерсти и uncertainty boundary; optional foreground refinement также борется с background color spill. Output всё ещё зависит от base mask.
- **Размер/ресурсы:** BEN/BEN2 около 94–95M; HF safetensors 381 MB, ONNX 223 MB, отдельный `.pth` 1.13 GB. Опубликованных latency/peak RAM/VRAM нет.
- **Runtime/deployment:** PyTorch и ONNX; публичный код декорирован CUDA autocast, поэтому CPU требует проверки/небольшой адаптации. Model card и package моложе BiRefNet, артефакты дублируются и велики.
- **Лицензия:** HF BEN2 обозначен MIT (BEN card — Apache-2.0 на момент обзора); коммерчески разрешительно, но из-за различий между cards нужно pin конкретный BEN2 revision и сохранить его LICENSE/model card.
- **Вердикт:** перспективный challenger для внутреннего bake-off, но пока не default.

### 4.9 SAM / SAM2 + matting refinement

- **Год/архитектура:** SAM (2023) — prompt encoder + ViT image encoder + lightweight mask decoder; SAM2 (2024, ICLR 2025) — Hiera encoder, promptable decoder и streaming memory для видео. SAM2.1: 38.9M–224.4M параметров.
- **Задача:** promptable instance segmentation, не automatic foreground matting. Автоматический mask generator создаёт все маски, но выбор «главного» foreground требует ranking/detector/text model.
- **Качество/latency:** SAM2.1 публикует J&F на SA-V/MOSE/LVOS и 91.2 FPS (tiny) — 39.5 FPS (large), A100 + compiled PyTorch; это video segmentation metric и hardware, не background-removal benchmark. SAM/SAM2 часто дают резкую binary boundary и пропускают полупрозрачность.
- **Границы:** SAM2 лучше выбирает объект и устойчив к novel categories, но волосы/шерсть требуют alpha refiner. MAM (SAM + 2.7M M2M) и Matte Anything (SAM + GroundingDINO + ViTMatte) превращают prompt mask в matte и улучшают transition regions/transparent materials.
- **Размер/ресурсы:** SAM ViT-H ~636M/~2.6 GB, ViT-B ~91M/~375 MB; SAM2.1 tiny 38.9M, large 224.4M. Полный Matte Anything pipeline включает несколько checkpoints, поэтому RAM/VRAM и cold start значительно выше одной DIS-модели.
- **Runtime/deployment:** CUDA — основной путь; CPU возможен, но тяжёл. SAM decoder эффективен после кэширования image embedding. Интеграция matting pipeline сложная, зато даёт UX с click/box correction.
- **Лицензия:** SAM и SAM2 code/checkpoints — Apache-2.0; MAM и Matte Anything repos — MIT. Для комбинации нужно отдельно проверить GroundingDINO, ViTMatte и конкретные weights.
- **Вердикт:** не primary one-click remover; лучшая архитектурная опция для второго этапа — интерактивного выбора/исправления объекта.

### 4.10 DiffDIS

- **Год/архитектура:** ICLR 2025; SD-Turbo U-Net, однократное denoising и дополнительная edge-generation branch.
- **Задача:** DIS.
- **Качество:** на DIS5K около `Fβmax 0.908` (DIS-VD), `0.911` (DIS-TE all) и MAE `0.027` в сравнительной таблице PDFNet; сильнее MVANet/BiRefNet по части метрик в том же протоколе.
- **Границы:** diffusion prior даёт global semantics, edge branch — тонкие детали. Это segmentation output, а не гарантированная оптика прозрачных объектов.
- **Размер/ресурсы:** около `865M + 84M` параметров с auxiliary components; около 0.8 FPS по сравнению в PDFNet. Peak VRAM не опубликован в универсальной форме, но класс заведомо тяжелее discriminative DIS.
- **Runtime/deployment:** требует SD-Turbo, fork diffusers, CUDA-oriented environment и несколько checkpoints; CPU fallback непрактичен. Сложность высокая.
- **Лицензия:** код repo MIT, но итоговый коммерческий статус определяется также лицензией SD-Turbo weights и зависимостей; нужен отдельный license audit.
- **Вердикт:** research SOTA 2025, вытеснен PDFNet по балансу и части качества.

### 4.11 PDFNet

- **Год/архитектура:** CVPR 2026; Prior-guided Depth Fusion Network со Swin-S/B/L, pseudo-depth от Depth Anything V2, cross-attention RGB-depth, integrity loss и adaptive fine-grained patches.
- **Задача:** automatic high-precision DIS.
- **Качество:** PDFNet-L + DAM-v2: `Fβmax=0.915` на DIS-VD и `0.915` на DIS-TE(all); это опубликованный SOTA result работы. Следует сравнивать только с результатами того же DIS5K protocol.
- **Границы:** patch enhancement и depth boundary prior специально повышают точность тонких структур и целостность объекта. Настоящая полупрозрачность не моделируется.
- **Размер/скорость:** PDFNet-L 94M + Depth Anything V2 ViT-L 335M; 3.9 FPS в paper. Официальный HF Space в CPU mode сообщает около минуты на изображение. Peak RAM/VRAM не опубликован.
- **Runtime/deployment:** два этапа, depth preprocessing, 1024², Swin и custom modules; GPU практически обязателен для SLA. MIT repo, но необходимо учитывать Apache-2.0 Depth Anything V2 code/weights и backbone licenses.
- **Лицензия:** PDFNet code MIT; коммерческая совместимость в целом выглядит разрешительной, но полный dependency/weights BOM нужно зафиксировать.
- **Вердикт:** **research SOTA automatic DIS**, пока не practical SOTA веб-сервиса.

### 4.12 Новейшие matting-сигналы: SDMatte и RenderMatte

- **SDMatte (ICCV 2025):** grafting Stable Diffusion для interactive matting; на AIM-500 natural paper сообщает, например, MSE `0.0027`, MAD `0.0087`, SAD `14.53` для полной модели, лучше опубликованных interactive baselines в той же таблице. Требует mask/interaction, diffusion backbone и тяжёлый deployment.
- **RenderMatte (arXiv, август 2026):** trimap-guided full fine-tune FLUX.1 Kontext, alpha-edge objective и group-relative alpha alignment на синтетическом exact-alpha RenderMatte dataset. Авторы заявляют SOTA across benchmarks и strand-level alpha. На дату среза нет зрелых открытых весов/репозитория и peer review; FLUX-лицензия и стоимость inference требуют отдельной проверки.
- **Вердикт:** это ориентир качества для будущей версии с ручной коррекцией, но не shortlist для первого production release.

## 5. Сравнительная таблица

`н/о` — не опубликовано первичным источником. Размер FP32, вычисленный как `params×4 bytes`, помечен `≈` и не равен peak RAM/VRAM.

| Модель | Год | Тип | Public quality (только свой benchmark) | Размер | CPU / GPU | Published latency | Лицензия code / weights | Deployment |
|---|---:|---|---|---:|---|---|---|---|
| U²-Net / U²-NetP | 2020 | SOD | 6 SOD benchmark; старый SOTA | 176.3 / 4.7 MB | да / да | 30 / 40 FPS, paper hardware | Apache-2.0 / Apache-2.0 | низкая |
| IS-Net | 2022 | DIS | DIS-TE: Fmax .799, MAE .070, HCE 1016 | 176.6 MB | да / да | 19.49 ms, paper setup | Apache-2.0 / repo release | низкая–средняя |
| MODNet | 2022 | portrait matting | PPM-100, Adobe Matting | <1M params; checkpoint несколько MB | да / да | ~67 FPS, 512², GTX 1080 Ti | Apache-2.0 / требует file audit | низкая |
| InSPyReNet | 2022 | HR-SOD/DIS | SOTA HR-SOD; competitive DIS | variant-dependent | да / CUDA/MPS | н/о для target setup | MIT / model-zoo provenance | низкая–средняя |
| MVANet | 2024 | DIS | DIS-VD .904; DIS-TE .908 Fmax | 93M, ≈372 MB FP32 | формально да / да | paper claims faster SOTA; target н/о | MIT / проверить artifact | высокая |
| BiRefNet full/lite | 2024 | DIS + matting variants | DIS-VD ~.897; DIS-TE ~.900; сильный general checkpoint | ~220M / 44.4M; ≈880/178 MB | да / да | н/о | MIT / MIT | **средняя** |
| BRIA RMBG-2.0 | 2024 | background removal | qualitative + закрытая 90% оценка | 221M; ONNX ~1.02 GB | да / да | н/о | custom / non-commercial | средняя + license blocker |
| BEN/BEN2 | 2025 | DIS + CGM | заявлен #1 DIS5K-val в paper | ~94–95M; 381 MB safetensors | требует адаптации / да | н/о | MIT (BEN2) / MIT card | средняя–высокая |
| SAM2.1 + MAM/MatAny | 2024–25 | prompt segmentation + matting | J&F или matting benchmark, не DIS | 38.9–224.4M + refiner(s) | формально да / да | 91.2–39.5 FPS на A100, video | Apache-2.0 + MIT / mixed BOM | высокая |
| DiffDIS | 2025 | diffusion DIS | DIS-VD .908; DIS-TE .911; MAE .027 | 865M+84M | непрактично / да | ~0.8 FPS, paper comparison | MIT / зависит от SD-Turbo | очень высокая |
| PDFNet-L + DAM-v2 | 2026 | RGB-depth DIS | **DIS-VD .915; DIS-TE .915 Fmax** | 94M+335M | ~1 мин / да | 3.9 FPS, paper | MIT + Apache-2.0 / mixed BOM | высокая |
| SDMatte / RenderMatte | 2025–26 | interactive/trimap matting | SOTA claims на AIM-500/AM-2K и др. | diffusion/FLUX-class | непрактично / да | не для low-latency endpoint | mixed / требует audit | очень высокая |

RAM/VRAM почти нигде не стандартизованы: они зависят от resolution, dtype, batch, allocator, framework и export backend. Поэтому в таблице не подменяются размером весов. Для sizing сервиса нужны собственные замеры peak RSS/VRAM при batch 1 и целевом максимальном разрешении.

## 6. Research SOTA

### Automatic foreground / DIS

**PDFNet-L + Depth Anything V2** — наиболее обоснованный research SOTA для рассматриваемого one-click сценария. Работа принята на CVPR 2026, код и weights опубликованы, а результат `0.915 Fβmax` показан и на DIS-VD, и на DIS-TE(all). Она превосходит приведённые в том же протоколе MVANet (`.904/.908`), BiRefNet (`.897/.900`) и DiffDIS (`.908/.911`).

Ограничения вывода: DIS5K — не alpha-matting benchmark; F-score не измеряет качество полупрозрачного стекла, дыма и color decontamination. Depth Anything добавляет собственные ошибки domain shift и существенную стоимость.

### Alpha matting

Для **trimap-guided open-world matting** наиболее свежий research claim — RenderMatte, но он пока слишком новый и закрыт с точки зрения воспроизводимого deployment. Из опубликованных и доступных interactive systems SDMatte показывает очень сильные результаты на AIM-500/AM-2K. Эти методы нельзя напрямую сравнивать с PDFNet: они получают mask/trimap и оптимизируют alpha metrics, тогда как PDFNet сам выбирает foreground.

## 7. Practical SOTA

**BiRefNet family** даёт лучший общий баланс:

- качество современных high-resolution DIS и отдельные general/portrait/matting/HR checkpoints;
- один forward pass без отдельного depth/diffusion foundation model;
- полный вариант для GPU и 44.4M `lite` для ограниченных ресурсов;
- официальный Transformers interface и ONNX exports;
- CPU fallback существует, хотя full не следует считать быстрым на CPU;
- MIT одновременно в официальном GitHub и HF cards — нет non-commercial blocker;
- активная экосистема и уже готовые интеграции.

Практический SOTA не означает, что BiRefNet выигрывает каждую фотографию. RMBG-2.0 может быть лучше обучен на e-commerce/advertising distribution, MODNet будет гораздо быстрее на портретах, а PDFNet даст лучший DIS5K score. Но первый требует коммерческого договора, второй не универсален, третий заметно тяжелее.

## 8. Рекомендуемая модель для проекта

Рекомендуется **`ZhengPeng7/BiRefNet` general, full (~0.2B)** как основной checkpoint веб-сервиса на GPU.

Почему:

1. Универсальная автоматическая постановка соответствует endpoint «загрузить фото → получить PNG с alpha» без click, text или trimap.
2. Качество тонких high-resolution границ ближе к современному SOTA, чем U²-Net/IS-Net/MODNet, а real-world general checkpoint полезнее чисто академического DIS5K checkpoint.
3. Один model service проще и дешевле PDFNet (BiRefNet против BiRefNet-like network + Depth Anything V2) и DiffDIS.
4. MIT не блокирует коммерческий self-hosting; у RMBG-2.0 этот blocker существенный.
5. Есть понятный путь оптимизации: FP16, pinned HF revision, ONNX/TensorRT после проверки parity; при дефиците ресурсов — drop-in переход на `BiRefNet_lite`.

Предлагаемый quality/production компромисс: начинать с full general на GPU ради качества; не объявлять CPU тем же SLA. Если обязательна быстрая работа без GPU, использовать **BiRefNet_lite как отдельный degraded/fallback tier**, а не незаметно отправлять full model на CPU.

### Практическая проверка выбранной модели (18 августа 2026)

Официальный `ZhengPeng7/BiRefNet` revision
`e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4` был реально загружен и выполнен в
локальном Windows/Python 3.11 окружении. Доступен только PyTorch 2.13.0 CPU
(`torch.cuda.is_available() == False`, 6 CPU threads). Получены 220 176 498
параметров и 880 705 992 байта FP32 parameter storage.

Технический smoke test на синтетическом RGB 320×240 при model input 512² дал:

- cached model load: 40.88 с;
- первый inference с формированием RGBA: 8.99 с;
- повторный inference с контрастным синтетическим объектом: 6.90 с, alpha
  охватил полный диапазон 0–255;
- output: RGBA исходного размера 320×240.

Это **не продуктовый benchmark качества или latency**: одно синтетическое
изображение, один прогон, уменьшенный model input и отсутствие GPU. Измерение
только подтверждает работоспособность full checkpoint и CPU fallback. Оно не
меняет выбор основной GPU-модели, но подтверждает, что full BiRefNet нельзя
обещать как low-latency CPU tier. Для CPU следует отдельно измерить
`BiRefNet_lite` на репрезентативных фотографиях.

При интеграции обнаружена неочевидная обязательная зависимость `einops` в
официальном remote model code; она добавлена в project dependencies. Веса
кэшируются на Windows без symlink при выключенном Developer Mode, что повышает
расход диска. Production должен использовать pinned revision, заранее
заполненный cache и проверенную vendored копию remote code.

До реализации следует принять модель только после собственного gate:

- quality set не менее 300–500 лицензированно доступных изображений по продуктовым категориям;
- ручные оценки foreground completeness, boundary halos, hair/fur, transparency, holes и wrong-object rate;
- автоматические SAD/MAD/Gradient там, где есть true alpha, и HCE/IoU/F-score для binary cases;
- latency p50/p95, cold start, peak RSS/VRAM на CPU и целевом GPU при 512/1024/2048 и batch 1/4;
- comparison: BiRefNet full, BiRefNet_lite, InSPyReNet, BEN2 и лицензированный trial RMBG-2.0;
- обязательный тест изображений «нет foreground» и нескольких равноправных объектов.

## 9. Лицензии и риски

Это технический обзор, не юридическое заключение.

- **Permissive:** U²-Net/IS-Net/SAM/SAM2 — Apache-2.0; BiRefNet/MVANet/InSPyReNet/BEN2/PDFNet/MAM/Matte Anything — MIT. Нужно сохранять license/copyright notices и фиксировать точные revisions.
- **BRIA RMBG-2.0:** HF weights non-commercial; production self-hosting только по отдельному соглашению либо через коммерческий API. Нельзя считать модель open source только из-за доступности файлов.
- **Составные pipelines:** лицензия верхнего repo не покрывает автоматически backbone, base weights и datasets. Для DiffDIS проверить SD-Turbo; для PDFNet — Swin и Depth Anything V2; для SAM-matting — SAM, GroundingDINO, ViTMatte/M2M; для RenderMatte — FLUX.1 Kontext.
- **Weights ≠ code:** для Google Drive checkpoints MVANet/InSPyReNet и исторических MODNet weights сохранить model card/README/LICENSE на дату скачивания. При отсутствии явной лицензии не распространять weight внутри коммерческого продукта до письменного подтверждения.
- **Remote code:** HF `trust_remote_code=True` позволяет выполнить код из model repository. В production pin commit SHA, использовать `safetensors`, vendor проверенный код и запретить runtime download/update.
- **Training data:** permissive weight license не гарантирует отсутствие privacy/copyright/bias рисков исходных данных. BRIA заявляет licensed dataset; у академических моделей происхождение смешанное. Это отдельный risk assessment.
- **Model behavior:** foreground selection неоднозначен. Ошибочное удаление людей/товаров — продуктовый риск; нужны preview, undo/manual correction и запрет silent destructive overwrite.
- **Benchmark leakage/overfitting:** general checkpoints часто смешивают DIS/SOD/COD data; результаты на закрытой BRIA evaluation и DIS5K не дают честной cross-domain гарантии.

## 10. Возможные альтернативы

1. **BiRefNet_lite (44.4M)** — первый запасной вариант. Существенно меньше weights и памяти, тот же API/лицензия; ожидаемая цена — потеря boundary/detail quality. Точные trade-offs нужно измерить на нашем наборе.
2. **InSPyReNet + `transparent-background`** — второй запасной вариант. Зрелый package, CPU/CUDA/MPS, MIT и проще эксплуатация. Подходит, если reliability/integration важнее максимального benchmark quality.
3. **MODNet** — отдельный fast path только если продукт станет строго portrait service. Не использовать как универсальный default.
4. **BEN2** — challenger для bake-off. Может выиграть на сложных границах, но нужен audit CPU path, artifacts и независимый quality test.
5. **BRIA RMBG-2.0 API/commercial weights** — если бюджет на лицензию приемлем и закрытая quality evaluation подтвердит выигрыш; юридически и операционно это иной вариант, чем open self-hosting.
6. **PDFNet** — quality tier/offline processing, если пользователи готовы ждать и качество DIS важнее стоимости. Не лучший стартовый realtime endpoint.
7. **SAM2.1 tiny/base + matting refiner** — будущий interactive mode: клик/box для выбора или исправления объекта. Это дополнение к автоматической модели, не замена в v1.
8. **RenderMatte/SDMatte** — наблюдать для premium/manual matting после появления зрелых weights, понятных лицензий и приемлемой оптимизации.

## 11. Sources

Основные первичные источники:

1. U²-Net: [paper (Pattern Recognition/arXiv)](https://arxiv.org/abs/2005.09007), [official GitHub](https://github.com/xuebinqin/U-2-Net).
2. IS-Net / DIS: [ECCV 2022 paper](https://arxiv.org/abs/2203.03041), [official GitHub](https://github.com/xuebinqin/DIS), [project page](https://xuebinqin.github.io/dis/index.html).
3. MODNet: [paper](https://arxiv.org/abs/2011.11961), [official GitHub and license statement](https://github.com/ZHKKKe/MODNet), [PPM benchmark](https://github.com/ZHKKKe/PPM).
4. InSPyReNet: [official GitHub](https://github.com/plemeri/InSPyReNet), [paper (ACCV)](https://openaccess.thecvf.com/content/ACCV2022/html/Kim_Revisiting_Image_Pyramid_Structure_for_High_Resolution_Salient_Object_Detection_ACCV_2022_paper.html), [deployment package](https://github.com/plemeri/transparent-background).
5. MVANet: [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Multi-view_Aggregation_Network_for_Dichotomous_Image_Segmentation_CVPR_2024_paper.pdf), [official GitHub](https://github.com/qianyu-dlut/MVANet).
6. BiRefNet: [official paper](https://doi.org/10.26599/AIR.2024.9150038), [official GitHub](https://github.com/ZhengPeng7/BiRefNet), [general model card](https://huggingface.co/ZhengPeng7/BiRefNet), [lite model card](https://huggingface.co/ZhengPeng7/BiRefNet_lite), [HR-matting model card](https://huggingface.co/ZhengPeng7/BiRefNet_HR-matting).
7. BRIA RMBG-2.0: [official model card](https://huggingface.co/briaai/RMBG-2.0), [official architecture/training announcement](https://blog.bria.ai/introducing-the-rmbg-v2.0-model-the-next-generation-in-background-removal-from-images), [license/API comparison](https://huggingface.co/briaai/RMBG-2.0).
8. BEN/BEN2: [paper](https://arxiv.org/abs/2501.06230), [official BEN card](https://huggingface.co/PramaLLC/BEN), [official BEN2 card/files](https://huggingface.co/PramaLLC/BEN2).
9. SAM: [paper](https://arxiv.org/abs/2304.02643), [official GitHub](https://github.com/facebookresearch/segment-anything).
10. SAM2/SAM2.1: [ICLR 2025 paper](https://openreview.net/forum?id=Ha6RTeWMd0), [official project](https://ai.meta.com/research/sam2/), [official GitHub with size/speed table](https://github.com/facebookresearch/sam2).
11. Matting Anything Model: [paper](https://arxiv.org/abs/2306.05399), [official GitHub](https://github.com/SHI-Labs/Matting-Anything).
12. Matte Anything: [paper](https://arxiv.org/abs/2306.04121), [official GitHub](https://github.com/hustvl/Matte-Anything).
13. DiffDIS: [ICLR 2025 paper](https://openreview.net/forum?id=vh1e2WJfZp), [official GitHub](https://github.com/qianyu-dlut/DiffDIS).
14. PDFNet: [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_High-Precision_Dichotomous_Image_Segmentation_via_Depth_Integrity-Prior_and_Fine-Grained_Patch_CVPR_2026_paper.html), [official GitHub](https://github.com/Tennine2077/PDFNet).
15. SDMatte: [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_SDMatte_Grafting_Diffusion_Models_for_Interactive_Matting_ICCV_2025_paper.pdf), [official GitHub](https://github.com/vivoCameraResearch/SDMatte).
16. RenderMatte: [arXiv preprint, 9 August 2026](https://arxiv.org/abs/2608.08487).

### Краткий итог

- **Research SOTA:** PDFNet-L + Depth Anything V2 для automatic DIS; RenderMatte — самый свежий matting research signal, но пока не production-ready.
- **Practical SOTA:** семейство BiRefNet.
- **Выбранная модель:** `ZhengPeng7/BiRefNet` general/full для основного GPU endpoint.
- **Запасные варианты:** `BiRefNet_lite`; InSPyReNet. BEN2 — дополнительный challenger в bake-off.
- **Почему:** современное качество тонких границ, один model pass, зрелые PyTorch/HF/ONNX пути и коммерчески разрешительная MIT-лицензия без обязательного proprietary API.
