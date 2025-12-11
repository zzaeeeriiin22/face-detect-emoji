<p align="center">
    <img src="docs/screenshot.png"/>
    <br>
    <img src='docs/description.svg'>
</p>

<span align="center">

# face-detect-emoji

</span>

### Overview

이 프로젝트는 카메라에서 받아온 프레임으로부터 얼굴 랜드마크를 추출하고, 이를 기반으로 얼굴 표정을 인식한 후 표정에 따라 이모지를 출력하는 프로젝트입니다.

모든 과정을 third-party 라이브러리 없이 순수한 C++ 로 구현하였으며, 같은 기능을 하는 JavaScript 버전도 구현하여 두 백엔드 간 성능을 비교할 수 있도록 하였습니다.

### 0. Architecture

![Architecture](docs/diagram.png)


### 1. Preprocessing

MediaPipe 를 사용하여 얼굴의 랜드마크를 추출하면 478개의 랜드마크를 얻을 수 있습니다.

사람의 얼굴이 정면으로 찍히지 않을 수 있으므로, Partial Affine Transformation 을 통해 얼굴의 자세를 정규화합니다.

```math
\begin{bmatrix}
s_x \cos \theta & -s_x \sin \theta & t_x \\
s_y \sin \theta  & s_y \cos \theta & t_y \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
x_0 \\
y_0 \\
1
\end{bmatrix}
=
\begin{bmatrix}
u_0 \\
v_0 \\
1
\end{bmatrix}
```

![](docs/normalization.png)

### 2. AI

Pytorch 를 통해 모델을 학습시키고, BatchNorm 을 fold 한 뒤 가중치와 편향을 C++ 헤더 파일로 추출하였습니다.

> [wasm/model_weights.h](wasm/model_weights.h)

preprocessing 과정에서 얻은 얼굴 랜드마크를 이 모델에 입력하면 얼굴 표정을 인식할 수 있습니다.

```cpp
void forward(const float* input, float* output) {
    // Layer 1
    alignas(64) float hidden1[HIDDEN1_DIM];
    for (int i = 0; i < HIDDEN1_DIM; ++i) {
        float sum = LAYER1_BIAS[i];
        for (int j = 0; j < INPUT_DIM; ++j) {
            sum += LAYER1_WEIGHT[i][j] * input[j];
        }
        hidden1[i] = leaky_relu(sum);
    }

    // Layer 2
    alignas(64) float hidden2[HIDDEN2_DIM];
    for (int i = 0; i < HIDDEN2_DIM; ++i) {
        float sum = LAYER2_BIAS[i];
        for (int j = 0; j < HIDDEN1_DIM; ++j) {
            sum += LAYER2_WEIGHT[i][j] * hidden1[j];
        }
        hidden2[i] = leaky_relu(sum);
    }

    // Layer 3 (output)
    for (int i = 0; i < OUTPUT_DIM; ++i) {
        float sum = LAYER3_BIAS[i];
        for (int j = 0; j < HIDDEN2_DIM; ++j) {
            sum += LAYER3_WEIGHT[i][j] * hidden2[j];
        }
        output[i] = sum;
    }
}
```

> [wasm/predict.cpp](wasm/predict.cpp)

`alignas(64)` 를 사용하여 64바이트 단위로 메모리를 정렬하고, `HIDDEN1_DIM`, `HIDDEN2_DIM`, `INPUT_DIM`, `OUTPUT_DIM` 을 `constexpr` 로 정의하여 컴파일러가 최대한 최적화할 수 있도록 하였습니다.

### 3. Emoji Overlay

얼굴 표정을 인식한 후 이모지를 얼굴에 출력하기 위해, 이모지 상 좌표와 얼굴 랜드마크 간 변환 식을 Full Affine Transformation 을 통해 계산합니다.

```math
\begin{bmatrix}
a & b & c \\
d & e & f \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
x_0 \\
y_0 \\
1
\end{bmatrix}
=
\begin{bmatrix}
u_0 \\
v_0 \\
1
\end{bmatrix}
```

```math
\begin{aligned}
u_0 = ax_0 + by_0 + c \\
v_0 = dx_0 + ey_0 + f \\
u_1 = ax_1 + by_1 + c \\
v_1 = dx_1 + ey_1 + f \\
u_2 = ax_2 + by_2 + c \\
v_2 = dx_2 + ey_2 + f \\
\end{aligned}
```

구해야하는 미지수가 6개이므로, 3개의 점을 이용하여 6개의 방정식을 세워 해를 구합니다.

![](docs/full_affine.png)

구한 Affine Matrix $M$ 은 emoji 에서 canvas 로의 변환 행렬입니다. 곧, $M^{-1}$ 은 canvas 에서 emoji 로의 변환 행렬입니다.

canvas 의 모든 픽셀을 $M^{-1}$ 을 이용하여 emoji 의 좌표로 변환한 후, 이모지의 픽셀 값을 가져오면 이모지를 얼굴에 출력할 수 있습니다.

이 과정에서 이모지의 픽셀 좌표가 소수점 이하의 값을 가질 수 있으므로, Bilinear Interpolation 을 통해 픽셀 값을 보간합니다.

![](docs/interpolation.png)

> [wasm/geometry.cpp](wasm/geometry.cpp)
