# Controle de Mão Robótica com Visão Computacional

Este projeto utiliza a câmera do computador para detectar os movimentos da mão de um usuário em tempo real e os traduz em comandos para controlar uma mão robótica feita com servo motores e conectada a um Arduino.

O script principal, escrito em Python, usa as bibliotecas **OpenCV** para capturar o vídeo da webcam e **MediaPipe** para identificar os 21 pontos-chave (*landmarks*) da mão. Com base nas coordenadas desses pontos, o programa calcula a distância entre a ponta dos dedos e a base da mão, convertendo essa medida em uma porcentagem de "abertura" para cada dedo.

Esses dados (cinco valores percentuais, um para cada dedo) são enviados via comunicação serial para uma placa Arduino, que por sua vez controla o ângulo de cinco servo motores correspondentes, replicando o movimento da mão do usuário em tempo real.

## ✨ Funcionalidades

* **Detecção de Mão em Tempo Real:** Utiliza o MediaPipe para um rastreamento rápido e preciso dos 21 pontos da mão.
* **Calibração Inteligente:** O sistema só é ativado após o usuário manter a mão dentro de um círculo guia por 3 segundos, evitando movimentos indesejados e inicializações acidentais.
* **Mapeamento Proporcional:** Calcula a abertura dos dedos de forma proporcional ao tamanho da mão, tornando o controle mais natural e independente da distância do usuário em relação à câmera.
* **Comunicação Serial Eficiente:** Envia os dados de forma otimizada para o Arduino em um formato de string simples.
* **Seleção de Porta Automática:** O script lista as portas seriais disponíveis e permite que o usuário escolha a correta, facilitando a configuração em diferentes computadores.

## ⚙️ Como Funciona

O fluxo de operação do sistema segue os seguintes passos:

1.  **Captura de Vídeo:** O OpenCV abre a webcam padrão e captura o vídeo quadro a quadro.
2.  **Detecção de Landmarks:** O MediaPipe processa cada quadro. Se uma mão for detectada, ele retorna as coordenadas X, Y e Z de seus 21 pontos-chave.
3.  **Calibração:** O usuário posiciona a mão dentro do círculo vermelho exibido na tela. Após 3 segundos com a mão totalmente dentro do círculo, o modo de controle é ativado.
4.  **Cálculo das Distâncias:** O script calcula a distância euclidiana entre a ponta de cada um dos cinco dedos e a base da palma.
5.  **Normalização:** Para tornar a medição independente do tamanho da mão ou da distância da câmera, a distância de cada dedo é dividida pela distância entre o pulso e a base do dedo do meio. O resultado é convertido em uma porcentagem de 0 a 100.
6.  **Envio de Dados:** Um pacote de dados contendo as porcentagens de cada dedo (Ex: `100,80,20,15,90`) é formatado e enviado via porta serial para o Arduino.
7.  **Controle dos Servos:** O Arduino recebe a string, interpreta os cinco valores e os mapeia para ângulos (geralmente de 0 a 180 graus) para posicionar cada servo motor correspondente.

## 🛠️ Requisitos

### Hardware
* Computador com webcam
* Placa Arduino (Uno, Nano, etc.)
* 5 Servo Motores (ex: SG90 ou MG996R)
* Uma mão robótica (impressa em 3D ou montada com outros materiais)
* Protoboard e Jumpers para as conexões elétricas

### Software
* Python 3.7+
* Arduino IDE
* Bibliotecas Python: `opencv-python`, `mediapipe`, `scipy`, `numpy`, `pyserial`

## 🚀 Instalação e Configuração

Siga os passos abaixo para configurar o ambiente e executar o projeto.

### 1. Clone o Repositório
Abra seu terminal ou Git Bash e execute o seguinte comando:
```bash
git clone [https://github.com/Bender223SCBR/M-oRobotica.git](https://github.com/Bender223SCBR/M-oRobotica.git)
cd M-oRobotica
