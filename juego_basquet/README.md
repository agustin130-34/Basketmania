# Básquet con el cuerpo

Juego para 2 jugadores con la cámara web, hecho con **Python + MediaPipe (Pose) + OpenCV**.

## Instalación (Windows)

1. Instalá Python 3.13: `py install 3.13`
2. Doble clic en **`instalar.bat`** (solo la primera vez)
3. Doble clic en **`iniciar.bat`**

La primera vez se descarga el modelo de cuerpo (~6 MB).

## Cómo se juega

- **Jugador 1** se para del lado **izquierdo** (azul) y **Jugador 2** del **derecho** (rojo).
- El juego espera a que la cámara detecte a **los dos jugadores**. Después arranca una cuenta regresiva de 5 segundos (si alguno sale de cámara, vuelve a esperar).
- Caen **varias pelotas a la vez** en cámara lenta: en cada tanda, entre 2 y 4 pelotas en lugares al azar. Cuando caen todas, sale la tanda siguiente.
- La pelota cae desde arriba. Golpeala con las **manos** (los círculos rojos) para mandarla al aro.
- **Canasta:** +1 punto para el último jugador que tocó la pelota.
- Si la pelota **toca el piso de tu lado**, perdés una vida (corazón).
- **Gana** el primero que llegue a 10 canastas, o el otro jugador si uno se queda sin vidas.

## Potenciadores y bomba

- **Rayo (violeta):** aparece cada tanto de un lado al azar y dura 6 segundos. Si lo tocás con la mano, **las pelotas del lado del rival caen el doble de rápido durante 5 segundos**.
- **x2 (dorado):** si lo tocás, **tus canastas valen el doble durante 10 segundos**.
- **Pelota negra (bomba):** cada tanto, una pelota de la tanda sale negra. **Si la tocás, explota y perdés 2 vidas.** Si cae al piso no pasa nada: ¡esquivala!
- Los efectos activos se ven debajo de los corazones de cada jugador, con los segundos que les quedan.

Teclas: `R` o `Espacio` reinicia · `Q` / `ESC` sale

## Configuración

Al principio de `juego.py`:

```python
CAMARA = 0              # cámara preferida
VIDAS = 10              # vidas por jugador
PUNTOS_PARA_GANAR = 10  # canastas para ganar
CUENTA_ATRAS = 5        # segundos de cuenta regresiva
MIN_PELOTAS = 2         # mínimo de pelotas por tanda
MAX_PELOTAS = 4         # máximo de pelotas por tanda
VELOCIDAD = 0.5         # cámara lenta: 1 = normal, 0.5 = mitad de velocidad
FACTOR_RAPIDO = 2.0     # cuánto se aceleran las pelotas del rival con el rayo
DURACION_RAPIDO = 5     # segundos del rayo
DURACION_X2 = 10        # segundos del x2
CADA_POTENCIADOR = (8, 14)   # cada cuántos segundos aparece un potenciador
PROBABILIDAD_BOMBA = 0.35    # chance de que una tanda traiga pelota negra
VIDAS_BOMBA = 2         # vidas que saca la bomba
GRAVEDAD = 820          # velocidad de caída (más alto = cae más rápido)
```

## Consejos

- Alejate de la cámara hasta que se te vea **al menos de la cintura para arriba** (mejor el cuerpo entero).
- Buena luz y espacio libre para moverse.
- Pegarle con un movimiento **hacia arriba** manda la pelota más alto.
