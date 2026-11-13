import os
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CARPETA = os.path.dirname(os.path.abspath(__file__))

_FUENTES = [
    "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
_cache_fuentes = {}


def _fuente(tam):
    if tam not in _cache_fuentes:
        f = None
        for ruta in _FUENTES:
            if os.path.exists(ruta):
                f = ImageFont.truetype(ruta, tam)
                break
        _cache_fuentes[tam] = f or ImageFont.load_default(size=tam)
    return _cache_fuentes[tam]


class Lienzo:
    def __init__(self):
        self.textos = []

    def texto(self, txt, pos, tam=28, color=(255, 255, 255), borde=3,
              centrado=False, derecha=False, fondo=None):
        self.textos.append((str(txt), pos, tam, color, borde, centrado, derecha, fondo))

    def dibujar(self, frame):
        if not self.textos:
            return frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        for txt, (x, y), tam, color, borde, centrado, derecha, fondo in self.textos:
            f = _fuente(tam)
            x0, y0, x1, y1 = d.textbbox((0, 0), txt, font=f, stroke_width=borde)
            if centrado:
                x, y = x - (x1 - x0) // 2, y - (y1 - y0) // 2
            elif derecha:
                x = x - (x1 - x0)
            if fondo:
                d.rectangle((x - 10, y - 8, x + (x1 - x0) + 10, y + (y1 - y0) + 12), fill=fondo[::-1])
            d.text((x - x0, y - y0), txt, font=f, fill=color[::-1],
                   stroke_width=borde, stroke_fill=(0, 0, 0))
        self.textos = []
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def abrir_camara(preferida=0, ancho=1280, alto=720):
    if sys.platform == "win32":
        metodos = [("DirectShow", cv2.CAP_DSHOW), ("Media Foundation", cv2.CAP_MSMF), ("Auto", cv2.CAP_ANY)]
    else:
        metodos = [("Auto", cv2.CAP_ANY)]
    for idx in list(dict.fromkeys([preferida, 0, 1, 2])):
        for nombre, api in metodos:
            print(f"Probando cámara {idx} con {nombre}...", end=" ", flush=True)
            cap = cv2.VideoCapture(idx, api)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
                for _ in range(10):
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        print(f"OK ({frame.shape[1]}x{frame.shape[0]})")
                        return cap
                    time.sleep(0.1)
                cap.release()
                cap = cv2.VideoCapture(idx, api)
                ok, frame = cap.read() if cap.isOpened() else (False, None)
                if ok and frame is not None:
                    print(f"OK ({frame.shape[1]}x{frame.shape[0]})")
                    return cap
            cap.release()
            print("no")
    raise SystemExit(
        "\nNo se pudo abrir ninguna cámara. Revisá:\n"
        "  1. Configuración de Windows > Privacidad y seguridad > Cámara:\n"
        "     activá 'Acceso a la cámara' y 'Permitir que las apps de escritorio accedan a la cámara'.\n"
        "  2. Que no quede otro Python abierto (Administrador de tareas > finalizar 'Python').\n"
        "  3. Que la cámara funcione en la app 'Cámara' de Windows (y después cerrala).")
