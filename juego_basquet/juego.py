import math
import os
import random
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from extras import CARPETA, Lienzo, abrir_camara

CAMARA = 0
ANCHO, ALTO = 1280, 720
VIDAS = 10
PUNTOS_PARA_GANAR = 10
GRAVEDAD = 820
CUENTA_ATRAS = 5
MIN_PELOTAS = 2
MAX_PELOTAS = 4
VELOCIDAD = 0.5
FACTOR_RAPIDO = 2.0
DURACION_RAPIDO = 5
DURACION_X2 = 10
CADA_POTENCIADOR = (8, 14)
DURACION_POTENCIADOR = 6
PROBABILIDAD_BOMBA = 0.35
VIDAS_BOMBA = 2

MODELO = os.path.join(CARPETA, "pose_landmarker_lite.task")
URL_MODELO = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
VENTANA = "Basquet con el cuerpo"

MANO_IZQ = (15, 17, 19)
MANO_DER = (16, 18, 20)
TORSO = (11, 12, 23, 24)

AZUL = (230, 140, 40)
ROJO = (60, 60, 230)
COLOR_MANO = (40, 40, 255)
NARANJA = (30, 110, 240)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (90, 90, 90)
COLOR_JUGADOR = [AZUL, ROJO]
DORADO = (0, 200, 255)
VIOLETA = (220, 60, 170)


def descargar_modelo():
    if not os.path.exists(MODELO):
        print("Descargando modelo de cuerpo (~6 MB)...")
        urllib.request.urlretrieve(URL_MODELO, MODELO)


def corazon(frame, cx, cy, s, color):
    r = int(s * 0.3)
    cv2.circle(frame, (cx - r, cy - r // 2), r, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx + r, cy - r // 2), r, color, -1, cv2.LINE_AA)
    pts = np.array([(cx - 2 * r, cy - r // 4), (cx + 2 * r, cy - r // 4), (cx, cy + int(s * 0.55))], np.int32)
    cv2.fillPoly(frame, [pts], color, cv2.LINE_AA)


def capa(frame, dibujar, alfa):
    copia = frame.copy()
    dibujar(copia)
    cv2.addWeighted(copia, alfa, frame, 1 - alfa, 0, frame)


class Pelota:
    def __init__(self, r):
        self.r = r
        self.activa = False
        self.x = self.y = self.vx = self.vy = 0.0
        self.angulo = 0.0
        self.ultimo = None
        self.encesto = False
        self.bomba = False

    def lanzar(self, x, y, vx, bomba=False):
        self.x, self.y, self.vx, self.vy = x, y, vx, 0.0
        self.activa, self.ultimo, self.encesto, self.bomba = True, None, False, bomba


class Potenciador:
    def __init__(self, tipo, x, y):
        self.tipo, self.x, self.y = tipo, x, y
        self.vida = DURACION_POTENCIADOR


class Juego:
    def __init__(self, detector=None, captura=None):
        if detector is None:
            descargar_modelo()
            opciones = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=MODELO),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=2,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            detector = vision.PoseLandmarker.create_from_options(opciones)
        self.detector = detector
        self.cap = captura if captura is not None else abrir_camara(CAMARA, ANCHO, ALTO)
        self.lienzo = Lienzo()
        self.pelotas = []
        self.reiniciar()

    def reiniciar(self):
        self.jugadores = [{"puntos": 0, "vidas": VIDAS} for _ in range(2)]
        self.estado = "espera"
        self.t_estado = 0.0
        self.t_ambos = 0.0
        self.lado_saque = random.randint(0, 1)
        self.t_tanda = 0.0
        self.potenciador = None
        self.t_potenciador = random.uniform(*CADA_POTENCIADOR)
        self.x2 = [0.0, 0.0]
        self.rapido = [0.0, 0.0]
        self.explosiones = []
        self.previas = {}
        self.mensaje, self.t_mensaje, self.color_mensaje = "", 0.0, BLANCO
        self.ganador = None
        for p in self.pelotas:
            p.activa = False

    def avisar(self, texto, color=BLANCO, seg=1.5):
        self.mensaje, self.t_mensaje, self.color_mensaje = texto, seg, color

    def geometria(self, w, h):
        s = h / 720
        self.s = s
        self.w, self.h = w, h
        self.radio_mano = int(70 * s)
        self.aro_y = int(0.30 * h)
        self.aro_izq = int(w / 2 - 70 * s)
        self.aro_der = int(w / 2 + 70 * s)
        self.piso = int(h - 0.07 * h)
        if len(self.pelotas) != MAX_PELOTAS or abs(self.pelotas[0].r - 34 * s) > 1:
            self.pelotas = [Pelota(34 * s) for _ in range(MAX_PELOTAS)]

    def personas(self, res):
        gente = []
        for lm in res.pose_landmarks or []:
            visibles = [lm[i].x for i in TORSO if getattr(lm[i], "visibility", 1) > 0.4]
            cx = sum(visibles) / len(visibles) if visibles else lm[0].x
            gente.append((cx, lm))
        gente.sort(key=lambda g: g[0])
        if len(gente) == 1:
            return {0 if gente[0][0] < 0.5 else 1: gente[0][1]}
        return {i: g[1] for i, g in enumerate(gente[:2])}

    def manos(self, personas, dt):
        resultado = []
        vistas = set()
        for j, lm in personas.items():
            for lado, idx in (("izq", MANO_IZQ), ("der", MANO_DER)):
                if getattr(lm[idx[0]], "visibility", 1) < 0.5:
                    continue
                x = sum(lm[i].x for i in idx) / 3 * self.w
                y = sum(lm[i].y for i in idx) / 3 * self.h
                clave = (j, lado)
                px, py = self.previas.get(clave, (x, y))
                vx, vy = (x - px) / dt, (y - py) / dt
                self.previas[clave] = (x, y)
                vistas.add(clave)
                resultado.append((j, x, y, vx, vy))
        for clave in list(self.previas):
            if clave not in vistas:
                del self.previas[clave]
        return resultado

    def lanzar_tanda(self):
        cantidad = random.randint(MIN_PELOTAS, MAX_PELOTAS)
        usadas = []
        separacion = 2.4 * self.pelotas[0].r / self.w
        for p in self.pelotas[:cantidad]:
            for _ in range(50):
                lado = random.randint(0, 1)
                x = random.uniform(0.06, 0.42) if lado == 0 else random.uniform(0.58, 0.94)
                if all(abs(x - u) > separacion for u in usadas):
                    break
            usadas.append(x)
            y = random.uniform(0.14, 0.22) * self.h
            vx = random.uniform(-60, 60) * self.s
            p.lanzar(x * self.w, y, vx)
        if cantidad >= 2 and random.random() < PROBABILIDAD_BOMBA:
            random.choice(self.pelotas[:cantidad]).bomba = True

    def crear_potenciador(self):
        lado = random.randint(0, 1)
        x = random.uniform(0.10, 0.38) if lado == 0 else random.uniform(0.62, 0.90)
        y = random.uniform(0.38, 0.62)
        tipo = random.choice(["rapido", "x2"])
        self.potenciador = Potenciador(tipo, x * self.w, y * self.h)

    def actualizar_potenciador(self, manos, dt):
        for i in (0, 1):
            self.x2[i] = max(0.0, self.x2[i] - dt)
            self.rapido[i] = max(0.0, self.rapido[i] - dt)
        pot = self.potenciador
        if pot is None:
            self.t_potenciador -= dt
            if self.t_potenciador <= 0:
                self.crear_potenciador()
            return
        pot.vida -= dt
        radio = 38 * self.s + self.radio_mano * 0.6
        for j, hx, hy, _, _ in manos:
            if math.hypot(pot.x - hx, pot.y - hy) < radio:
                if pot.tipo == "x2":
                    self.x2[j] = DURACION_X2
                    self.avisar(f"¡Jugador {j + 1}: puntos x2 por {DURACION_X2} s!", DORADO, 2.5)
                else:
                    self.rapido[1 - j] = DURACION_RAPIDO
                    self.avisar(f"¡Jugador {j + 1} aceleró las pelotas del rival!", VIOLETA, 2.5)
                pot.vida = 0
                break
        if pot.vida <= 0:
            self.potenciador = None
            self.t_potenciador = random.uniform(*CADA_POTENCIADOR)

    def explotar(self, p, j):
        p.activa = False
        self.explosiones.append([p.x, p.y, 0.0])
        self.jugadores[j]["vidas"] = max(0, self.jugadores[j]["vidas"] - VIDAS_BOMBA)
        self.avisar(f"¡BOOM! Jugador {j + 1} pierde {VIDAS_BOMBA} vidas", (0, 80, 255), 2.0)

    def fisica(self, p, dt, manos):
        s = self.s
        for _ in range(3):
            paso = dt / 3
            p.vy += GRAVEDAD * s * paso
            y_antes = p.y
            p.x += p.vx * paso
            p.y += p.vy * paso
            p.angulo += p.vx * paso / p.r
            if p.x < p.r:
                p.x, p.vx = p.r, abs(p.vx) * 0.8
            if p.x > self.w - p.r:
                p.x, p.vx = self.w - p.r, -abs(p.vx) * 0.8
            if p.y < p.r:
                p.y, p.vy = p.r, abs(p.vy) * 0.6
            for ex in (self.aro_izq, self.aro_der):
                dx, dy = p.x - ex, p.y - self.aro_y
                d = math.hypot(dx, dy)
                limite = p.r + 6 * s
                if 0 < d < limite:
                    nx, ny = dx / d, dy / d
                    p.x, p.y = ex + nx * limite, self.aro_y + ny * limite
                    vn = p.vx * nx + p.vy * ny
                    if vn < 0:
                        p.vx -= 1.7 * vn * nx
                        p.vy -= 1.7 * vn * ny
            if (not p.encesto and not p.bomba and p.vy > 0 and y_antes < self.aro_y <= p.y
                    and self.aro_izq + p.r * 0.4 < p.x < self.aro_der - p.r * 0.4):
                p.encesto = True
                if p.ultimo is not None:
                    valor = 2 if self.x2[p.ultimo] > 0 else 1
                    self.jugadores[p.ultimo]["puntos"] += valor
                    self.avisar(f"¡Canasta del Jugador {p.ultimo + 1}!  +{valor}", COLOR_JUGADOR[p.ultimo])
            for j, hx, hy, hvx, hvy in manos:
                dx, dy = p.x - hx, p.y - hy
                d = math.hypot(dx, dy)
                limite = p.r + self.radio_mano * 0.6
                if d < limite and p.bomba:
                    self.explotar(p, j)
                    return
                if d < limite:
                    nx, ny = (dx / d, dy / d) if d > 0 else (0.0, -1.0)
                    p.x, p.y = hx + nx * limite, hy + ny * limite
                    rvx, rvy = p.vx - hvx, p.vy - hvy
                    vn = rvx * nx + rvy * ny
                    if vn < 0:
                        p.vx -= 1.6 * vn * nx
                        p.vy -= 1.6 * vn * ny
                    p.vx += hvx * 0.35
                    p.vy += hvy * 0.35
                    p.vy = min(p.vy, -800 * s)
                    g = GRAVEDAD * s
                    caida = p.vy * p.vy - 2 * g * (p.y - self.aro_y)
                    if caida > 0:
                        t = (-p.vy + math.sqrt(caida)) / g
                        p.vx = 0.65 * (self.w / 2 - p.x) / t + 0.35 * p.vx
                    v = math.hypot(p.vx, p.vy)
                    tope = 1500 * s
                    if v > tope:
                        p.vx, p.vy = p.vx / v * tope, p.vy / v * tope
                    p.ultimo = j
                    p.encesto = False
            if p.y >= self.piso - p.r:
                p.activa = False
                if not p.encesto and not p.bomba:
                    lado = 0 if p.x < self.w / 2 else 1
                    self.jugadores[lado]["vidas"] = max(0, self.jugadores[lado]["vidas"] - 1)
                    self.avisar(f"Jugador {lado + 1} pierde una vida", COLOR_JUGADOR[lado])
                break

    def revisar_fin(self):
        for i, j in enumerate(self.jugadores):
            if j["puntos"] >= PUNTOS_PARA_GANAR:
                self.ganador = i
            elif j["vidas"] <= 0:
                self.ganador = 1 - i
        if self.ganador is not None:
            self.estado = "fin"
            self.potenciador = None
            for p in self.pelotas:
                p.activa = False

    def choques_entre_pelotas(self):
        activas = [p for p in self.pelotas if p.activa]
        for i, a in enumerate(activas):
            for b in activas[i + 1:]:
                dx, dy = b.x - a.x, b.y - a.y
                d = math.hypot(dx, dy)
                limite = a.r + b.r
                if 0 < d < limite:
                    nx, ny = dx / d, dy / d
                    mitad = (limite - d) / 2
                    a.x, a.y = a.x - nx * mitad, a.y - ny * mitad
                    b.x, b.y = b.x + nx * mitad, b.y + ny * mitad
                    vn = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny
                    if vn < 0:
                        a.vx += vn * nx * 0.9
                        a.vy += vn * ny * 0.9
                        b.vx -= vn * nx * 0.9
                        b.vy -= vn * ny * 0.9

    def actualizar(self, personas, manos, dt):
        self.t_estado += dt
        if self.t_mensaje > 0:
            self.t_mensaje -= dt
        ambos = 0 in personas and 1 in personas
        if self.estado == "espera":
            self.t_ambos = self.t_ambos + dt if ambos else 0.0
            if self.t_ambos >= 0.5:
                self.estado, self.t_estado = "cuenta", 0.0
        elif self.estado == "cuenta":
            if not ambos:
                self.estado, self.t_estado, self.t_ambos = "espera", 0.0, 0.0
            elif self.t_estado >= CUENTA_ATRAS:
                self.estado, self.t_estado = "juego", 0.0
                for p in self.pelotas:
                    p.activa = False
                self.t_tanda = 0.0
        elif self.estado == "juego":
            self.actualizar_potenciador(manos, dt)
            for p in self.pelotas:
                if p.activa:
                    lado = 0 if p.x < self.w / 2 else 1
                    factor = FACTOR_RAPIDO if self.rapido[lado] > 0 else 1.0
                    self.fisica(p, dt * VELOCIDAD * factor, manos)
            self.choques_entre_pelotas()
            if not any(p.activa for p in self.pelotas):
                self.t_tanda -= dt
                if self.t_tanda <= 0:
                    self.lanzar_tanda()
                    self.t_tanda = 1.2
            self.revisar_fin()

    def dibujar_piso(self, frame):
        w, h, piso = self.w, self.h, self.piso

        def pintar(img):
            cv2.rectangle(img, (0, piso), (w // 2, h), AZUL, -1)
            cv2.rectangle(img, (w // 2, piso), (w, h), ROJO, -1)

        capa(frame, pintar, 0.45)
        paso = max(20, w // 16)
        for x in range(0, w + 1, paso):
            cv2.line(frame, (x, piso), (x + (x - w // 2) // 6, h), BLANCO, 1, cv2.LINE_AA)
        for y in (piso, piso + (h - piso) // 2):
            cv2.line(frame, (0, y), (w, y), BLANCO, 1, cv2.LINE_AA)

    def dibujar_tablero(self, frame):
        s, cx = self.s, self.w // 2
        x0, x1 = int(cx - 165 * s), int(cx + 165 * s)
        y0, y1 = int(0.07 * self.h), int(self.aro_y + 40 * s)
        cv2.rectangle(frame, (x0, y0), (x1, y1), BLANCO, -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), NEGRO, max(3, int(6 * s)), cv2.LINE_AA)
        cv2.rectangle(frame, (int(cx - 55 * s), int(self.aro_y - 85 * s)), (int(cx + 55 * s), self.aro_y),
                      NEGRO, max(2, int(5 * s)), cv2.LINE_AA)

    def dibujar_aro(self, frame):
        s, ay = self.s, self.aro_y
        izq, der = self.aro_izq, self.aro_der
        fondo = int(ay + 95 * s)
        abajo_izq, abajo_der = int(self.w / 2 - 40 * s), int(self.w / 2 + 40 * s)
        grosor = max(1, int(2 * s))
        for i in range(7):
            t = i / 6
            xa = int(izq + (der - izq) * t)
            xb = int(abajo_izq + (abajo_der - abajo_izq) * t)
            cv2.line(frame, (xa, ay), (xb, fondo), NEGRO, grosor, cv2.LINE_AA)
            cv2.line(frame, (xa, ay), (int(abajo_izq + (abajo_der - abajo_izq) * min(1, t + 0.33)), fondo),
                     NEGRO, grosor, cv2.LINE_AA)
        for k in range(1, 4):
            t = k / 4
            y = int(ay + (fondo - ay) * t)
            xi = int(izq + (abajo_izq - izq) * t)
            xd = int(der + (abajo_der - der) * t)
            cv2.line(frame, (xi, y), (xd, y), NEGRO, grosor, cv2.LINE_AA)
        cv2.ellipse(frame, (self.w // 2, ay), ((der - izq) // 2, int(10 * s)), 0, 0, 360, NEGRO,
                    max(3, int(7 * s)), cv2.LINE_AA)

    def dibujar_pelota(self, frame, p):
        if not p.activa:
            return
        c, r = (int(p.x), int(p.y)), int(p.r)
        if p.bomba:
            self.dibujar_bomba(frame, c, r)
            return
        cv2.circle(frame, c, r, NARANJA, -1, cv2.LINE_AA)
        a = p.angulo
        for ang in (a, a + math.pi / 2):
            dx, dy = int(math.cos(ang) * r), int(math.sin(ang) * r)
            cv2.line(frame, (c[0] - dx, c[1] - dy), (c[0] + dx, c[1] + dy), NEGRO, 2, cv2.LINE_AA)
        grados = math.degrees(a)
        cv2.ellipse(frame, c, (int(r * 0.55), r), grados, 0, 360, NEGRO, 2, cv2.LINE_AA)
        cv2.circle(frame, c, r, NEGRO, 3, cv2.LINE_AA)

    def dibujar_bomba(self, frame, c, r):
        parpadeo = int(time.time() * 6) % 2 == 0
        cv2.circle(frame, c, r, (25, 25, 25), -1, cv2.LINE_AA)
        cv2.circle(frame, (c[0] - r // 3, c[1] - r // 3), max(2, r // 5), (110, 110, 110), -1, cv2.LINE_AA)
        cv2.circle(frame, c, r, (0, 0, 255) if parpadeo else NEGRO, 3, cv2.LINE_AA)
        mecha = (c[0] + int(r * 0.5), c[1] - int(r * 1.15))
        cv2.line(frame, (c[0] + int(r * 0.35), c[1] - int(r * 0.85)), mecha, (60, 120, 160), 4, cv2.LINE_AA)
        cv2.circle(frame, mecha, max(3, r // 5), (0, 200, 255) if parpadeo else (0, 80, 255), -1, cv2.LINE_AA)

    def dibujar_potenciador(self, frame):
        pot = self.potenciador
        if pot is None:
            return
        s = self.s
        r = int(38 * s * (1 + 0.08 * math.sin(time.time() * 8)))
        c = (int(pot.x), int(pot.y))
        if pot.vida < 2 and int(time.time() * 8) % 2 == 0:
            return
        if pot.tipo == "x2":
            cv2.circle(frame, c, r, DORADO, -1, cv2.LINE_AA)
            cv2.circle(frame, c, r, BLANCO, 3, cv2.LINE_AA)
            self.lienzo.texto("x2", c, int(34 * s), BLANCO, centrado=True)
            self.lienzo.texto("Puntos x2", (c[0], c[1] + int(58 * s)), int(20 * s), DORADO, centrado=True)
        else:
            cv2.circle(frame, c, r, VIOLETA, -1, cv2.LINE_AA)
            cv2.circle(frame, c, r, BLANCO, 3, cv2.LINE_AA)
            k = r / 30
            rayo = np.array([(4, -24), (-12, 4), (0, 4), (-6, 24), (12, -6), (0, -6), (6, -24)], np.float32)
            pts = (rayo * k + np.array(c, np.float32)).astype(np.int32)
            cv2.fillPoly(frame, [pts], (0, 230, 255), cv2.LINE_AA)
            self.lienzo.texto("Rival rápido", (c[0], c[1] + int(58 * s)), int(20 * s), VIOLETA, centrado=True)

    def dibujar_explosiones(self, frame, dt):
        vivas = []
        for e in self.explosiones:
            e[2] += dt
            t = e[2] / 0.6
            if t >= 1:
                continue
            vivas.append(e)
            c = (int(e[0]), int(e[1]))
            radio = int((40 + 160 * t) * self.s)

            def pintar(img, c=c, radio=radio):
                cv2.circle(img, c, radio, (0, 120, 255), -1, cv2.LINE_AA)
                cv2.circle(img, c, int(radio * 0.6), (0, 220, 255), -1, cv2.LINE_AA)

            capa(frame, pintar, 0.75 * (1 - t))
            if t < 0.6:
                self.lienzo.texto("BOOM", c, int(48 * self.s), (0, 80, 255), borde=5, centrado=True)
        self.explosiones = vivas

    def dibujar_manos(self, frame, personas, manos):
        def pintar(img):
            for j, x, y, _, _ in manos:
                cv2.circle(img, (int(x), int(y)), self.radio_mano, COLOR_MANO, -1, cv2.LINE_AA)

        capa(frame, pintar, 0.45)
        for j, lm in personas.items():
            nariz = lm[0]
            self.lienzo.texto(f"J{j + 1}", (int(nariz.x * self.w), max(30, int(nariz.y * self.h - 110 * self.s))),
                              34, COLOR_JUGADOR[j], centrado=True)

    def dibujar_marcador(self, frame):
        lz, s, w = self.lienzo, self.s, self.w
        tam = int(26 * s)
        for i, j in enumerate(self.jugadores):
            texto = f"Jugador {i + 1}  ·  {j['puntos']} pts"
            if i == 0:
                lz.texto(texto, (int(30 * s), int(10 * s)), int(32 * s), COLOR_JUGADOR[0])
                x0 = int(30 * s + tam / 2)
            else:
                lz.texto(texto, (int(w - 30 * s), int(10 * s)), int(32 * s), COLOR_JUGADOR[1], derecha=True)
                x0 = int(w - 30 * s - tam / 2 - (VIDAS - 1) * tam * 1.15)
            for k in range(VIDAS):
                color = (40, 40, 220) if k < j["vidas"] else GRIS
                corazon(frame, int(x0 + k * tam * 1.15), int(70 * s), tam, color)
            efectos = []
            if self.x2[i] > 0:
                efectos.append((f"x2 puntos · {math.ceil(self.x2[i])} s", DORADO))
            if self.rapido[i] > 0:
                efectos.append((f"pelotas rápidas · {math.ceil(self.rapido[i])} s", VIOLETA))
            for n, (txt, color) in enumerate(efectos):
                y = int((95 + 32 * n) * s)
                if i == 0:
                    lz.texto(txt, (int(30 * s), y), int(22 * s), color, fondo=(0, 0, 0))
                else:
                    lz.texto(txt, (int(w - 30 * s), y), int(22 * s), color, derecha=True, fondo=(0, 0, 0))

    def dibujar_textos(self, personas):
        lz, w, h, s = self.lienzo, self.w, self.h, self.s
        if self.estado == "espera":
            lz.texto("Esperando a los dos jugadores", (w // 2, int(h * 0.5)), int(44 * s), centrado=True,
                     fondo=(0, 0, 0))
            for j in (0, 1):
                listo = j in personas
                texto = f"Jugador {j + 1}: {'¡listo!' if listo else 'parate de este lado'}"
                color = (120, 255, 0) if listo else (200, 200, 200)
                lz.texto(texto, (w // 4 if j == 0 else 3 * w // 4, int(h * 0.62)), int(30 * s), color,
                         centrado=True, fondo=(0, 0, 0))
        elif self.estado == "cuenta":
            n = CUENTA_ATRAS - int(self.t_estado)
            lz.texto("¡Prepárense!", (w // 2, int(h * 0.42)), int(40 * s), centrado=True)
            lz.texto(str(max(n, 1)), (w // 2, int(h * 0.6)), int(180 * s), (120, 255, 0), borde=8, centrado=True)
        elif self.estado == "fin":
            g = self.ganador
            lz.texto(f"¡Ganó el Jugador {g + 1}!", (w // 2, int(h * 0.55)), int(70 * s), COLOR_JUGADOR[g],
                     borde=6, centrado=True, fondo=(0, 0, 0))
            lz.texto("R para jugar de nuevo  ·  Q para salir", (w // 2, int(h * 0.68)), int(30 * s), centrado=True)
        if self.t_mensaje > 0 and self.estado == "juego":
            lz.texto(self.mensaje, (w // 2, int(h * 0.5)), int(40 * s), self.color_mensaje, borde=4,
                     centrado=True)
        if self.estado == "juego" and not personas:
            lz.texto("No se ve a ningún jugador", (w // 2, int(h * 0.62)), int(30 * s), centrado=True,
                     fondo=(0, 0, 0))
        lz.texto(f"Gana quien llegue a {PUNTOS_PARA_GANAR} canastas  ·  Si la pelota cae en tu lado, perdés una vida"
                 "  ·  ¡No toques la pelota negra!  ·  R reinicia  ·  Q sale", (w // 2, h - int(16 * s)), int(17 * s), borde=2, centrado=True)

    def paso(self, frame, ts_ms, dt):
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        self.geometria(w, h)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
        personas = self.personas(res)
        manos = self.manos(personas, max(dt, 1e-3))
        self.actualizar(personas, manos, dt)
        self.dibujar_piso(frame)
        cv2.line(frame, (w // 2, self.piso), (w // 2, h), BLANCO, 2)
        self.dibujar_tablero(frame)
        for p in self.pelotas:
            self.dibujar_pelota(frame, p)
        self.dibujar_aro(frame)
        self.dibujar_potenciador(frame)
        self.dibujar_explosiones(frame, dt)
        self.dibujar_manos(frame, personas, manos)
        self.dibujar_marcador(frame)
        self.dibujar_textos(personas)
        return self.lienzo.dibujar(frame)

    def tecla(self, k):
        if k in (27, ord("q")):
            return False
        if k in (ord("r"), ord(" ")):
            self.reiniciar()
        return True

    def correr(self):
        cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(VENTANA, 1280, 720)
        inicio = previo = time.time()
        ultimo_ts = -1
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("No llegan imágenes de la cámara.")
                    break
                ahora = time.time()
                dt = min(max(ahora - previo, 1e-3), 0.1)
                previo = ahora
                ts = max(int((ahora - inicio) * 1000), ultimo_ts + 1)
                ultimo_ts = ts
                cv2.imshow(VENTANA, self.paso(frame, ts, dt))
                k = cv2.waitKey(1)
                if k != -1 and not self.tecla(k & 0xFF):
                    break
                if cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            try:
                self.detector.close()
            except Exception:
                pass


if __name__ == "__main__":
    os.chdir(CARPETA)
    Juego().correr()
